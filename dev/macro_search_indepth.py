#!/usr/bin/env python3
"""In-depth per-stage macro search with exhaustive logging (v11 development).

For ONE dataset, this:
  1. builds the uniform-incumbent baseline (all stages = DilatedConv3, the tcn2d
     primitive and v10's strongest general op) and trains it at the arm budget;
  2. scans EVERY (stage, candidate-op) trial -- logging each trial's curve,
     best val, and delta vs baseline, not just the winners (this is the
     stage x op utility map the analysis needs);
  3. greedily adopts per stage with default-incumbent + verify-before-adopt
     (a swap is kept only if it beats the incumbent beyond noise at short-train
     AND survives a longer verification re-train);
  4. trains the final per-stage arch and records its test accuracy.

Everything lands in <out>/<codename>.json (+ <codename>_curves.npz). Reuses the
v10 model zoo (build_cnn / OP_VOCAB) and DataProcessor, so archs and data
preprocessing are identical to the pipeline. Self-contained train/eval loop
(mirrors v10's _timed_train) so there is no hidden coupling to NAS internals.

Run ONE dataset (SLURM-array friendly); dev/run_macro_search.sh loops all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- reuse the v10 model zoo + data pipeline --------------------------------
V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from helpers import build_cnn, OP_VOCAB          # noqa: E402
from data_processor import DataProcessor          # noqa: E402

# --- search constants (match v10) -------------------------------------------
DEFAULT_OP = "DilatedConv3"      # uniform incumbent = tcn2d primitive
DEFAULT_DEPTH = 6
DEFAULT_WIDTH = 48
STEM_EMBED_MIN_CH = 8
# curated candidate set: the three v10 families + conv variants + attentions +
# a recurrent op (flagged risky). DilatedConv3 (= tcn2d) is the incumbent.
DEFAULT_CANDIDATES = ["DilatedConv3", "Conv3", "Conv5", "SepConv3",
                      "ResNeXt", "SEConv3", "SpatialSelfAttn2D", "RowBiLSTM"]


class Clock:
    def __init__(self, hours: float):
        self.start = time.perf_counter()
        self.time_limit = self.start + hours * 3600

    def check(self) -> float:
        return self.time_limit - time.perf_counter()


def load_dataset(d: Path):
    tr = (np.load(d / "train_x.npy"), np.load(d / "train_y.npy"))
    va = (np.load(d / "valid_x.npy"), np.load(d / "valid_y.npy"))
    test_x = np.load(d / "test_x.npy")
    test_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    return tr, va, test_x, test_y, meta


def train_arm(ops, in_ch, nc, h, w, width, stem, loaders, device,
              epochs, cap_s):
    """Train for a fixed number of EPOCHS with a COSINE LR schedule -- v10's real
    train-out recipe, not the fixed-LR quick probe. Epoch-based so the comparison
    is GPU-portable and fair across ops; cosine annealing so each arm actually
    reaches its potential (a fixed LR plateaus far below it on datasets that need
    annealing, e.g. Sudoku 34% -> ~90%). Same optimiser/augmentation-free setup
    as v10's trainer. `cap_s` is a wall-clock safety only. Returns per-epoch val
    series + timing."""
    train_loader, val_loader = loaders
    model = build_cnn(in_ch, nc, h, w, ops=ops, base_width=width,
                      drop_path_rate=0.0, stem=stem).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9,
                          weight_decay=3e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    vals, t0 = [], time.perf_counter()
    cap = t0 + cap_s
    for _ in range(epochs):
        if time.perf_counter() > cap:
            break
        model.train()
        for xb, yb in train_loader:
            if time.perf_counter() > cap:
                break
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN in macro-search train")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vals.append(_eval(model, val_loader, device))
        sched.step()
    elapsed = time.perf_counter() - t0
    return model, (vals or [0.0]), elapsed


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).argmax(1).cpu()
        correct += int((pred == yb).sum()); total += int(yb.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(xb.to(device, non_blocking=True)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _noise(vals):
    tail = vals[-10:]
    if len(tail) < 3:
        return 0.02
    m = sum(tail) / len(tail)
    return float(np.sqrt(sum((v - m) ** 2 for v in tail) / len(tail)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-epochs", type=int, default=150,
                    help="fixed epochs per arm, trained with a cosine LR schedule")
    ap.add_argument("--arm-cap-seconds", type=float, default=1800.0,
                    help="wall-clock safety cap per arm so nothing runs away")
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cand_ops = [o for o in args.candidates.split(",") if o]
    bad = [o for o in cand_ops if o not in OP_VOCAB]
    if bad:
        sys.exit(f"unknown ops {bad}; valid: {list(OP_VOCAB)}")

    d = Path(args.dataset_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (tr, va, test_x, test_y, meta) = load_dataset(d)
    code = meta.get("codename", d.name)
    print(f"[{code}] device={device} cosine({args.arm_epochs} epochs/arm) "
          f"candidates={cand_ops}", flush=True)

    # data pipeline identical to v10 (also stamps fp_corr1/fp_encoded into meta)
    clock = Clock(hours=999)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock)
    train_loader, val_loader, test_loader = dp.process()
    loaders = (train_loader, val_loader)

    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])
    stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
    D = args.depth
    n_st = max(1, min(args.stages, D))
    bounds = [round(i * D / n_st) for i in range(n_st + 1)]

    rec = {"codename": code, "dataset": d.name, "device": str(device),
           "fp_corr1": meta.get("fp_corr1"), "fp_encoded": meta.get("fp_encoded"),
           "in_ch": in_ch, "h": h, "w": w, "num_classes": nc, "stem": stem,
           "depth": D, "width": args.width, "n_stages": n_st, "bounds": bounds,
           "default_op": DEFAULT_OP, "candidates": cand_ops,
           "arm_epochs": args.arm_epochs, "lr_schedule": "cosine",
           "arm_cap_seconds": args.arm_cap_seconds, "seed": args.seed,
           "baseline": None, "trials": [], "adoptions": [], "final": None}
    curves = {}

    def test_acc(model):
        if test_y is None:
            return None
        pred = _predict(model, test_loader, device)
        n = min(len(pred), len(test_y))
        return float((pred[:n] == test_y[:n]).mean()) if n else None

    # ---- 1. baseline (uniform incumbent) -----------------------------------
    base_ops = [DEFAULT_OP] * D
    model, bvals, bt = train_arm(base_ops, in_ch, nc, h, w, args.width, stem,
                                 loaders, device, args.arm_epochs,
                                 args.arm_cap_seconds)
    inc = max(bvals); noise = _noise(bvals)
    rec["baseline"] = {"ops": base_ops, "best_val": round(inc, 4),
                       "noise": round(noise, 4), "epochs": len(bvals),
                       "seconds": round(bt, 1), "test_acc": test_acc(model)}
    curves["baseline"] = np.array(bvals, np.float32)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"[{code}] baseline best_val={inc:.4f} noise={noise:.4f}", flush=True)

    # ---- 2. full scan + 3. greedy verified adoption ------------------------
    stage_ops = list(base_ops)
    for gi in range(n_st):
        a, b = bounds[gi], bounds[gi + 1]
        cur = stage_ops[a]
        best_op, best_v = cur, inc
        for op in cand_ops:
            if op == cur:
                continue
            trial = list(stage_ops)
            for k in range(a, b):
                trial[k] = op
            try:
                m, vv, et = train_arm(trial, in_ch, nc, h, w, args.width, stem,
                                      loaders, device, args.arm_epochs,
                                      args.arm_cap_seconds)
                v = max(vv)
            except Exception as e:
                m, vv, et, v = None, [0.0], 0.0, 0.0
                print(f"[{code}] stage{gi} op={op} FAILED: {e}", flush=True)
            t = {"stage": gi, "op": op, "best_val": round(v, 4),
                 "delta_vs_baseline": round(v - inc, 4),
                 "beats_incumbent": bool(v > inc + noise),
                 "epochs": len(vv), "seconds": round(et, 1)}
            rec["trials"].append(t)
            curves[f"s{gi}_{op}"] = np.array(vv, np.float32)
            print(f"[{code}] stage{gi} op={op:18s} val={v:.4f} "
                  f"d={v-inc:+.4f}{'  WIN' if v > inc + noise else ''}", flush=True)
            if v > best_v + noise:
                best_op, best_v = op, v
            if m is not None:
                del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # verify-before-adopt the stage winner
        adopted = False
        if best_op != cur and best_v > inc + noise:
            trial = list(stage_ops)
            for k in range(a, b):
                trial[k] = best_op
            try:
                m, vvals, vt = train_arm(trial, in_ch, nc, h, w, args.width, stem,
                                         loaders, device, args.arm_epochs,
                                         args.arm_cap_seconds)
                verified = max(vvals)
            except Exception:
                m, verified = None, 0.0
            if verified > inc + noise:
                for k in range(a, b):
                    stage_ops[k] = best_op
                inc = verified
                adopted = True
            rec["adoptions"].append(
                {"stage": gi, "winner": best_op, "short_best": round(best_v, 4),
                 "verified": round(verified, 4), "adopted": adopted,
                 "verify_seconds": round(vt, 1)})
            if m is not None:
                del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[{code}] stage{gi} -> {stage_ops[a]} "
              f"({'ADOPTED ' + best_op if adopted else 'kept default'})", flush=True)

    # ---- 4. final arch, trained to convergence + test ----------------------
    model, fvals, ft = train_arm(stage_ops, in_ch, nc, h, w, args.width, stem,
                                 loaders, device, args.arm_epochs,
                                 args.arm_cap_seconds)
    rec["final"] = {"ops": stage_ops, "best_val": round(max(fvals), 4),
                    "delta_vs_baseline": round(max(fvals) - rec["baseline"]["best_val"], 4),
                    "heterogeneous": bool(any(o != DEFAULT_OP for o in stage_ops)),
                    "epochs": len(fvals), "seconds": round(ft, 1),
                    "test_acc": test_acc(model)}
    curves["final"] = np.array(fvals, np.float32)

    # timing summary: the SEARCH/decision cost (baseline + every scan arm + the
    # verifies) vs the final training. Per-arm seconds are already in trials[]/
    # baseline/adoptions; this aggregate lets us later estimate the deployment
    # budget at which running this search actually pays off. device is logged for
    # cross-GPU normalisation (wall-clock is GB10-specific).
    search_s = (rec["baseline"]["seconds"]
                + sum(t["seconds"] for t in rec["trials"])
                + sum(a.get("verify_seconds", 0.0) for a in rec["adoptions"]))
    rec["timing"] = {"device": str(device),
                     "search_seconds": round(search_s, 1),
                     "n_search_arms": 1 + len(rec["trials"]) + len(rec["adoptions"]),
                     "final_seconds": round(ft, 1),
                     "total_seconds": round(search_s + ft, 1)}
    print(f"[{code}] FINAL ops={stage_ops} val={max(fvals):.4f} "
          f"test={rec['final']['test_acc']}  search={search_s/60:.1f}min "
          f"final={ft/60:.1f}min", flush=True)

    (out / f"{code}.json").write_text(json.dumps(rec, indent=2))
    np.savez_compressed(out / f"{code}_curves.npz", **curves)
    print(f"[{code}] wrote {out / (code + '.json')}", flush=True)


if __name__ == "__main__":
    main()
