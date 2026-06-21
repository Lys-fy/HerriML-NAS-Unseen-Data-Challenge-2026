#!/usr/bin/env python3
"""v11.2: EGNAS-style heterogeneous micro-macro NAS with verify-before-adopt.

EGNAS searches depth/width/per-layer-op over CONV blocks by a greedy hill-climb,
trusting short-train rankings. v11.2 keeps EGNAS's hill-climb STRUCTURE but (1)
the per-stage op vocabulary spans ALL families (conv, dilated/TCN, separable,
squeeze-excite, attention) -> genuinely heterogeneous stages, (2) the depth range
goes DEEP (the cap that cost us CIFARTile), and (3) every move is committed only
after a VERIFY re-train, never on the short-train ranking (our reliability fix for
EGNAS's proxy-wall weakness).

The search is a coordinate-descent hill-climb that WALKS the huge space rather
than sampling it:  depth (macro, deep) -> per-stage op (micro, heterogeneous) ->
width (macro). At each decision: short-train the small candidate set to SCREEN
(cheap rung), then VERIFY the winner at a longer horizon and adopt only if it
beats the incumbent beyond noise (expensive rung). Finally the searched arch is
compared to the v10-route family-default arch -- adopted only if it wins, so v11.2
can never lose to v10 on a fair test.

Epoch-based (hardware-agnostic: same arch+epochs ranks identically on any GPU).
Reuses v10's build_cnn / OP_VOCAB / DataProcessor; the search + training are new.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- reuse only the v10 model builder + data pipeline (not test code) --------
V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from helpers import build_cnn, OP_VOCAB, build_tcn   # noqa: E402
from data_processor import DataProcessor              # noqa: E402

# Per-stage candidate ops = the PROVEN v10 2-D building blocks + the two
# heterogeneity levers. DilatedConv3 IS the tcn2d primitive. Conv5/Conv7/
# GroupConv3 dropped (redundant); ResNeXt dropped (the "modern" arm, gauntlet
# loser, CIFARTile -25); LSTMs excluded (a short proxy can't rank them -> v11.3).
# tcn1d is NOT here: it flattens H*W to a 1-D sequence and cannot be one stage
# among 2-D stages -- it is a whole-NETWORK family, compared at the end instead.
OP_CANDS = ["Conv3", "DilatedConv3", "SepConv3", "SEConv3", "SpatialSelfAttn2D"]
DEPTHS = [6, 10, 14, 20]                          # macro: includes DEEP
WIDTHS = [32, 48, 64]
STEM_EMBED_MIN_CH = 8


class Clock:
    def __init__(self, hours):
        self.t0 = time.perf_counter(); self.limit = self.t0 + hours * 3600

    def check(self):
        return self.limit - time.perf_counter()


def load_dataset(d):
    tr = (np.load(d / "train_x.npy"), np.load(d / "train_y.npy"))
    va = (np.load(d / "valid_x.npy"), np.load(d / "valid_y.npy"))
    test_x = np.load(d / "test_x.npy")
    test_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    return tr, va, test_x, test_y, meta


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(device, non_blocking=True)).argmax(1).cpu()
        correct += int((pred == yb).sum()); total += int(yb.numel())
    return correct / max(total, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(xb.to(device, non_blocking=True)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def noise_of(curve):
    tail = curve[-10:]
    if len(tail) < 3:
        return 0.02
    m = sum(tail) / len(tail)
    return float(np.sqrt(sum((v - m) ** 2 for v in tail) / len(tail)))


class Search:
    """Holds the data/context and trains arbitrary (ops, width) archs to a given
    epoch budget with v10's recipe (SGD + cosine + label smoothing + grad clip)."""

    def __init__(self, in_ch, nc, h, w, stem, loaders, device, cap_per_epoch=60.0):
        self.in_ch, self.nc, self.h, self.w = in_ch, nc, h, w
        self.stem, self.loaders, self.device = stem, loaders, device
        self.cap_per_epoch = cap_per_epoch
        self.curves = {}
        self.n_arms = 0
        self.arm_seconds = 0.0

    def _fit(self, model, epochs, key):
        """Train a prebuilt model for `epochs` epochs (cosine + v10 recipe);
        return (model, val_curve, secs, capped). Epoch-based -> portable."""
        tl, vl = self.loaders
        model = model.to(self.device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9,
                              weight_decay=3e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        vals, t0 = [], time.perf_counter(); cap = t0 + epochs * self.cap_per_epoch
        for _ in range(epochs):
            if time.perf_counter() > cap:
                break
            model.train()
            for xb, yb in tl:
                if time.perf_counter() > cap:
                    break
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vals.append(evaluate(model, vl, self.device)); sched.step()
        secs = time.perf_counter() - t0
        self.curves[key] = np.array(vals or [0.0], np.float32)
        self.n_arms += 1; self.arm_seconds += secs
        return model, (vals or [0.0]), secs, len(vals) < epochs

    def train(self, ops, width, epochs, key):
        """Build a heterogeneous build_cnn arch (per-layer ops) and train it."""
        model = build_cnn(self.in_ch, self.nc, self.h, self.w, ops=ops,
                          base_width=width, drop_path_rate=0.0, stem=self.stem)
        return self._fit(model, epochs, key)

    def train_tcn(self, kind, epochs, key):
        """Whole-network proven v10 family backbone (kind = 'tcn1d' / 'tcn2d')."""
        model = build_tcn(kind, self.in_ch, self.nc, self.h, self.w,
                          base_width=48, stem=self.stem)
        return self._fit(model, epochs, key)

    def score(self, ops, width, epochs, key):
        """Train and return (best_val, capped), freeing the model."""
        try:
            m, vals, secs, capped = self.train(ops, width, epochs, key)
            v = max(vals); del m
        except Exception as e:
            print(f"   arm FAILED ({key}): {e}", flush=True)
            v, capped = 0.0, False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return v, capped


def stage_bounds(depth, n_stages):
    n = max(1, min(n_stages, depth))
    return [round(i * depth / n) for i in range(n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--screen-epochs", type=int, default=12)   # cheap rung
    ap.add_argument("--verify-epochs", type=int, default=40)   # expensive rung (commit)
    ap.add_argument("--final-epochs", type=int, default=120)   # train-out the winner
    ap.add_argument("--n-stages", type=int, default=4)         # micro op-search granularity
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name)

    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, Clock(999))
    train_loader, val_loader, test_loader = dp.process()
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])
    stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
    encoded = bool(meta.get("fp_encoded", False))
    base_op = "Conv3" if encoded else "DilatedConv3"   # v10 family default
    S = Search(in_ch, nc, h, w, stem, (train_loader, val_loader), device)
    SE, VE, FE = args.screen_epochs, args.verify_epochs, args.final_epochs
    print(f"[{code}] device={device} encoded={encoded} stem={stem} base_op={base_op} "
          f"screen={SE} verify={VE} final={FE}", flush=True)

    def test_acc(ops, width, key):
        if test_y is None:
            return None, 0.0
        m, vals, secs, _ = S.train(ops, width, FE, key)
        pred = predict(m, test_loader, device)
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        n = min(len(pred), len(test_y))
        return (float((pred[:n] == test_y[:n]).mean()) if n else None, max(vals))

    trace = {"steps": []}

    # ---- incumbent: uniform family default at depth 6, verified -------------
    ops = [base_op] * 6; width = 48
    _, ivals, _, _ = S.train(ops, width, VE, "inc_init")
    inc_score = max(ivals); inc_noise = noise_of(ivals)
    print(f"[{code}] incumbent {base_op}x6 W48  verify_val={inc_score:.4f} noise={inc_noise:.4f}", flush=True)

    def try_moves(name, candidates):
        """candidates: list of dict(ops,width,tag). Screen all, verify the best,
        adopt iff verified > incumbent + noise. Updates ops/width/inc_score."""
        nonlocal ops, width, inc_score, inc_noise
        screened = []
        for j, c in enumerate(candidates):
            sv, cap = S.score(c["ops"], c["width"], SE, f"{name}_s{j}")
            screened.append((sv, c))
            print(f"[{code}]   {name} screen {c['tag']:20s} val={sv:.4f}{' CAP' if cap else ''}", flush=True)
        screened.sort(reverse=True, key=lambda x: x[0])
        best_sv, best_c = screened[0]
        m, vvals, _, _ = S.train(best_c["ops"], best_c["width"], VE, f"{name}_verify")
        verified = max(vvals); del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        adopt = verified > inc_score + inc_noise
        step = {"move": name, "n_candidates": len(candidates), "winner": best_c["tag"],
                "screen_best": round(best_sv, 4), "verified": round(verified, 4),
                "incumbent": round(inc_score, 4), "noise": round(inc_noise, 4),
                "adopted": adopt}
        trace["steps"].append(step)
        if adopt:
            ops, width = best_c["ops"], best_c["width"]
            inc_score, inc_noise = verified, noise_of(vvals)
        print(f"[{code}]   {name} -> {'ADOPT ' + best_c['tag'] if adopt else 'keep incumbent'} "
              f"(verified {verified:.4f} vs inc {step['incumbent']:.4f}+{step['noise']:.4f})", flush=True)

    # ---- 1. depth hill-climb (macro, deep) ---------------------------------
    try_moves("depth", [{"ops": [base_op] * D, "width": width, "tag": f"depth{D}"}
                        for D in DEPTHS if D != len(ops)])

    # ---- 2. per-stage op hill-climb (micro, heterogeneous) -----------------
    bounds = stage_bounds(len(ops), args.n_stages)
    for gi in range(len(bounds) - 1):
        a, b = bounds[gi], bounds[gi + 1]
        cur = ops[a]
        cands = []
        for op in OP_CANDS:
            if op == cur:
                continue
            trial = list(ops)
            for k in range(a, b):
                trial[k] = op
            cands.append({"ops": trial, "width": width, "tag": f"st{gi}_{op}"})
        try_moves(f"op_stage{gi}", cands)

    # ---- 3. width hill-climb (macro) ---------------------------------------
    try_moves("width", [{"ops": ops, "width": W, "tag": f"W{W}"}
                        for W in WIDTHS if W != width])

    # ---- 4. final: train-out the searched arch + the PROVEN whole-net family
    #         backbones (tcn1d / tcn2d) + the v10-route, then pick the best ------
    searched_test, searched_val = test_acc(ops, width, "final_searched")

    fam = {}
    for kind in ("tcn1d", "tcn2d"):
        try:
            m, vals, _, _ = S.train_tcn(kind, FE, f"final_{kind}")
            pred = predict(m, test_loader, device); del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            n = min(len(pred), len(test_y)) if test_y is not None else 0
            fam[kind] = {"final_val": round(max(vals), 4),
                         "test_acc": (float((pred[:n] == test_y[:n]).mean()) if n else None)}
            print(f"[{code}] family {kind} backbone  val={fam[kind]['final_val']:.4f} "
                  f"test={fam[kind]['test_acc']}", flush=True)
        except Exception as e:
            print(f"[{code}] {kind} backbone FAILED: {e}", flush=True)
            fam[kind] = {"final_val": 0.0, "test_acc": None}

    m, v10_vals, _, _ = S.train([base_op] * 6, 48, FE, "final_v10")
    v10_pred = predict(m, test_loader, device); del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    v10_val = max(v10_vals); v10_noise = noise_of(v10_vals)
    v10_test = None
    if test_y is not None:
        n = min(len(v10_pred), len(test_y)); v10_test = float((v10_pred[:n] == test_y[:n]).mean())

    # selection: best of {searched heterogeneous arch, tcn1d, tcn2d}; adopt iff it
    # beats the v10-route default beyond noise (so v11.2 can never lose to v10).
    finalists = {"searched": (searched_val, searched_test),
                 "tcn1d": (fam["tcn1d"]["final_val"], fam["tcn1d"]["test_acc"]),
                 "tcn2d": (fam["tcn2d"]["final_val"], fam["tcn2d"]["test_acc"])}
    winner = max(finalists, key=lambda k: finalists[k][0])
    win_val, win_test = finalists[winner]
    adopt = win_val > v10_val + v10_noise

    rec = {"codename": code, "dataset": d.name, "device": str(device),
           "search_mode": "v11.2_hillclimb_verify", "fp_corr1": meta.get("fp_corr1"),
           "fp_encoded": encoded, "in_ch": in_ch, "h": h, "w": w, "num_classes": nc,
           "stem": stem, "base_op": base_op, "seed": args.seed,
           "epochs": {"screen": SE, "verify": VE, "final": FE},
           "n_stages": args.n_stages, "trace": trace["steps"],
           "searched": {"ops": ops, "width": width, "depth": len(ops),
                        "heterogeneous": len(set(ops)) > 1,
                        "final_val": round(searched_val, 4), "test_acc": searched_test},
           "family_alternatives": fam,
           "v10_route": {"ops": [base_op] * 6, "width": 48, "final_val": round(v10_val, 4),
                         "noise": round(v10_noise, 4), "test_acc": v10_test},
           "decision": {"winner": winner, "winner_val": round(win_val, 4),
                        "winner_test_acc": win_test, "adopt_over_v10": bool(adopt),
                        "margin_val": round(win_val - v10_val, 4),
                        "rule": "winner = best of {searched,tcn1d,tcn2d}; adopt iff > v10 + noise"},
           "timing": {"device": str(device), "search_seconds": round(S.arm_seconds, 1),
                      "n_arms": S.n_arms}}
    (out / f"{code}.json").write_text(json.dumps(rec, indent=2))
    np.savez_compressed(out / f"{code}_curves.npz", **S.curves)
    print(f"[{code}] DONE  winner={winner} val={win_val:.4f}  vs v10 {v10_val:.4f} "
          f"(margin {win_val-v10_val:+.4f}, adopt={adopt})  searched={ops} W{width}  "
          f"test win/v10={win_test}/{v10_test}  search={S.arm_seconds/60:.0f}min "
          f"({S.n_arms} arms)", flush=True)


if __name__ == "__main__":
    main()
