#!/usr/bin/env python3
"""Surplus-budget accuracy test: Model Soups + logit ensembling + flip-TTA, as a
WRAPPER around the unmodified v10 Trainer (no submission-code surgery).

From the research scout's top-payoff recommendations (arXiv 2203.05482 Model
Soups; ensembling): spend leftover budget by training K runs of the SAME default
architecture and combining them. We compare, on held-out TEST:
  single_best     best-val single model (baseline)
  soup_uniform    average the K weight tensors (Model Soups, uniform)
  soup_greedy     greedy soup: add models in val-desc order only if val improves
  logit_ensemble  average softmax over the K models
  *_tta           each of the above with horizontal-flip test-time augmentation

IMPORTANT caveat (honest): weight-averaging independently-INITIALISED nets fails
(different loss basins / neuron permutations). So all K seeds SHARE the same
weight init (same construction seed) and differ only in training trajectory
(data order / dropout / augmentation) -- the regime where soups are valid. Flip-
TTA only makes sense for natural images; for board/sequence datasets (Chesseract,
Language) a flip is semantically wrong, so *_tta there is expected to HURT and is
reported only for completeness.

Reuses v10 build_cnn / DataProcessor / Trainer / Clock, like validate_winner.py.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
import trainer as trainer_mod                       # noqa: E402  -- to override its SEED
from helpers import build_cnn                        # noqa: E402
from data_processor import DataProcessor             # noqa: E402
from trainer import Trainer                           # noqa: E402

V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
            "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
            "Sloan": 84.60, "Cryptic": 72.08, "AddNIST": 96.33, "MultNIST": 95.52,
            "GeoClassing": 95.36, "GameOfLife": 100.0, "Myofibre": 91.09,
            "Voxel": 81.56, "Windspeed": 19.36}
INIT_SEED = 1234   # shared weight init across all K seeds (soup validity)


class Clock:
    def __init__(self, hours):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + hours * 3600

    def check(self):
        return self.time_limit - time.perf_counter()


def load_dataset(d):
    tr = (np.load(d / "train_x.npy"), np.load(d / "train_y.npy"))
    va = (np.load(d / "valid_x.npy"), np.load(d / "valid_y.npy"))
    test_x = np.load(d / "test_x.npy")
    test_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    return tr, va, test_x, test_y, meta


def fresh_model(in_ch, nc, h, w, width, blocks):
    torch.manual_seed(INIT_SEED)                     # SHARED init across seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INIT_SEED)
    return build_cnn(in_ch, nc, h, w, ops=["Conv3"] * blocks, base_width=width,
                     drop_path_rate=0.1)


@torch.no_grad()
def _logits(model, loader, device, tta):
    model.eval(); outs = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device, memory_format=torch.channels_last)
        lo = model(xb).softmax(1)
        if tta:
            lo = lo + model(torch.flip(xb, dims=[3])).softmax(1)
        outs.append(lo.cpu())
    return torch.cat(outs)


def acc_of(prob, y):
    n = min(len(prob), len(y))
    return round(100.0 * float((prob[:n].argmax(1).numpy() == y[:n]).mean()), 3)


def avg_states(states):
    """Average floating-point tensors elementwise; keep non-float buffers from #0."""
    out = copy.deepcopy(states[0])
    for k, v in out.items():
        if v.is_floating_point():
            out[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours-per-seed", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--width", type=int, default=48)
    args = ap.parse_args()

    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name); nc = int(meta["num_classes"])
    ref = V10_REAL.get(code) or V10_REAL.get(d.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # one DataProcessor / loaders reused across seeds (a full budget-clock per seed)
    clock0 = Clock(args.hours_per_seed * args.k + 0.2)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock0)
    train_loader, valid_loader, test_loader = dp.process()
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]

    print(f"[{code}] SOUP test K={args.k} x {args.hours_per_seed}h  nc={nc}  v10={ref}", flush=True)
    states, val_accs = [], []
    for k in range(args.k):
        model = fresh_model(in_ch, nc, h, w, args.width, args.blocks)
        trainer_mod.SEED = INIT_SEED + 1 + k          # vary trajectory only
        clock = Clock(args.hours_per_seed)
        tr_ = Trainer(model, device, train_loader, valid_loader, meta, clock)
        tr_.train()                                   # restores best_state into model
        states.append({kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()})
        val_accs.append(tr_.best_val_acc)
        print(f"  seed {k}: best_val={tr_.best_val_acc:.4f}", flush=True)

    order = sorted(range(args.k), key=lambda i: -val_accs[i])   # val-desc

    def build_and_load(state):
        m = fresh_model(in_ch, nc, h, w, args.width, args.blocks).to(device, memory_format=torch.channels_last)
        m.load_state_dict(state); return m

    # greedy soup: start from best-val, add a model only if val (no TTA) improves
    greedy_idx = [order[0]]
    greedy_state = states[order[0]]
    greedy_val = acc_of(_logits(build_and_load(greedy_state), valid_loader, device, False),
                        va[1].astype(np.int64))
    for i in order[1:]:
        cand = avg_states([states[j] for j in greedy_idx + [i]])
        v = acc_of(_logits(build_and_load(cand), valid_loader, device, False), va[1].astype(np.int64))
        if v >= greedy_val:
            greedy_idx.append(i); greedy_state = cand; greedy_val = v

    uniform_state = avg_states(states)
    models = [build_and_load(s) for s in states]

    def report(prob):
        return dict(acc=acc_of(prob, test_y)) if test_y is not None else dict(acc=None)

    res = {}
    if test_y is not None:
        ty = test_y.astype(np.int64)
        best_i = order[0]
        for tta in (False, True):
            suf = "_tta" if tta else ""
            single = _logits(models[best_i], test_loader, device, tta)
            uni = _logits(build_and_load(uniform_state), test_loader, device, tta)
            grd = _logits(build_and_load(greedy_state), test_loader, device, tta)
            ens = sum(_logits(m, test_loader, device, tta) for m in models) / len(models)
            res[f"single_best{suf}"] = acc_of(single, ty)
            res[f"soup_uniform{suf}"] = acc_of(uni, ty)
            res[f"soup_greedy{suf}"] = acc_of(grd, ty)
            res[f"logit_ensemble{suf}"] = acc_of(ens, ty)

    rec = {"codename": code, "dataset": d.name, "pipeline": "soup_ab", "k": args.k,
           "hours_per_seed": args.hours_per_seed, "val_accs": [round(v, 4) for v in val_accs],
           "greedy_members": len(greedy_idx), "v10_real_test_acc": ref, "results": res}
    (out / f"{code}_soup.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] SOUP DONE  v10={ref}  {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
