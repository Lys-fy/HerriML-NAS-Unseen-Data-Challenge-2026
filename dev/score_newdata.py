#!/usr/bin/env python3
"""Score the 5 custom datasets: v10 (3 seeds) and the ResNet-18 baseline,
test accuracy + adjusted score using ResNet-18 as the zero-point (the custom
datasets ship benchmark=0.0). Run on spark (needs predictions + test_y)."""
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

DD = Path.home() / "local/nas-comp-datasets"
REPO = Path.home() / "projects/HerriML-NAS-Unseen-Data-Challenge-2026"
CODES = ["Quincy", "Magnus", "Cornelius", "Pavel", "Dominik"]
SEEDS = ["42", "1337", "2024"]


def raw(pred_dir, code):
    pf = REPO / pred_dir / f"{code}.npy"
    if not pf.exists():
        return None
    ty = np.load(DD / code / "test_y.npy")
    pr = np.load(pf)
    return 100 * accuracy_score(ty[:len(pr)], pr)


hdr = f"{'dataset':10s} {'resnet':>7s} {'v10_42':>7s} {'v10_1337':>8s} {'v10_2024':>8s} {'v10mean':>8s}  adj(resnet zero-point)"
print(hdr)
tot = [0.0, 0.0, 0.0]
for c in CODES:
    rn = raw("dev/predictions-resnet18-newdata-s42", c)
    vs = [raw(f"dev/predictions-v10nd-v10_s{s}", c) for s in SEEDS]
    vm = np.mean([v for v in vs if v is not None]) if any(v is not None for v in vs) else float("nan")
    adj = [((v - rn) * 10 / max(100 - rn, 1e-6)) if (v is not None and rn is not None) else None for v in vs]
    for i, a in enumerate(adj):
        if a is not None:
            tot[i] += a
    astr = "  ".join(f"{a:+.2f}" if a is not None else "  -  " for a in adj)
    rns = f"{rn:7.2f}" if rn is not None else "   -   "
    vss = "  ".join(f"{v:7.2f}" if v is not None else "   -   " for v in vs)
    print(f"{c:10s} {rns} {vss} {vm:8.2f}  {astr}")
print(f"\ntotal adj per seed (resnet zero-point): {[round(t,2) for t in tot]}")
print(f"mean total: {np.mean(tot):+.2f} ± {np.std(tot):.2f}")
