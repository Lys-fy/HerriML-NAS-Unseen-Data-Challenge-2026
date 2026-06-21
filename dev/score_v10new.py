#!/usr/bin/env python3
"""Score the v10 3-seed runs on the 3 new datasets vs their ResNet-18 benchmark.
adj = (raw - benchmark) * 10 / (100 - benchmark), capped at -10. Run on spark."""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

DD = Path.home() / "local/nas-comp-datasets"
REPO = Path.home() / "projects/HerriML-NAS-Unseen-Data-Challenge-2026"
CODES = ["Sedol", "Sloan", "Cauchy"]
SEEDS = [42, 1337, 2024]


def adj(raw, bench):
    if bench >= 100:
        return 0.0
    return max(-10.0, (raw - bench) * 10.0 / (100.0 - bench))


print(f"{'dataset':8s} {'bench':>7s} {'s42':>7s} {'s1337':>7s} {'s2024':>7s} "
      f"{'mean':>7s} {'adj':>7s}")
totals = []
for code in CODES:
    meta = json.loads((DD / code / "metadata").read_text())
    bench = float(meta.get("benchmark", 0.0))
    ty = np.load(DD / code / "test_y.npy")
    accs = []
    for s in SEEDS:
        pf = REPO / f"dev/predictions-v10new-s{s}" / f"{code}.npy"
        if not pf.exists():
            accs.append(None); continue
        pr = np.load(pf); n = min(len(pr), len(ty))
        accs.append(100.0 * accuracy_score(ty[:n], pr[:n]))
    good = [a for a in accs if a is not None]
    m = float(np.mean(good)) if good else float("nan")
    cells = " ".join(f"{a:7.2f}" if a is not None else "   -   " for a in accs)
    a = adj(m, bench)
    totals.append(a)
    print(f"{code:8s} {bench:7.2f} {cells} {m:7.2f} {a:+7.2f}")
print(f"\nmean adjusted over the 3 new datasets: {np.mean(totals):+.2f}")
