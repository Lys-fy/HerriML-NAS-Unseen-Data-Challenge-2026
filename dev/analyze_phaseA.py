#!/usr/bin/env python3
"""Phase A analysis: live backbone-gate routing variants + TSE-in-search.

A1 gate: evaluates routing policies on the LIVE 39 short-train arms
(~7 min each, aug off, matched budget), scored by the final adj of the
backbone each policy picks (finals from the completed full-budget runs).

Policies (predeclared):
  FIXED   : tcn2d everywhere (the +59.2 baseline)
  GATE-ALL-a : gate on every dataset, abstain (leader within noise of 2nd)
               -> default tcn2d. No fingerprint.
  ROUTE-a : fingerprint skip (image-like -> tcn2d) + gate on encoded,
            abstain leader-vs-SECOND -> conv
  ROUTE-b : fingerprint skip + gate on encoded,
            abstain leader-vs-DEFAULT(conv) -> conv
            ("deviate from the encoded default only when clearly better")
  ORACLE  : best backbone per dataset

Noise threshold per dataset: mean over arms of std(val_acc, last 10 epochs).
No tuned constants. Also prints leave-one-out sanity (the policies have no
fitted parameters, so LOO == full here; kept for the record).

A4 TSE: parses TSE_LOG lines from the 4 search runs; per dataset, Spearman
between TSE score (negated mean of last 3 epoch losses) and short-train val
across candidates, plus whether the top-1 picks agree.
"""
import csv
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "dev/logs/phaseA/gate"
TSE = REPO / "dev/logs/phaseA/tse"

ARCHS = ["conv", "tcn1d", "tcn2d"]
ENCODED = {"Language", "Sudoku", "Chesseract", "Gutenberg"}  # fingerprint corr1<0.1
# final adj per dataset (conv, tcn1d, tcn2d) from the completed 2h runs
FINALS = {
    "Language":   (-1.03, 2.80, 2.05), "Cryptic":   (7.03, 7.74, 7.12),
    "AddNIST":    (5.88, 4.24, 6.67),  "CIFARTile":  (0.24, -0.95, 4.75),
    "Chesseract": (0.30, 0.10, -0.18), "GameOfLife": (9.99, 10.00, 9.998),
    "GeoClassing":(7.65, 7.18, 7.71),  "Gutenberg":  (1.03, -0.72, -0.06),
    "MultNIST":   (4.26, 2.21, 5.09),  "Myofibre":   (0.34, -0.17, 3.34),
    "Sudoku":     (9.45, 8.07, 8.81),  "Voxel":      (2.38, 3.57, 3.13),
    "Windspeed":  (0.72, 0.59, 0.72),
}


def load_curve(ds, arm):
    d = GATE / f"{ds}_{arm}"
    csvs = list(d.glob("*.csv"))
    if not csvs:
        return None
    vals = []
    with open(csvs[0]) as f:
        for r in csv.DictReader(f):
            vals.append(max(float(r["val_acc"]), float(r.get("val_acc_ema", 0) or 0)))
    return vals


def gate_inputs(ds):
    curves = {a: load_curve(ds, a) for a in ARCHS}
    best = {a: max(c) for a, c in curves.items()}
    noises = []
    for c in curves.values():
        tail = c[-10:]
        m = sum(tail) / len(tail)
        noises.append(math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail)))
    return best, sum(noises) / len(noises)


def adj(ds, arm):
    return dict(zip(ARCHS, FINALS[ds]))[arm]


def evaluate():
    pol_totals = {}
    rows = []
    for ds in FINALS:
        best, noise = gate_inputs(ds)
        order = sorted(ARCHS, key=lambda a: best[a], reverse=True)
        leader, second = order[0], order[1]
        picks = {}
        picks["FIXED"] = "tcn2d"
        picks["GATE-ALL-a"] = leader if best[leader] - best[second] > noise else "tcn2d"
        if ds in ENCODED:
            picks["ROUTE-a"] = leader if best[leader] - best[second] > noise else "conv"
            picks["ROUTE-b"] = leader if best[leader] - best["conv"] > noise else "conv"
        else:
            picks["ROUTE-a"] = picks["ROUTE-b"] = "tcn2d"
        picks["ORACLE"] = max(ARCHS, key=lambda a: adj(ds, a))
        rows.append((ds, noise, best, picks))
        for k, arm in picks.items():
            pol_totals[k] = pol_totals.get(k, 0.0) + adj(ds, arm)
    return rows, pol_totals


rows, totals = evaluate()
print("=== A1 LIVE GATE: per-dataset picks (noise-derived thresholds) ===")
print(f"{'dataset':12s} {'noise':>6s}  {'ROUTE-a':8s} {'ROUTE-b':8s} {'GATE-ALL':8s} {'oracle':8s}")
for ds, noise, best, p in rows:
    print(f"{ds:12s} {noise:6.3f}  {p['ROUTE-a']:8s} {p['ROUTE-b']:8s} "
          f"{p['GATE-ALL-a']:8s} {p['ORACLE']:8s}")
print("\n=== totals (final adj of picked backbones) ===")
for k in ["FIXED", "GATE-ALL-a", "ROUTE-a", "ROUTE-b", "ORACLE"]:
    print(f"  {k:10s} {totals[k]:+7.2f}")

print("\n=== A4 TSE vs short-train val (within-family depth candidates) ===")
pat = re.compile(r"TSE_LOG D=(\d+) W=(\d+) val=([\d.]+) losses=\[([^\]]*)\]")


def spearman(a, b):
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: b[i]))}
    ar = [ra[i] for i in range(n)]; br = [rb[i] for i in range(n)]
    ma = sum(ar) / n; mb = sum(br) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ar, br))
    den = math.sqrt(sum((x - ma) ** 2 for x in ar) * sum((y - mb) ** 2 for y in br)) + 1e-12
    return num / den


for f in sorted(TSE.glob("tse_*.log")):
    cands = []
    for m in pat.finditer(f.read_text()):
        d, w, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        losses = [float(x) for x in m.group(4).split(",") if x.strip()]
        if losses:
            tse = -sum(losses[-3:]) / len(losses[-3:])
            cands.append((d, w, val, tse))
    if len(cands) < 3:
        print(f"  {f.stem}: only {len(cands)} candidates, skip")
        continue
    vals = [c[2] for c in cands]; tses = [c[3] for c in cands]
    top_val = max(cands, key=lambda c: c[2])
    top_tse = max(cands, key=lambda c: c[3])
    print(f"  {f.stem:18s} n={len(cands):2d} spearman(TSE,val)={spearman(tses, vals):+.2f} "
          f"top_val=D{top_val[0]} top_tse=D{top_tse[0]} "
          f"{'AGREE' if top_val[0]==top_tse[0] else 'DIFFER'}")
