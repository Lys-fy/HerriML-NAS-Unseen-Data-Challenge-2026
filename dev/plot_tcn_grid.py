#!/usr/bin/env python3
"""conv vs TCN-1d vs TCN-2d learning curves, one panel per finished dataset.

v8 conv (the deployed backbone) vs the two TCN excursion variants, seed 42,
all at full 2 h budget. best-val solid, raw-val faint, benchmark dashed red
(skipped where the shipped benchmark is a 0.0 placeholder, e.g. Cryptic).
Style matches plot_tier_results.py.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/architecture"
OUT.mkdir(parents=True, exist_ok=True)

CONV = "#1f77b4"; T1 = "#2c8a3a"; T2 = "#ff7f0e"; BENCHC = "#b53030"

# name, codename, benchmark, final TEST raw % per arch (conv, tcn1d, tcn2d)
DATASETS = [
    ("Language",    "LaMelo",  85.200, (83.68, 89.34, 88.24)),
    ("Cryptic",     "Cryptic",  0.000, (70.27, 77.36, 71.18)),
    ("AddNIST",     "Adaline", 89.850, (95.82, 94.15, 96.62)),
    ("CIFARTile",   "Caitie",  47.008, (48.30, 41.99, 72.20)),
    ("Chesseract",  "Chester", 57.826, (59.10, 58.25, 57.08)),
    ("GameOfLife",  "conway",  47.530, (99.96, 100.00, 99.99)),
    ("GeoClassing", "Sadie",   80.330, (95.38, 94.46, 95.49)),
    ("Gutenberg",   "Gutenberg", 40.980, (47.08, 36.75, 40.62)),
    ("MultNIST",    "Mateo",   90.870, (94.76, 92.89, 95.52)),
    ("Myofibre",    "Myopia",  87.928, (88.33, 87.73, 91.96)),
    ("Sudoku",      "Sokoto",   0.000, (94.46, 80.68, 88.14)),
    ("Voxel",       "Volga",   71.350, (78.16, 81.58, 80.33)),
    ("Windspeed",   "windspeed", 13.492, (19.74, 18.59, 19.74)),
]


def load(p):
    ep, val, best = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(ep), np.array(val), np.array(best)


def paths(name, code):
    return {
        "conv":  REPO / f"dev/logs/v8/v8_s42/{code}.csv",
        "tcn1d": REPO / f"dev/logs/tcn_probe/tcn1d_{name}_s42/{code}.csv",
        "tcn2d": REPO / f"dev/logs/tcn_probe/tcn2d_{name}_s42/{code}.csv",
    }


fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
flat = axes.flatten()

for ax, (name, code, bench, raws) in zip(flat, DATASETS):
    pr = paths(name, code)
    series = [("conv", CONV, raws[0]), ("tcn1d", T1, raws[1]), ("tcn2d", T2, raws[2])]
    lo, hi = 100, 0
    for key, c, raw in series:
        if not pr[key].exists():
            continue
        ep, val, best = load(pr[key])
        ax.plot(ep, val, color=c, alpha=0.18, linewidth=0.9)
        ax.plot(ep, best, color=c, linewidth=2.0,
                label=f"{key:6s} {raw:.1f}%")
        lo, hi = min(lo, best.min()), max(hi, best.max())
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.1,
                   label=f"bench {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("val acc (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

# text summary in the first free cell; turn off the rest
SUMMARY_CELL = len(DATASETS)  # 13 -> flat[13]
ax = flat[SUMMARY_CELL]; ax.axis("off")
ax.text(0.02, 0.98,
        "conv vs TCN   seed 42, full 2h\n"
        "(solid = best-val, faint = raw-val)\n\n"
        "TOTAL adj over 13 datasets:\n"
        "  conv (v8)  +48.2\n"
        "  TCN-1d     +44.7\n"
        "  TCN-2d     +59.2   (best, +10.9)\n\n"
        "TCN-2d vs conv: win 6 / tie 4 / lose 3\n"
        "  big wins: CIFARTile +4.5,\n"
        "    Language +3.1, Myofibre +3.0\n"
        "  losses (all small/odd):\n"
        "    Gutenberg, Sudoku, Chesseract\n\n"
        "TCN-1d = sequence specialist\n"
        "  (Cryptic, Language, Voxel), but\n"
        "  weak overall -> not a backbone",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(flat)):
    if j >= len(DATASETS) and j != SUMMARY_CELL:
        flat[j].axis("off")

fig.suptitle("v8 conv vs TCN-1d vs TCN-2d, learning curves, all 13 datasets (seed 42)  |  "
             "TCN-2d total +59.2 vs conv +48.2",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT / "tcn_curves_grid.png"
fig.savefig(out, dpi=125)
print(f"wrote {out}")
