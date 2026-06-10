#!/usr/bin/env python3
"""v9 (searched dilated backbone) learning curves vs the two references:
conv (v8, searched) and the fixed TCN-2d probe. Seed 42, all 13 datasets.

v9 final TEST scores are not printed until the seed run completes, so the
v9 label shows best-val plus the searched arch (D, W). conv and tcn2d labels
show their known final test raw. Style matches plot_tcn_grid.py.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/tier2a"
OUT.mkdir(parents=True, exist_ok=True)

CONV = "#1f77b4"; T2 = "#ff7f0e"; V9 = "#2c8a3a"; BENCHC = "#b53030"

# name, codename, benchmark, conv test raw, tcn2d-fixed test raw, v9 arch,
# v9 FINAL test raw (seed-42 run complete, total +52.4)
DATASETS = [
    ("Language",    "LaMelo",    85.200, 83.68, 88.24, "D7 W48", 85.21),
    ("Cryptic",     "Cryptic",    0.000, 70.27, 71.18, "D6 W48", 73.70),
    ("AddNIST",     "Adaline",   89.850, 95.82, 96.62, "D8 W48", 96.56),
    ("CIFARTile",   "Caitie",    47.008, 48.30, 72.20, "D5 W48", 70.65),
    ("Chesseract",  "Chester",   57.826, 59.10, 57.08, "D7 W48", 57.42),
    ("GameOfLife",  "conway",    47.530, 99.96, 99.99, "D6 W48", 99.99),
    ("GeoClassing", "Sadie",     80.330, 95.38, 95.49, "D4 W48", 95.51),
    ("Gutenberg",   "Gutenberg", 40.980, 47.08, 40.62, "D4 W48", 40.03),
    ("MultNIST",    "Mateo",     90.870, 94.76, 95.52, "D8 W32", 94.80),
    ("Myofibre",    "Myopia",    87.928, 88.33, 91.96, "D5 W48", 88.78),
    ("Sudoku",      "Sokoto",     0.000, 94.46, 88.14, "D8 W48", 75.44),
    ("Voxel",       "Volga",     71.350, 78.16, 80.33, "D8 W48", 80.14),
    ("Windspeed",   "windspeed", 13.492, 19.74, 19.74, "D7 W48", 20.73),
]


def load(p):
    ep, val, best = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(ep), np.array(val), np.array(best)


fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
flat = axes.flatten()

for ax, (name, code, bench, conv_raw, t2_raw, arch, v9_raw) in zip(flat, DATASETS):
    series = [
        ("conv (v8)", REPO / f"dev/logs/tier2a_p1_v8/p1v8_s42/{code}.csv",
         CONV, f"conv  {conv_raw:.1f}%", 1.6),
        ("tcn2d fixed", REPO / f"dev/logs/tcn_probe/tcn2d_{name}_s42/{code}.csv",
         T2, f"tcn2d-fix {t2_raw:.1f}%", 1.6),
        ("v9", REPO / f"dev/logs/tier2a_p1_v9/p1v9_s42/{code}.csv",
         V9, f"v9 search {arch}  {v9_raw:.1f}%", 2.4),
    ]
    lo, hi = 100.0, 0.0
    for key, path, c, lab, lw in series:
        if not path.exists():
            continue
        ep, val, best = load(path)
        ax.plot(ep, val, color=c, alpha=0.16, linewidth=0.9)
        ax.plot(ep, best, color=c, linewidth=lw, label=lab)
        lo, hi = min(lo, float(best.min())), max(hi, float(best.max()))
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.1,
                   label=f"bench {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("val acc (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

# summary cell
ax = flat[len(DATASETS)]; ax.axis("off")
ax.text(0.02, 0.98,
        "v9 = dilated TCN-2d backbone,\n"
        "depth/width SEARCHED (seed 42, 2h)\n"
        "all numbers = FINAL TEST raw\n\n"
        "TOTAL adj (13 datasets, seed 42):\n"
        "  conv searched (v8)   +48.2\n"
        "  v9 dilated searched  +52.4\n"
        "  dilated FIXED D=6    +59.2\n"
        "-> the search gives back 7 of the\n"
        "   fixed backbone's 11 points\n\n"
        "search HURTS on encoded data:\n"
        "  Sudoku D8  75.4 vs fixed 88.1\n"
        "  Language D7  85.2 vs fixed 88.2\n"
        "  depth mis-ranked BOTH directions\n"
        "v10 fix: default-incumbent D=6 +\n"
        "  route encoded data to the gate",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(DATASETS) + 1, len(flat)):
    flat[j].axis("off")

fig.suptitle("v9 searched dilated backbone vs conv (v8) and fixed TCN-2d, seed 42",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT / "tier2a_v9_curves_grid.png"
fig.savefig(out, dpi=125)
print(f"wrote {out}")
