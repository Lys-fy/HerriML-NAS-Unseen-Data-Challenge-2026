#!/usr/bin/env python3
"""Proposal results figure: our pipeline vs the searched-CNN baseline,
13 public datasets, 3x5 grid, the selected configuration under each panel.
No internal version labels (corrector-facing)."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OURS = REPO / "dev/logs/v10_sweep/v10_s42"
BASE = REPO / "dev/logs/v8/v8_s42"
OUT = REPO / "docs/proposal/results_grid.png"

CB = "#1f77b4"; CO = "#2c8a3a"; BENCHC = "#b53030"

# name, codename, benchmark (0 = not published), baseline test raw,
# ours test raw, selected-config line
DATA = [
    ("AddNIST",     "Adaline",   89.850, 95.82, 96.33, "2-D TCN, depth 6, aug on"),
    ("CIFARTile",   "Caitie",    47.008, 48.30, 73.65, "2-D TCN, depth 6, aug on"),
    ("Chesseract",  "Chester",   57.826, 59.10, 57.66, "gate: CNN, depth 6 (embed stem)"),
    ("Cryptic",     "Cryptic",    0.000, 70.27, 74.25, "2-D TCN, depth 8 verified"),
    ("GameOfLife",  "conway",    47.530, 99.96, 100.00, "2-D TCN, depth 6"),
    ("GeoClassing", "Sadie",     80.330, 95.38, 95.36, "2-D TCN, depth 4 verified, aug on"),
    ("Gutenberg",   "Gutenberg", 40.980, 47.08, 47.28, "gate: CNN, depth 6, aug on"),
    ("Language",    "LaMelo",    85.200, 83.68, 90.03, "gate: 1-D TCN, aug off + mixup"),
    ("MultNIST",    "Mateo",     90.870, 94.76, 95.52, "2-D TCN, depth 6, aug on"),
    ("Myofibre",    "Myopia",    87.928, 88.33, 91.09, "2-D TCN, depth 6, aug on"),
    ("Sudoku",      "Sokoto",     0.000, 94.46, 97.70, "gate: CNN, then random arch adopted"),
    ("Voxel",       "Volga",     71.350, 78.16, 81.56, "2-D TCN, depth 6 (embed stem)"),
    ("Windspeed",   "windspeed", 13.492, 19.74, 19.36, "2-D TCN, depth 6"),
]


def load(p):
    ep, val, best = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(ep), np.array(val), np.array(best)


fig, axes = plt.subplots(5, 3, figsize=(13, 15.5), squeeze=False)
flat = axes.flatten()

for ax, (name, code, bench, base_raw, ours_raw, cfg) in zip(flat, DATA):
    lo, hi = 100.0, 0.0
    for path, c, lab, lw in (
            (BASE / f"{code}.csv", CB, f"searched CNN (test {base_raw:.1f}%)", 1.5),
            (OURS / f"{code}.csv", CO, f"ours (test {ours_raw:.1f}%)", 2.3)):
        if not path.exists():
            continue
        ep, val, best = load(path)
        ax.plot(ep, val, color=c, alpha=0.15, linewidth=0.8)
        ax.plot(ep, best, color=c, linewidth=lw, label=lab)
        lo, hi = min(lo, float(best.min())), max(hi, float(best.max()))
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.0,
                   label=f"official benchmark {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("val acc (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    ax.set_xlabel(f"selected: {cfg}", fontsize=8.5, style="italic",
                  color="#333333")

for j in range(len(DATA), len(flat)):
    flat[j].axis("off")

fig.tight_layout(h_pad=2.2)
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
