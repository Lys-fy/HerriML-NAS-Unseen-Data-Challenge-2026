#!/usr/bin/env python3
"""3-seed bake-off, seed 42: v11.1 (informed-defaults floor + verified attention
probe) vs the FAITHFUL EGNAS replica vs the v11.3 EVOLUTION, all full-budget
(1.5h) validated against v10's known s42 test acc. One figure that tells the
story: v11.1 tracks v10 and captures the Chesseract attention lever; both real
searches are net-negative, EGNAS catastrophically so (Cryptic, CIFARTile).

Data hardcoded (project convention, cf. plot_v11_testing.py); regenerate the
numbers from ~/local/{v11_1,egnas_faithful_validated,evo_validated} on spark.
Output -> images/pipeline/bakeoff_s42.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/bakeoff_s42.png"

# v10 s42 real test acc (codename-mapped); Cryptic/Sudoku ship placeholder
# benchmarks so only raw acc is meaningful.
V10 = {
    "Chesseract": 57.66, "Sedol": 56.55, "CIFARTile": 73.65, "Sudoku": 97.70,
    "Cryptic": 74.25, "Language": 90.03, "Gutenberg": 47.28, "Sloan": 84.60,
    "Cauchy": 93.91, "AddNIST": 96.33, "MultNIST": 95.52, "GeoClassing": 95.36,
    "GameOfLife": 100.0, "Myofibre": 91.09, "Voxel": 81.56, "Windspeed": 19.36,
}
# (v11.1, egnas_faithful, evo) full-budget test acc, seed 42
RES = {
    "Chesseract": (62.48, 57.54, 61.97), "Sedol": (57.02, 55.42, 57.24),
    "CIFARTile": (70.81, 47.59, 60.36), "Sudoku": (94.38, 98.21, 96.03),
    "Cryptic": (69.32, 52.45, 72.65), "Language": (88.65, 79.03, 79.38),
    "Gutenberg": (47.07, 48.00, 47.63), "Sloan": (84.34, 84.52, 84.69),
    "Cauchy": (94.11, 92.01, 91.76), "AddNIST": (96.51, 96.20, 96.31),
    "MultNIST": (95.06, 94.84, 94.55), "GeoClassing": (94.93, 94.39, 95.67),
    "GameOfLife": (99.99, 99.99, 99.97), "Myofibre": (90.64, 86.96, 89.18),
    "Voxel": (80.72, 79.25, 80.65), "Windspeed": (18.79, 18.59, 20.90),
}
# board/encoded levers first, ceilinged image sets last
ORDER = ["Chesseract", "Sedol", "Cryptic", "Sudoku", "Gutenberg", "Language",
         "CIFARTile", "Cauchy", "Sloan", "Myofibre", "Voxel", "GeoClassing",
         "MultNIST", "AddNIST", "Windspeed", "GameOfLife"]
LABELS = ["v11.1 (floor)", "EGNAS-faithful", "evolution"]
COLORS = ["#2ca02c", "#d62728", "#1f77b4"]

dn = [d - V10[k] for k in ORDER for d in [RES[k][0]]]  # noqa (clarity below)
margins = np.array([[RES[k][m] - V10[k] for k in ORDER] for m in range(3)])

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.1], hspace=0.38)

# --- Panel A: per-dataset margin vs v10 -----------------------------------
ax = fig.add_subplot(gs[0])
x = np.arange(len(ORDER)); w = 0.26
for m in range(3):
    ax.bar(x + (m - 1) * w, margins[m], w, label=LABELS[m], color=COLORS[m])
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=40, ha="right")
ax.set_ylabel("test acc minus v10 (pp)")
ax.set_title("3-seed bake-off (seed 42): margin vs v10 per dataset  "
             "[positive = beats v10]")
ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-28, 8)
# annotate the two big EGNAS craters so the clipped bars are readable
for k, lab in [("Cryptic", "EGNAS -21.8"), ("CIFARTile", "EGNAS -26.1")]:
    xi = ORDER.index(k)
    ax.annotate(lab, (xi, -27.5), ha="center", va="bottom", fontsize=8,
                color=COLORS[1])

# --- Panel B: summary (mean margin + win/tie/loss) ------------------------
ax2 = fig.add_subplot(gs[1])
means = margins.mean(axis=1)
def wtl(row):
    w = int((row > 0.3).sum()); l = int((row < -0.3).sum())
    return w, len(row) - w - l, l
y = np.arange(3)[::-1]
bars = ax2.barh(y, means, color=COLORS, height=0.6)
ax2.axvline(0, color="black", lw=1)
ax2.set_yticks(y); ax2.set_yticklabels(LABELS)
ax2.set_xlabel("mean margin vs v10 (pp), averaged over 16 datasets")
ax2.set_title("summary")
for i, (yy, mn) in enumerate(zip(y, means)):
    w, t, l = wtl(margins[i])
    ax2.text(mn + (0.4 if mn >= 0 else -0.4), yy,
             f"{mn:+.2f} pp   (W {w} / T {t} / L {l})",
             va="center", ha="left" if mn >= 0 else "right", fontsize=9)
ax2.set_xlim(-7, 3); ax2.grid(axis="x", alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
for i in range(3):
    w, t, l = wtl(margins[i])
    print(f"  {LABELS[i]:16s} mean {means[i]:+.2f}pp  total {margins[i].sum():+.1f}pp  "
          f"W{w}/T{t}/L{l}")
