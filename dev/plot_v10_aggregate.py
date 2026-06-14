#!/usr/bin/env python3
"""Aggregated v10 result across the 3 seeds (42/1337/2024): per-dataset
adjusted score with variance marked (mean, +/- std, and the individual seed
points), plus the 3-seed total. Adjusted score = (raw - benchmark) * 10 /
(100 - benchmark), the competition metric.

Numbers are the scored test results (dev/mock_harness scoring) of the main
v10 2h sweep. GameOfLife s2024's prediction was lost to a spark reboot
between training and predict (its own val hit 100%); imputed +10.0 from the
other two seeds' 100% test and flagged.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v10_aggregate_3seed.png"

SEEDS = ["42", "1337", "2024"]
# adjusted score per dataset, [s42, s1337, s2024]
ADJ = {
    "AddNIST":     [6.384, 5.862, 6.581],
    "CIFARTile":   [5.028, 5.143, 4.797],
    "Chesseract":  [-0.040, 1.534, 0.915],
    "Cryptic":     [7.425, 6.902, 7.296],
    "GameOfLife":  [10.000, 9.998, 10.000],  # s2024 imputed (pred lost to reboot)
    "GeoClassing": [7.641, 7.781, 7.647],
    "Gutenberg":   [1.068, 1.014, 1.037],
    "Language":    [3.264, 3.345, 2.838],
    "MultNIST":    [5.093, 4.721, 4.797],
    "Myofibre":    [2.617, 3.020, 3.020],
    "Sudoku":      [9.770, 8.614, 9.183],
    "Voxel":       [3.564, 3.312, 3.438],
    "Windspeed":   [0.678, 0.811, 0.596],
}
IMPUTED = {"GameOfLife"}  # mark on the plot

items = sorted(ADJ.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
names = [k for k, _ in items]
means = np.array([np.mean(v) for _, v in items])
stds = np.array([np.std(v, ddof=0) for _, v in items])
y = np.arange(len(names))

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(y, means, color="#2c8a3a", alpha=0.30, height=0.66, zorder=1)
ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#1f5f28",
            elinewidth=2, capsize=5, zorder=3)
# individual seed points
markers = {"42": "o", "1337": "s", "2024": "^"}
colors = {"42": "#1f77b4", "1337": "#ff7f0e", "2024": "#9467bd"}
for si, s in enumerate(SEEDS):
    xs = [ADJ[n][si] for n in names]
    ax.scatter(xs, y, marker=markers[s], s=34, color=colors[s], zorder=4,
               label=f"seed {s}", edgecolor="white", linewidth=0.5)
for i, n in enumerate(names):
    tag = " *" if n in IMPUTED else ""
    ax.text(means[i] + stds[i] + 0.15, i,
            f"{means[i]:.2f}±{stds[i]:.2f}{tag}", va="center", fontsize=8.5)

ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("adjusted score  (raw - ResNet18 benchmark, rescaled to 0-10)", fontsize=10)
ax.set_xlim(-1.2, 11.6)
ax.axvline(0, color="#888", linewidth=0.8)
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

totals = [sum(ADJ[n][si] for n in ADJ) for si in range(3)]
tmean, tstd = np.mean(totals), np.std(totals, ddof=0)
ax.set_title(
    "v10 (measure-gate-train): aggregated 3-seed adjusted score per dataset\n"
    f"total per seed: s42 {totals[0]:+.2f}  s1337 {totals[1]:+.2f}  "
    f"s2024 {totals[2]:+.2f}  ->  mean {tmean:+.2f} ± {tstd:.2f}   "
    "(* GameOfLife s2024 imputed: pred lost to reboot, val=100%)",
    fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
print(f"total per seed: {[round(t,2) for t in totals]}  mean {tmean:.2f} +/- {tstd:.2f}")
