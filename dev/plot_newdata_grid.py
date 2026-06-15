#!/usr/bin/env python3
"""Custom-datasets result grid, 5 rows x 4 cols.
Rows: Quincy, Magnus, Cornelius, Pavel, Dominik (one dataset per row).
Cols: seed 42, seed 1337, seed 2024, 3-seed aggregate (mean +/- std).
Each panel shows the v10 best-val curve (green) and, the same way in every
panel, the ResNet-18 baseline (blue, seed 42) which is the zero-point for
these custom datasets. Missing cells (sweep still running) are annotated.
Re-run to fill in as the sweep + resnet baseline + Cornelius catch-up land.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/newdata_curves_grid_4x5.png"
NDLOG = REPO / "dev/logs/v10_newdata_sweep"
SEEDS = ["42", "1337", "2024"]
GRID = np.linspace(0.0, 1.0, 100)
V10C = "#2c8a3a"; RNC = "#1f77b4"
COLS = [("Quincy", "QR, encoded"), ("Magnus", "polynomial degree"),
        ("Cornelius", "connect-4 win col"), ("Pavel", "collision, temporal"),
        ("Dominik", "domino mismatch")]


def best_curve(path):
    best = []
    if not path.exists():
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            best.append(float(r["best_val_acc"]) * 100)
    if len(best) < 2:
        return None
    return np.interp(GRID, np.linspace(0, 1, len(best)), np.array(best))


def resnet_curve(code):
    return best_curve(NDLOG / "resnet18_s42" / f"{code}.csv")


fig, axes = plt.subplots(5, 4, figsize=(16, 18), squeeze=False)
col_labels = ["seed 42", "seed 1337", "seed 2024", "3-seed aggregate"]

for ri, (code, hint) in enumerate(COLS):
    rn = resnet_curve(code)
    for ci in range(4):
        ax = axes[ri][ci]
        lo, hi = 100.0, 0.0
        # ResNet baseline (same in every column)
        if rn is not None:
            ax.plot(GRID, rn, color=RNC, linewidth=1.8, label=f"ResNet-18 {rn[-1]:.1f}%")
            lo, hi = min(lo, float(rn.min())), max(hi, float(rn.max()))
        # v10
        if ci < 3:
            c = best_curve(NDLOG / f"v10_s{SEEDS[ci]}" / f"{code}.csv")
            if c is not None:
                ax.plot(GRID, c, color=V10C, linewidth=2.4, label=f"v10 {c[-1]:.1f}%")
                lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
        else:
            curves = [best_curve(NDLOG / f"v10_s{s}" / f"{code}.csv") for s in SEEDS]
            curves = [c for c in curves if c is not None]
            if curves:
                a = np.array(curves); m, sd = a.mean(0), a.std(0)
                ax.fill_between(GRID, m - sd, m + sd, color=V10C, alpha=0.25)
                ax.plot(GRID, m, color=V10C, linewidth=2.6,
                        label=f"v10 {m[-1]:.1f}±{sd[-1]:.1f}% (n={len(curves)})")
                lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
        if hi <= lo:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="#999", fontsize=12)
        else:
            pad = max(2.0, (hi - lo) * 0.12)
            ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.grid(alpha=0.25)
        if ri == 0:
            ax.set_title(col_labels[ci], fontsize=12, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{code}\n({hint})\nval acc (%)", fontsize=10)
        if ri == 4:
            ax.set_xlabel("fraction of training")

fig.suptitle("v10 on the 5 custom datasets vs ResNet-18 baseline  "
             "(rows: datasets; cols: seeds 42/1337/2024 + aggregate; blue = ResNet-18 zero-point)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
