#!/usr/bin/env python3
"""Custom-datasets result grid, 5 rows x 4 cols.
Rows: Quincy, Magnus, Cornelius, Pavel, Dominik (one dataset per row).
Cols: seed 42, seed 1337, seed 2024, 3-seed aggregate (mean +/- std).
Each panel shows the v10 best-val curve (green) and, the same way in every
panel, the ResNet-18 baseline (blue, seed 42) which is the zero-point for
these custom datasets.

x-axis is EPOCHS (raw, no resampling). For the aggregate column the seeds have
different epoch counts, so each seed's monotone best-val is forward-filled at
its final value to the longest seed before averaging. Missing cells (sweep
still running) are annotated; re-run to fill in.
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
V10C = "#2c8a3a"; RNC = "#1f77b4"
COLS = [("Quincy", "QR, encoded"), ("Magnus", "polynomial degree"),
        ("Cornelius", "connect-4 win col"), ("Pavel", "collision, temporal"),
        ("Dominik", "domino mismatch"),
        ("Sedol", "Go outcome, encoded"), ("Sloan", "photometry, image-like"),
        ("Cauchy", "convergence, encoded")]


def best_epochs(path):
    """best_val_acc (%) per epoch, raw."""
    if not path.exists():
        return None
    best = []
    with open(path) as f:
        for r in csv.DictReader(f):
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(best) if len(best) >= 2 else None


def pad(arr, n):
    return arr if len(arr) >= n else np.concatenate([arr, np.full(n - len(arr), arr[-1])])


def resnet_curve(code):
    return best_epochs(NDLOG / "resnet18_s42" / f"{code}.csv")


fig, axes = plt.subplots(len(COLS), 4, figsize=(16, 3.6 * len(COLS)), squeeze=False)
col_labels = ["seed 42", "seed 1337", "seed 2024", "3-seed aggregate"]

for ri, (code, hint) in enumerate(COLS):
    rn = resnet_curve(code)
    for ci in range(4):
        ax = axes[ri][ci]
        lo, hi = 100.0, 0.0
        # ResNet baseline (same in every column), over its own epochs
        if rn is not None:
            ax.plot(np.arange(len(rn)), rn, color=RNC, linewidth=1.8,
                    label=f"ResNet-18 {rn[-1]:.1f}%")
            lo, hi = min(lo, float(rn.min())), max(hi, float(rn.max()))
        # v10
        if ci < 3:
            c = best_epochs(NDLOG / f"v10_s{SEEDS[ci]}" / f"{code}.csv")
            if c is not None:
                ax.plot(np.arange(len(c)), c, color=V10C, linewidth=2.4,
                        label=f"v10 {c[-1]:.1f}%")
                lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
        else:
            cs = [best_epochs(NDLOG / f"v10_s{s}" / f"{code}.csv") for s in SEEDS]
            cs = [c for c in cs if c is not None]
            if cs:
                n = max(len(c) for c in cs)
                a = np.array([pad(c, n) for c in cs]); m, sd = a.mean(0), a.std(0)
                x = np.arange(n)
                ax.fill_between(x, m - sd, m + sd, color=V10C, alpha=0.25)
                ax.plot(x, m, color=V10C, linewidth=2.6,
                        label=f"v10 {m[-1]:.1f}±{sd[-1]:.1f}% (n={len(cs)})")
                lo, hi = min(lo, float(m.min())), max(hi, float(m.max()))
        if hi <= lo:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="#999", fontsize=12)
        else:
            pad_y = max(2.0, (hi - lo) * 0.12)
            ax.set_ylim(lo - pad_y, hi + pad_y)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.grid(alpha=0.25)
        if ri == 0:
            ax.set_title(col_labels[ci], fontsize=12, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{code}\n({hint})\nval acc (%)", fontsize=10)
        if ri == len(COLS) - 1:
            ax.set_xlabel("epoch")

fig.suptitle("v10 on the 8 custom datasets vs ResNet-18 baseline  "
             "(rows: datasets; cols: seeds 42/1337/2024 + aggregate; blue = ResNet-18 zero-point)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
