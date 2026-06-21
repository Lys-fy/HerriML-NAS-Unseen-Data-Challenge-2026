#!/usr/bin/env python3
"""v11 gauntlet curves: v10 vs the fixed alternative families (attn / recur /
modern) on the 4 gauntlet datasets. The 'lines' companion to the delta heatmap:
shows that the alternatives only catch v10 on Chesseract and fall far below it
everywhere else. best_val_acc per epoch, single seed (s42).
Output -> images/pipeline/v11_gauntlet_curves.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "dev/logs"
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v11_gauntlet_curves.png"

# (display name, codename)
PANELS = [("Chesseract", "Chester"), ("Sudoku", "Sokoto"),
          ("Language", "LaMelo"), ("CIFARTile", "Caitie")]
# (label, path-template, colour)
SERIES = [
    ("v10",      LOG / "v10_sweep/v10_s42",   "#2c8a3a"),
    ("attention", LOG / "arch_probe/attn",    "#e8843c"),
    ("recurrent", LOG / "arch_probe/recur",   "#8a5cc8"),
    ("modern",   LOG / "arch_probe/modern",   "#1f77b4"),
]


def best_curve(path):
    if not path.exists():
        return None
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.append(float(r["best_val_acc"]) * 100)
            except (KeyError, ValueError):
                pass
    return np.array(out) if len(out) >= 2 else None


fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
for ax, (name, code) in zip(axes.flat, PANELS):
    lo, hi = 100.0, 0.0
    for label, d, color in SERIES:
        c = best_curve(d / f"{code}.csv")
        if c is None:
            continue
        ax.plot(np.arange(len(c)), c, color=color, linewidth=2.4 if label == "v10" else 1.8,
                label=f"{label} {c[-1]:.1f}%", zorder=3 if label == "v10" else 2)
        lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    if hi > lo:
        ax.set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("best val acc (%)")
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

fig.suptitle("v11 gauntlet: v10 (green) vs fixed alternative families, 4 datasets (s42)\n"
             "alternatives only reach v10 on Chesseract; far below it on the rest",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
