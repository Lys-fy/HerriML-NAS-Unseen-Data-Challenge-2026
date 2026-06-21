#!/usr/bin/env python3
"""v11-testing summary: delta vs v10 (s42 test acc) for every extension
experiment x dataset. One heatmap that shows the whole story: Chesseract is the
only column that is green across the board, everything else is at ceiling or
worse. Green = the alternative beats v10, red = worse, grey = not tested.

Deltas are from dev/score_overview.py (single seed). The forced-macro probe has
no gate, so its big reds (Sudoku -33.6, CIFARTile -22) are the isolation working
as designed, not a bug. Output -> images/pipeline/v11_testing_delta_heatmap.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v11_testing_delta_heatmap.png"

DATASETS = ["AddNIST", "CIFARTile", "Chesseract", "Cryptic", "GameOfLife",
            "GeoClassing", "Gutenberg", "Language", "MultNIST", "Myofibre",
            "Sudoku", "Voxel", "Windspeed"]
ROWS = ["attn (fixed)", "recur (fixed)", "modern (fixed)", "ensemble",
        "width search", "macro (forced)"]
# delta vs v10 s42 (percentage points); None = not tested
D = {
 "attn (fixed)":   {"CIFARTile": -20.53, "Chesseract": 1.55, "Language": -12.46, "Sudoku": -10.74},
 "recur (fixed)":  {"CIFARTile": -28.18, "Chesseract": 0.37, "Language": -11.84, "Sudoku": -3.57},
 "modern (fixed)": {"CIFARTile": -25.63, "Chesseract": 0.51, "Language": -7.20, "Sudoku": -2.25},
 "ensemble":       {"Chesseract": 1.42, "Sudoku": -16.04},
 "width search":   {"Chesseract": 1.17, "Gutenberg": -1.32, "Language": -0.17, "Sudoku": 0.15},
 "macro (forced)": {"AddNIST": -0.84, "CIFARTile": -21.98, "Chesseract": 5.94, "Cryptic": -4.12,
                    "GameOfLife": -0.10, "GeoClassing": -3.02, "Gutenberg": -1.52, "Language": -7.90,
                    "MultNIST": -0.89, "Myofibre": -2.82, "Sudoku": -33.57, "Voxel": -3.16,
                    "Windspeed": 0.65},
}

M = np.full((len(ROWS), len(DATASETS)), np.nan)
for ri, r in enumerate(ROWS):
    for ci, c in enumerate(DATASETS):
        if c in D[r]:
            M[ri, ci] = D[r][c]

fig, ax = plt.subplots(figsize=(13, 5.0))
cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#e8e8e8")          # grey = not tested
norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=6)
ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")

for ri in range(len(ROWS)):
    for ci in range(len(DATASETS)):
        v = M[ri, ci]
        if not np.isnan(v):
            ax.text(ci, ri, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold" if v > 0 else "normal")

ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels(DATASETS, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS, fontsize=10)
ax.set_xticks(np.arange(-.5, len(DATASETS), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
ax.set_title("v11 testing: delta vs v10 (s42 test acc, pp) per experiment x dataset\n"
             "green = beats v10  |  red = worse  |  grey = not tested  -- "
             "Chesseract is the only consistent win; everything else is at ceiling",
             fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}")
