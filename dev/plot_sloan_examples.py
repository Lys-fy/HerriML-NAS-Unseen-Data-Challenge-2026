"""Sloan example figure -> docs/notes/03 - Datasets/examples/Sloan.png.

One row per class (GALAXY / QSO / STAR), the four (5x5) encoding channels across
the columns, so the note shows what the network actually sees: a colour matrix,
a brightness (sum) matrix, and the two broadcast position planes. numpy +
matplotlib only, runs locally.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
D = REPO / "datasets" / "Sloan"
OUT = REPO / "docs/notes/03 - Datasets/examples"
CLASSES = ["GALAXY", "QSO", "STAR"]
CH = ["colour  b_i - b_j", "sum  b_i + b_j"]
PER_CLASS = 2

x = np.load(D / "train_x.npy", mmap_mode="r")
y = np.load(D / "train_y.npy")

rows = []
for c in range(3):
    idx = np.where(y == c)[0][:PER_CLASS]
    rows.extend((c, i) for i in idx)

fig, axes = plt.subplots(len(rows), 2, figsize=(4.2, 1.9 * len(rows)), squeeze=False)
for r, (c, i) in enumerate(rows):
    img = np.asarray(x[i])
    for ch in range(2):
        ax = axes[r][ch]
        ax.imshow(img[ch], cmap="coolwarm")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(CH[ch], fontsize=8)
    axes[r][0].set_ylabel(CLASSES[c], fontsize=10, rotation=90, labelpad=6)

fig.suptitle(
    "Sloan - example train samples (2x5x5, balanced 3-way)\n"
    "classify GALAXY / QSO / STAR from PHOTOMETRY ALONE (u,g,r,i,z); "
    "redshift and the leaked label columns are dropped on purpose",
    fontsize=10, y=1.0)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "Sloan.png"
fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"wrote {p}")
