#!/usr/bin/env python3
"""Example-sample figure for the Quincy (QR) custom dataset.

Row 1: six clean codes of the SAME class (intra-class variation from the
       random payload suffix + random mask pattern).
Row 2: six clean codes of six different classes.
Row 3: corrupted samples, three module-flip and three occlusion.
Run with an env that has the qrcode package (see dev/make_qr_dataset.py).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_qr_dataset import make_code, corrupt, FLIP_P, OCCLUDE

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/03 - Datasets/additional/Quincy.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(3)
fig, axes = plt.subplots(3, 6, figsize=(11, 6))

for j in range(6):
    axes[0, j].imshow(make_code(rng, 3), cmap="binary")
    axes[0, j].set_title("class C03", fontsize=9)

for j, cls in enumerate((0, 4, 7, 11, 15, 19)):
    axes[1, j].imshow(make_code(rng, cls), cmap="binary")
    axes[1, j].set_title(f"class C{cls:02d}", fontsize=9)

for j in range(6):
    m = make_code(rng, int(rng.integers(20)))
    if j < 3:
        flips = rng.random(m.shape) < FLIP_P
        m[flips] = 1.0 - m[flips]
        t = f"flips p={FLIP_P}"
    else:
        r, c = rng.integers(0, 21 - OCCLUDE, size=2)
        m[r:r + OCCLUDE, c:c + OCCLUDE] = float(rng.integers(0, 2))
        t = f"occlusion {OCCLUDE}x{OCCLUDE}"
    axes[2, j].imshow(m, cmap="binary")
    axes[2, j].set_title(t, fontsize=9)

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
for i, lab in enumerate(("same class,\nclean", "different classes,\nclean",
                         "corrupted\n(10% of samples)")):
    axes[i, 0].set_ylabel(lab, fontsize=10)

fig.suptitle("Quincy: version-1 QR codes (1x21x21), 20-way classification",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
