"""Sedol example figure -> docs/notes/03 - Datasets/examples/Sedol.png.

Renders sampled mid-to-late board positions as real gobans (stones on a 19x19
grid), titled by the LABEL = who eventually won the game. Two Black-wins and two
White-wins so the note shows the task: read the position, predict the outcome.
Decodes the (4,19,19) planes (black, white, empty, side-to-move). Local-only.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
D = REPO / "datasets" / "Sedol"
OUT = REPO / "docs/notes/03 - Datasets/examples"
S = 19
STAR_PTS = [(r, c) for r in (3, 9, 15) for c in (3, 9, 15)]  # hoshi

x = np.load(D / "train_x.npy", mmap_mode="r")
y = np.load(D / "train_y.npy")

# two of each class
idx = []
for c in (0, 1):
    idx.extend(np.where(y == c)[0][:2].tolist())


def goban(ax, planes, label):
    ax.set_facecolor("#e3b964")                       # board colour
    for k in range(S):
        ax.plot([0, S - 1], [k, k], color="black", lw=0.5, zorder=1)
        ax.plot([k, k], [0, S - 1], color="black", lw=0.5, zorder=1)
    for (r, c) in STAR_PTS:
        ax.plot(c, r, "o", color="black", ms=3, zorder=2)
    blk, wht = planes[0], planes[1]
    for r in range(S):
        for c in range(S):
            if blk[r, c]:
                ax.add_patch(plt.Circle((c, r), 0.46, color="black", zorder=3))
            elif wht[r, c]:
                ax.add_patch(plt.Circle((c, r), 0.46, facecolor="white",
                                        edgecolor="black", lw=0.6, zorder=3))
    ax.set_xlim(-0.6, S - 0.4); ax.set_ylim(S - 0.4, -0.6)   # flip y
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    nstones = int(blk.sum() + wht.sum())
    ax.set_title(f"{label}\n({nstones} stones on board)", fontsize=9)


fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, i in zip(axes, idx):
    label = "Black wins" if int(y[i]) == 0 else "White wins"
    goban(ax, np.asarray(x[i]), label)

fig.suptitle(
    "Sedol - example board positions (2x19x19: black / white stone planes)\n"
    "predict the EVENTUAL game WINNER from one mid-to-late position "
    "(dan-level KGS games) - the hard ENCODED Go sibling of Chesseract",
    fontsize=11, y=1.06)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "Sedol.png"
fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"wrote {p}")
