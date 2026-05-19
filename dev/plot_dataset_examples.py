"""Render example samples from every local dataset → docs/notes/03 - Datasets/examples/.

Gives a visual idea of what each (often non-obvious) input tensor actually
looks like. Renderer branches on channel count:

  * C == 3  → RGB grid of K samples (per-image min-max normalised). NB: for
              AddNIST/MultNIST the 3 "channels" are 3 stacked digits, so the
              RGB composite looks odd — that *is* what the model sees; the
              per-channel montage (C>3 path is not taken here) is in the note text.
  * C == 1, not too wide → grayscale grid of K samples.
  * C == 1, very wide (Cryptic 6×768 BERT) → stacked heatmaps (aspect='auto').
  * C  > 3  → channel montage of 2 samples (reveals Chesseract/Voxel/Windspeed
              structure: board planes / z-slices / 6h×2vars).

Reads raw `train_x.npy` (mmap) + `train_y.npy`; picks samples spanning
distinct labels. numpy + matplotlib only — no torch, runs locally.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import json

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "datasets"
OUT = REPO / "docs" / "notes" / "03 - Datasets" / "examples"

K = 2  # samples per dataset (user: "maybe 1 or 2")

# "What needs to be classified" — concise task per dataset, from the
# notebase Datasets Overview + verified READMEs/figshare descriptions.
# num_classes is read live from each dataset's metadata (authoritative).
TASKS = {
    "AddNIST": "label derived from 3 stacked MNIST-style digit channels (sum)",
    "MultNIST": "label from 3 stacked MNIST-style digit channels (product)",
    "CIFARTile": "# of distinct CIFAR-10 classes in the 2×2 tile (1–4)",
    "GeoClassing": "which region/country — Sentinel-2 satellite RGB",
    "Myofibre": "muscle-fibre medical-image class",
    "Language": "which language — encoded character grid",
    "Gutenberg": "which author — encoded text grid",
    "Chesseract": "game outcome from board state (12 piece-plane channels)",
    "Sudoku": "value (1–9, zero-indexed) of the single flagged target cell",
    "GameOfLife": "next-generation alive-cell count (Conway's Game of Life)",
    "Voxel": "3D object category — 20 z-slices stored as channels",
    "Windspeed": "binned avg wind speed (Beaufort) from 6 h temp+pressure (ERA5)",
    "Cryptic": "binary label of a 6-word cryptic-crossword clue (BERT embeddings)",
}


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _pick_indices(y, n):
    """Indices spanning distinct labels (then fill in order)."""
    idx, seen = [], set()
    for i, v in enumerate(y):
        v = int(v)
        if v not in seen:
            seen.add(v); idx.append(i)
        if len(idx) >= n:
            break
    i = 0
    while len(idx) < n and i < len(y):
        if i not in idx:
            idx.append(i)
        i += 1
    return idx[:n]


def render(name: str):
    d = DATA / name
    xp, yp = d / "train_x.npy", d / "train_y.npy"
    ncls = "?"
    try:
        ncls = json.loads((d / "metadata").read_text()).get("num_classes", "?")
    except Exception:
        pass
    task = TASKS.get(name, "(see dataset note)")
    split = "train"
    if not xp.exists():       # local train_x missing (e.g. CIFARTile .tmp) → use valid
        xp, yp, split = d / "valid_x.npy", d / "valid_y.npy", "valid"
        if not xp.exists():
            print(f"  skip {name} (no train_x or valid_x)")
            return
    x = np.load(xp, mmap_mode="r")
    y = np.load(yp, mmap_mode="r")
    if x.ndim == 3:           # (N,H,W) → add channel
        x = x[:, None]
    N, C, H, W = x.shape
    yl = np.asarray(y[: min(len(y), 200000)])
    idx = _pick_indices(yl, K)

    # Honest colormap rule: discrete/sparse encodings (few unique values) have
    # no colour → grayscale, never a false palette. Continuous scalar fields
    # (many unique values: Windspeed temp/pressure, Cryptic embeddings) → a
    # perceptual colormap is the standard truthful magnitude rendering.
    _u = np.unique(np.asarray(x[idx[0]]))
    discrete = _u.size <= 12
    CM = "gray_r" if discrete else "viridis"

    if C == 3:
        # K rows × 4 cols: RGB composite + the 3 channels split out. The
        # split reveals independent-channel datasets (AddNIST/MultNIST are
        # 3 stacked digits → label = sum/product of the 3).
        fig, axes = plt.subplots(K, 4, figsize=(8.0, 3.0 * K), squeeze=False)
        fig.subplots_adjust(hspace=0.45)
        for r, i in enumerate(idx):
            img = np.asarray(x[i])
            axes[r][0].imshow(_norm(img).transpose(1, 2, 0))
            axes[r][0].set_title(f"label={int(y[i])} · RGB", fontsize=8)
            for c in range(3):
                axes[r][c + 1].imshow(_norm(img[c]), cmap="gray")
                axes[r][c + 1].set_title(f"ch{c}", fontsize=7)
            for c in range(4):
                axes[r][c].axis("off")
        sub = f"{N} {split} samples · 3×{H}×{W} (RGB + per-channel split)"
    elif C == 1 and W <= 8 * H:                      # small grid (digit/grid-like)
        cols = min(K, 4)
        rows = (K + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows),
                                 squeeze=False)
        annot = (H <= 12 and W <= 12)                # tiny grids → show raw values
        for ax, i in zip(axes.flat, idx):
            raw = np.asarray(x[i])[0].astype(np.float32)
            ax.imshow(raw, cmap=CM)            # neutral: white=0, dark=high
            if annot:
                for rr in range(H):
                    for cc in range(W):
                        v = raw[rr, cc]
                        ax.text(cc, rr, f"{v:g}", ha="center", va="center",
                                fontsize=6,
                                color="red" if v > 0.95 else
                                ("black" if v < 0.5 else "white"))
                ax.set_xticks(np.arange(-.5, W, 1), minor=True)
                ax.set_yticks(np.arange(-.5, H, 1), minor=True)
                ax.grid(which="minor", color="0.7", lw=.5)
            ax.set_title(f"label={int(y[i])}  (raw cell values shown)"
                         if annot else f"label={int(y[i])}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.flat[len(idx):]:
            ax.axis("off")
        sub = (f"{N} {split} samples · 1×{H}×{W} — real data "
               f"({'raw values overlaid' if annot else 'grayscale, white=0'})")
    elif C == 1:                                     # Cryptic-like wide matrix
        fig, axes = plt.subplots(K, 1, figsize=(12, 2.0 * K), squeeze=False)
        for ax, i in zip(axes.flat, idx):
            ax.imshow(_norm(np.asarray(x[i])[0]), cmap="magma", aspect="auto")
            ax.set_title(f"label={int(y[i])}  (1×{H}×{W} — BERT-style embeddings)",
                         fontsize=8)
            ax.set_yticks([]); ax.set_xticks([])
        sub = f"{N} {split} samples · 1×{H}×{W} (wide encoded matrix)"
    else:                                            # many-channel montage
        nshow = min(K, 2)
        cols = min(C, 8)
        rows_per = (C + cols - 1) // cols
        fig, axes = plt.subplots(nshow * rows_per, cols,
                                 figsize=(1.5 * cols, 1.5 * nshow * rows_per),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for s in range(nshow):
            img = np.asarray(x[idx[s]])
            for c in range(C):
                ax = axes[s * rows_per + c // cols][c % cols]
                ax.imshow(_norm(img[c]), cmap=CM)
                ax.set_title(f"s{s}·ch{c} (y={int(y[idx[s]])})", fontsize=6)
                ax.axis("off")
        sub = (f"{N} {split} samples · {C}×{H}×{W} — {nshow} samples, "
               f"all {C} channels (montage)")

    fig.suptitle(
        f"{name} — example train samples   ({sub})\n"
        f"▸ classify: {task}  →  {ncls} classes",
        fontsize=11, y=1.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}  [{C}x{H}x{W}, N={N}]")


def main():
    names = sorted(p.name for p in DATA.iterdir()
                   if p.is_dir() and ((p / "train_x.npy").exists()
                                      or (p / "valid_x.npy").exists()))
    print(f"datasets: {names}")
    for n in names:
        try:
            render(n)
        except Exception as ex:
            print(f"  FAILED {n}: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
