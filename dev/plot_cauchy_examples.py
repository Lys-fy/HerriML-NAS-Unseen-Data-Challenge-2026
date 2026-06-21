"""Cauchy example figure -> docs/notes/03 - Datasets/examples/Cauchy.png.

Shows the (2,16,16) encoding = real and imaginary parts of a Hermitian matrix A,
for two CONVERGES and two DIVERGES samples. The title prints lambda_max(A) (the
hidden spectral quantity that decides the label) to make the point: the planes
look alike to the eye, the answer is a nonlinear function of the entries. Local.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
D = REPO / "datasets" / "Cauchy"
OUT = REPO / "docs/notes/03 - Datasets/examples"

x = np.load(D / "train_x.npy", mmap_mode="r")
y = np.load(D / "train_y.npy")

idx = []
for c in (1, 0):                                  # converges first, then diverges
    idx.extend(np.where(y == c)[0][:2].tolist())

fig, axes = plt.subplots(len(idx), 2, figsize=(5.0, 2.3 * len(idx)), squeeze=False)
for r, i in enumerate(idx):
    planes = np.asarray(x[i])
    A = planes[0] + 1j * planes[1]
    lmax = float(np.linalg.eigvalsh(A).max())
    lab = "CONVERGES" if int(y[i]) == 1 else "DIVERGES"
    for c, (title, plane) in enumerate([("real(A)", planes[0]),
                                        ("imag(A)", planes[1])]):
        ax = axes[r][c]
        ax.imshow(plane, cmap="PuOr")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=9)
    axes[r][0].set_ylabel(f"{lab}\n$\\lambda_{{max}}$={lmax:.2f}",
                          fontsize=9, rotation=90, labelpad=6)

fig.suptitle(
    "Cauchy - example samples (2x16x16: real and imag of Hermitian A)\n"
    "predict whether fixed-step gradient descent CONVERGES "
    "(label is lambda_max < threshold); the spectrum is hidden in the entries",
    fontsize=10, y=1.0)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "Cauchy.png"
fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"wrote {p}")
