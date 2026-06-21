#!/usr/bin/env python3
"""Cauchy: predict convergence of an iterative solver in complex space.

Framing 2 (the ENCODED one). A convex quadratic f(x) = x^H A x on C^n with A
Hermitian positive-definite. Fixed-step gradient descent x <- x - eta0 A x has
iteration matrix (I - eta0 A); it converges iff lambda_max(A) < 2/eta0. So the
label is: does GD with a fixed global step converge? The network sees only the
raw matrix entries and must in effect estimate the top eigenvalue -- a nonlinear
spectral quantity, no spatial-image prior. Hence ENCODED and hard.

We fix the trace (sum of eigenvalues = n) so the answer is NOT readable from the
trace; it depends on the spectral SHAPE (how concentrated the spectrum is). eta0
is set to the median lambda_max over the pool, giving a balanced 2-way split.

Encoding: (2, n, n) float32 = real(A), imag(A). Hermitian => real part
symmetric, imag part antisymmetric. Regenerable from the seed. Runs locally.
"""
import json
from pathlib import Path

import numpy as np

SEED = 17
N = 16                                   # matrix dimension
POOL = 50_000                            # generate, then threshold for balance
SPLIT = {"train": 15_000, "valid": 3_500, "test": 3_500}   # per class
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "dev/datasets/Cauchy"

rng = np.random.default_rng(SEED)


def random_hermitian_pd():
    """Hermitian PD A with trace fixed to N; spectral shape varies."""
    # random unitary U via QR of a complex Gaussian
    z = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))          # fix QR phase ambiguity
    # eigenvalues: vary concentration. mixture of broad and spiky spectra so
    # lambda_max is decorrelated from the (fixed) trace.
    a = rng.uniform(0.2, 6.0)
    lam = rng.gamma(shape=a, scale=1.0, size=N) + 1e-3
    lam *= N / lam.sum()                                # fix trace = N
    A = (q * lam) @ q.conj().T
    A = 0.5 * (A + A.conj().T)                          # exact Hermitian
    return A.astype(np.complex64), float(lam.max())


def main():
    print(f"generating {POOL} Hermitian PD matrices ({N}x{N}) ...")
    mats = np.empty((POOL, N, N), np.complex64)
    lmax = np.empty(POOL, np.float64)
    for k in range(POOL):
        mats[k], lmax[k] = random_hermitian_pd()
    thresh = float(np.median(lmax))                     # eta0 = 2/thresh
    label = (lmax < thresh).astype(np.int64)            # 1 = converges, 0 = diverges
    print(f"  median lambda_max = {thresh:.3f}  (eta0 = {2/thresh:.4f}); "
          f"balance {np.bincount(label).tolist()}")

    def encode(idx):
        A = mats[idx]
        x = np.stack([A.real, A.imag], axis=1).astype(np.float32)   # (n,2,N,N)
        return x

    OUT.mkdir(parents=True, exist_ok=True)
    pos = rng.permutation(np.where(label == 1)[0])
    neg = rng.permutation(np.where(label == 0)[0])
    off = 0
    for name, per in SPLIT.items():
        ip, ineg = pos[off:off + per], neg[off:off + per]
        off += per
        idx = np.concatenate([ip, ineg])
        sh = rng.permutation(len(idx)); idx = idx[sh]
        x = encode(idx); y = label[idx]
        np.save(OUT / f"{name}_x.npy", x)
        np.save(OUT / f"{name}_y.npy", y)
        # stash lambda_max for the example plot / sanity (test split only)
        if name == "test":
            np.save(OUT / "test_lmax.npy", lmax[idx])
        print(f"  {name}: x{x.shape} balance {np.bincount(y).tolist()}")

    meta = {"num_classes": 2, "input_shape": [2 * SPLIT['train'], 2, N, N],
            "codename": "Cauchy", "benchmark": 0.0}
    (OUT / "metadata").write_text(json.dumps(meta))
    print(f"wrote {OUT}  meta={meta}")


if __name__ == "__main__":
    main()
