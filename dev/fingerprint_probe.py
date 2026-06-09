#!/usr/bin/env python3
"""Dataset fingerprint probe: long-range correlation decay on raw tensors.

The novel claim (Gap 2 in [[SOTA Scan - Novel Directions]]): a cheap, pure-CPU
measurement of the data's spatial dependence structure can vote for the
backbone family (long-range dependence -> dilated/sequence, local -> plain
conv) without training anything, sidestepping the proxy wall entirely.

Validation target: the per-dataset final margin (tcn2d adj minus conv adj)
from the completed full-budget runs. If the fingerprint's long-range score
rank-correlates with that margin, the probe carries real signal.

Measurement: Pearson correlation between pixels at horizontal/vertical
distance tau (per channel, standardized, over a sample of images), for tau up
to half the grid. Summary stats per dataset:
  corr1      : correlation at tau=1 (local smoothness)
  half_dist  : first tau where corr < 0.5 * corr1 (decay length)
  lr_score   : mean correlation at tau in [4, 8, 16] relative to corr1
               (the long-range tail: high = slow decay = long-range structure)
  pow_vs_exp : R^2(power-law fit) - R^2(exponential fit) on the decay curve
               (positive = power-law tail = scale-free long-range structure)
Cost: a few seconds per dataset, numpy only.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

DATA = Path.home() / "local" / "nas-comp-datasets"
N_SAMPLES = 256
RNG = np.random.default_rng(42)

# final adj margins (tcn2d - conv) from the completed runs = validation target
MARGIN = {
    "Language": 3.08, "Cryptic": 0.09, "AddNIST": 0.79, "CIFARTile": 4.51,
    "Chesseract": -0.48, "GameOfLife": 0.01, "GeoClassing": 0.06,
    "Gutenberg": -1.09, "MultNIST": 0.83, "Myofibre": 3.00, "Sudoku": -0.64,
    "Voxel": 0.75, "Windspeed": 0.00,
}


def corr_at_tau(x, tau):
    """Mean Pearson correlation between pixel pairs at distance tau,
    horizontal and vertical, over samples and channels. x: (n, c, h, w),
    already standardized per channel."""
    h_pairs = (x[..., :, :-tau] * x[..., :, tau:]).mean()
    v_pairs = (x[..., :-tau, :] * x[..., tau:, :]).mean()
    return float((h_pairs + v_pairs) / 2.0)


def fit_r2(xs, ys):
    """R^2 of a linear least-squares fit."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 3:
        return 0.0
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def fingerprint(name):
    path = DATA / name / "train_x.npy"
    x = np.load(path, mmap_mode="r")
    n = x.shape[0]
    idx = np.sort(RNG.choice(n, size=min(N_SAMPLES, n), replace=False))
    x = np.asarray(x[idx], dtype=np.float64)        # (s, c, h, w)
    if x.ndim == 3:
        x = x[:, None]
    h, w = x.shape[-2], x.shape[-1]
    # standardize per channel (over samples and space)
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    sd = x.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    x = (x - mu) / sd

    max_tau = max(2, min(h, w) // 2)
    taus = [t for t in [1, 2, 3, 4, 6, 8, 12, 16, 24] if t <= max_tau]
    decay = {t: corr_at_tau(x, t) for t in taus}

    corr1 = decay[1]
    # decay length
    half_dist = taus[-1]
    for t in taus:
        if decay[t] < 0.5 * corr1:
            half_dist = t
            break
    # long-range tail score (relative, so smoothness does not dominate)
    tail_taus = [t for t in (4, 8, 16) if t in decay]
    lr_score = (sum(max(decay[t], 0.0) for t in tail_taus) / len(tail_taus)
                / (abs(corr1) + 1e-8)) if tail_taus else 0.0
    # power-law vs exponential fit on positive part of the curve
    pos = [(t, c) for t, c in decay.items() if c > 1e-4]
    if len(pos) >= 3:
        ts = [t for t, _ in pos]
        cs = [c for _, c in pos]
        r2_exp = fit_r2(ts, [math.log(c) for c in cs])
        r2_pow = fit_r2([math.log(t) for t in ts], [math.log(c) for c in cs])
        pow_vs_exp = r2_pow - r2_exp
    else:
        pow_vs_exp = 0.0
    return dict(name=name, h=h, w=w, corr1=round(corr1, 4),
                half_dist=half_dist, lr_score=round(lr_score, 4),
                pow_vs_exp=round(pow_vs_exp, 4),
                decay={t: round(c, 4) for t, c in decay.items()})


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def main():
    results = []
    for name in sorted(MARGIN):
        try:
            fp = fingerprint(name)
        except Exception as ex:
            print(f"  {name}: FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        results.append(fp)
        print(f"  {name:11s} {fp['h']:3d}x{fp['w']:<3d} corr1={fp['corr1']:+.3f} "
              f"half_dist={fp['half_dist']:>2d} lr_score={fp['lr_score']:.3f} "
              f"pow_vs_exp={fp['pow_vs_exp']:+.3f}", flush=True)

    names = [r["name"] for r in results]
    margins = [MARGIN[n] for n in names]
    print("\n=== rank correlation with (tcn2d - conv) final margin ===")
    for key in ("lr_score", "half_dist", "pow_vs_exp", "corr1"):
        vals = [r[key] for r in results]
        print(f"  {key:10s} spearman={spearman(vals, margins):+.3f}")

    out = Path(__file__).resolve().parent / "logs" / "fingerprint" / "fingerprint.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
