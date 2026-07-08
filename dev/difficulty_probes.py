#!/usr/bin/env python3
"""C3, done right -- a CPU battery of dataset-difficulty / predictability probes.

Motivation (BDH paper, Kosowski 2025 sec 6.5 / Fig 14): activation sparsity in a
TRAINED BDH tracks the PREDICTABILITY (surprisal-given-context) of the input, not
its classification difficulty, and it is measured PER TOKEN in a sequence. Our
original C3 (dev/sparsity_fingerprint.py) used an UNTRAINED random ReLU code and
correlated its sparsity against task difficulty -- a triple category error, and it
failed. This script instead throws EVERY cheap CPU signal we can think of at the
real target -- difficulty = 100 - v10_real_test_acc -- and reports which, if any,
actually rank the datasets. Signals fall in three families:

  UNSUP  training-free complexity / predictability (the paper's true axis):
         |corr1|, raw PCA dim, TwoNN intrinsic dim, gzip compression ratio,
         byte conditional entropy (surprisal), PCA reconstruction error,
         untrained wide-ReLU sparsity + code dim (the old C3, for reference).
  SUP    cheap supervised probes (LEGIT for routing -- train labels exist at
         runtime): 1-NN LOO error, k-NN error, nearest-class-centroid error,
         linear ridge probe error, random-ReLU-feature ridge probe error,
         Fisher separability, label silhouette.
  All numpy + stdlib only (no torch, no sklearn). Pure CPU, a few seconds/dataset.

Run:  python3 dev/difficulty_probes.py            (all local datasets)
      python3 dev/difficulty_probes.py --datasets Sudoku Chesseract Language
"""
import argparse
import glob
import gzip
import json
import math
import os

import numpy as np

DATA = os.path.expanduser("~/local/nas-comp-datasets")
N = 640           # subsample size (probes are O(N^2) but N is small)
WIDE = 256        # untrained wide-ReLU width (the C3 code)
BIAS = 1.0

# v10 real 3-seed test acc -> difficulty = 100 - acc is the ground truth target.
V10 = {"Sudoku": 97.70, "Chesseract": 57.66, "CIFARTile": 73.65, "Gutenberg": 47.28,
       "Language": 90.03, "Cryptic": 72.08, "Sedol": 56.55, "Sloan": 84.60,
       "Cauchy": 93.91, "AddNIST": 96.33, "MultNIST": 95.52, "GeoClassing": 95.36,
       "GameOfLife": 100.0, "Myofibre": 91.09, "Voxel": 81.56, "Windspeed": 19.36}


# ------------------------------------------------------------------ utilities
def part_ratio(Z):
    """Effective dimensionality = (sum eig)^2 / sum(eig^2) via the N x N Gram."""
    Zc = Z - Z.mean(0, keepdims=True)
    G = Zc @ Zc.T
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 1e-9]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else 0.0


def pca_eigs(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Zc @ Zc.T)
    return np.sort(ev[ev > 1e-9])[::-1]


def corr1_of(x):
    if x.ndim == 3:
        x = x[:, None]
    mu = x.mean((0, 2, 3), keepdims=True); sd = x.std((0, 2, 3), keepdims=True) + 1e-8
    x = (x - mu) / sd
    h = float((x[..., :, :-1] * x[..., :, 1:]).mean())
    v = float((x[..., :-1, :] * x[..., 1:, :]).mean())
    return (h + v) / 2.0


def pdist2(A, B=None):
    """Squared euclidean pairwise distances."""
    B = A if B is None else B
    aa = (A * A).sum(1)[:, None]; bb = (B * B).sum(1)[None, :]
    d = aa + bb - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def twonn_id(Xf):
    """TwoNN intrinsic-dimension estimator (Facco et al. 2017): ID from the
    ratio mu = r2/r1 of 2nd- to 1st-nearest-neighbour distances."""
    D = np.sqrt(pdist2(Xf)); np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    r1, r2 = r[:, 0], r[:, 1]
    ok = (r1 > 1e-12)
    mu = np.sort((r2[ok] / r1[ok]))
    n = len(mu)
    if n < 10:
        return 0.0
    Femp = np.arange(1, n + 1) / n
    x = np.log(mu); y = -np.log(1 - Femp * (n / (n + 1)))
    m = x > 0
    return float((x[m] @ y[m]) / (x[m] @ x[m] + 1e-12))   # slope through origin = ID


def gzip_ratio(tok):
    """Mean per-sample gzip compression ratio of the quantized bytes (lower =
    more compressible = more predictable). Kolmogorov-ish, fully training-free."""
    ratios = []
    for row in tok[:256]:
        b = row.astype(np.uint8).tobytes()
        ratios.append(len(gzip.compress(b, 6)) / max(len(b), 1))
    return float(np.mean(ratios))


def byte_cond_entropy(tok):
    """1st-order conditional entropy H(next|prev) of the flattened byte stream
    (bits/byte). Low = predictable sequence (the paper's surprisal axis)."""
    flat = tok.reshape(-1).astype(np.int64)
    if flat.size < 2:
        return 0.0
    prev, nxt = flat[:-1], flat[1:]
    C = np.zeros((256, 256), dtype=np.float64)
    np.add.at(C, (prev, nxt), 1.0)
    row = C.sum(1, keepdims=True)
    P = C / np.where(row > 0, row, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(P * np.where(P > 0, np.log2(P), 0.0)).sum(1)          # H(next|prev=r)
    w = (row[:, 0] / row.sum())
    return float((w * H).sum())


def relu_code(Xf, rng, bias):
    Din = Xf.shape[1]
    W = rng.standard_normal((Din, WIDE)) / math.sqrt(Din)
    pre = Xf @ W
    pre = (pre - pre.mean(0, keepdims=True)) / (pre.std(0, keepdims=True) + 1e-8)
    return np.maximum(pre - bias, 0.0)


# ------------------------------------------------------------- supervised probes
def _split(N_, seed=0):
    perm = np.random.default_rng(seed).permutation(N_)
    ntr = max(2, int(0.7 * N_))
    return perm[:ntr], perm[ntr:]


def ridge_probe_err(H, y, K):
    tr, te = _split(len(y))
    Y = np.eye(K)[y]
    lam = 1e-1 * H.shape[1]
    W = np.linalg.solve(H[tr].T @ H[tr] + lam * np.eye(H.shape[1]), H[tr].T @ Y[tr])
    acc = (H[te] @ W).argmax(1) == y[te]
    return 1.0 - float(acc.mean())


def knn_err(Xf, y, k):
    tr, te = _split(len(y))
    D = pdist2(Xf[te], Xf[tr])
    nn = np.argsort(D, axis=1)[:, :k]
    votes = y[tr][nn]
    pred = np.array([np.bincount(v).argmax() for v in votes])
    return 1.0 - float((pred == y[te]).mean())


def nn1_loo_err(Xf, y):
    D = pdist2(Xf); np.fill_diagonal(D, np.inf)
    pred = y[np.argmin(D, axis=1)]
    return 1.0 - float((pred == y).mean())


def ncc_err(Xf, y, K):
    tr, te = _split(len(y))
    cents = np.stack([Xf[tr][y[tr] == c].mean(0) if (y[tr] == c).any()
                      else np.zeros(Xf.shape[1]) for c in range(K)])
    pred = np.argmin(pdist2(Xf[te], cents), axis=1)
    return 1.0 - float((pred == y[te]).mean())


def fisher_sep(Xf, y, K):
    """trace(between) / trace(within) class scatter -- higher = more separable."""
    mu = Xf.mean(0)
    sw = sb = 0.0
    for c in range(K):
        Xc = Xf[y == c]
        if len(Xc) < 2:
            continue
        mc = Xc.mean(0)
        sw += ((Xc - mc) ** 2).sum()
        sb += len(Xc) * ((mc - mu) ** 2).sum()
    return float(sb / (sw + 1e-9))


def silhouette(Xf, y, K):
    """Mean silhouette of the TRUE labels (subsample) -- higher = cleaner clusters."""
    D = np.sqrt(pdist2(Xf)); n = len(y)
    s = np.zeros(n)
    for i in range(n):
        same = (y == y[i]); same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in range(K):
            if c == y[i]:
                continue
            m = (y == c)
            if m.any():
                b = min(b, D[i, m].mean())
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


# ------------------------------------------------------------------ per dataset
def probe(name):
    d = os.path.join(DATA, name)
    meta = json.load(open(os.path.join(d, "metadata")))
    x = np.load(os.path.join(d, "train_x.npy"), mmap_mode="r")
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(x.shape[0], size=min(N, x.shape[0]), replace=False))
    xs = np.asarray(x[idx], dtype=np.float64)
    if xs.ndim == 3:
        xs = xs[:, None]
    c1 = corr1_of(xs)
    mu = xs.mean((0, 2, 3), keepdims=True); sd = xs.std((0, 2, 3), keepdims=True) + 1e-8
    xf = ((xs - mu) / sd).reshape(xs.shape[0], -1)

    # quantize to bytes for the compression / surprisal probes
    flat = xs.reshape(xs.shape[0], -1)
    lo, hi = float(flat.min()), float(flat.max())
    tok = np.clip(np.rint((flat - lo) / (hi - lo) * 255.0) if hi > lo else flat * 0,
                  0, 255).astype(np.int64)

    # unsupervised
    eigs = pca_eigs(xf); cum = np.cumsum(eigs) / eigs.sum()
    pca90 = float(np.searchsorted(cum, 0.90) + 1) / len(eigs)       # frac comps for 90% var
    k = min(16, len(eigs))
    pca_recon = float(1.0 - cum[k - 1])                              # residual var at 16 comps
    h = relu_code(xf, rng, BIAS)

    out = dict(
        name=name, T=xf.shape[1], nc=int(meta.get("num_classes", 0)),
        abs_corr1=abs(round(c1, 4)),
        raw_dim=round(part_ratio(xf), 2),
        twonn_id=round(twonn_id(xf), 2),
        gzip_ratio=round(gzip_ratio(tok), 4),
        byte_condH=round(byte_cond_entropy(tok), 4),
        pca90_frac=round(pca90, 4),
        pca_recon16=round(pca_recon, 4),
        fire_frac=round(float((h > 0).mean()), 4),          # old C3
        code_dim=round(part_ratio(h), 2),                   # old C3
    )

    yp = os.path.join(d, "train_y.npy")
    if os.path.exists(yp):
        yraw = np.asarray(np.load(yp, mmap_mode="r")[idx]).reshape(-1)
        uniq = {v: i for i, v in enumerate(sorted(set(yraw.tolist())))}
        y = np.array([uniq[v] for v in yraw.tolist()]); K = len(uniq)
        # keep the probe dimensionality sane on huge inputs (project to <=256 PCs)
        if xf.shape[1] > 256:
            Zc = xf - xf.mean(0, keepdims=True)
            _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
            Xp = Zc @ Vt[:256].T
        else:
            Xp = xf
        out.update(
            nn1_err=round(nn1_loo_err(Xp, y), 4),
            knn5_err=round(knn_err(Xp, y, 5), 4),
            ncc_err=round(ncc_err(Xp, y, K), 4),
            lin_err=round(ridge_probe_err(Xp, y, K), 4),
            rf_err=round(ridge_probe_err(relu_code(xf, rng, BIAS), y, K), 4),
            fisher=round(fisher_sep(Xp, y, K), 4),
            silhouette=round(silhouette(Xp, y, K), 4),
        )
    return out


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (math.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    names = args.datasets or sorted(os.path.basename(os.path.dirname(m))
                                    for m in glob.glob(os.path.join(DATA, "*", "metadata")))
    rows = []
    for n in names:
        try:
            r = probe(n); rows.append(r)
            print(f"  probed {n:12s} T={r['T']:6d} nc={r['nc']:3d}", flush=True)
        except Exception as e:
            print(f"  {n}: FAILED {type(e).__name__}: {e}")

    ev = [r for r in rows if V10.get(r["name"]) is not None]
    diff = [100 - V10[r["name"]] for r in ev]
    signals = ["abs_corr1", "raw_dim", "twonn_id", "gzip_ratio", "byte_condH",
               "pca90_frac", "pca_recon16", "fire_frac", "code_dim",
               "nn1_err", "knn5_err", "ncc_err", "lin_err", "rf_err",
               "fisher", "silhouette"]
    n = len(ev)
    # ~5% two-sided Spearman significance threshold for this n (t-approx)
    from math import sqrt
    tcrit = 2.16 if n <= 14 else 2.04
    rho_sig = tcrit / sqrt(n - 2 + tcrit ** 2)

    print(f"\n=== Spearman |rho| vs DIFFICULTY (100 - v10 acc), n={n} datasets ===")
    print(f"    (|rho| >= {rho_sig:.2f} ~ p<0.05; SUP probes use train labels, legit for routing)\n")
    print(f"{'signal':12s} {'rho_vs_diff':>11s} {'|rho|':>6s} {'sig?':>5s}   family")
    print("-" * 52)
    ranked = []
    for s in signals:
        vals = [r.get(s) for r in ev]
        if any(v is None for v in vals):
            continue
        rho = spearman(vals, diff)
        ranked.append((s, rho))
    fam = {s: ("SUP" if s in ("nn1_err", "knn5_err", "ncc_err", "lin_err", "rf_err",
                              "fisher", "silhouette") else "UNSUP") for s in signals}
    for s, rho in sorted(ranked, key=lambda t: -abs(t[1])):
        sig = "***" if abs(rho) >= rho_sig else ""
        print(f"{s:12s} {rho:>+11.3f} {abs(rho):>6.3f} {sig:>5s}   {fam[s]}")

    out = os.path.join(os.path.dirname(__file__), "logs", "difficulty_probes.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows, "n_eval": n, "rho_sig": rho_sig}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
