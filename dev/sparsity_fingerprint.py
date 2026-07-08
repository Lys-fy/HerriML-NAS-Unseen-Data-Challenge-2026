#!/usr/bin/env python3
"""C3 -- activation-sparsity fingerprint (training-free), from the BDH paper
(Kosowski 2025, Sec 6.4 / Fig 14): a wide ReLU code's activation sparsity /
richness tracks input complexity. Pure-CPU, no training: project a sample of
each dataset through an UNTRAINED wide random ReLU block and measure code
statistics, then compare against complexity references (v10 task difficulty,
raw intrinsic dimension, num_classes) and our existing corr1 fingerprint.

Signals per dataset (averaged over a few random projections):
  corr1        local pixel correlation (our routing fingerprint; ~0 = encoded)
  raw_dim      effective dimensionality of the RAW input (PCA participation ratio)
  fire_frac    fraction of wide-ReLU neurons that fire (higher ~ heavier-tailed)
  fire_var     sample-to-sample variation in firing (higher ~ diverse inputs)
  code_dim     effective dimensionality of the sparse CODE (PR) -- the C3 richness
  code_ent     neuron-usage entropy of the code (0..1; higher ~ more neurons used)

Run:  python3 dev/sparsity_fingerprint.py            (all local datasets)
"""
import argparse
import glob
import json
import math
import os

import numpy as np

DATA = os.path.expanduser("~/local/nas-comp-datasets")
N_SAMPLES = 512
WIDE = 256
BIAS = 1.0          # negative bias before ReLU -> induces sparsity (~16% baseline)
SEEDS = (0, 1, 2)

# v10 real raw test acc = task-difficulty reference (low acc = hard = complex)
V10 = {"Sudoku": 97.70, "Chesseract": 57.66, "CIFARTile": 73.65, "Gutenberg": 47.28,
       "Language": 90.03, "Cryptic": 74.25, "Sedol": 56.55, "Sloan": 84.60,
       "Cauchy": 93.91, "AddNIST": 96.33, "MultNIST": 95.52, "GeoClassing": 95.36,
       "GameOfLife": 100.0, "Myofibre": 91.09, "Voxel": 81.56, "Windspeed": 19.36}


def part_ratio(Z):
    """Effective dimensionality = (sum lambda)^2 / sum(lambda^2) of the sample
    covariance, computed via the N x N Gram matrix (cheap for wide inputs)."""
    Zc = Z - Z.mean(0, keepdims=True)
    G = Zc @ Zc.T
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 1e-9]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else 0.0


def corr1_of(x):
    if x.ndim == 3:
        x = x[:, None]
    mu = x.mean((0, 2, 3), keepdims=True); sd = x.std((0, 2, 3), keepdims=True) + 1e-8
    x = (x - mu) / sd
    h = float((x[..., :, :-1] * x[..., :, 1:]).mean())
    v = float((x[..., :-1, :] * x[..., 1:, :]).mean())
    return (h + v) / 2.0


def _relu_code(Xflat, rng, bias):
    Din = Xflat.shape[1]
    W = rng.standard_normal((Din, WIDE)) / math.sqrt(Din)
    pre = Xflat @ W
    pre = (pre - pre.mean(0, keepdims=True)) / (pre.std(0, keepdims=True) + 1e-8)
    return np.maximum(pre - bias, 0.0)


def rf_probe(H, y, split=0.7):
    """Random-feature ridge linear-probe held-out accuracy -> cheap DIFFICULTY
    estimate (low acc = hard = complex). Uses labels (available at runtime)."""
    N = len(y)
    perm = np.random.default_rng(0).permutation(N)
    H, y = H[perm], y[perm]
    uniq = {v: i for i, v in enumerate(sorted(set(y.tolist())))}
    yy = np.array([uniq[v] for v in y.tolist()]); K = len(uniq)
    ntr = max(2, int(split * N))
    Htr, Hte, ytr, yte = H[:ntr], H[ntr:], yy[:ntr], yy[ntr:]
    if len(Hte) < 2:
        return 0.0
    Ytr = np.eye(K)[ytr]
    lam = 1e-1 * Htr.shape[1]
    Wc = np.linalg.solve(Htr.T @ Htr + lam * np.eye(Htr.shape[1]), Htr.T @ Ytr)
    return float(((Hte @ Wc).argmax(1) == yte).mean())


def sparse_stats(Xflat, rng, y=None):
    h = _relu_code(Xflat, rng, BIAS)          # ~16% fire (baseline)
    h5 = _relu_code(Xflat, np.random.default_rng(rng.integers(1 << 30)), 2.0)  # ~2-5% fire (sparser)
    hd = np.maximum(_relu_code(h, rng, 0.5), 0.0)   # 2-layer deep code
    fire = h > 0
    rate = fire.mean(0); p = rate / (rate.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum() / math.log(len(p)))
    out = dict(fire_frac=float(fire.mean()), fire_var=float(fire.mean(1).std()),
               code_dim=part_ratio(h), code_ent=ent,
               code_dim_sparse=part_ratio(h5), code_dim_deep=part_ratio(hd))
    if y is not None:
        out["rf_acc"] = rf_probe(h, y)
    return out


def fingerprint(name):
    d = os.path.join(DATA, name)
    meta = json.load(open(os.path.join(d, "metadata")))
    x = np.load(os.path.join(d, "train_x.npy"), mmap_mode="r")
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(x.shape[0], size=min(N_SAMPLES, x.shape[0]), replace=False))
    xs = np.asarray(x[idx], dtype=np.float64)
    if xs.ndim == 3:
        xs = xs[:, None]
    c1 = corr1_of(xs)
    # per-channel standardize, then flatten
    mu = xs.mean((0, 2, 3), keepdims=True); sd = xs.std((0, 2, 3), keepdims=True) + 1e-8
    xf = ((xs - mu) / sd).reshape(xs.shape[0], -1)
    raw_dim = part_ratio(xf)
    yy = None
    yp = os.path.join(d, "train_y.npy")
    if os.path.exists(yp):
        yy = np.asarray(np.load(yp, mmap_mode="r")[idx]).reshape(-1)
    accs = [sparse_stats(xf, np.random.default_rng(s), yy) for s in SEEDS]
    agg = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
    return dict(name=name, T=xf.shape[1], nc=int(meta.get("num_classes", 0)),
                corr1=round(c1, 4), raw_dim=round(raw_dim, 1), **{k: round(v, 4) for k, v in agg.items()})


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
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
            rows.append(fingerprint(n))
        except Exception as e:
            print(f"  {n}: FAILED {type(e).__name__}: {e}")
    # sort by code richness (the C3 complexity signal)
    rows.sort(key=lambda r: r["code_dim"], reverse=True)
    print(f"\n{'dataset':12s} {'T':>6s} {'nc':>3s} {'corr1':>7s} {'raw_dim':>8s} "
          f"{'fire%':>6s} {'fireVar':>7s} {'code_dim':>8s} {'code_ent':>8s} {'v10':>6s}")
    print("-" * 86)
    for r in rows:
        v10 = V10.get(r["name"])
        print(f"{r['name']:12s} {r['T']:6d} {r['nc']:3d} {r['corr1']:+7.3f} {r['raw_dim']:8.1f} "
              f"{100*r['fire_frac']:6.2f} {r['fire_var']:7.4f} {r['code_dim']:8.1f} {r['code_ent']:8.3f} "
              f"{(f'{v10:.1f}' if v10 else '  -  '):>6s}")
    # correlations vs complexity references (subset with a v10 number)
    ev = [r for r in rows if V10.get(r["name"]) is not None]
    diff = [100 - V10[r["name"]] for r in ev]   # difficulty = 100 - acc (higher = harder)
    print("\n=== Spearman of each sparsity signal vs references (n=%d official) ===" % len(ev))
    print(f"{'signal':10s} {'vs difficulty':>14s} {'vs raw_dim':>11s} {'vs num_classes':>15s} {'vs |corr1|':>11s}")
    refs = {"difficulty": diff,
            "raw_dim": [r["raw_dim"] for r in ev],
            "num_classes": [r["nc"] for r in ev],
            "|corr1|": [abs(r["corr1"]) for r in ev]}
    for sig in ("fire_frac", "fire_var", "code_dim", "code_dim_sparse", "code_dim_deep",
                "code_ent", "rf_acc", "raw_dim"):
        vals = [r.get(sig, 0.0) for r in ev]
        print(f"{sig:10s} {spearman(vals, refs['difficulty']):>14.3f} "
              f"{spearman(vals, refs['raw_dim']):>11.3f} {spearman(vals, refs['num_classes']):>15.3f} "
              f"{spearman(vals, refs['|corr1|']):>11.3f}")
    out = os.path.join(os.path.dirname(__file__), "logs", "sparsity_fingerprint.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
