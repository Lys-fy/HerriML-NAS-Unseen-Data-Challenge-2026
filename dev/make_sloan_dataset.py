#!/usr/bin/env python3
"""Sloan: photometric source classification (built from Kaggle Playground S6E6).

The custom RQ2 generalisation set #6. We deliberately make it HARD and HONEST:
we DROP the giveaway columns (redshift, spectral_type, galaxy_population -- the
first alone nearly solves the task, the other two are leaked labels) and keep
only raw photometry u,g,r,i,z plus sky position alpha,delta. Predicting
GALAXY / QSO / STAR from photometry alone is the real historical astronomy
problem and is genuinely hard (it is why spectroscopic surveys exist).

Encoding (physics, not an arbitrary reshape): the five bands are a coarse SED
ordered by wavelength u<g<r<i<z. We build, per source, a (2, 5, 5) tensor:
  ch0  colour matrix  C[i,j] = b_i - b_j   (antisymmetric; all 10 colour indices)
  ch1  sum matrix     S[i,j] = b_i + b_j   (symmetric; absolute brightness scale)
Sky position (alpha, delta) is dropped: it is a weak, possibly survey-leaking
feature, and as a constant broadcast plane it only inflated the fingerprint. The
colour/sum matrices are smooth across adjacent wavelengths, so Sloan routes
image-like. Giveaways gone, this lands a moderate-to-hard benchmark, not the
~95% the raw table allows.

Regenerable from the seed; arrays are gitignored, this generator is committed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
REPO = Path(__file__).resolve().parent.parent
SRC = Path.home() / "local/downloads/playground-series-s6e6.zip"
OUT = REPO / "dev/datasets/Sloan"
BANDS = ["u", "g", "r", "i", "z"]
POS = ["alpha", "delta"]
LABELS = {"GALAXY": 0, "QSO": 1, "STAR": 2}
# balanced split: equal per class (chance = 1/3, accuracy is meaningful)
PER_CLASS = {"train": 20_000, "valid": 5_000, "test": 5_000}

rng = np.random.default_rng(SEED)


def load():
    import zipfile
    with zipfile.ZipFile(SRC) as z:
        with z.open("train.csv") as f:
            df = pd.read_csv(f)
    df = df[df["class"].isin(LABELS)].copy()
    df["y"] = df["class"].map(LABELS).astype(np.int64)
    # robust clip per feature (guard SDSS sentinels / outliers), then we
    # standardise on the TRAIN split only (fit below).
    for c in BANDS + POS:
        lo, hi = df[c].quantile([0.001, 0.999])
        df[c] = df[c].clip(lo, hi)
    return df


def balanced_idx(y, per_class, taken):
    """Pick `per_class` indices for EACH class from the pool not in `taken`."""
    out = []
    for c in sorted(LABELS.values()):
        pool = np.setdiff1d(np.where(y == c)[0], taken, assume_unique=False)
        k = min(len(pool), per_class)
        out.extend(rng.choice(pool, size=k, replace=False).tolist())
    return np.array(out, dtype=np.int64)


def encode(feat):
    """feat: (N, 5) standardised bands [u,g,r,i,z] -> (N, 2, 5, 5) float32."""
    b = feat                                 # (N,5) bands
    color = b[:, :, None] - b[:, None, :]    # (N,5,5) b_i - b_j
    ssum = b[:, :, None] + b[:, None, :]     # (N,5,5) b_i + b_j
    x = np.stack([color, ssum], axis=1).astype(np.float32)  # (N,2,5,5)
    return x


def main():
    df = load()
    y = df["y"].to_numpy()
    print(f"loaded {len(df)} rows; class balance "
          f"{ {k: int((y==v).sum()) for k,v in LABELS.items()} }")

    tr = balanced_idx(y, PER_CLASS["train"], np.array([], dtype=np.int64))
    va = balanced_idx(y, PER_CLASS["valid"], tr)
    te = balanced_idx(y, PER_CLASS["test"], np.concatenate([tr, va]))

    raw = df[BANDS].to_numpy(dtype=np.float64)        # photometry only
    mu, sd = raw[tr].mean(0), raw[tr].std(0) + 1e-8   # fit on TRAIN only
    std = (raw - mu) / sd

    # sanity: is there signal above chance from photometry alone? (1/3 = chance)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        lr = LogisticRegression(max_iter=200, multi_class="multinomial")
        lr.fit(std[tr], y[tr])
        acc = accuracy_score(y[te], lr.predict(std[te]))
        print(f"  [sanity] logistic-regression test acc = {100*acc:.1f}% (chance 33.3%)")
    except Exception as e:
        print(f"  [sanity] skipped ({e})")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, idx in [("train", tr), ("valid", va), ("test", te)]:
        x = encode(std[idx])
        yy = y[idx].astype(np.int64)
        np.save(OUT / f"{name}_x.npy", x)
        np.save(OUT / f"{name}_y.npy", yy)
        print(f"  {name}: x{x.shape} y{yy.shape} balance "
              f"{np.bincount(yy, minlength=3).tolist()}")

    meta = {"num_classes": 3,
            "input_shape": [len(tr), 2, 5, 5],
            "codename": "Sloan",
            "benchmark": 0.0}
    (OUT / "metadata").write_text(json.dumps(meta))
    print(f"wrote {OUT}  meta={meta}")


if __name__ == "__main__":
    main()
