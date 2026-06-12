#!/usr/bin/env python3
"""Generate the QR custom dataset (codename Quincy).

Task: 20-way classification. Each sample is a version-1 QR code (21x21
modules, no quiet zone) encoding "C<class:02d>-<8 random alnum>" at ECC
level Q with a random mask pattern, so intra-class samples share only the
class token's contribution to the bit layout. ~10% of samples are corrupted
(random module flips OR an occlusion patch), within what ECC-Q could in
principle recover, so the label is always recoverable from the code.

Format matches the competition datasets: train/valid/test_x.npy float32,
*_y.npy int64, metadata JSON. Shape (N, 1, 21, 21), values {0,1}.
"""
import argparse
import json
import string
from pathlib import Path

import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_Q

N_CLASSES = 20
SIZE = 21  # QR version 1
CORRUPT_FRAC = 0.10
FLIP_P = 0.06          # per-module flip prob for noise corruption
OCCLUDE = 5            # occlusion patch side
ALNUM = string.ascii_uppercase + string.digits


def make_code(rng, cls):
    suffix = "".join(rng.choice(list(ALNUM), size=8))
    payload = f"C{cls:02d}-{suffix}"
    qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_Q,
                       box_size=1, border=0,
                       mask_pattern=int(rng.integers(0, 8)))
    qr.add_data(payload)
    qr.make(fit=False)
    m = np.array(qr.get_matrix(), dtype=np.float32)
    assert m.shape == (SIZE, SIZE), m.shape
    return m


def corrupt(rng, m):
    m = m.copy()
    if rng.random() < 0.5:
        flips = rng.random(m.shape) < FLIP_P
        m[flips] = 1.0 - m[flips]
    else:
        r, c = rng.integers(0, SIZE - OCCLUDE, size=2)
        m[r:r + OCCLUDE, c:c + OCCLUDE] = float(rng.integers(0, 2))
    return m


def gen_split(rng, n):
    ys = rng.integers(0, N_CLASSES, size=n)
    xs = np.empty((n, 1, SIZE, SIZE), dtype=np.float32)
    corrupted = 0
    for i, y in enumerate(ys):
        m = make_code(rng, int(y))
        if rng.random() < CORRUPT_FRAC:
            m = corrupt(rng, m)
            corrupted += 1
        xs[i, 0] = m
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n}")
    return xs, ys.astype(np.int64), corrupted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dev/datasets/Quincy")
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--valid", type=int, default=10000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for split, n in (("train", args.train), ("valid", args.valid),
                     ("test", args.test)):
        print(f"{split}: {n}")
        xs, ys, ncor = gen_split(rng, n)
        np.save(out / f"{split}_x.npy", xs)
        np.save(out / f"{split}_y.npy", ys)
        print(f"  corrupted: {ncor} ({100 * ncor / n:.1f}%)")

    meta = {"num_classes": N_CLASSES,
            "input_shape": [args.train, 1, SIZE, SIZE],
            "codename": "Quincy", "benchmark": 0.000}
    (out / "metadata").write_text(json.dumps(meta))
    print("wrote", out / "metadata", meta)


if __name__ == "__main__":
    main()
