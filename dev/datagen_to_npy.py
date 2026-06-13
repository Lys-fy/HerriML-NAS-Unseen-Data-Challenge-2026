#!/usr/bin/env python3
"""Convert a teammate-generated dataset (PNG + labels.csv, from dev/datagen/)
into the competition .npy format our pipeline consumes:

    train/valid/test_x.npy   float32, channels-first (N, C, H, W), values in [0,1]
    train/valid/test_y.npy   int64, class indices 0..K-1
    metadata                 JSON: num_classes, input_shape, codename, benchmark

Label values are remapped to contiguous 0..K-1 using the TRAIN split's sorted
unique labels (so mathvision's degrees 1..5 become classes 0..4); the mapping is
recorded in metadata.

Special case: polygon_collision (Pavel) is rendered as a horizontal film strip
of T frames (H x T*H x 3). With --frames_as_channels each strip is split into
its T frames, greyscaled, and stacked as a (T, H, H) channel tensor, turning the
wide RGB strip into a clean per-timestep channel stack (the competition idiom,
cf. Windspeed 12x32x32 / GameOfLife multi-step).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# dataset -> (label column, codename)
SPEC = {
    "mathvision": ("degree", "Magnus"),
    "connect4": ("winning_column", "Cornelius"),
    "polygon_collision": ("collision", "Pavel"),
    "domino": ("label", "Dominik"),
}
SPLITS = [("train", "train"), ("val", "valid"), ("test", "test")]


def strip_to_channels(rgb):
    """(H, W=T*H, 3) film strip -> (T, H, H) greyscale channel stack."""
    h, w, _ = rgb.shape
    t = w // h
    g = rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    frames = [g[:, i * h:(i + 1) * h] for i in range(t)]
    return np.stack(frames, axis=0) / 255.0  # (T, H, H)


def rgb_to_chw(rgb):
    return np.transpose(rgb[..., :3].astype(np.float32) / 255.0, (2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPEC))
    ap.add_argument("--in_dir", required=True, help="generated PNG/CSV dataset root")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frames_as_channels", action="store_true",
                    help="[polygon_collision] split the strip into T channels")
    args = ap.parse_args()

    label_col, codename = SPEC[args.dataset]
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the label remap from the train split.
    train_csv = list(csv.DictReader(open(in_dir / "train" / "labels.csv")))
    uniq = sorted({int(r[label_col]) for r in train_csv})
    remap = {v: i for i, v in enumerate(uniq)}

    stack = args.dataset == "polygon_collision" and args.frames_as_channels

    def to_arr(rgb):
        return strip_to_channels(rgb) if stack else rgb_to_chw(rgb)

    shape = None
    for src, dst in SPLITS:
        sdir = in_dir / src
        if not (sdir / "labels.csv").exists():
            continue
        rows = list(csv.DictReader(open(sdir / "labels.csv")))
        n = len(rows)
        # Preallocate from the first image so we never hold a Python list of
        # arrays (large wide images like Dominik 3x44x374 OOM otherwise).
        first = to_arr(np.array(Image.open(sdir / rows[0]["filename"]).convert("RGB")))
        x = np.empty((n,) + first.shape, dtype=np.float32)
        y = np.empty((n,), dtype=np.int64)
        x[0] = first
        y[0] = remap[int(rows[0][label_col])]
        for i in range(1, n):
            r = rows[i]
            rgb = np.array(Image.open(sdir / r["filename"]).convert("RGB"))
            x[i] = to_arr(rgb)
            y[i] = remap[int(r[label_col])]
        np.save(out_dir / f"{dst}_x.npy", x)
        np.save(out_dir / f"{dst}_y.npy", y)
        if shape is None:
            shape = list(x.shape)
        print(f"  {dst}: x={x.shape} y={y.shape} classes={len(uniq)}")

    meta = {"num_classes": len(uniq), "input_shape": shape,
            "codename": codename, "benchmark": 0.0,
            "label_remap": {str(k): v for k, v in remap.items()},
            "source_label": label_col}
    (out_dir / "metadata").write_text(json.dumps(meta))
    print("wrote", out_dir / "metadata", meta)


if __name__ == "__main__":
    main()
