#!/usr/bin/env python3
"""Fixed-arch A/B for the BDH-harvested ops (C1 LinearAttn2D, C2 SparseGated,
C4 weight-shared depth). Isolates the marginal value of ONE harvested op by
training a fixed default backbone (all-Conv3, v10 recipe: full Trainer, cosine
train-to-budget + EMA + aug) with and without the op injected, and comparing
full-budget test accuracy against v10's known real number.

This answers "do the new ops help at all?" BEFORE any search -- if a single
injected op doesn't move the needle in a controlled A/B, it won't rescue the
search. Reuses v10's build_cnn / DataProcessor / Trainer, exactly like
validate_winner.py.

Variants (via flags, so the queue picks combos without editing this file):
  --inject OPNAME   replace the block at --at with this op (default: none)
  --at IDX          stage index to inject at (default: n_blocks // 2)
  --repeat K        C4 weight-shared depth: apply each shape-preserving block xK
  --tag STR         label for the output file ({code}_ab_{tag}.json / .csv)

Examples:
  default : --tag default
  C1      : --inject LinearAttn2D --tag C1
  C2      : --inject SparseGated  --tag C2
  C4      : --repeat 3            --tag C4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from helpers import build_cnn, build_tcn, OP_NAMES  # noqa: E402
from data_processor import DataProcessor          # noqa: E402
from trainer import Trainer                        # noqa: E402

# v10's known real test accuracies (3-seed s42), by dataset name AND codename.
V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
            "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
            "Sloan": 84.60, "Cryptic": 72.08, "AddNIST": 96.33, "MultNIST": 95.52,
            "GeoClassing": 95.36, "GameOfLife": 100.0, "Myofibre": 91.09,
            "Voxel": 81.56, "Windspeed": 19.36}


class Clock:
    def __init__(self, hours):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + hours * 3600

    def check(self):
        return self.time_limit - time.perf_counter()


def load_dataset(d):
    tr = (np.load(d / "train_x.npy"), np.load(d / "train_y.npy"))
    va = (np.load(d / "valid_x.npy"), np.load(d / "valid_y.npy"))
    test_x = np.load(d / "test_x.npy")
    test_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    return tr, va, test_x, test_y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--width", type=int, default=48)
    # backbone family: cnn = build_cnn (width-doubling res-blocks, supports
    # --inject/--repeat); tcn2d/tcn1d = the fixed dilated backbones (depth via
    # --blocks; inject/repeat not applicable there).
    ap.add_argument("--family", choices=("cnn", "tcn2d", "tcn1d"), default="cnn")
    ap.add_argument("--inject", default=None, help="op to inject at --at (default none)")
    ap.add_argument("--at", type=int, default=None, help="stage index (default n_blocks//2)")
    ap.add_argument("--repeat", type=int, default=1, help="C4 weight-shared depth k")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.inject is not None and args.inject not in OP_NAMES:
        raise SystemExit(f"--inject {args.inject!r} not in OP_VOCAB {OP_NAMES}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # per-tag subdir for the Trainer per-epoch CSV so A/B variants of the same
    # dataset (which the Trainer writes as {codename}.csv) do not clobber.
    csv_dir = out / args.tag; csv_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BASELINE_LOG_DIR"] = str(csv_dir)
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clock = Clock(args.hours)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock)
    train_loader, valid_loader, test_loader = dp.process()
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])

    ops = ["Conv3"] * args.blocks
    at = args.at if args.at is not None else args.blocks // 2
    if args.family != "cnn":
        if args.inject is not None or args.repeat > 1:
            raise SystemExit("--inject/--repeat only apply to --family cnn")
        model = build_tcn(args.family, in_ch, nc, h, w,
                          base_width=args.width, n_blocks=args.blocks)
        desc = f"{args.family} {args.blocks}L W{args.width}"
    else:
        if args.inject is not None:
            at = max(0, min(args.blocks - 1, at))
            ops[at] = args.inject
        model = build_cnn(in_ch, nc, h, w, ops=ops, base_width=args.width,
                          drop_path_rate=0.1, shared_repeat=args.repeat)
        desc = (f"default {args.blocks}L W{args.width}"
                + (f" +{args.inject}@{at}" if args.inject else "")
                + (f" xrepeat{args.repeat}" if args.repeat > 1 else ""))
    n_params = sum(p.numel() for p in model.parameters())

    ref = V10_REAL.get(code) or V10_REAL.get(d.name)
    print(f"[{code}] A/B tag={args.tag} ({desc}) {args.hours}h  "
          f"params={n_params/1e6:.2f}M  v10 real={ref}", flush=True)

    trainer = Trainer(model, device, train_loader, valid_loader, meta, clock)
    trainer.train()
    pred = np.asarray(trainer.predict(test_loader))
    test_acc = None
    if test_y is not None:
        n = min(len(pred), len(test_y))
        test_acc = round(100.0 * float((pred[:n] == test_y[:n]).mean()), 3)

    rec = {"codename": code, "dataset": d.name, "pipeline": "harvest_ab",
           "tag": args.tag, "family": args.family, "blocks": args.blocks,
           "inject": args.inject, "at": at if args.inject else None,
           "repeat": args.repeat, "ops": ops if args.family == "cnn" else None,
           "width": args.width,
           "seed": args.seed, "hours": args.hours, "params_M": round(n_params / 1e6, 3),
           "full_budget_test_acc": test_acc, "v10_real_test_acc": ref,
           "beats_v10_real": (test_acc > ref) if (test_acc is not None and ref) else None,
           "margin_vs_v10": round(test_acc - ref, 3) if (test_acc is not None and ref) else None}
    (out / f"{code}_ab_{args.tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] A/B {args.tag} DONE: test {test_acc}%  vs v10 {ref}%  "
          f"margin {rec['margin_vs_v10']}  beats_v10={rec['beats_v10_real']}", flush=True)


if __name__ == "__main__":
    main()
