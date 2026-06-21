#!/usr/bin/env python3
"""Full-budget validation of a v11.2 winner. The v11.2 search ran at 120-epoch
fidelity, which UNDERSTATES slow-training ops (attention). This rebuilds the
winning architecture and trains it with v10's REAL recipe -- the full Trainer
(cosine train-to-budget + EMA + aug) at a real wall-clock budget -- to get the
true test accuracy, and compares it to v10's known real number.

Reads the v11.2 decision JSON for the winner. Reuses v10's build_cnn / build_tcn
/ DataProcessor / Trainer."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from helpers import build_cnn, build_tcn          # noqa: E402
from data_processor import DataProcessor           # noqa: E402
from trainer import Trainer                         # noqa: E402

# v10's known real test accuracies (3-seed s42), by codename AND dataset name.
V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
            "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
            "Sloan": 84.60}


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
    ap.add_argument("--winner-json", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    wj = json.loads(Path(args.winner_json).read_text())
    winner = wj["decision"]["winner"]
    sr = wj["searched"]; stem = wj["stem"]
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clock = Clock(args.hours)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock)
    train_loader, valid_loader, test_loader = dp.process()
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])

    if winner == "searched":
        model = build_cnn(in_ch, nc, h, w, ops=sr["ops"], base_width=sr["width"],
                          drop_path_rate=0.1, stem=stem)
        desc = f"searched {sr['depth']}L W{sr['width']} ops={sr['ops']}"
    else:
        model = build_tcn(winner, in_ch, nc, h, w, base_width=48, stem=stem)
        desc = f"{winner} backbone"

    ref = V10_REAL.get(code) or V10_REAL.get(d.name)
    print(f"[{code}] FULL-BUDGET validate WINNER={winner} ({desc}) at {args.hours}h "
          f"(v10 real = {ref})", flush=True)
    trainer = Trainer(model, device, train_loader, valid_loader, meta, clock)
    trainer.train()
    pred = np.asarray(trainer.predict(test_loader))
    test_acc = None
    if test_y is not None:
        n = min(len(pred), len(test_y))
        test_acc = round(100.0 * float((pred[:n] == test_y[:n]).mean()), 3)

    rec = {"codename": code, "dataset": d.name, "winner": winner, "arch": desc,
           "hours": args.hours, "full_budget_test_acc": test_acc,
           "v10_real_test_acc": ref,
           "beats_v10_real": (test_acc > ref) if (test_acc is not None and ref) else None,
           "margin_vs_v10": round(test_acc - ref, 3) if (test_acc is not None and ref) else None}
    (out / f"{code}_validated.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] VALIDATED {winner}: full-budget test {test_acc}%  vs v10 real "
          f"{ref}%  margin {rec['margin_vs_v10']}  beats_v10={rec['beats_v10_real']}", flush=True)


if __name__ == "__main__":
    main()
