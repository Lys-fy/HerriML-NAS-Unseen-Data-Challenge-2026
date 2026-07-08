#!/usr/bin/env python3
"""Run the v11.1 pipeline end-to-end on one dataset and report the true
full-budget test accuracy -- the informed-defaults FLOOR of the budget-adaptive
ladder, for the 3-seed bake-off against egnas_faithful / evo.

v11.1 = v10 (fingerprint route -> family default -> family-matched aug probe ->
default-incumbent depth -> train-to-budget, cosine-to-0) with one change: the
fragile random arm is replaced by a verified early-attention probe on board/
encoded data (baselines/v11_1/nas.py). This runner mirrors validate_winner.py
(DataProcessor -> search -> Trainer.train -> predict) so its number is directly
comparable to the egnas/evo validated results and to v10's real numbers.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # seed must be set BEFORE importing trainer/nas (read at module load)
    os.environ["BASELINE_SEED"] = str(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BASELINE_LOG_DIR", str(out))

    V11 = Path(__file__).resolve().parent.parent / "baselines" / "v11_1"
    sys.path.insert(0, str(V11))
    from nas import NAS                       # noqa: E402
    from data_processor import DataProcessor   # noqa: E402
    from trainer import Trainer                 # noqa: E402

    V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
                "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
                "Sloan": 84.60, "Cryptic": 72.08}

    class Clock:
        def __init__(self, hours):
            self.t0 = time.perf_counter(); self.limit = self.t0 + hours * 3600

        def check(self):
            return self.limit - time.perf_counter()

    d = Path(args.dataset_dir)
    tr = (np.load(d / "train_x.npy"), np.load(d / "train_y.npy"))
    va = (np.load(d / "valid_x.npy"), np.load(d / "valid_y.npy"))
    test_x = np.load(d / "test_x.npy")
    test_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    code = meta.get("codename", d.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clock = Clock(args.hours)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock)
    train_loader, valid_loader, test_loader = dp.process()

    ref = V10_REAL.get(code) or V10_REAL.get(d.name)
    print(f"[{code}] v11.1 run seed={args.seed} at {args.hours}h (v10 real={ref})", flush=True)

    # search (consumes part of the clock) then train the winner to the rest
    model = NAS(train_loader, valid_loader, meta, clock).search()
    trainer = Trainer(model, device, train_loader, valid_loader, meta, clock)
    trainer.train()
    pred = np.asarray(trainer.predict(test_loader))

    test_acc = None
    if test_y is not None:
        n = min(len(pred), len(test_y))
        test_acc = round(100.0 * float((pred[:n] == test_y[:n]).mean()), 3)
    # recover the final arch string from the decision record if present
    arch = "?"
    rec_p = out / f"{code}_decisions.json"
    if rec_p.exists():
        try:
            stages = json.loads(rec_p.read_text()).get("stages", [])
            fin = [s for s in stages if s.get("stage") == "final"]
            arch = fin[-1].get("arch", "?") if fin else "?"
        except Exception:
            pass

    rec = {"codename": code, "dataset": d.name, "pipeline": "v11.1", "seed": args.seed,
           "hours": args.hours, "arch": arch, "full_budget_test_acc": test_acc,
           "v10_real_test_acc": ref,
           "beats_v10_real": (test_acc > ref) if (test_acc is not None and ref) else None,
           "margin_vs_v10": round(test_acc - ref, 3) if (test_acc is not None and ref) else None}
    (out / f"{code}_v11_1.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] v11.1 DONE: test {test_acc}%  vs v10 {ref}%  "
          f"margin {rec['margin_vs_v10']}  arch={arch}", flush=True)


if __name__ == "__main__":
    main()
