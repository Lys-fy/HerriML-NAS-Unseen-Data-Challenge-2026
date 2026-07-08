#!/usr/bin/env python3
"""v12 integration test: the FULL v11.1 pipeline (DataProcessor -> NAS.search()
-> Trainer, exact grader call sequence) plus the two screen winners layered on
top, at a real total budget:

  - TrivialAugment train-time aug (set TRAIN_TRIVIAL_AUG=1 in the env BEFORE
    launching -- the trainer reads it at import time);
  - K-seed training of the NAS-chosen architecture (SHARED init, trajectories
    varied via the trainer seed -- the R2-validated regime) followed by
    VAL-GATED FINAL SELECTION: enumerate {each single seed, K-seed logit
    ensemble} x {no TTA, flip-TTA}, score each on validation, ship the winner
    to test. Mixed-result techniques (TTA, ensembling) become safe by
    construction: nothing deploys without a measured val win on THIS dataset.

This answers the campaign's decisive question: do the screen deltas survive on
top of the strong deployed pipeline, i.e. does v12 beat v10-real/v11.1-s42?

--plain gives the pure v11.1 path (k=1, no selection, no TTA) for controlled
arms (e.g. the C1/C4 in-pipeline op tests via MACRO_OPS / V11_SHARED_REPEAT).

Budget layout: NAS probes spend from the shared clock exactly as in v11.1;
the REMAINING clock after search is split evenly across the K seed trainings
(each seed gets its own sub-clock, so train-to-budget sizes each schedule
correctly). Selection itself costs ~2(K+1) val passes, negligible.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V11 = Path(__file__).resolve().parent.parent / "baselines" / "v11_1"
sys.path.insert(0, str(V11))
import trainer as trainer_mod                        # noqa: E402
from data_processor import DataProcessor              # noqa: E402
from nas import NAS                                    # noqa: E402
from trainer import Trainer                             # noqa: E402

V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
            "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
            "Sloan": 84.60, "Cryptic": 72.08, "AddNIST": 96.33, "MultNIST": 95.52,
            "GeoClassing": 95.36, "GameOfLife": 100.0, "Myofibre": 91.09,
            "Voxel": 81.56, "Windspeed": 19.36}
SAFETY_S = 90.0


class Clock:
    def __init__(self, hours=None, deadline=None):
        self.start_time = time.perf_counter()
        self.time_limit = deadline if deadline is not None else self.start_time + hours * 3600

    def check(self):
        return self.time_limit - time.perf_counter()


@torch.no_grad()
def softmax_probs(model, loader, device, tta):
    model.eval()
    outs = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device, memory_format=torch.channels_last)
        p = model(xb).softmax(1)
        if tta:
            p = 0.5 * (p + model(torch.flip(xb, dims=[3])).softmax(1))
        outs.append(p.cpu())
    return torch.cat(outs).numpy()


def acc(probs, y):
    n = min(len(probs), len(y))
    return round(100.0 * float((probs[:n].argmax(1) == y[:n]).mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=2.0, help="TOTAL budget (search + all seeds)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tag", default="v12")
    ap.add_argument("--plain", action="store_true",
                    help="pure v11.1 path: k=1, no selection, no TTA")
    args = ap.parse_args()
    if args.plain:
        args.k = 1

    d = Path(args.dataset_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr_x = np.load(d / "train_x.npy"); tr_y = np.load(d / "train_y.npy")
    va_x = np.load(d / "valid_x.npy"); va_y = np.load(d / "valid_y.npy")
    te_x = np.load(d / "test_x.npy")
    te_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    code = meta.get("codename", d.name)
    ref = V10_REAL.get(code) or V10_REAL.get(d.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clock = Clock(hours=args.hours)
    meta["time_remaining"] = clock.check()
    import os as _os
    print(f"[{code}] v12 integration tag={args.tag} total={args.hours}h k={args.k} "
          f"plain={args.plain} TA={_os.environ.get('TRAIN_TRIVIAL_AUG')=='1'} v10={ref}", flush=True)

    # --- exact grader sequence: DataProcessor -> NAS.search() ----------------
    dp = DataProcessor(tr_x, tr_y, va_x, va_y, te_x, meta, clock)
    train_loader, valid_loader, test_loader = dp.process()
    meta["time_remaining"] = clock.check()
    model = NAS(train_loader, valid_loader, meta, clock).search()
    n_params = int(sum(p.numel() for p in model.parameters()))
    search_spent = args.hours * 3600 - clock.check()
    print(f"[{code}] search done in {search_spent/60:.1f}min, params={n_params/1e6:.2f}M, "
          f"{clock.check()/60:.1f}min left for {args.k} seed(s)", flush=True)

    # --- K seed trainings, SHARED init, even split of the remaining clock ----
    init_state = copy.deepcopy(model.state_dict())
    states, seed_vals = [], []
    for i in range(args.k):
        remaining = clock.check() - SAFETY_S
        slice_s = max(60.0, remaining / (args.k - i))
        sub = Clock(deadline=time.perf_counter() + slice_s)
        model.load_state_dict(init_state)
        trainer_mod.SEED = 42 + i                      # vary trajectory only
        _os.environ["BASELINE_LOG_DIR"] = str(out / args.tag / f"seed{i}")
        t = Trainer(model, device, train_loader, valid_loader, meta, sub)
        t.train()                                      # leaves best_state loaded
        states.append({k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()})
        seed_vals.append(round(t.best_val_acc, 4))
        print(f"[{code}] seed {i}: best_val={t.best_val_acc:.4f} "
              f"({slice_s/60:.1f}min slice)", flush=True)

    yva = va_y.astype(np.int64)
    yte = te_y.astype(np.int64) if te_y is not None else None

    if args.plain:
        model.load_state_dict(states[0])
        model = model.to(device, memory_format=torch.channels_last)
        test_probs = softmax_probs(model, test_loader, device, tta=False)
        chosen, val_scores, test_scores = "single0", {}, {}
        test_acc = acc(test_probs, yte) if yte is not None else None
    else:
        # --- val-gated final selection ---------------------------------------
        # cache per-(state, tta) probs once; every config is derived from these
        val_p, test_p = {}, {}
        for i, st in enumerate(states):
            model.load_state_dict(st)
            model = model.to(device, memory_format=torch.channels_last)
            for tta in (False, True):
                key = (i, tta)
                val_p[key] = softmax_probs(model, valid_loader, device, tta)
                test_p[key] = softmax_probs(model, test_loader, device, tta)
        configs = {}
        for tta in (False, True):
            suf = "_tta" if tta else ""
            for i in range(args.k):
                configs[f"single{i}{suf}"] = (val_p[(i, tta)], test_p[(i, tta)])
            configs[f"ensemble{suf}"] = (
                np.mean([val_p[(i, tta)] for i in range(args.k)], axis=0),
                np.mean([test_p[(i, tta)] for i in range(args.k)], axis=0))
        val_scores = {name: acc(v, yva) for name, (v, _) in configs.items()}
        # deterministic tie-break: iterate simple-first (singles, then ensemble,
        # then TTA variants -- dict insertion order above), strict improvement only
        chosen = max(configs, key=lambda n: val_scores[n])
        test_scores = {name: (acc(t_, yte) if yte is not None else None)
                       for name, (_, t_) in configs.items()}
        test_acc = test_scores[chosen]
        print(f"[{code}] selection: {val_scores} -> chose {chosen}", flush=True)

    rec = {"codename": code, "dataset": d.name, "pipeline": "v12_integration",
           "tag": args.tag, "hours": args.hours, "k": args.k, "plain": args.plain,
           "trivial_aug": _os.environ.get("TRAIN_TRIVIAL_AUG") == "1",
           "params_M": round(n_params / 1e6, 3),
           "search_spent_min": round(search_spent / 60, 1),
           "seed_vals": seed_vals, "chosen": chosen,
           "val_scores": val_scores, "test_scores": test_scores,
           "test_acc": test_acc, "v10_real_test_acc": ref,
           "beats_v10_real": (test_acc > ref) if (test_acc is not None and ref) else None,
           "margin_vs_v10": round(test_acc - ref, 3) if (test_acc is not None and ref) else None}
    (out / f"{code}_{args.tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] {args.tag} DONE: test {test_acc}% ({chosen})  vs v10 {ref}%  "
          f"margin {rec['margin_vs_v10']}", flush=True)


if __name__ == "__main__":
    main()
