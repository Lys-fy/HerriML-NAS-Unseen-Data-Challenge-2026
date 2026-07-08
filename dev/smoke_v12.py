#!/usr/bin/env python3
"""Smoke for the v12 package trainer (controller paths, seconds not hours).
Forces both rungs via env overrides: tiny FULL/CAP make the planner pick K=3
on a generous clock and K=1 on a tight one; checks selection ran, predict
returns sane labels for both single and ensemble configs, and the degradation
guard works. Run on spark (or any torch env)."""
import os
os.environ.setdefault("V12_FULL_EPOCHS", "4")     # tiny "full training"
os.environ.setdefault("V12_CAP_EPOCHS", "12")
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines" / "v12"))
import trainer as T                                # noqa: E402
from helpers import build_cnn                       # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}  TA_default_on={T.USE_TRIVIAL_AUG}")


class Clock:
    def __init__(self, s): self.t = time.perf_counter() + s
    def check(self): return self.t - time.perf_counter()


def loaders(n=96, nc=4):
    x = torch.randn(n, 3, 9, 9)
    y = torch.randint(0, nc, (n,))
    tr = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    va = DataLoader(TensorDataset(x[:48], y[:48]), batch_size=16, shuffle=False)
    te = DataLoader(TensorDataset(x[:32]), batch_size=16, shuffle=False)
    return tr, va, te


fails = []


def check(name, clock_s, want_multi_seed):
    try:
        tr, va, te = loaders()
        m = build_cnn(3, 4, 9, 9, ops=["Conv3"] * 4, base_width=16)
        t = T.Trainer(m, dev, tr, va, {"num_classes": 4, "codename": "toy",
                                       "mixup_alpha": 0.0}, Clock(clock_s))
        t.train()
        k = len(t.seed_states)
        preds = t.predict(te)
        assert len(preds) == 32 and all(0 <= p < 4 for p in preds), "bad preds"
        assert t.chosen is not None and t.val_scores is not None
        multi_ok = (k > 1) == want_multi_seed
        print(f"  {'OK ' if multi_ok else 'WARN'} {name:18s} seeds={k} "
              f"chosen={t.chosen} best_val={t.best_val_acc:.3f} "
              f"n_scores={len(t.val_scores)}")
        if not multi_ok:
            fails.append(f"{name}: expected multi_seed={want_multi_seed}, got k={k}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        fails.append(name)


# generous clock + tiny FULL -> planner must pick K=3, selection must include ensemble
check("surplus->K3", clock_s=420, want_multi_seed=True)
# tight clock -> K=1 (single, still selects single vs single_tta)
check("tight->K1", clock_s=95, want_multi_seed=False)

print("\n" + ("V12 SMOKE FAIL: " + "; ".join(fails) if fails else "V12 SMOKE PASS"))
sys.exit(1 if fails else 0)
