#!/usr/bin/env python3
"""Smoke for the v12.1 package trainer. Forces the NEW controller paths on toy
data in seconds: reactive top-up seeds, top-m ensemble configs, noise-banded
selection, heterogeneous alt-family seed, independent-init mode, plus the v12
regression paths (K plan, tight-budget fallback, predict)."""
import os
os.environ.setdefault("V12_FULL_EPOCHS", "4")
os.environ.setdefault("V12_CAP_EPOCHS", "8")
os.environ.setdefault("V12_HARD_CAP_SEEDS", "5")
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines" / "v12_1"))
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


def check(name, clock_s, want_min_seeds, alt=False, indep=False):
    T.INDEP_INIT = indep
    try:
        tr, va, te = loaders()
        m = build_cnn(3, 4, 9, 9, ops=["Conv3"] * 4, base_width=16)
        if alt:  # heterogeneous alt-family + capacity seeds via the side-channel
            m._v12_meta = {"family": "conv", "alt_family": "toy-alt",
                           "gate_margin": 0.01, "gate_noise": 0.02,
                           "alt_model": build_cnn(3, 4, 9, 9, ops=["SepConv3"] * 3,
                                                  base_width=12),
                           "cap_model": build_cnn(3, 4, 9, 9, ops=["Conv3"] * 4,
                                                  base_width=32)}
        t = T.Trainer(m, dev, tr, va, {"num_classes": 4, "codename": "toy",
                                       "mixup_alpha": 0.0}, Clock(clock_s))
        t.train()
        k = len(t.seed_states)
        preds = t.predict(te)
        assert len(preds) == 32 and all(0 <= p < 4 for p in preds), "bad preds"
        n_topm = sum(1 for nm in t.val_scores if nm.startswith("top"))
        distinct_models = len({id(mm) for mm in t.seed_models})
        ok = k >= want_min_seeds
        if alt:
            ok = ok and distinct_models >= 3   # primary + alt-family + capacity
        print(f"  {'OK ' if ok else 'FAIL'} {name:16s} seeds={k} models={distinct_models} "
              f"topm_cfgs={n_topm} chosen={t.chosen[0]}/{t.chosen[2]} "
              f"best_val={t.best_val_acc:.3f} n_scores={len(t.val_scores)}")
        if not ok:
            fails.append(name)
    except Exception as e:
        import traceback; traceback.print_exc()
        fails.append(name)


# 1. surplus: planned K=3 (cap 8 epochs) + reactive TOP-UPS up to hard cap 5
check("topup->K>=4", clock_s=600, want_min_seeds=4)
# 2. tight: K=1 fallback survives
check("tight->K1", clock_s=95, want_min_seeds=1)
# 3. heterogeneous alt-family seed occupies one slot
check("alt-seed", clock_s=420, want_min_seeds=3, alt=True)
# 4. independent inits train and select fine
check("indep-init", clock_s=300, want_min_seeds=2, indep=True)

print("\n" + ("V12.1 SMOKE FAIL: " + ",".join(fails) if fails else "V12.1 SMOKE PASS"))
sys.exit(1 if fails else 0)
