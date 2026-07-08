#!/usr/bin/env python3
"""Fast smoke for the Trainer recipe toggles (SAM / self-distillation /
TrivialAugment). Runs a few epochs of the real Trainer on toy data with each
toggle set, checking the loop runs and loss stays finite. Bounds epochs via
TRAIN_MAX_EPOCHS so it finishes in seconds."""
import os
os.environ.setdefault("TRAIN_MAX_EPOCHS", "3")
os.environ.setdefault("BASELINE_SEED", "0")
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
import trainer as T                                  # noqa: E402
from helpers import build_cnn                         # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}")


class Clock:
    def __init__(self, h): self.t = time.perf_counter() + h * 3600
    def check(self): return self.t - time.perf_counter()


def toy_loaders():
    x = torch.randn(64, 3, 8, 8); y = torch.randint(0, 4, (64,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


configs = {
    "baseline": {},
    "SAM": {"USE_SAM": True},
    "SelfDistill": {"USE_SELF_DISTILL": True},
    "TrivialAug": {"USE_TRIVIAL_AUG": True},
    "AdamW": {"USE_ADAMW": True},
    "mixup+SAM": {"USE_SAM": True, "_mixup": True},
}
fails = []
for name, flags in configs.items():
    T.USE_SAM = T.USE_SELF_DISTILL = T.USE_TRIVIAL_AUG = T.USE_ADAMW = False   # reset
    for k, v in flags.items():
        if not k.startswith("_"):
            setattr(T, k, v)
    meta = {"num_classes": 4, "codename": "toy",
            "mixup_alpha": 0.2 if flags.get("_mixup") else 0.0}
    try:
        m = build_cnn(3, 4, 8, 8, ops=["Conv3"] * 4, base_width=16)
        tl, vl = toy_loaders()
        tr = T.Trainer(m, dev, tl, vl, meta, Clock(0.05))
        tr.train()
        assert np.isfinite(tr.best_val_acc)
        print(f"  OK   {name:12s} best_val={tr.best_val_acc:.3f}")
    except Exception as e:
        print(f"  FAIL {name:12s} {type(e).__name__}: {e}")
        fails.append(name)

print("\n" + ("RECIPE SMOKE FAIL: " + ",".join(fails) if fails else "RECIPE SMOKE PASS"))
sys.exit(1 if fails else 0)
