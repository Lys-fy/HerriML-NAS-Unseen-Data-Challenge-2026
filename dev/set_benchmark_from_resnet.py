#!/usr/bin/env python3
"""Set the metadata `benchmark` (test-accuracy %) for the custom datasets from
their ResNet-18 predictions. The benchmark is the zero-point for the adjusted
score: adj = (raw - benchmark) * 10 / (100 - benchmark). Run on spark after the
resnet18 baseline writes dev/predictions-resnet18/<codename>.npy."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

DD = Path.home() / "local/nas-comp-datasets"
PRED = Path.home() / "projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/predictions-resnet18"
CODES = ["Sedol", "Sloan", "Cauchy"]

for code in CODES:
    pf = PRED / f"{code}.npy"
    ty = DD / code / "test_y.npy"
    mf = DD / code / "metadata"
    if not pf.exists() or not ty.exists():
        print(f"{code}: missing predictions or test_y, skip")
        continue
    pred = np.load(pf)
    y = np.load(ty)
    n = min(len(pred), len(y))
    acc = 100.0 * accuracy_score(y[:n], pred[:n])
    meta = json.loads(mf.read_text())
    old = meta.get("benchmark")
    meta["benchmark"] = round(float(acc), 3)
    mf.write_text(json.dumps(meta))
    print(f"{code}: benchmark {old} -> {acc:.3f}%  (n={n})")
