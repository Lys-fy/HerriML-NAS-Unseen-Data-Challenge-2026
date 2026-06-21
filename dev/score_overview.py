#!/usr/bin/env python3
"""Overview of all current extension-sweep test results vs the v10 s42 baseline.
Auto-maps codename->dataset dir via metadata. Run on spark."""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

DD = Path.home() / "local/nas-comp-datasets"
REPO = Path.home() / "projects/HerriML-NAS-Unseen-Data-Challenge-2026"
BASE = REPO / "dev/predictions-v10-v10_s42"   # v10 single-seed baseline

code2dir, code2name = {}, {}
for d in DD.iterdir():
    mf = d / "metadata"
    if mf.exists():
        c = json.loads(mf.read_text())["codename"]
        code2dir[c] = d
        code2name[c] = d.name


def acc(pred_dir, code):
    pf = Path(pred_dir) / f"{code}.npy"
    if not pf.exists():
        return None
    ty = np.load(code2dir[code] / "test_y.npy")
    pr = np.load(pf)
    return 100 * accuracy_score(ty[:len(pr)], pr)


PROBES = [
    ("arch:attn",   "dev/predictions-arch_probe-attn"),
    ("arch:recur",  "dev/predictions-arch_probe-recur"),
    ("arch:modern", "dev/predictions-arch_probe-modern"),
    ("ensemble",    "dev/predictions-ensemble_probe"),
    ("width",       "dev/predictions-width_search"),
    ("macro",       "dev/predictions-macro_probe"),
]

for label, pd in PROBES:
    pdp = REPO / pd
    npys = sorted(pdp.glob("*.npy")) if pdp.exists() else []
    if not npys:
        print(f"\n### {label:12s}  (no results yet)")
        continue
    print(f"\n### {label:12s}  ({pd})")
    print(f"  {'dataset':14s} {'codename':10s} {'v10_s42':>8s} {label.split(':')[0]:>8s} {'delta':>8s}")
    for f in npys:
        code = f.stem
        if code not in code2dir:
            continue
        base = acc(BASE, code)
        x = acc(pdp, code)
        bs = f"{base:7.2f}" if base is not None else "   -   "
        d = f"{x-base:+7.2f}" if (x is not None and base is not None) else "   -   "
        print(f"  {code2name[code]:14s} {code:10s} {bs} {x:8.2f} {d}  {'WIN' if (base is not None and x>base) else ''}")
