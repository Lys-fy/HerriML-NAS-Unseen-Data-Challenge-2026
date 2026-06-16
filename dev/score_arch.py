#!/usr/bin/env python3
"""Score the architecture gauntlet (attn / recur / modern fixed-arch probes,
seed 42) on its 4 datasets vs v10 seed-42 test accuracy. Run on spark."""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

DD = Path.home() / "local/nas-comp-datasets"
REPO = Path.home() / "projects/HerriML-NAS-Unseen-Data-Challenge-2026"
CODES = {"Chesseract": "Chester", "Sudoku": "Sokoto",
         "Language": "LaMelo", "CIFARTile": "Caitie"}
V10_S42 = {"Chesseract": 57.66, "Sudoku": 97.70, "Language": 90.03, "CIFARTile": 73.65}

# codename -> dataset dir
code2dir = {}
for d in DD.iterdir():
    mf = d / "metadata"
    if mf.exists():
        code2dir[json.loads(mf.read_text())["codename"]] = d


def raw(pred_dir, code):
    pf = REPO / pred_dir / f"{code}.npy"
    if not pf.exists():
        return None
    ty = np.load(code2dir[code] / "test_y.npy")
    pr = np.load(pf)
    return 100 * accuracy_score(ty[:len(pr)], pr)


print(f"{'dataset':11s} {'v10(s42)':>9s} {'attn':>8s} {'recur':>8s} {'modern':>8s}")
for ds, code in CODES.items():
    a = raw("dev/predictions-arch_probe-attn", code)
    r = raw("dev/predictions-arch_probe-recur", code)
    m = raw("dev/predictions-arch_probe-modern", code)

    def cell(x, base):
        if x is None:
            return "    -   "
        return f"{x:6.2f}{'+' if x > base else ' '}"
    print(f"{ds:11s} {V10_S42[ds]:9.2f} {cell(a, V10_S42[ds])} {cell(r, V10_S42[ds])} {cell(m, V10_S42[ds])}")
print("\n('+' = beats v10 s42 on that dataset)")
