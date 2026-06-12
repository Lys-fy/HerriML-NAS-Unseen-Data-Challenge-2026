#!/usr/bin/env python3
"""v10 sweep curves (seed 42) vs the two key references: conv (v8) and the
fixed TCN-2d probe. v10 labels show the ROUTED architecture from the
decision records. While the seed is still scoring, v10 shows best-val.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
V10 = REPO / "dev/logs/v10_sweep/v10_s42"
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/tier2a"
OUT.mkdir(parents=True, exist_ok=True)

CONV = "#1f77b4"; T2 = "#ff7f0e"; V10C = "#2c8a3a"; BENCHC = "#b53030"

# name, codename, benchmark, conv(v8) test raw, tcn2d-fixed test raw
DATASETS = [
    ("Language",    "LaMelo",    85.200, 83.68, 88.24),
    ("Cryptic",     "Cryptic",    0.000, 70.27, 71.18),
    ("AddNIST",     "Adaline",   89.850, 95.82, 96.62),
    ("CIFARTile",   "Caitie",    47.008, 48.30, 72.20),
    ("Chesseract",  "Chester",   57.826, 59.10, 57.08),
    ("GameOfLife",  "conway",    47.530, 99.96, 99.99),
    ("GeoClassing", "Sadie",     80.330, 95.38, 95.49),
    ("Gutenberg",   "Gutenberg", 40.980, 47.08, 40.62),
    ("MultNIST",    "Mateo",     90.870, 94.76, 95.52),
    ("Myofibre",    "Myopia",    87.928, 88.33, 91.96),
    ("Sudoku",      "Sokoto",     0.000, 94.46, 88.14),
    ("Voxel",       "Volga",     71.350, 78.16, 80.33),
    ("Windspeed",   "windspeed", 13.492, 19.74, 19.74),
]


def load(p):
    ep, val, best = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(ep), np.array(val), np.array(best)


def v10_arch(code):
    f = V10 / f"{code}_decisions.json"
    if not f.exists():
        return "?"
    j = json.loads(f.read_text())
    fin = next((s for s in j["stages"] if s["stage"] == "final"), {})
    return fin.get("arch", "?").replace(" W=48", "")


fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
flat = axes.flatten()

for ax, (name, code, bench, conv_raw, t2_raw) in zip(flat, DATASETS):
    series = [
        (REPO / f"dev/logs/tier2a_p1_v8/p1v8_s42/{code}.csv",
         CONV, f"conv(v8) {conv_raw:.1f}%", 1.5),
        (REPO / f"dev/logs/tcn_probe/tcn2d_{name}_s42/{code}.csv",
         T2, f"tcn2d-fix {t2_raw:.1f}%", 1.5),
        (V10 / f"{code}.csv", V10C, None, 2.5),
    ]
    lo, hi = 100.0, 0.0
    for path, c, lab, lw in series:
        if not path.exists():
            continue
        ep, val, best = load(path)
        if lab is None:
            lab = f"v10 [{v10_arch(code)}] bv={best.max():.1f}%"
        ax.plot(ep, val, color=c, alpha=0.15, linewidth=0.9)
        ax.plot(ep, best, color=c, linewidth=lw, label=lab)
        lo, hi = min(lo, float(best.min())), max(hi, float(best.max()))
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.1,
                   label=f"bench {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("val acc (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

ax = flat[len(DATASETS)]; ax.axis("off")
ax.text(0.02, 0.98,
        "v10 measure-gate-train, seed 42\n"
        "(2h, sweep in progress: v10 shows\n"
        "best-val, refs show final test)\n\n"
        "routing highlights (live decisions):\n"
        "  Language  -> gate picked tcn1d\n"
        "    + family aug probe  bv=90.4\n"
        "  Sudoku    -> conv, then RANDOM\n"
        "    arm adopted 7-op W=24  bv=98.2\n"
        "  Cryptic   -> tcn2d, depth D=8\n"
        "    verified + adopted\n"
        "  GeoClassing -> tcn2d D=4 verified\n"
        "  9 image-like -> skip to tcn2d\n"
        "    (Voxel/Chesseract: embed stem)",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(DATASETS) + 1, len(flat)):
    flat[j].axis("off")

fig.suptitle("v10 (measure-gate-train) vs conv (v8) and fixed TCN-2d, seed 42",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT / "tier2a_v10_curves_grid.png"
fig.savefig(out, dpi=125)
print(f"wrote {out}")
