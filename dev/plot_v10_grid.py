#!/usr/bin/env python3
"""v10 sweep curves, ONE PNG PER SEED (v10_curves_grid_s<seed>.png),
vs the two key references: conv (v8, same seed when available) and the fixed
TCN-2d probe (s42 only). v10 labels show the ROUTED architecture from that
seed's decision records, plus the final test score where the seed has
finished scoring (s42), or best-val while the sweep is still running."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline"
OUT.mkdir(parents=True, exist_ok=True)

CONV = "#1f77b4"; T2 = "#ff7f0e"; V10C = "#2c8a3a"; BENCHC = "#b53030"

# name, codename, benchmark, conv(v8) s42 test raw, tcn2d-fixed s42 test raw
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

# v10 FINAL test raw per seed (only seeds that finished scoring)
V10_TEST = {
    "42": {"Adaline": 96.33, "Caitie": 73.65, "Chester": 57.66,
           "Cryptic": 74.25, "conway": 100.00, "Sadie": 95.36,
           "Gutenberg": 47.28, "LaMelo": 90.03, "Mateo": 95.52,
           "Myopia": 91.09, "Sokoto": 97.70, "Volga": 81.56,
           "windspeed": 19.36},
}

SUMMARY = {
    "42": ("v10 measure-gate-train, 2h, seed 42 (COMPLETE)\n"
           "green label = FINAL TEST score\n\n"
           "totals (adj):\n"
           "  conv (v8)     +48.2\n"
           "  tcn2d fixed   +59.2\n"
           "  v10           +62.49  (all-time best)"),
    "2024": ("v10 measure-gate-train, 2h, seed 2024 (PARTIAL,\n"
             "green label = best-val; test scoring pending)\n\n"
             "5/13 datasets done before a spark reboot;\n"
             "the remaining 8 are re-running (resumed),\n"
             "then the 4 new datasets are queued.\n\n"
             "references (conv v8, fixed tcn2d) are the\n"
             "seed-42 baselines, marked (s42); only the\n"
             "v10 curve is seed 2024"),
    "1337": ("v10 measure-gate-train, 2h, seed 1337 (13/13,\n"
             "green label = best-val; test scoring pending)\n\n"
             "routing matches s42 on 11/13 datasets.\n"
             "the two random-arm adoptions diverge:\n"
             "  Chesseract  64.8 bv  (+6.1 vs s42)  WIN\n"
             "    attn-after-stem, 6 ops W=16; above the\n"
             "    all-time competition best (63.0)\n"
             "  Sudoku      86.2 bv  (-12.0 vs s42) LOSS\n"
             "    12 ops W=32 probed 0.95 but realised\n"
             "    0.86, below the conv-6 incumbent 0.87:\n"
             "    random arm trusted a noisy probe score\n\n"
             "references (conv v8, fixed tcn2d) are the\n"
             "seed-42 baselines, marked (s42); only the\n"
             "v10 curve is seed 1337"),
}


def load(p):
    ep, val, best = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(ep), np.array(val), np.array(best)


def v10_arch(v10_dir, code):
    f = v10_dir / f"{code}_decisions.json"
    if not f.exists():
        return "?"
    j = json.loads(f.read_text())
    fin = next((s for s in j["stages"] if s["stage"] == "final"), {})
    return fin.get("arch", "?").replace(" W=48", "")


def plot_seed(seed):
    v10_dir = REPO / f"dev/logs/v10_sweep/v10_s{seed}"
    fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
    flat = axes.flatten()
    n_panels = 0

    # only datasets the seed has reached (compact, no gaps)
    todo = [d for d in DATASETS if (v10_dir / f"{d[1]}.csv").exists()]

    for ax, (name, code, bench, conv_raw, t2_raw) in zip(flat, todo):
        v10_csv = v10_dir / f"{code}.csv"
        test = V10_TEST.get(seed, {}).get(code)
        score_lab = f"{test:.1f}%" if test is not None else None
        # References are the FIXED s42 baselines on every grid (the v8 sweep
        # and the tcn2d-fixed probe are only complete for s42; their seed
        # variation is negligible). Only the v10 curve varies by seed.
        ref = " (s42)" if seed != "42" else ""
        series = [
            (REPO / f"dev/logs/v8/v8_s42/{code}.csv", CONV,
             f"conv(v8) {conv_raw:.1f}%{ref}", 1.5),
            (REPO / f"dev/logs/tcn_probe/tcn2d_{name}_s42/{code}.csv", T2,
             f"tcn2d-fix {t2_raw:.1f}%{ref}", 1.5),
            (v10_csv, V10C, None, 2.5),
        ]
        lo, hi = 100.0, 0.0
        plotted = False
        for path, c, lab, lw in series:
            if not path.exists():
                continue
            ep, val, best = load(path)
            if lab is None:
                s = score_lab if score_lab else f"bv={best.max():.1f}%"
                lab = f"v10 [{v10_arch(v10_dir, code)}] {s}"
            ax.plot(ep, val, color=c, alpha=0.15, linewidth=0.9)
            ax.plot(ep, best, color=c, linewidth=lw, label=lab)
            lo, hi = min(lo, float(best.min())), max(hi, float(best.max()))
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        n_panels += 1
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

    ax = flat[len(todo)]; ax.axis("off")
    ax.text(0.02, 0.98, SUMMARY.get(seed, ""), transform=ax.transAxes,
            fontsize=10.5, va="top", family="monospace")
    for j in range(len(todo) + 1, len(flat)):
        flat[j].axis("off")

    fig.suptitle(f"v10 (measure-gate-train) vs references, seed {seed}",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / f"v10_curves_grid_s{seed}.png"
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"wrote {out} ({n_panels} panels)")


for d in sorted((REPO / "dev/logs/v10_sweep").glob("v10_s*")):
    if d.is_dir():
        plot_seed(d.name.replace("v10_s", ""))
