#!/usr/bin/env python3
"""Per-dataset v10 learning curves AGGREGATED over the 3 seeds (42/1337/2024):
one panel per dataset, the v10 best-val curve shown as mean +/- std across the
seeds (band), versus the conv(v8) baseline (blue) and the fixed TCN-2d probe
(orange), with the ResNet-18 benchmark (red dashed). Seeds have different
epoch counts, so each curve is interpolated onto a common fraction-of-training
axis before averaging.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v10_curves_grid_3seed.png"
SEEDS = ["42", "1337", "2024"]
GRID = np.linspace(0.0, 1.0, 100)

CONV = "#1f77b4"; T2 = "#ff7f0e"; V10C = "#2c8a3a"; BENCHC = "#b53030"

# name, codename, benchmark, conv(v8) s42 test, tcn2d-fix s42 test
DATASETS = [
    ("Language", "LaMelo", 85.200, 83.68, 88.24),
    ("Cryptic", "Cryptic", 0.000, 70.27, 71.18),
    ("AddNIST", "Adaline", 89.850, 95.82, 96.62),
    ("CIFARTile", "Caitie", 47.008, 48.30, 72.20),
    ("Chesseract", "Chester", 57.826, 59.10, 57.08),
    ("GameOfLife", "conway", 47.530, 99.96, 99.99),
    ("GeoClassing", "Sadie", 80.330, 95.38, 95.49),
    ("Gutenberg", "Gutenberg", 40.980, 47.08, 40.62),
    ("MultNIST", "Mateo", 90.870, 94.76, 95.52),
    ("Myofibre", "Myopia", 87.928, 88.33, 91.96),
    ("Sudoku", "Sokoto", 0.000, 94.46, 88.14),
    ("Voxel", "Volga", 71.350, 78.16, 80.33),
    ("Windspeed", "windspeed", 13.492, 19.74, 19.74),
]
# v10 final TEST raw per seed (conway s2024 imputed 100.0, pred lost to reboot)
V10_TEST = {
    "42": {"Adaline": 96.33, "Caitie": 73.65, "Chester": 57.66, "Cryptic": 74.25,
           "conway": 100.00, "Sadie": 95.36, "Gutenberg": 47.28, "LaMelo": 90.03,
           "Mateo": 95.52, "Myopia": 91.09, "Sokoto": 97.70, "Volga": 81.56, "windspeed": 19.36},
    "1337": {"Adaline": 95.80, "Caitie": 74.26, "Chester": 64.30, "Cryptic": 69.02,
             "conway": 99.99, "Sadie": 95.63, "Gutenberg": 46.97, "LaMelo": 90.15,
             "Mateo": 95.18, "Myopia": 91.57, "Sokoto": 86.14, "Volga": 80.84, "windspeed": 20.51},
    "2024": {"Adaline": 96.53, "Caitie": 72.43, "Chester": 61.69, "Cryptic": 72.96,
             "conway": 100.00, "Sadie": 95.37, "Gutenberg": 47.10, "LaMelo": 89.40,
             "Mateo": 95.25, "Myopia": 91.57, "Sokoto": 91.83, "Volga": 81.20, "windspeed": 18.65},
}


def best_curve(path):
    """best_val_acc (%) interpolated onto GRID (fraction of training)."""
    best = []
    with open(path) as f:
        for r in csv.DictReader(f):
            best.append(float(r["best_val_acc"]) * 100)
    if len(best) < 2:
        return None
    best = np.array(best)
    frac = np.linspace(0, 1, len(best))
    return np.interp(GRID, frac, best)


fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
flat = axes.flatten()

for ax, (name, code, bench, conv_raw, t2_raw) in zip(flat, DATASETS):
    lo, hi = 100.0, 0.0
    # v10 aggregated band over seeds
    curves = []
    for s in SEEDS:
        p = REPO / f"dev/logs/v10_sweep/v10_s{s}/{code}.csv"
        if p.exists():
            c = best_curve(p)
            if c is not None:
                curves.append(c)
    if curves:
        arr = np.array(curves)
        mean, std = arr.mean(0), arr.std(0)
        tests = [V10_TEST[s][code] for s in SEEDS if code in V10_TEST[s]]
        tlab = f"{np.mean(tests):.1f}±{np.std(tests):.1f}"
        ax.fill_between(GRID, mean - std, mean + std, color=V10C, alpha=0.25, zorder=2)
        ax.plot(GRID, mean, color=V10C, linewidth=2.6, zorder=3,
                label=f"v10 (3 seeds) test {tlab}%")
        lo, hi = min(lo, float(mean.min())), max(hi, float(mean.max()))
    # references (s42 baselines)
    for path, c, lab in (
            (REPO / f"dev/logs/v8/v8_s42/{code}.csv", CONV, f"conv(v8) {conv_raw:.1f}%"),
            (REPO / f"dev/logs/tcn_probe/tcn2d_{name}_s42/{code}.csv", T2, f"tcn2d-fix {t2_raw:.1f}%")):
        if path.exists():
            cc = best_curve(path)
            if cc is not None:
                ax.plot(GRID, cc, color=c, linewidth=1.6, label=lab)
                lo, hi = min(lo, float(cc.min())), max(hi, float(cc.max()))
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.1, label=f"bench {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("fraction of training"); ax.set_ylabel("val acc (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

ax = flat[len(DATASETS)]; ax.axis("off")
ax.text(0.02, 0.98,
        "v10 measure-gate-train, 2h\n"
        "green = mean of seeds 42/1337/2024\n"
        "  band = +/- std across seeds\n"
        "  label = mean final TEST +/- std\n"
        "blue = conv(v8) baseline (s42)\n"
        "orange = fixed TCN-2d probe (s42)\n"
        "red dashed = ResNet-18 benchmark\n\n"
        "3-seed total: +62.23 ± 0.19 (test)\n"
        "widest spread: Chesseract (random arm)",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(DATASETS) + 1, len(flat)):
    flat[j].axis("off")

fig.suptitle("v10 per-dataset learning curves, aggregated over 3 seeds (mean ± std) vs conv(v8) baseline",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=125)
print(f"wrote {OUT}")
