#!/usr/bin/env python3
"""Aggregated learning curve: total adjusted score (summed over the 13
datasets) as training progresses, v10 vs the conv(v8) baseline (the blue
reference in the per-seed grids).

Each dataset's per-epoch best-val is converted to its adjusted score
(raw - benchmark) * 10 / (100 - benchmark), then interpolated onto a common
"fraction of training budget" axis (0..1, from clock_remaining), and summed
across datasets to give a total-score trajectory. v10 is the mean +/- std
over seeds 42/1337/2024; conv(v8) is seed 42 (the baseline shown in blue).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v10_aggregate_curves.png"

# name, codename, benchmark (ResNet-18 zero-point; 0 for the two placeholders)
DATASETS = [
    ("AddNIST", "Adaline", 89.850), ("CIFARTile", "Caitie", 47.008),
    ("Chesseract", "Chester", 57.826), ("Cryptic", "Cryptic", 0.0),
    ("GameOfLife", "conway", 47.530), ("GeoClassing", "Sadie", 80.330),
    ("Gutenberg", "Gutenberg", 40.980), ("Language", "LaMelo", 85.200),
    ("MultNIST", "Mateo", 90.870), ("Myofibre", "Myopia", 87.928),
    ("Sudoku", "Sokoto", 0.0), ("Voxel", "Volga", 71.350),
    ("Windspeed", "windspeed", 13.492),
]
GRID = np.linspace(0.0, 1.0, 120)


def adj_curve(path, bench):
    """Return adjusted-score interpolated onto GRID (fraction of training)."""
    ep, best, cr = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            best.append(float(r["best_val_acc"]) * 100)
            cr.append(float(r["clock_remaining_s"]))
    if len(best) < 2:
        return None
    best = np.array(best); cr = np.array(cr)
    adj = np.maximum(-10.0, (best - bench) * 10.0 / max(100.0 - bench, 1e-6))
    span = cr[0] - cr[-1]
    if span <= 0:
        frac = np.linspace(0, 1, len(adj))
    else:
        frac = (cr[0] - cr) / span
    return np.interp(GRID, frac, adj)


def total_traj(logdir):
    """Sum adjusted-score trajectories across datasets present in logdir."""
    tot = np.zeros_like(GRID)
    n = 0
    for name, code, bench in DATASETS:
        p = logdir / f"{code}.csv"
        if not p.exists():
            continue
        c = adj_curve(p, bench)
        if c is None:
            continue
        tot += c
        n += 1
    return tot, n


fig, ax = plt.subplots(figsize=(10, 6.5))

# v10: 3 seeds
seed_tots = []
for s in ("42", "1337", "2024"):
    t, n = total_traj(REPO / f"dev/logs/v10_sweep/v10_s{s}")
    seed_tots.append(t)
    print(f"v10 s{s}: {n} datasets, final total {t[-1]:.1f}")
seed_tots = np.array(seed_tots)
v10_mean = seed_tots.mean(0)
v10_std = seed_tots.std(0)

# conv(v8) baseline, seed 42 (blue)
conv_tot, nconv = total_traj(REPO / "dev/logs/v8/v8_s42")
print(f"conv(v8) s42: {nconv} datasets, final total {conv_tot[-1]:.1f}")

ax.plot(GRID, conv_tot, color="#1f77b4", linewidth=2.4,
        label=f"conv (v8) baseline, s42  (final {conv_tot[-1]:.1f})")
ax.fill_between(GRID, v10_mean - v10_std, v10_mean + v10_std,
                color="#2c8a3a", alpha=0.22, zorder=2)
ax.plot(GRID, v10_mean, color="#2c8a3a", linewidth=2.6,
        label=f"v10 mean of 3 seeds  (final {v10_mean[-1]:.1f} ± {v10_std[-1]:.1f})",
        zorder=3)

ax.set_xlabel("fraction of training budget (per dataset, 2h)", fontsize=11)
ax.set_ylabel("total adjusted score over 13 datasets (validation)", fontsize=11)
ax.set_title("Aggregated learning curve: total adjusted score vs the conv(v8) baseline\n"
             f"validation: v10 +{v10_mean[-1]:.1f} vs conv(v8) +{conv_tot[-1]:.1f} "
             f"(+{v10_mean[-1]-conv_tot[-1]:.1f})   |   final TEST: v10 +62.2, conv(v8) +48.2",
             fontsize=11)
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
