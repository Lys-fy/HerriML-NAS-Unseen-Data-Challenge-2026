#!/usr/bin/env python3
"""v10 vs EGNAS (Tier 2b, the 2025-winner reimplementation = our method
baseline), aggregated over 3 seeds (42/1337/2024), on all 13 datasets.

Two figures:
  1. v10_vs_egnas_curves_grid_3seed.png - one panel per dataset, each method's
     best-val curve as mean +/- std across seeds (v10 green, EGNAS blue),
     ResNet-18 benchmark red dashed.
  2. v10_vs_egnas_aggregate_curve.png - total adjusted score (summed over the
     13 datasets) as training progresses, both methods as 3-seed mean +/- std.

x-axis is EPOCHS. Seeds within a method have different epoch counts, so each
seed's best-val (monotone) is forward-filled at its final value to the longest
seed before averaging. Validation-based curves; adjusted score uses the
metadata ResNet-18 benchmark, same convention for both methods.
EGNAS CSVs: dev/logs/egnas_full/t2bfull_s<seed> (11) and t2bfull_new_s<seed>
(Cryptic, windspeed).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline"
SEEDS = ["42", "1337", "2024"]
V10C = "#2c8a3a"; EGC = "#1f77b4"; BENCHC = "#b53030"

DATASETS = [
    ("Language", "LaMelo", 85.200), ("Cryptic", "Cryptic", 0.0),
    ("AddNIST", "Adaline", 89.850), ("CIFARTile", "Caitie", 47.008),
    ("Chesseract", "Chester", 57.826), ("GameOfLife", "conway", 47.530),
    ("GeoClassing", "Sadie", 80.330), ("Gutenberg", "Gutenberg", 40.980),
    ("MultNIST", "Mateo", 90.870), ("Myofibre", "Myopia", 87.928),
    ("Sudoku", "Sokoto", 0.0), ("Voxel", "Volga", 71.350),
    ("Windspeed", "windspeed", 13.492),
]
V10_TEST = {
    "42": {"Adaline":96.33,"Caitie":73.65,"Chester":57.66,"Cryptic":74.25,"conway":100.0,"Sadie":95.36,"Gutenberg":47.28,"LaMelo":90.03,"Mateo":95.52,"Myopia":91.09,"Sokoto":97.70,"Volga":81.56,"windspeed":19.36},
    "1337": {"Adaline":95.80,"Caitie":74.26,"Chester":64.30,"Cryptic":69.02,"conway":99.99,"Sadie":95.63,"Gutenberg":46.97,"LaMelo":90.15,"Mateo":95.18,"Myopia":91.57,"Sokoto":86.14,"Volga":80.84,"windspeed":20.51},
    "2024": {"Adaline":96.53,"Caitie":72.43,"Chester":61.69,"Cryptic":72.96,"conway":100.0,"Sadie":95.37,"Gutenberg":47.10,"LaMelo":89.40,"Mateo":95.25,"Myopia":91.57,"Sokoto":91.83,"Volga":81.20,"windspeed":18.65},
}


def best_epochs(path):
    """best_val_acc (%) per epoch, raw (no resampling)."""
    best = []
    with open(path) as f:
        for r in csv.DictReader(f):
            best.append(float(r["best_val_acc"]) * 100)
    return np.array(best) if len(best) >= 2 else None


def pad(arr, n):
    return arr if len(arr) >= n else np.concatenate([arr, np.full(n - len(arr), arr[-1])])


def v10_path(code, s):
    return REPO / f"dev/logs/v10_sweep/v10_s{s}/{code}.csv"


def egnas_path(code, s):
    p = REPO / f"dev/logs/egnas_full/t2bfull_s{s}/{code}.csv"
    return p if p.exists() else REPO / f"dev/logs/egnas_full/t2bfull_new_s{s}/{code}.csv"


def seed_curves(pathfn, code):
    cs = []
    for s in SEEDS:
        p = pathfn(code, s)
        if p.exists():
            c = best_epochs(p)
            if c is not None:
                cs.append(c)
    return cs


def band(pathfn, code):
    """Per-method curve over epochs: x, mean, std (seeds forward-filled)."""
    cs = seed_curves(pathfn, code)
    if not cs:
        return None
    n = max(len(c) for c in cs)
    arr = np.array([pad(c, n) for c in cs])
    return np.arange(n), arr.mean(0), arr.std(0)


# ---- Figure 1: per-dataset grid ---------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(19, 17), squeeze=False)
flat = axes.flatten()
for ax, (name, code, bench) in zip(flat, DATASETS):
    lo, hi = 100.0, 0.0
    for pathfn, col, lab in ((v10_path, V10C, "v10"), (egnas_path, EGC, "EGNAS")):
        b = band(pathfn, code)
        if b is None:
            continue
        x, mean, std = b
        ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.22, zorder=2)
        if lab == "v10":
            tests = [V10_TEST[s][code] for s in SEEDS]
            tag = f"test {np.mean(tests):.1f}±{np.std(tests):.1f}%"
        else:
            tag = f"val {mean[-1]:.1f}%"
        ax.plot(x, mean, color=col, linewidth=2.4, zorder=3, label=f"{lab} ({tag})")
        lo, hi = min(lo, float(mean.min())), max(hi, float(mean.max()))
    if bench > 0:
        ax.axhline(bench, color=BENCHC, linestyle="--", linewidth=1.1, label=f"bench {bench:.1f}%")
        lo, hi = min(lo, bench), max(hi, bench)
    pad_y = max(2.0, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad_y, hi + pad_y)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("val acc (%)")
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
ax = flat[len(DATASETS)]; ax.axis("off")
ax.text(0.02, 0.98,
        "v10 vs EGNAS (Tier 2b,\n2025-winner reimplementation)\n\n"
        "green = v10, blue = EGNAS\n"
        "  band = +/- std over 3 seeds\n"
        "  (shorter seeds held at final)\n"
        "red dashed = ResNet-18 benchmark\n\n"
        "total adjusted (test):\n"
        "  v10    +62.2\n"
        "  EGNAS  +35.5\n"
        "  -> +26.7 over the baseline",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(DATASETS) + 1, len(flat)):
    flat[j].axis("off")
fig.suptitle("v10 vs EGNAS per-dataset learning curves, aggregated over 3 seeds (mean ± std)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out1 = IMG / "v10_vs_egnas_curves_grid_3seed.png"
fig.savefig(out1, dpi=125); plt.close(fig)
print(f"wrote {out1}")


# ---- Figure 2: aggregated total curve over epochs ---------------------------
def total_per_seed(pathfn):
    """Per seed: total adjusted score over 13 datasets at each epoch.
    Each dataset's curve is forward-filled to the seed's longest dataset."""
    out = []
    for s in SEEDS:
        cols = []
        for name, code, bench in DATASETS:
            p = pathfn(code, s)
            if not p.exists():
                continue
            c = best_epochs(p)
            if c is None:
                continue
            cols.append(np.maximum(-10.0, (c - bench) * 10.0 / max(100.0 - bench, 1e-6)))
        if not cols:
            continue
        n = max(len(c) for c in cols)
        out.append(np.sum([pad(c, n) for c in cols], axis=0))
    n = max(len(c) for c in out)
    return np.array([pad(c, n) for c in out])


v10_t = total_per_seed(v10_path)
eg_t = total_per_seed(egnas_path)
fig, ax = plt.subplots(figsize=(10, 6.5))
for arr, col, lab in ((eg_t, EGC, "EGNAS"), (v10_t, V10C, "v10")):
    m, sd = arr.mean(0), arr.std(0)
    x = np.arange(len(m))
    ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.22)
    ax.plot(x, m, color=col, linewidth=2.6, label=f"{lab}  (val {m[-1]:.1f} ± {sd[-1]:.1f})")
ax.set_xlabel("epoch", fontsize=11)
ax.set_ylabel("total adjusted score over 13 datasets (validation)", fontsize=11)
gap = v10_t.mean(0)[-1] - eg_t.mean(0)[-1]
ax.set_title("Aggregated learning curve: v10 vs EGNAS (Tier 2b baseline), 3-seed mean ± std\n"
             f"validation +{v10_t.mean(0)[-1]:.1f} vs +{eg_t.mean(0)[-1]:.1f} (+{gap:.1f})   |   "
             "final TEST: v10 +62.2, EGNAS +35.5", fontsize=11)
ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
fig.tight_layout()
out2 = IMG / "v10_vs_egnas_aggregate_curve.png"
fig.savefig(out2, dpi=140); plt.close(fig)
print(f"wrote {out2}")
print(f"v10 val totals {[round(t[-1],1) for t in v10_t]}  egnas {[round(t[-1],1) for t in eg_t]}")
