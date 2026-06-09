#!/usr/bin/env python3
"""Language (LaMelo) — v8 conv vs TCN learning curves, seed 42.

Three architectures on the one dataset where v8 conv regresses:
  - v8 conv (the 4-op macro-winner that scored adj -1.03, below benchmark)
  - tcn2d   (2-D dilated-conv TCN-style net)
  - tcn1d   (true 1-D sequence TCN)

Left panel: val-acc vs epoch. Right panel: best-val vs wall-clock minutes —
shows the convergence-SPEED story (the TCN nails Language in ~7 min, so the
2 h budget barely improves on the smoke). Style matches plot_tier_results.py
(benchmark dashed red, best-val solid, raw-val faint).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "notes" / "09 - Implementation" / "implementation" / "images" / "language"
IMG.mkdir(parents=True, exist_ok=True)
BENCH = 85.20

# (label, csv, colour, final TEST raw % from the score line)
SERIES = [
    ("v8 conv",  REPO / "dev/logs/tier2a_p1_v8/p1v8_s42/LaMelo.csv",            "#1f77b4", 83.68),
    ("TCN-2d",   REPO / "dev/logs/tcn_probe/tcn2d_Language_s42/LaMelo.csv",     "#ff7f0e", 88.24),
    ("TCN-1d",   REPO / "dev/logs/tcn_probe/tcn1d_Language_s42/LaMelo.csv",     "#2c8a3a", 89.34),
]


def load(p):
    ep, val, best, dt = [], [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            val.append(float(r["val_acc"]) * 100)
            best.append(float(r["best_val_acc"]) * 100)
            dt.append(float(r["epoch_time_s"]))
    return (np.array(ep), np.array(val), np.array(best),
            np.cumsum(np.array(dt)) / 60.0)  # cumulative wall-clock minutes


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

for label, path, c, raw in SERIES:
    ep, val, best, mins = load(path)
    # Left: vs epoch
    axL.plot(ep, val, color=c, alpha=0.22, linewidth=1.0)
    axL.plot(ep, best, color=c, linewidth=2.0,
             label=f"{label}  (test {raw:.1f}%, {ep[-1]} ep)")
    # Right: best-val vs wall-clock minutes
    axR.plot(mins, best, color=c, linewidth=2.0, label=f"{label}  (test {raw:.1f}%)")
    # mark the 7-min smoke point on the right panel
    i7 = int(np.searchsorted(mins, 7.0))
    if i7 < len(mins):
        axR.plot(mins[i7], best[i7], "o", color=c, markersize=6, zorder=5)

for ax in (axL, axR):
    ax.axhline(BENCH, color="#b53030", linestyle="--", linewidth=1.2,
               label=f"benchmark = {BENCH:.1f}%")
    ax.set_ylabel("validation accuracy (%)")
    ax.grid(alpha=0.25)
    ax.set_ylim(60, 92)

axL.set_xlabel("epoch")
axL.set_title("Learning curve vs epoch  (faint = val, solid = best-so-far)")
axR.set_xlabel("wall-clock training time (minutes)")
axR.axvline(7.0, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
axR.text(7.2, 61, "7-min smoke", color="grey", fontsize=8, rotation=90, va="bottom")
axR.set_title("Best-val vs wall-clock  (TCN converges in minutes)")
axR.set_xlim(0, 120)

# de-dup the benchmark label: legend only once per axis is fine, drop on right
handles, labels = axL.get_legend_handles_labels()
axL.legend(loc="lower right", fontsize=9, framealpha=0.92)
axR.legend(loc="lower right", fontsize=9, framealpha=0.92)

fig.suptitle("Language (LaMelo) — v8 conv vs TCN, seed 42  |  "
             "TCN-1d 89.3% > TCN-2d 88.2% > conv 83.7% (bench 85.2)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = IMG / "lang_tcn_curves.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
