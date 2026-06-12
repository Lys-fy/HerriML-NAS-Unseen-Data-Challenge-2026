#!/usr/bin/env python3
"""Phase A figures.

1. phaseA_gate_grid.png : the 39 live bake-off curves (13 datasets x 3 arms,
   ~7 min matched budget, aug off), per-panel verdict line.
2. phaseA_verdicts.png  : the verdict bars: routing-policy totals, A5 stem
   retest, A5 aug-transfer retest, TCN block A/B.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "dev/logs/phaseA/gate"
IMG = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline"
IMG.mkdir(parents=True, exist_ok=True)

C = {"conv": "#1f77b4", "tcn1d": "#2c8a3a", "tcn2d": "#ff7f0e"}
ENCODED = {"Language", "Sudoku", "Chesseract", "Gutenberg"}
ORACLE = {"Language": "tcn1d", "Cryptic": "tcn1d", "AddNIST": "tcn2d",
          "CIFARTile": "tcn2d", "Chesseract": "conv", "GameOfLife": "tcn1d",
          "GeoClassing": "tcn2d", "Gutenberg": "conv", "MultNIST": "tcn2d",
          "Myofibre": "tcn2d", "Sudoku": "conv", "Voxel": "tcn1d",
          "Windspeed": "conv"}
ROUTE = {"Language": "tcn2d", "Cryptic": "tcn2d", "AddNIST": "tcn2d",
         "CIFARTile": "tcn2d", "Chesseract": "tcn1d", "GameOfLife": "tcn2d",
         "GeoClassing": "tcn2d", "Gutenberg": "conv", "MultNIST": "tcn2d",
         "Myofibre": "tcn2d", "Sudoku": "conv", "Voxel": "tcn2d",
         "Windspeed": "tcn2d"}
DATASETS = list(ORACLE)


def curve(ds, arm):
    """x-axis = cumulative wall-clock minutes (the matched quantity).
    Epoch counts differ 1..244 across datasets/arms because epoch TIME
    differs; plotting vs epochs made matched runs look inconsistent."""
    d = GATE / f"{ds}_{arm}"
    csvs = list(d.glob("*.csv"))
    if not csvs:
        return None, None
    mins, val, t = [], [], 0.0
    with open(csvs[0]) as f:
        for r in csv.DictReader(f):
            t += float(r["epoch_time_s"])
            mins.append(t / 60.0)
            val.append(max(float(r["val_acc"]), float(r.get("val_acc_ema", 0) or 0)) * 100)
    return np.array(mins), np.array(val)


# ---- figure 1: gate grid ----------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(18, 15), squeeze=False)
flat = axes.flatten()
for ax, ds in zip(flat, DATASETS):
    for arm in C:
        ep, val = curve(ds, arm)
        if ep is None:
            continue
        run = np.maximum.accumulate(val)
        n_ep = len(ep)
        ax.plot(ep, val, color=C[arm], alpha=0.25, linewidth=0.9)
        ax.plot(ep, run, color=C[arm], linewidth=1.9, label=f"{arm} ({n_ep} ep)")
    routed, truth = ROUTE[ds], ORACLE[ds]
    mark = "OK" if routed == truth else "miss"
    tag = "encoded -> gate" if ds in ENCODED else "image-like -> skip"
    ax.set_title(f"{ds}  ({tag})\nrouted={routed}  truth={truth}  [{mark}]",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_xlabel("wall-clock training minutes (matched ~6 min)")
    ax.set_ylabel("val acc (%)")
    ax.legend(loc="lower right", fontsize=8)
ax = flat[13]; ax.axis("off")
ax.text(0.02, 0.97,
        "A1 LIVE BACKBONE GATE\n"
        "39 short-train arms, matched\n"
        "~7 min budget, aug off, seed 42\n\n"
        "routing-policy totals (final adj):\n"
        "  fixed tcn2d        +59.2\n"
        "  gate everywhere    +58.5\n"
        "  fingerprint+gate   +61.2\n"
        "  oracle             +63.2\n\n"
        "-> the fingerprint SKIP is\n"
        "   load-bearing: gating every\n"
        "   dataset falls into the\n"
        "   Myofibre/GeoClassing traps;\n"
        "   routing only encoded data\n"
        "   through the gate adds +2.0",
        transform=ax.transAxes, fontsize=10.5, va="top", family="monospace")
for j in range(len(DATASETS) + 1, len(flat)):
    flat[j].axis("off")
fig.suptitle("Phase A1: live backbone bake-off curves + fingerprint routing verdict",
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(IMG / "phaseA_gate_grid.png", dpi=125)
print("wrote", IMG / "phaseA_gate_grid.png")

# ---- figure 2: verdict bars --------------------------------------------------
fig2, axs = plt.subplots(1, 4, figsize=(18, 4.6))

ax = axs[0]
pols = ["fixed\ntcn2d", "gate\neverywhere", "fingerprint\n+ gate", "oracle"]
vals = [59.15, 58.54, 61.16, 63.17]
cols = ["#999999", "#b53030", "#2c8a3a", "#1f77b4"]
ax.bar(pols, vals, color=cols)
ax.set_ylim(55, 65)
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:+.1f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("A1: routing policy totals (adj)")
ax.grid(axis="y", alpha=0.25)

ax = axs[1]
x = np.arange(2); w = 0.35
ax.bar(x - w / 2, [79.66, 58.81], w, label="conv3 stem", color="#999999")
ax.bar(x + w / 2, [80.75, 60.17], w, label="channel-embed", color="#2c8a3a")
ax.set_xticks(x); ax.set_xticklabels(["Voxel", "Chesseract"])
for i, (a, b) in enumerate([(79.66, 80.75), (58.81, 60.17)]):
    ax.text(i - w / 2, a + 0.2, f"{a:.1f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.2, f"{b:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_title("A5a: stem x dilated backbone (raw %)\nchannel-embed WINS both -> ship")
ax.set_ylim(55, 84); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

ax = axs[2]
ax.bar(["aug OFF\n(probe said)", "aug ON"], [86.61, 89.34],
       color=["#b53030", "#2c8a3a"])
ax.set_ylim(84, 91)
ax.text(0, 86.61 + 0.1, "86.6", ha="center", fontsize=10)
ax.text(1, 89.34 + 0.1, "89.3", ha="center", fontsize=10, fontweight="bold")
ax.set_title("A5b: Language, tcn1d backbone (raw %)\nthe conv-derived aug decision does NOT transfer")
ax.grid(axis="y", alpha=0.25)

ax = axs[3]
x = np.arange(3); w = 0.26
ds3 = ["CIFARTile", "Myofibre", "Language"]
bn = [68.34, 88.82, 88.09]; ca = [44.87, 85.03, 88.74]; ge = [67.46, 88.50, 89.63]
ax.bar(x - w, bn, w, label="ours (BN+ReLU)", color="#1f77b4")
ax.bar(x, ca, w, label="canonical (WN+drop)", color="#b53030")
ax.bar(x + w, ge, w, label="GELU", color="#ff7f0e")
ax.set_xticks(x); ax.set_xticklabels(ds3)
ax.set_title("Block A/B (raw %, 1h)\nboth challengers REJECTED (bar: +0.5 on 2/3)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

fig2.suptitle("Phase A verdicts", fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(IMG / "phaseA_verdicts.png", dpi=130)
print("wrote", IMG / "phaseA_verdicts.png")
