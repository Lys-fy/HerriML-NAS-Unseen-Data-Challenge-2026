#!/usr/bin/env python3
"""Per-epoch curves for ALL v12 testing, in the style of the v10/v11 pipeline
figures: val accuracy (%) per epoch (solid, left axis) + training loss (dashed,
faint, right axis) per panel.

Figures (output: images/pipeline/v12/):
  v12_2h_curves.png        T2 prototype @2h total -- 9-dataset grid, 3 seed
                           lines each (the deployed-context integration test)
  v12_budget_ladder.png    Chesseract + CIFARTile across budgets (0.5h / 2h /
                           6h, + the plain-6h single-train control) -- the
                           throughput-keyed ladder made visible
  v12_inpipe_ops.png       C1/C4 in-pipeline arms vs the v12 seeds on
                           Language + Sudoku (the mirage check)
  v12_package_curves.png   the CURRENT v12 package validation curves from
                           spark ~/local/v12val/curves (latest tier per
                           dataset; seed column separates the K seeds)

Curve sources (rsync from spark first, see dev/logs/v12curves/):
  proto/<tag>/seed<i>/<code>.csv   v11_1-trainer 8-col CSVs from dev/v12_integration.py
  package/<code>.csv               v12-package 9-col CSVs (incl. seed column)
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "dev/logs/v12curves"
OUTDIR = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v12"
OUTDIR.mkdir(parents=True, exist_ok=True)

CODE2DS = {"Chester": "Chesseract", "Caitie": "CIFARTile", "Sokoto": "Sudoku",
           "LaMelo": "Language", "Adaline": "AddNIST", "Mateo": "MultNIST",
           "Sadie": "GeoClassing", "Myopia": "Myofibre", "Volga": "Voxel",
           "conway": "GameOfLife", "windspeed": "Windspeed", "Cryptic": "Cryptic"}
ORDER9 = ["Chesseract", "Sudoku", "Language", "GeoClassing", "CIFARTile",
          "AddNIST", "GameOfLife", "Myofibre", "Windspeed"]
SEED_COL = ["#2ca02c", "#1f77b4", "#9467bd"]
V10_REAL = {"Chesseract": 57.66, "Sudoku": 97.70, "Language": 90.03,
            "GeoClassing": 95.36, "CIFARTile": 73.65, "AddNIST": 96.33,
            "GameOfLife": 100.0, "Myofibre": 91.09, "Windspeed": 19.36,
            "Cryptic": 72.08}


def read_csv(p):
    ep, val, loss, seed = [], [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                ep.append(int(row["epoch"])); val.append(100 * float(row["val_acc"]))
                loss.append(float(row["train_loss"]))
                seed.append(int(row.get("seed", 0)))
            except (ValueError, KeyError):
                continue
    return np.array(ep), np.array(val), np.array(loss), np.array(seed)


def proto_seeds(tag, code):
    """[(seed_idx, ep, val, loss)] for proto/<tag>/seed*/<code>.csv"""
    out = []
    for i in range(4):
        p = ROOT / "proto" / tag / f"seed{i}" / f"{code}.csv"
        if p.exists():
            ep, val, loss, _ = read_csv(p)
            if len(ep):
                out.append((i, ep, val, loss))
    return out


def panel(ax, curves, title, v10=None):
    """curves: [(label, color, ep, val, loss)] -- val solid left, loss dashed right."""
    ax2 = ax.twinx()
    for label, color, ep, val, loss in curves:
        ax.plot(ep, val, color=color, lw=1.2, label=label)
        ax2.plot(ep, loss, color=color, lw=0.7, ls="--", alpha=0.35)
    if v10 is not None:
        ax.axhline(v10, color="#d62728", lw=1.0, ls=":", alpha=0.9)
        ax.text(0.99, v10, " v10", color="#d62728", fontsize=7,
                transform=ax.get_yaxis_transform(), va="bottom", ha="right")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("val acc %", fontsize=7)
    ax2.set_ylabel("train loss", fontsize=7, alpha=0.6)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7, colors="0.5")
    ax.grid(alpha=0.25)


def fig_v12_2h():
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, ds in zip(axes.flat, ORDER9):
        code = {v: k for k, v in CODE2DS.items()}[ds]
        curves = [(f"seed{i}", SEED_COL[i], ep, val, loss)
                  for i, ep, val, loss in proto_seeds("v12", code)]
        panel(ax, curves, ds, V10_REAL.get(ds))
        if ds == ORDER9[0]:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("v12 prototype @2h total (T2): K=3 shared-init seeds -- val acc (solid) + train loss (dashed)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / "v12_2h_curves.png", dpi=130)
    plt.close(fig)


def fig_budget_ladder():
    budgets = [("v12h05", "0.5h", "#7fc97f"), ("v12", "2h", "#2ca02c"),
               ("v12h6", "6h", "#00441b"), ("plainh6", "plain 6h (K=1, no TA)", "#ff7f0e")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ds, code) in zip(axes, [("Chesseract", "Chester"), ("CIFARTile", "Caitie")]):
        curves = []
        for tag, lab, color in budgets:
            for i, ep, val, loss in proto_seeds(tag, code):
                curves.append((lab if i == 0 else None, color, ep, val, loss))
        curves = [(l if l else "_nolegend_", c, e, v, lo) for l, c, e, v, lo in curves]
        panel(ax, curves, ds, V10_REAL.get(ds))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("v12 budget ladder: same pipeline at 0.5h / 2h / 6h (all seeds) vs plain-6h control", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTDIR / "v12_budget_ladder.png", dpi=130)
    plt.close(fig)


def fig_inpipe_ops():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ds, code) in zip(axes, [("Language", "LaMelo"), ("Sudoku", "Sokoto")]):
        curves = []
        for i, ep, val, loss in proto_seeds("v12", code):
            curves.append(("v12 seeds" if i == 0 else "_nolegend_", "#2ca02c", ep, val, loss))
        for tag, lab, color in [("c1pipe", "v11.1 + C1 (LinearAttn2D probe)", "#d62728"),
                                ("c4pipe", "v11.1 + C4 (shared-repeat x3)", "#ff7f0e")]:
            for i, ep, val, loss in proto_seeds(tag, code):
                curves.append((lab, color, ep, val, loss))
        panel(ax, curves, ds, V10_REAL.get(ds))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("C1/C4 in-pipeline arms vs v12 @2h -- the screen-mirage check (both ops rejected)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTDIR / "v12_inpipe_ops.png", dpi=130)
    plt.close(fig)


def fig_package():
    pkg = ROOT / "package"
    codes = [p.stem for p in sorted(pkg.glob("*.csv")) if p.stem in CODE2DS]
    n = len(codes)
    if not n:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows), squeeze=False)
    for k, code in enumerate(codes):
        ax = axes[k // cols][k % cols]
        ep, val, loss, seed = read_csv(pkg / f"{code}.csv")
        curves = []
        for s in sorted(set(seed.tolist())):
            m = seed == s
            curves.append((f"seed{s}", SEED_COL[int(s) % 3], ep[m], val[m], loss[m]))
        panel(ax, curves, CODE2DS[code], V10_REAL.get(CODE2DS[code]))
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("v12 PACKAGE validation curves (spark, latest tier per dataset -- queue still running)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "v12_package_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_v12_2h()
    fig_budget_ladder()
    fig_inpipe_ops()
    fig_package()
    print(f"wrote 4 figures to {OUTDIR}")
