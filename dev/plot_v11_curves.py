#!/usr/bin/env python3
"""Per-epoch training curves (val acc vs epoch) for the v11 bake-off, like the
v10_curves_grid figures -- to see WHAT THE TRAINING DID, not just final acc.

Produces, for one seed:
  - v11.1_curves_s<seed>.png   16-panel grid, the informed-defaults floor
  - v11.2_curves_s<seed>.png   the heterogeneous hill-climb (datasets it has)
  - v11.3_curves_s<seed>.png   16-panel grid, the evolution
  - v11_compare_s<seed>.png    per-dataset overlay of v11.1 / v11.2 / v11.3 (+egnas)

Curve sources under --root (rsync from spark first):
  v11_1/<code>.csv          (Trainer CSV: epoch,lr,train_loss,val_acc,val_acc_ema,...)
  evo/<code>.log            (validate_winner stdout: "epoch N ... val= ema= best=")
  egnas/<code>.log          same
  v112/<code>.log + v112c/<code>.log   the hill-climb's validated winners

Aggregated-across-seeds figure: run with --aggregate after >=2 seeds are synced
into root/<approach>_s<seed>/ subdirs (mean +/- std band per dataset).
Output dir: images/pipeline/v11/
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "docs/notes/09 - Implementation/implementation/images/pipeline/v11"

CODE2DS = {"Chester": "Chesseract", "Caitie": "CIFARTile", "Sokoto": "Sudoku",
           "LaMelo": "Language", "Adaline": "AddNIST", "Mateo": "MultNIST",
           "Sadie": "GeoClassing", "Myopia": "Myofibre", "Volga": "Voxel",
           "conway": "GameOfLife", "windspeed": "Windspeed", "Cryptic": "Cryptic",
           "Gutenberg": "Gutenberg", "Sedol": "Sedol", "Sloan": "Sloan",
           "Cauchy": "Cauchy"}
DS2CODE = {v: k for k, v in CODE2DS.items()}
ORDER = ["Chesseract", "Sedol", "Cryptic", "Sudoku", "Gutenberg", "Language",
         "CIFARTile", "Cauchy", "Sloan", "Myofibre", "Voxel", "GeoClassing",
         "MultNIST", "AddNIST", "Windspeed", "GameOfLife"]
COL = {"v11.1": "#2ca02c", "v11.2": "#9467bd", "v11.3": "#1f77b4", "egnas": "#d62728"}

_LOG = re.compile(r"epoch\s+(\d+)\s+loss=[\d.nan]+\s+val=([\d.]+)\s+ema=([-\d.]+)\s+best=([\d.]+)")


def csv_curve(p):
    ep, val, ema, best = [], [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"])); val.append(float(row["val_acc"]))
            ema.append(float(row["val_acc_ema"])); best.append(float(row["best_val_acc"]))
    return np.array(ep), np.array(val), np.array(ema), np.array(best)


def log_curve(p):
    ep, val, ema, best = [], [], [], []
    for line in Path(p).read_text().splitlines():
        m = _LOG.search(line)
        if m:
            ep.append(int(m.group(1))); val.append(float(m.group(2)))
            ema.append(float(m.group(3))); best.append(float(m.group(4)))
    return np.array(ep), np.array(val), np.array(ema), np.array(best)


def load(root, approach, code):
    """Return (ep, val, ema, best) or None. v11.1 from CSV; others from .log
    (v11.2 lives in either v112/ or v112c/)."""
    root = Path(root)
    try:
        if approach == "v11.1":
            p = root / "v11_1" / f"{code}.csv"
            return csv_curve(p) if p.exists() else None
        if approach == "v11.3":
            p = root / "evo" / f"{code}.log"
            return log_curve(p) if p.exists() else None
        if approach == "egnas":
            p = root / "egnas" / f"{code}.log"
            return log_curve(p) if p.exists() else None
        if approach == "v11.2":
            for sub in ("v112", "v112c"):
                p = root / sub / f"{code}.log"
                if p.exists():
                    return log_curve(p)
            return None
    except Exception:
        return None
    return None


def grid(root, approach, seed, datasets):
    n = len(datasets); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.7 * rows), squeeze=False)
    fig.suptitle(f"{approach} training curves (val acc vs epoch), seed {seed}", y=1.0, fontsize=14)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        c = load(root, approach, DS2CODE[ds])
        if c is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                    color="grey"); ax.set_title(ds, fontsize=10); continue
        ep, val, ema, best = c
        ax.plot(ep, val, color=COL[approach], lw=1.0, label="val")
        ax.plot(ep, ema, color=COL[approach], lw=0.8, alpha=0.4, ls="--", label="ema")
        ax.plot(ep, best, color="black", lw=0.7, alpha=0.5, label="best")
        ax.set_title(f"{ds}  (best {best.max():.3f})", fontsize=10)
        ax.set_ylim(min(0.0, val.min()), 1.02); ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.supxlabel("epoch"); fig.supylabel("val acc")
    fig.tight_layout()
    out = OUTDIR / f"{approach.replace('.', '_')}_curves_s{seed}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def compare(root, seed, datasets, approaches):
    cols = 4; rows = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.7 * rows), squeeze=False)
    fig.suptitle(f"v11.1 / v11.2 / v11.3 best-val curves, seed {seed}", y=1.0, fontsize=14)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        for ap in approaches:
            c = load(root, ap, DS2CODE[ds])
            if c is None:
                continue
            ep, val, ema, best = c
            ax.plot(ep, best, color=COL[ap], lw=1.2, label=ap)
        ax.set_title(ds, fontsize=10); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    for j in range(len(datasets), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.supxlabel("epoch"); fig.supylabel("best val acc")
    fig.tight_layout()
    out = OUTDIR / f"v11_compare_s{seed}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with v11_1/ evo/ egnas/ v112/ v112c/ subdirs")
    ap.add_argument("--seed", default="42")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    grid(args.root, "v11.1", args.seed, ORDER)
    grid(args.root, "v11.3", args.seed, ORDER)
    # v11.2 only has the validated-winner datasets
    v112_ds = [d for d in ORDER if load(args.root, "v11.2", DS2CODE[d]) is not None]
    grid(args.root, "v11.2", args.seed, v112_ds)
    compare(args.root, args.seed, ORDER, ["v11.1", "v11.2", "v11.3"])


if __name__ == "__main__":
    main()
