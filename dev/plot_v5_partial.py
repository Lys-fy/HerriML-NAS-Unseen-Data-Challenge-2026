#!/usr/bin/env python3
"""Plot v5 partial-run comparison.

For each dataset that v5 s42 has finished training, compare the best
validation accuracy against the corresponding v4 and EGNAS-full runs on
the same seed. Drop a bar chart with benchmark lines. Used to get a
quick early read on whether v5's changes (driver C + Mixup-on-OFF +
GroupConv3 + ResNeXt) are moving things, before the harness writes its
final Scoring section.

Note: best val_acc is a strong proxy for the eventual test raw acc but
not identical. Bars are ordered alphabetic to match the other paired
plots in the notebase.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "notes" / "09 - Implementation" / "implementation" / "images"
IMG.mkdir(parents=True, exist_ok=True)

CODENAME_TO_DATASET = {
    "Adaline": "AddNIST", "Caitie": "CIFARTile", "Chester": "Chesseract",
    "Cryptic": "Cryptic", "conway": "GameOfLife", "Sadie": "GeoClassing",
    "Gutenberg": "Gutenberg", "LaMelo": "Language", "Mateo": "MultNIST",
    "Myopia": "Myofibre", "Sokoto": "Sudoku", "Volga": "Voxel",
    "windspeed": "Windspeed",
}


def best_val(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    last = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if not last:
        return None
    return float(last["best_val_acc"]) * 100


def egnas_picks(pool_path: Path) -> str:
    if not pool_path.exists():
        return ""
    pool = json.loads(pool_path.read_text())
    # Walk forward applying the incumbent rule to reconstruct the final ops.
    # We can also just read the last micro entry's ops field.
    micro_entries = [x for x in pool if x.get("phase") == "micro" and "ops" in x]
    if not micro_entries:
        # Macro-only (no micro changes accepted) — return implied all-Conv3
        # from the macro phase.
        for x in pool:
            if x.get("phase") in ("depth", "width", "bootstrap"):
                return f"D={x['D']} W={x['W']} (all Conv3)"
        return ""
    # Final state is the last entry's ops (since the trace shows ALL tries,
    # not just accepts, we need to be careful — but the LAST micro entry
    # always reflects the most-recent ops list state).
    last_micro = micro_entries[-1]
    return f"D={last_micro['D']} W={last_micro['W']} ops={last_micro['ops']}"


def parse_egnas_final_ops_from_log(log_path: Path, codename: str) -> str:
    """Extract the 'EGNAS full done: ... ops=[...]' line for a codename
    from the per-seed log."""
    pat = re.compile(
        r"Dataset\s+" + re.escape(codename) +
        r".*?EGNAS full done:\s+D=(\d+)\s+W=(\d+).*?ops=(\[[^\]]+\])",
        re.DOTALL,
    )
    if not log_path.exists():
        return ""
    m = pat.search(log_path.read_text())
    if not m:
        return ""
    try:
        ops = json.loads(m.group(3).replace("'", '"'))
    except Exception:
        ops = []
    return f"D={m.group(1)} W={m.group(2)}  ops={ops}"


def harness_benchmark(codename: str) -> float | None:
    """Read the dataset's metadata benchmark."""
    ds = CODENAME_TO_DATASET.get(codename, codename)
    md_path = REPO / "datasets" / ds / "metadata"
    if not md_path.exists():
        return None
    try:
        md = json.loads(md_path.read_text())
        b = float(md.get("benchmark", 0.0) or 0.0)
        return b if b > 0.5 else None  # skip placeholder benchmarks
    except Exception:
        return None


def main():
    seed = 42
    base_v5  = REPO / "dev" / "logs" / "v5" / f"v5_s{seed}"
    base_v4  = REPO / "dev" / "logs" / "v4" / f"v4_s{seed}"
    base_eg  = REPO / "dev" / "logs" / "egnas_full" / f"t2bfull_s{seed}"
    base_egn = REPO / "dev" / "logs" / "egnas_full" / f"t2bfull_new_s{seed}"
    v5_log   = REPO / "dev" / "logs" / "v5" / f"v5_s{seed}.log"

    if not base_v5.exists():
        print(f"v5 dir not found: {base_v5}")
        return
    codenames = sorted({p.stem for p in base_v5.glob("*.csv")},
                       key=lambda c: CODENAME_TO_DATASET.get(c, c).lower())
    if not codenames:
        print("no v5 CSVs yet")
        return

    rows = []
    for c in codenames:
        ds = CODENAME_TO_DATASET.get(c, c)
        v5_acc = best_val(base_v5 / f"{c}.csv")
        v4_acc = best_val(base_v4 / f"{c}.csv")
        eg_acc = best_val(base_eg / f"{c}.csv") or best_val(base_egn / f"{c}.csv")
        bench  = harness_benchmark(c)
        arch   = parse_egnas_final_ops_from_log(v5_log, c)
        rows.append({
            "code": c, "ds": ds,
            "v5": v5_acc, "v4": v4_acc, "eg": eg_acc,
            "bench": bench, "arch": arch,
        })
        print(f"{ds:14s} v5={v5_acc:.2f}% v4={v4_acc and f'{v4_acc:.2f}%' or '—'}  EG={eg_acc and f'{eg_acc:.2f}%' or '—'}  bench={bench or '—'}  arch={arch}")

    n = len(rows)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(2.0 * n + 4, 7))

    eg_vals = [r["eg"] for r in rows]
    v4_vals = [r["v4"] for r in rows]
    v5_vals = [r["v5"] for r in rows]

    ax.bar(x - w, eg_vals, w, color="#2c8a3a", edgecolor="black", linewidth=.5,
           label="EGNAS-full s42 (paired baseline)")
    ax.bar(x,     v4_vals, w, color="#d97706", edgecolor="black", linewidth=.5,
           label="P1 v4 s42")
    ax.bar(x + w, v5_vals, w, color="#1f77b4", edgecolor="black", linewidth=.5,
           label="P1 v5 s42 (this run, ★ on bars with new-op picks)")

    # Benchmark + value annotations
    for i, r in enumerate(rows):
        if r["bench"]:
            ax.plot([i - 1.5 * w, i + 1.5 * w], [r["bench"]] * 2,
                    color="#b53030", linestyle="--", lw=1.4,
                    label="harness benchmark" if i == 0 else None)
        for off, val, color in [(-w, r["eg"], "#2c8a3a"),
                                 (0,   r["v4"], "#d97706"),
                                 (w,   r["v5"], "#1f77b4")]:
            if val is not None:
                ax.text(i + off, val + 0.7, f"{val:.1f}", ha="center",
                        fontsize=8, color="black")
        # Mark v5 bar with star if its EGNAS pick includes one of the new ops
        arch = r["arch"] or ""
        if "GroupConv3" in arch or "ResNeXt" in arch:
            ax.text(i + w, max(r["v5"] or 0, r["bench"] or 0) + 4,
                    "★", ha="center", fontsize=16, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels([r["ds"] for r in rows], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("best val_acc (%) — proxy for test raw acc")
    ax.set_ylim(35, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_title(
        f"Tier 2a / P1 v5 — s42 PARTIAL ({n} of 13 datasets finished training)\n"
        "★ above v5 bar = EGNAS micro picked GroupConv3 and/or ResNeXt for that dataset",
        fontsize=11,
    )
    fig.tight_layout()
    out = IMG / "pipeline" / "v5_s42_partial_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
