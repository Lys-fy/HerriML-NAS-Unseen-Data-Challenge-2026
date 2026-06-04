"""Plot per-tier results: per-dataset learning curves + adjusted-score bar chart.

Generic across tiers — pass --tier tier0 / tier1 / etc.

Reads:
  - dev/logs/<csv_dir>/<codename>.csv     per-epoch val/loss/lr/clock
  - dev/logs/<sweep_log>                  final adj scores from the
                                          "========== Scoring ==========" section

Writes:
  - baselines/<tier_dir>/<tier>_scores.png  bar chart of adj per dataset
  - baselines/<tier_dir>/<tier>_curves.png  learning-curve grid
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-GUI backend — avoids Gdk segfaults on headless / Wayland
import matplotlib.pyplot as plt
import numpy as np

from internal_benchmarks import resolve_benchmark  # Sokoto=70.95 override etc.

REPO_ROOT = Path(__file__).resolve().parent.parent

# All plots live in the notebase so Obsidian can embed them with ![[name.png]].
IMPL_DIR = REPO_ROOT / "docs" / "notes" / "09 - Implementation" / "implementation"
# All PNGs go into an `images/` subfolder to keep the implementation/ note
# folder uncluttered. Obsidian's `![[name.png]]` resolves by basename, so
# embeds in the notes don't need updating after moving files here.
IMG_DIR = IMPL_DIR / "images"


def _img_out(name):
    """Route a figure basename into its topic subfolder under images/.
    Obsidian embeds resolve by basename, so ![[name.png]] keeps working
    regardless of which subfolder the file lives in."""
    name = Path(name).name
    if name.startswith("tier0"):
        sub = "tier0"
    elif name.startswith("tier1"):
        sub = "tier1"
    elif name.startswith("tier2a"):
        sub = "tier2a"
    elif name.startswith("tier2b"):
        sub = "tier2b"
    elif name.startswith("e1"):
        sub = "e1"
    elif name.startswith("multiseed"):
        sub = "multiseed"
    elif name.startswith("lang"):
        sub = "language"
    else:
        sub = ""
    d = (IMG_DIR / sub) if sub else IMG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / name

# Machine-readable aggregate produced by dev/aggregate_multiseed.py.
MULTISEED_JSON = REPO_ROOT / "dev" / "logs" / "multiseed" / "_aggregate.json"
# Machine-readable aggregate produced by dev/aggregate_e1.py.
E1_JSON = REPO_ROOT / "dev" / "logs" / "e1val" / "_e1_aggregate.json"
# Machine-readable aggregate produced by dev/aggregate_tier2bfull.py.
TIER2BFULL_JSON = REPO_ROOT / "dev" / "logs" / "tier2bfull" / "_tier2bfull_aggregate.json"
TIER2A_P1_V4_JSON = (REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4"
                     / "_tier2a_p1_v4_aggregate.json")
TIER2A_P1_V6_JSON = (REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6"
                     / "_tier2a_p1_v6_aggregate.json")
TIER2A_P1_V7_JSON = (REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7"
                     / "_tier2a_p1_v7_aggregate.json")

CODENAME_TO_DATASET = {
    "Adaline": "AddNIST",
    "Caitie": "CIFARTile",
    "Chester": "Chesseract",
    "conway": "GameOfLife",
    "Sadie": "GeoClassing",
    "Gutenberg": "Gutenberg",
    "LaMelo": "Language",
    "Mateo": "MultNIST",
    "Myopia": "Myofibre",
    "Sokoto": "Sudoku",
    "Volga": "Voxel",
    "Cryptic": "Cryptic",
    "windspeed": "Windspeed",
}

# Per-tier path resolution. Add new tiers here. All outputs land in IMPL_DIR
# (the Obsidian implementation/ folder) so notes can embed them.
TIER_CONFIG = {
    "tier0": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "training-2h",
        log_file=REPO_ROOT / "dev" / "logs" / "tier0-sweep-2h.log",
        out_dir=IMG_DIR,
        title="Tier 0 — Hardcoded CNN",
    ),
    "tier1": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "training-tier1-2h",
        log_file=REPO_ROOT / "dev" / "logs" / "tier1-sweep-2h.log",
        out_dir=IMG_DIR,
        title="Tier 1 — Augmentation + Random Op Search",
    ),
    # Single-seed snapshot of the full-EGNAS sweep (seed 42). NOT a verdict —
    # the 3-seed paired analysis is the trustworthy read; this is the same
    # tier0/tier1-style overview the user asked for.
    # Per-seed Tier-2b configs pull from both the original 11-ds sweep dir and
    # the Windspeed+Cryptic follow-on dir, so the per-seed score bars + curves
    # cover all 13 datasets.
    "tier2b_s42": dict(
        csv_dir=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s42",
                 REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s42"),
        log_file=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s42.log",
                  REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s42.log"),
        out_dir=IMG_DIR,
        title="Tier 2b — Full EGNAS (macro+micro), seed 42 — single seed, NOT a verdict",
    ),
    "tier2b_s1337": dict(
        csv_dir=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s1337",
                 REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s1337"),
        log_file=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s1337.log",
                  REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s1337.log"),
        out_dir=IMG_DIR,
        title="Tier 2b — Full EGNAS (macro+micro), seed 1337 — single seed",
    ),
    "tier2b_s2024": dict(
        csv_dir=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s2024",
                 REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s2024"),
        log_file=(REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_s2024.log",
                  REPO_ROOT / "dev" / "logs" / "tier2bfull" / "t2bfull_new_s2024.log"),
        out_dir=IMG_DIR,
        title="Tier 2b — Full EGNAS (macro+micro), seed 2024 — single seed",
    ),
    # Tier-2a / P1 v4 — full EGNAS + 3-probe shadow comparison (A drives,
    # B+C shadow). Per-seed view; v4 verdict lives in Tier 2a - Results.md.
    "tier2a_p1_v4_s42": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s42",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s42.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v4 (3-probe shadow, A drives), seed 42",
    ),
    "tier2a_p1_v4_s1337": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s1337",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s1337.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v4 (3-probe shadow, A drives), seed 1337",
    ),
    "tier2a_p1_v4_s2024": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s2024",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v4" / "p1v4_s2024.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v4, seed 2024 — PARTIAL (sweep still running)",
    ),
    # Tier-2a / P1 v5 — driver C + Mixup-on-OFF + GroupConv3 + ResNeXt in OP_VOCAB.
    "tier2a_p1_v5_s42": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s42",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s42.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v5 (driver C + Mixup-on-OFF + GroupConv3 + ResNeXt), seed 42",
    ),
    "tier2a_p1_v5_s1337": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s1337",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s1337.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v5, seed 1337",
    ),
    "tier2a_p1_v5_s2024": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s2024",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v5" / "p1v5_s2024.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v5, seed 2024",
    ),
    # Tier-2a / P1 v6 — C-only probe + 9-op vocab + K=12 + verify-of-accepts
    # + clock-extend + zero-shot shadow. Per-seed view; v6 verdict in notes.
    "tier2a_p1_v6_s42": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s42",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s42.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v6 (C-only + 9 ops + verify + clock-extend), seed 42",
    ),
    "tier2a_p1_v6_s1337": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s1337",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s1337.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v6, seed 1337",
    ),
    "tier2a_p1_v6_s2024": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s2024",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6" / "p1v6_s2024.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v6, seed 2024 — PARTIAL (sweep still running)",
    ),
    # Tier-2a / P1 v7 — simplify pivot: EGNAS-full + 4 conv ops + macro-prune-verify + aug-probe.
    "tier2a_p1_v7_s42": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s42",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s42.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v7 (EGNAS-full + 4-op + macro-prune-verify + aug-probe), seed 42",
    ),
    "tier2a_p1_v7_s1337": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s1337",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s1337.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v7, seed 1337 — PARTIAL (sweep running)",
    ),
    "tier2a_p1_v7_s2024": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s2024",
        log_file=REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7" / "p1v7_s2024.log",
        out_dir=IMG_DIR,
        title="Tier 2a / P1 v7, seed 2024 — PARTIAL (sweep running)",
    ),
}

# Tier 1 ablation: full baseline vs single-feature-disabled configs, on the
# 3 representative datasets. Baseline numbers from the tier1 full sweep.
ABLATION_BASELINE = {  # codename: adj from tier1-sweep-2h.log
    "Adaline": 1.862,
    "Caitie": -0.498,
    "LaMelo": -6.642,
}
ABLATION_LOGS = {
    "Tier 1 full": None,  # uses ABLATION_BASELINE
    "search-off": REPO_ROOT / "dev" / "logs" / "ablation-search-off.log",
    "ema-off": REPO_ROOT / "dev" / "logs" / "ablation-ema-off.log",
    "aug-off": REPO_ROOT / "dev" / "logs" / "ablation-aug-off.log",
}
ABLATION_DATASETS = ["Adaline", "Caitie", "LaMelo"]


def _as_paths(x):
    """Normalize a Path or sequence of Paths to a list."""
    if isinstance(x, Path):
        return [x]
    return list(x)


def parse_scores(log_path):
    """Parse score lines from one log file or several (tuple/list of Paths).

    Later files override earlier ones if a codename appears in both."""
    pat = re.compile(
        r"\s*(\w+):\s*raw=([\d.]+)%\s+benchmark=\s*([\d.]+)%\s+adj=\s*([+-]?[\d.]+)"
    )
    scores = {}
    for p in _as_paths(log_path):
        if not p.exists():
            continue
        for m in pat.finditer(p.read_text()):
            scores[m.group(1)] = {
                "raw": float(m.group(2)),
                "benchmark": float(m.group(3)),
                "adj": float(m.group(4)),
            }
    return scores


def _ds_name(code):
    """Alphabetic-sort key: case-insensitive dataset name (or codename fallback)."""
    return CODENAME_TO_DATASET.get(code, code).lower()


def load_csv(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_score_bars(scores, out_path, title_prefix):
    items = sorted(scores.items(), key=lambda kv: _ds_name(kv[0]))
    names = [CODENAME_TO_DATASET.get(c, c) for c, _ in items]
    adj = np.array([s["adj"] for _, s in items])
    raw = np.array([s["raw"] for _, s in items])
    bench = np.array([s["benchmark"] for _, s in items])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2c8a3a" if a > 0 else "#b53030" for a in adj]
    bars = ax.barh(names, adj, color=colors, edgecolor="black", linewidth=0.6)

    for bar, a, r, b, name in zip(bars, adj, raw, bench, names):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = 0.15 if x >= 0 else -0.15
        if name == "Sudoku":
            label = f"{r:.1f}% (b={b:.1f}*)"
        else:
            label = f"{r:.1f}% (b={b:.1f})"
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(10, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axvline(-10, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.text(10, -0.5, "+10 cap", color="grey", fontsize=8, ha="center")
    ax.text(-10, -0.5, "-10 floor", color="grey", fontsize=8, ha="center")

    ax.set_xlabel("Adjusted score (raw - benchmark) × 10 / (100 - benchmark), clipped [-10, +10]")
    total = adj.sum()
    ax.set_title(f"{title_prefix} — Per-dataset adjusted scores  (Total: {total:+.3f})  "
                 f"[* Sudoku benchmark=0.0 is a placeholder]", fontsize=11)
    ax.set_xlim(-12, 12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_learning_curves(scores, csv_dir, out_path, title_prefix):
    codenames_in_order = sorted(scores.keys(), key=_ds_name)
    csv_dirs = _as_paths(csv_dir)
    n = len(codenames_in_order)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.0 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, codename in zip(axes_flat, codenames_in_order):
        csv_path = next((d / f"{codename}.csv" for d in csv_dirs
                         if (d / f"{codename}.csv").exists()), None)
        if csv_path is None:
            ax.set_title(f"{codename} (no CSV)")
            ax.axis("off")
            continue
        rows_ = load_csv(csv_path)
        epochs = np.array([r["epoch"] for r in rows_])
        val_acc = np.array([r["val_acc"] for r in rows_]) * 100
        best_val = np.array([r["best_val_acc"] for r in rows_]) * 100
        train_loss = np.array([r["train_loss"] for r in rows_])

        ax.plot(epochs, val_acc, color="#1f77b4", alpha=0.45, linewidth=1.0, label="val_acc")
        ax.plot(epochs, best_val, color="#1f77b4", linewidth=1.6, label="best_val_acc")

        # Tier 1 CSVs also have val_acc_ema; plot it if present.
        if "val_acc_ema" in rows_[0]:
            val_acc_ema = np.array([r["val_acc_ema"] for r in rows_]) * 100
            ax.plot(epochs, val_acc_ema, color="#9467bd", alpha=0.65, linewidth=1.0,
                    linestyle="--", label="val_acc_ema")

        bench = scores[codename]["benchmark"]
        if bench > 0.5:
            ax.axhline(bench, color="#b53030", linestyle="--", linewidth=1.0,
                       label=f"benchmark={bench:.1f}%")

        best_idx = int(np.argmax(best_val))
        ax.plot(epochs[best_idx], best_val[best_idx], marker="*",
                markersize=11, color="#2c8a3a", linestyle="")

        ax2 = ax.twinx()
        ax2.plot(epochs, train_loss, color="#ff7f0e", linewidth=0.9, alpha=0.7,
                 label="train_loss")
        ax2.set_ylabel("train loss", color="#ff7f0e", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=7)
        ax2.set_yscale("log")

        adj = scores[codename]["adj"]
        raw = scores[codename]["raw"]
        ds_name = CODENAME_TO_DATASET.get(codename, codename)
        title_color = "#2c8a3a" if adj > 0 else "#b53030"
        ax.set_title(f"{ds_name}  raw={raw:.1f}%  adj={adj:+.2f}",
                     fontsize=10, color=title_color)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("val acc %", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{title_prefix} — Per-dataset learning curves "
                 "(bold blue = best_val so far; purple dashed = val_acc_ema; "
                 "star = best epoch; orange = train loss, log scale)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_ablation(out_path):
    """Grouped bars: Tier 1 full vs single-feature-disabled, per dataset."""
    configs = list(ABLATION_LOGS.keys())  # full, search-off, ema-off, aug-off
    # adj[config][codename]
    adj = {}
    for cfg, log in ABLATION_LOGS.items():
        if log is None:
            adj[cfg] = dict(ABLATION_BASELINE)
        else:
            parsed = parse_scores(log)
            adj[cfg] = {c: parsed[c]["adj"] for c in ABLATION_DATASETS if c in parsed}

    ds_names = [CODENAME_TO_DATASET[c] for c in ABLATION_DATASETS]
    n_groups = len(ABLATION_DATASETS)
    n_cfg = len(configs)
    bar_w = 0.8 / n_cfg
    x = np.arange(n_groups)

    palette = {
        "Tier 1 full": "#2c8a3a",
        "search-off": "#1f77b4",
        "ema-off": "#9467bd",
        "aug-off": "#d4881f",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, cfg in enumerate(configs):
        vals = [adj[cfg].get(c, np.nan) for c in ABLATION_DATASETS]
        offset = (i - (n_cfg - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=cfg,
                      color=palette.get(cfg, "#888"), edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.15 if v >= 0 else -0.35),
                    f"{v:+.2f}", ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("Adjusted score (clipped [-10, +10])")
    ax.set_title("Tier 1 ablation — full vs single-feature-disabled "
                 "(higher = better; full is the green reference)", fontsize=12)
    ax.legend(title="config", loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-11, 4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

    # Print the value table to stdout too — handy for the results note.
    print("\n  Per-feature value (Tier 1 full minus ablated; + means the feature helps):")
    for c in ABLATION_DATASETS:
        full = adj["Tier 1 full"][c]
        line = f"    {CODENAME_TO_DATASET[c]:<12}"
        for cfg in configs[1:]:
            line += f"  {cfg}={full - adj[cfg].get(c, np.nan):+.3f}"
        print(line)


def plot_multiseed(out_path):
    """Three-panel multi-seed figure from dev/aggregate_multiseed.py's JSON.

      A) `full` per-dataset mean±std internal adj + the 3 seed dots
         (visualises which datasets are stable vs. seed-fragile).
      B) aug effect on the shared 7: full vs aug-off grouped bars ±std.
      C) per-config seed-total scatter — the ~16-pt full-config spread.
    """
    agg = json.loads(MULTISEED_JSON.read_text())
    c2d = agg["codename_to_dataset"]
    aug_codes = agg["aug_codenames"]
    fbd = agg["full_by_dataset"]
    aeff = agg["aug_effect"]
    tot = agg["totals"]

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.5], hspace=0.42)

    # ---- A: full per-dataset mean±std + per-seed dots --------------------
    axA = fig.add_subplot(gs[0])
    items = sorted(fbd.items(), key=lambda kv: c2d.get(kv[0], kv[0]).lower())
    names = [c2d.get(c, c) for c, _ in items]
    means = np.array([d["adj"]["mean"] for _, d in items])
    stds = np.array([d["adj"]["std"] for _, d in items])
    x = np.arange(len(items))
    colors = ["#2c8a3a" if m > 0 else "#b53030" for m in means]
    axA.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.6,
            capsize=4, error_kw=dict(ecolor="#333", lw=1.2), alpha=0.85)
    rng = np.random.default_rng(0)
    for xi, (_, d) in zip(x, items):
        vals = d["adj"]["vals"]
        axA.scatter(xi + rng.uniform(-0.13, 0.13, len(vals)), vals,
                    s=34, color="black", zorder=5, alpha=0.8)
    for xi, m, s in zip(x, means, stds):
        axA.text(xi, m + (0.3 if m >= 0 else -0.3) + np.sign(m) * s,
                 f"{m:+.2f}\n±{s:.2f}", ha="center",
                 va="bottom" if m >= 0 else "top", fontsize=8)
    axA.axhline(0, color="black", linewidth=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axA.set_ylabel("internal adj (clipped [-10,+10])")
    t = tot["full"]["internal"]
    axA.set_title(
        f"A. `full` per-dataset — mean ± std over seeds {agg['seeds']} "
        f"(black dots = individual seeds)\n"
        f"full total (internal) = {t['mean']:+.2f} ± {t['std']:.2f}  "
        f"[seeds {', '.join(f'{v:+.1f}' for v in t['vals'])}; spread {t['spread']:.1f}] "
        f"— seed variance ≈ the whole signal", fontsize=11)
    axA.grid(axis="y", alpha=0.3)

    # ---- B: aug effect, shared 7, full vs aug-off -----------------------
    axB = fig.add_subplot(gs[1])
    bn = [c2d[c] for c in aug_codes]
    fm = np.array([aeff[c]["full"]["mean"] for c in aug_codes])
    fs = np.array([aeff[c]["full"]["std"] for c in aug_codes])
    om = np.array([aeff[c]["augoff"]["mean"] for c in aug_codes])
    os_ = np.array([aeff[c]["augoff"]["std"] for c in aug_codes])
    xb = np.arange(len(aug_codes))
    w = 0.38
    axB.bar(xb - w / 2, fm, w, yerr=fs, capsize=3, label="full (aug on)",
            color="#2c8a3a", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    axB.bar(xb + w / 2, om, w, yerr=os_, capsize=3, label="aug-off",
            color="#d4881f", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    for xi, c in zip(xb, aug_codes):
        dlt = aeff[c]["delta"]["mean"]
        top = max(fm[list(aug_codes).index(c)], om[list(aug_codes).index(c)])
        axB.text(xi, top + 1.0, f"Δ{dlt:+.1f}", ha="center", fontsize=8,
                 color="#2c8a3a" if dlt >= 0 else "#b53030", fontweight="bold")
    axB.axhline(0, color="black", linewidth=0.8)
    axB.set_xticks(xb)
    axB.set_xticklabels(bn, rotation=20, ha="right", fontsize=9)
    axB.set_ylabel("internal adj")
    axB.set_ylim(-11.5, 11.5)
    f7, o7 = tot["full"]["internal_aug7"], tot["augoff"]["internal"]
    axB.set_title(
        f"B. Augmentation effect on the shared 7 (Δ = full − aug-off)  "
        f"total: full {f7['mean']:+.1f} vs aug-off {o7['mean']:+.1f} "
        f"→ aug worth {f7['mean']-o7['mean']:+.1f}  "
        f"(Language is the lone large reversal)", fontsize=11)
    axB.legend(loc="lower left", framealpha=0.9)
    axB.grid(axis="y", alpha=0.3)

    # ---- C: per-config seed-total scatter -------------------------------
    axC = fig.add_subplot(gs[2])
    for yi, cfg in enumerate(["full", "augoff"]):
        s = tot[cfg]["internal"]
        axC.scatter(s["vals"], [yi] * len(s["vals"]), s=90, zorder=5,
                    color="#2c8a3a" if cfg == "full" else "#d4881f",
                    edgecolor="black")
        axC.errorbar(s["mean"], yi, xerr=s["std"], fmt="D", ms=9,
                     color="black", capsize=5, zorder=6)
        axC.text(s["mean"], yi + 0.25,
                 f"{cfg}: {s['mean']:+.2f} ± {s['std']:.2f}  "
                 f"(spread {s['spread']:.1f})", ha="center", fontsize=9)
    axC.set_yticks([0, 1])
    axC.set_yticklabels(["full\n(11 ds)", "aug-off\n(7 ds)"])
    axC.set_ylim(-0.6, 1.7)
    axC.set_xlabel("internal total adj per seed (♦ = mean ± std; dots = seeds)")
    axC.set_title("C. Per-config seed-total spread — single-seed deltas are noise",
                  fontsize=11)
    axC.grid(axis="x", alpha=0.3)

    fig.suptitle("Multi-seed sweep (seeds 42/1337/2024, --time-limit 2.0, "
                 "Tier 1 recipe) — variance + augmentation effect",
                 fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_e1(out_path):
    """Two-panel E1-vs-Tier1 figure from dev/aggregate_e1.py's JSON.

      A) per-dataset mean ± std, Tier-1 vs E1 grouped bars + per-seed dots
         (does the bar shrink AND not drop — the success picture).
      B) σ (std) per dataset, Tier-1 → E1 — the variance collapse, the
         headline (AddNIST target σ < 2).
    """
    if not E1_JSON.exists():
        print(f"  {E1_JSON.relative_to(REPO_ROOT)} not found — run "
              f"dev/aggregate_e1.py after the E1 sweep lands.")
        return
    agg = json.loads(E1_JSON.read_text())
    rows = agg.get("rows", [])
    if not rows:
        print("  E1 aggregate has no rows yet (sweep not landed) — skipping.")
        return
    names = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 10))

    bm = np.array([r["base"]["mean"] for r in rows])
    bs = np.array([r["base"]["std"] for r in rows])
    em = np.array([r["e1"]["mean"] for r in rows])
    es = np.array([r["e1"]["std"] for r in rows])
    axA.bar(x - w / 2, bm, w, yerr=bs, capsize=3, label="Tier 1 (paired)",
            color="#1f77b4", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    axA.bar(x + w / 2, em, w, yerr=es, capsize=3, label="E1 ensemble",
            color="#2c8a3a", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    for xi, r in zip(x, rows):
        for v in r["base"]["vals"]:
            axA.scatter(xi - w / 2, v, s=22, color="black", zorder=5, alpha=.7)
        for v in r["e1"]["vals"]:
            axA.scatter(xi + w / 2, v, s=22, color="black", zorder=5, alpha=.7)
        axA.text(xi, max(bm[list(x).index(xi)], em[list(x).index(xi)]) + 1.0,
                 f"Δ{r['d_mean']:+.2f}", ha="center", fontsize=8,
                 color="#2c8a3a" if r["d_mean"] >= 0 else "#b53030",
                 fontweight="bold")
    axA.axhline(0, color="black", linewidth=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels(names, fontsize=9)
    axA.set_ylabel("internal adj (mean ± std, 3 seeds)")
    axA.set_title("A. Tier-1 vs E1 per dataset — mean ± std + per-seed dots "
                  "(Δ = E1 − Tier-1 mean; want σ↓ without mean↓)", fontsize=11)
    axA.legend(loc="lower left", framealpha=.9)
    axA.grid(axis="y", alpha=.3)

    axB.bar(x - w / 2, bs, w, label="Tier 1 σ", color="#1f77b4",
            edgecolor="black", linewidth=.5)
    axB.bar(x + w / 2, es, w, label="E1 σ", color="#2c8a3a",
            edgecolor="black", linewidth=.5)
    for xi, r in zip(x, rows):
        axB.text(xi, max(bs[list(x).index(xi)], es[list(x).index(xi)]) + 0.1,
                 f"×{r['sigma_ratio']:.2f}", ha="center", fontsize=8,
                 fontweight="bold")
    axB.axhline(2.0, color="#b53030", linestyle=":", linewidth=1.0,
                label="AddNIST σ target (<2)")
    axB.set_xticks(x)
    axB.set_xticklabels(names, fontsize=9)
    axB.set_ylabel("σ of internal adj across 3 seeds")
    axB.set_title("B. Variance collapse — Tier-1 σ → E1 σ "
                  "(the headline; ×ratio annotated)", fontsize=11)
    axB.legend(loc="upper right", framealpha=.9)
    axB.grid(axis="y", alpha=.3)

    fig.suptitle("E1 budget-filling ensemble vs Tier-1 (paired: same 5 "
                 "datasets × seeds 42/1337/2024 × 2 h) — thesis Q4",
                 fontsize=13, y=0.997)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2bfull(out_path):
    """Two-panel figure for the Tier-2b FULL EGNAS sweep (11 ds × 3 seeds).

      A) per-dataset Tier-1 vs EGNAS-full mean ± std grouped bars + per-seed
         dots; sorted by Δmean desc; Δ annotated above each pair.
      B) per-dataset σ comparison — the variance-collapse picture.

    Reads dev/logs/tier2bfull/_tier2bfull_aggregate.json. Mirrors plot_e1's
    style for visual consistency with [[E1 - Results]] and the multi-seed
    figure.
    """
    if not TIER2BFULL_JSON.exists():
        print(f"  {TIER2BFULL_JSON.relative_to(REPO_ROOT)} not found — run "
              f"dev/aggregate_tier2bfull.py after the sweep lands.")
        return
    agg = json.loads(TIER2BFULL_JSON.read_text())
    rows = agg.get("rows", {})
    if not rows:
        print("  no rows yet — skipping.")
        return
    # Sort by Δmean desc so big wins read first.
    names = sorted(rows.keys(), key=str.lower)
    x = np.arange(len(names))
    bm = np.array([rows[n]["tier1"]["mean"] for n in names])
    bs = np.array([rows[n]["tier1"]["std"]  for n in names])
    em = np.array([rows[n]["egnas"]["mean"] for n in names])
    es = np.array([rows[n]["egnas"]["std"]  for n in names])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 11))
    w = 0.38

    # A — paired means
    axA.bar(x - w / 2, bm, w, yerr=bs, capsize=3, label="Tier 1 (paired)",
            color="#1f77b4", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    axA.bar(x + w / 2, em, w, yerr=es, capsize=3, label="EGNAS-full (macro+micro)",
            color="#2c8a3a", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    for i, n in enumerate(names):
        for v in rows[n]["tier1"]["vals"]:
            axA.scatter(i - w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        for v in rows[n]["egnas"]["vals"]:
            axA.scatter(i + w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        d = rows[n]["d_mean"]
        top = max(bm[i] + bs[i], em[i] + es[i])
        axA.text(i, top + 0.5, f"Δ{d:+.1f}", ha="center", fontsize=8,
                 fontweight="bold",
                 color="#2c8a3a" if d > 0 else "#b53030")
    axA.axhline(0, color="black", lw=.8)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axA.set_ylabel("internal adj (mean ± std, 3 seeds)")
    T = agg.get("totals") or {}
    if T.get("tier1") and T.get("egnas"):
        Tb, Te = T["tier1"], T["egnas"]
        ratio = Te['std'] / Tb['std'] if Tb['std'] > 1e-9 else float('nan')
        axA.set_title(
            "A. Tier-1 vs EGNAS-full per dataset — alphabetic order "
            "(black dots = individual seeds; Δ annotated above each pair)\n"
            f"Total (11 ds): Tier-1 {Tb['mean']:+.2f} ± {Tb['std']:.2f}  →  "
            f"EGNAS-full {Te['mean']:+.2f} ± {Te['std']:.2f}   "
            f"(Δmean {Te['mean']-Tb['mean']:+.2f}, σ ×{ratio:.2f})",
            fontsize=11)
    axA.legend(loc="lower left", framealpha=.9)
    axA.grid(axis="y", alpha=.3)

    # B — σ collapse
    axB.bar(x - w / 2, bs, w, label="Tier 1 σ",
            color="#1f77b4", edgecolor="black", linewidth=.5)
    axB.bar(x + w / 2, es, w, label="EGNAS-full σ",
            color="#2c8a3a", edgecolor="black", linewidth=.5)
    for i in range(len(names)):
        ratio = es[i] / max(bs[i], 1e-9)
        top = max(bs[i], es[i])
        axB.text(i, top + 0.12, f"×{ratio:.2f}", ha="center", fontsize=8,
                 fontweight="bold")
    axB.set_xticks(x)
    axB.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axB.set_ylabel("σ of internal adj across 3 seeds")
    axB.set_title("B. Variance collapse per dataset — Tier-1 σ → EGNAS-full σ "
                  "(×ratio annotated)", fontsize=11)
    axB.legend(loc="upper right", framealpha=.9)
    axB.grid(axis="y", alpha=.3)

    fig.suptitle("Tier-2b FULL EGNAS (macro + micro) — 3-seed × 11-dataset "
                 f"paired vs Tier-1   (seeds {agg.get('seeds')})",
                 fontsize=13, y=0.997)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2bfull_curves(out_path):
    """Per-dataset aggregated learning curves across the 3 EGNAS-full seeds.

    For each dataset: thin grey lines = each seed's val_acc trajectory; bold
    green = mean across seeds (NaN-aware so early-stopped seeds don't bias
    later epochs); shaded band = ±std. Reads
    `dev/logs/tier2bfull/t2bfull_s{42,1337,2024}/<codename>.csv` (the
    per-epoch CSVs the Tier-1 trainer writes via BASELINE_LOG_DIR).
    """
    base = REPO_ROOT / "dev" / "logs" / "tier2bfull"
    seeds = [42, 1337, 2024]
    # Original 11-ds sweep lives in t2bfull_s{seed}/; the Windspeed+Cryptic
    # follow-on landed in t2bfull_new_s{seed}/. Union both so the figure now
    # covers all 13 datasets.
    s42_dirs = [base / "t2bfull_s42", base / "t2bfull_new_s42"]
    s42_dirs = [d for d in s42_dirs if d.exists()]
    if not s42_dirs:
        print(f"  no t2bfull s42 dirs found under {base.relative_to(REPO_ROOT)}.")
        return
    codes = sorted({p.stem for d in s42_dirs for p in d.glob("*.csv")},
                   key=_ds_name)
    if not codes:
        print("  no per-codename CSVs found.")
        return

    def _egnas_paths(code):
        """Return candidate CSV paths for `code` across the 3 seeds, looking
        in both the original sweep dir and the new-datasets follow-on dir."""
        return [base / f"{tag}_s{s}" / f"{code}.csv"
                for s in seeds
                for tag in ("t2bfull", "t2bfull_new")]

    cols = 4
    rows = (len(codes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                             squeeze=False)
    flat = axes.flatten()

    base_t1 = REPO_ROOT / "dev" / "logs" / "multiseed"  # paired Tier-1 CSVs

    def _aggregate(paths):
        """Load val_acc curves from CSV paths → per-seed list + max_ep."""
        per = []
        mx = 0
        for p in paths:
            if not p.exists():
                continue
            rs = load_csv(p)
            va = np.array([r["val_acc"] for r in rs]) * 100
            per.append(va)
            mx = max(mx, len(va))
        return per, mx

    for ax, code in zip(flat, codes):
        ds = CODENAME_TO_DATASET.get(code, code)

        # --- dataset benchmark (ResNet-18 zero-point of adj-score) ---
        try:
            md_p = REPO_ROOT / "datasets" / ds / "metadata"
            if md_p.exists():
                md = json.loads(md_p.read_text())
                bench = resolve_benchmark(code, float(md.get("benchmark", 0.0) or 0.0))
                if bench > 0.5:  # skip placeholder/zero benchmarks
                    ax.axhline(bench, color="#b53030", linestyle="--", lw=1.0,
                               alpha=0.75, label=f"benchmark {bench:.1f}%")
        except Exception:
            pass

        # --- Tier-1 baseline (multiseed full_s*) underneath ---
        t1_per, t1_max = _aggregate([base_t1 / f"full_s{s}" / f"{code}.csv"
                                     for s in seeds])
        if t1_per:
            mat = np.full((len(t1_per), t1_max), np.nan)
            for i, v in enumerate(t1_per):
                mat[i, : len(v)] = v
            m1, s1 = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(t1_max), m1 - s1, m1 + s1,
                            color="#1f77b4", alpha=0.16, label="Tier-1 ±std")
            ax.plot(np.arange(t1_max), m1, color="#1f77b4", lw=1.4, alpha=0.9,
                    label="Tier-1 mean")

        # --- EGNAS-full (on top) ---
        eg_per, eg_max = _aggregate(_egnas_paths(code))
        for v in eg_per:
            ax.plot(np.arange(len(v)), v, color="#3a3", alpha=0.30, lw=0.5)
        if eg_per:
            mat = np.full((len(eg_per), eg_max), np.nan)
            for i, v in enumerate(eg_per):
                mat[i, : len(v)] = v
            me, se = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(eg_max), me - se, me + se,
                            color="#2c8a3a", alpha=0.24, label="EGNAS-full ±std")
            ax.plot(np.arange(eg_max), me, color="#2c8a3a", lw=2.0,
                    label="EGNAS-full mean")

        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("val acc %", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=6, framealpha=0.85)

    for ax in flat[len(codes):]:
        ax.axis("off")

    fig.suptitle("Tier-2b Full EGNAS — per-dataset learning curves vs Tier-1 "
                 "(blue = Tier-1 mean ± std across 3 seeds; "
                 "green = EGNAS-full mean ± std; thin green = each seed)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2a_p1_v6(out_path):
    """Two-panel figure for Tier-2a / P1 v6 (paired vs EGNAS-full, 13 ds × 3 seeds).

    Same shape as plot_tier2bfull but reads _tier2a_p1_v4_aggregate.json,
    where rows have `egnas` and `p1` fields instead of `tier1` and `egnas`.
    Panel A: per-dataset EG vs +P1 mean ± std + seed dots + Δ annotation.
    Panel B: σ change EG → +P1 (×ratio annotated).
    """
    if not TIER2A_P1_V6_JSON.exists():
        print(f"  {TIER2A_P1_V6_JSON.relative_to(REPO_ROOT)} not found — run "
              "dev/aggregate_tier2a_p1_v6.py after the sweep lands.")
        return
    agg = json.loads(TIER2A_P1_V6_JSON.read_text())
    rows = agg.get("rows", {})
    if not rows:
        print("  no rows yet — skipping.")
        return
    names = sorted(rows.keys(), key=str.lower)
    x = np.arange(len(names))
    em = np.array([rows[n]["egnas"]["mean"] for n in names])
    es = np.array([rows[n]["egnas"]["std"]  for n in names])
    pm = np.array([rows[n]["p1"]["mean"]    for n in names])
    ps = np.array([rows[n]["p1"]["std"]     for n in names])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 11))
    w = 0.38

    # A — paired means
    axA.bar(x - w / 2, em, w, yerr=es, capsize=3, label="EGNAS-full (paired baseline)",
            color="#2c8a3a", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    axA.bar(x + w / 2, pm, w, yerr=ps, capsize=3, label="+P1 v6 (C-only + 9 ops + verify)",
            color="#d97706", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    for i, n in enumerate(names):
        for v in rows[n]["egnas"]["vals"]:
            axA.scatter(i - w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        for v in rows[n]["p1"]["vals"]:
            axA.scatter(i + w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        d = rows[n]["d_mean"]
        top = max(em[i] + es[i], pm[i] + ps[i])
        axA.text(i, top + 0.5, f"Δ{d:+.1f}", ha="center", fontsize=8,
                 fontweight="bold",
                 color="#2c8a3a" if d > 0 else "#b53030")
    # Mark which datasets had the probe applied vs skipped (no-aug metadata).
    probes_field = agg.get("probe_decisions", {})
    for i, n in enumerate(names):
        # codename lookup: invert
        code = next((c for c, dn in {"Adaline":"AddNIST","Caitie":"CIFARTile",
                "Chester":"Chesseract","conway":"GameOfLife","Sadie":"GeoClassing",
                "Gutenberg":"Gutenberg","LaMelo":"Language","Mateo":"MultNIST",
                "Myopia":"Myofibre","Sokoto":"Sudoku","Volga":"Voxel",
                "Cryptic":"Cryptic","windspeed":"Windspeed"}.items() if dn == n), n)
        decs = probes_field.get(code, {})
        if not decs:
            axA.text(i, axA.get_ylim()[0] + 0.3, "(skip-probe)", ha="center",
                     fontsize=7, color="#888", style="italic")
    axA.axhline(0, color="black", lw=.8)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axA.set_ylabel("internal adj (mean ± std across seeds)")
    T = agg.get("totals") or {}
    if T.get("egnas") and T.get("p1"):
        Te, Tp = T["egnas"], T["p1"]
        ratio = Tp['std'] / Te['std'] if Te['std'] > 1e-9 else float('nan')
        axA.set_title(
            "A. EGNAS-full vs +P1 v6 per dataset — alphabetic order "
            "(black dots = individual seeds; Δ annotated above each pair)\n"
            f"Total (13 ds): EGNAS {Te['mean']:+.2f} ± {Te['std']:.2f}  →  "
            f"+P1 v6 {Tp['mean']:+.2f} ± {Tp['std']:.2f}   "
            f"(Δmean {Tp['mean']-Te['mean']:+.2f}, σ ×{ratio:.2f})",
            fontsize=11)
    axA.legend(loc="lower left", framealpha=.9)
    axA.grid(axis="y", alpha=.3)

    # B — σ change
    axB.bar(x - w / 2, es, w, label="EGNAS-full σ",
            color="#2c8a3a", edgecolor="black", linewidth=.5)
    axB.bar(x + w / 2, ps, w, label="+P1 v6 σ",
            color="#d97706", edgecolor="black", linewidth=.5)
    for i in range(len(names)):
        ratio = ps[i] / max(es[i], 1e-9)
        top = max(es[i], ps[i])
        axB.text(i, top + 0.12, f"×{ratio:.2f}", ha="center", fontsize=8,
                 fontweight="bold",
                 color="#b53030" if ratio > 1 else "#2c8a3a")
    axB.set_xticks(x)
    axB.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axB.set_ylabel("σ of internal adj across seeds")
    axB.set_title("B. σ change per dataset — EGNAS-full σ → +P1 v6 σ "
                  "(×ratio annotated; red = P1 noisier than EG)", fontsize=11)
    axB.legend(loc="upper right", framealpha=.9)
    axB.grid(axis="y", alpha=.3)

    fig.suptitle("Tier-2a / P1 v6 (C-only + 9 ops + verify) — multi-seed × 13-dataset "
                 f"paired vs EGNAS-full   (seeds {agg.get('seeds')})",
                 fontsize=13, y=0.997)  # v6: s42+s1337 complete, s2024 partial
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2a_p1_v6_curves(out_path):
    """Per-dataset aggregated learning curves: EGNAS-full (green) vs +P1 v6 (orange).

    Mirrors plot_tier2bfull_curves but compares EGNAS-full baseline (the
    paired EG sweep dirs) against +P1 v6. 13 panels alphabetic; benchmark
    line per panel; thin lines = each seed, bold = across-seed mean, band
    = ±std.
    """
    base_eg = REPO_ROOT / "dev" / "logs" / "tier2bfull"
    base_p1 = REPO_ROOT / "dev" / "logs" / "tier2a_p1_v6"
    seeds = [42, 1337, 2024]
    s42_dirs = [base_eg / "t2bfull_s42", base_eg / "t2bfull_new_s42"]
    s42_dirs = [d for d in s42_dirs if d.exists()]
    p1_seed_dir = base_p1 / "p1v6_s42"
    if not p1_seed_dir.exists():
        print(f"  {p1_seed_dir.relative_to(REPO_ROOT)} not found.")
        return
    # Union codenames (EG sweep + P1 v6 sweep).
    codes = sorted({p.stem for d in s42_dirs for p in d.glob("*.csv")}
                   | {p.stem for p in p1_seed_dir.glob("*.csv")},
                   key=_ds_name)
    if not codes:
        print("  no per-codename CSVs found.")
        return

    def _aggregate(paths):
        per = []
        mx = 0
        for p in paths:
            if not p.exists():
                continue
            rs = load_csv(p)
            va = np.array([r["val_acc"] for r in rs]) * 100
            per.append(va)
            mx = max(mx, len(va))
        return per, mx

    def _eg_paths(code):
        return [base_eg / f"{tag}_s{s}" / f"{code}.csv"
                for s in seeds
                for tag in ("t2bfull", "t2bfull_new")]

    def _p1_paths(code):
        return [base_p1 / f"p1v6_s{s}" / f"{code}.csv" for s in seeds]

    cols = 4
    rows = (len(codes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                             squeeze=False)
    flat = axes.flatten()

    for ax, code in zip(flat, codes):
        ds = CODENAME_TO_DATASET.get(code, code)

        # benchmark reference (dashed red)
        try:
            md_p = REPO_ROOT / "datasets" / ds / "metadata"
            if md_p.exists():
                md = json.loads(md_p.read_text())
                bench = resolve_benchmark(code, float(md.get("benchmark", 0.0) or 0.0))
                if bench > 0.5:
                    ax.axhline(bench, color="#b53030", linestyle="--", lw=1.0,
                               alpha=0.75, label=f"benchmark {bench:.1f}%")
        except Exception:
            pass

        # EGNAS-full baseline (green, under)
        eg_per, eg_max = _aggregate(_eg_paths(code))
        if eg_per:
            mat = np.full((len(eg_per), eg_max), np.nan)
            for i, v in enumerate(eg_per):
                mat[i, : len(v)] = v
            m1, s1 = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(eg_max), m1 - s1, m1 + s1,
                            color="#2c8a3a", alpha=0.16, label="EGNAS-full ±std")
            ax.plot(np.arange(eg_max), m1, color="#2c8a3a", lw=1.4, alpha=0.9,
                    label="EGNAS-full mean")

        # +P1 v6 (orange, on top)
        p1_per, p1_max = _aggregate(_p1_paths(code))
        for v in p1_per:
            ax.plot(np.arange(len(v)), v, color="#d97706", alpha=0.30, lw=0.5)
        if p1_per:
            mat = np.full((len(p1_per), p1_max), np.nan)
            for i, v in enumerate(p1_per):
                mat[i, : len(v)] = v
            me, se = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(p1_max), me - se, me + se,
                            color="#d97706", alpha=0.24, label="+P1 v6 ±std")
            ax.plot(np.arange(p1_max), me, color="#d97706", lw=2.0,
                    label="+P1 v6 mean")

        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("val acc %", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=6, framealpha=0.85)

    for ax in flat[len(codes):]:
        ax.axis("off")

    fig.suptitle("Tier-2a / P1 v6 vs EGNAS-full — per-dataset learning curves "
                 "(green = EGNAS-full mean ± std across seeds; "
                 "orange = +P1 v6 mean ± std; thin orange = each seed)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2a_p1_v7(out_path):
    """Two-panel figure for Tier-2a / P1 v7 (paired vs EGNAS-full, 13 ds × 3 seeds).

    Same shape as plot_tier2bfull but reads _tier2a_p1_v4_aggregate.json,
    where rows have `egnas` and `p1` fields instead of `tier1` and `egnas`.
    Panel A: per-dataset EG vs +P1 mean ± std + seed dots + Δ annotation.
    Panel B: σ change EG → +P1 (×ratio annotated).
    """
    if not TIER2A_P1_V7_JSON.exists():
        print(f"  {TIER2A_P1_V7_JSON.relative_to(REPO_ROOT)} not found — run "
              "dev/aggregate_tier2a_p1_v7.py after the sweep lands.")
        return
    agg = json.loads(TIER2A_P1_V7_JSON.read_text())
    rows = agg.get("rows", {})
    if not rows:
        print("  no rows yet — skipping.")
        return
    names = sorted(rows.keys(), key=str.lower)
    x = np.arange(len(names))
    em = np.array([rows[n]["egnas"]["mean"] for n in names])
    es = np.array([rows[n]["egnas"]["std"]  for n in names])
    pm = np.array([rows[n]["p1"]["mean"]    for n in names])
    ps = np.array([rows[n]["p1"]["std"]     for n in names])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 11))
    w = 0.38

    # A — paired means
    axA.bar(x - w / 2, em, w, yerr=es, capsize=3, label="EGNAS-full (paired baseline)",
            color="#2c8a3a", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    axA.bar(x + w / 2, pm, w, yerr=ps, capsize=3, label="+P1 v7 (C-only + 9 ops + verify)",
            color="#d97706", edgecolor="black", linewidth=.5,
            error_kw=dict(ecolor="#333", lw=1.0))
    for i, n in enumerate(names):
        for v in rows[n]["egnas"]["vals"]:
            axA.scatter(i - w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        for v in rows[n]["p1"]["vals"]:
            axA.scatter(i + w / 2, v, s=18, color="black", zorder=5, alpha=.6)
        d = rows[n]["d_mean"]
        top = max(em[i] + es[i], pm[i] + ps[i])
        axA.text(i, top + 0.5, f"Δ{d:+.1f}", ha="center", fontsize=8,
                 fontweight="bold",
                 color="#2c8a3a" if d > 0 else "#b53030")
    # Mark which datasets had the probe applied vs skipped (no-aug metadata).
    probes_field = agg.get("probe_decisions", {})
    for i, n in enumerate(names):
        # codename lookup: invert
        code = next((c for c, dn in {"Adaline":"AddNIST","Caitie":"CIFARTile",
                "Chester":"Chesseract","conway":"GameOfLife","Sadie":"GeoClassing",
                "Gutenberg":"Gutenberg","LaMelo":"Language","Mateo":"MultNIST",
                "Myopia":"Myofibre","Sokoto":"Sudoku","Volga":"Voxel",
                "Cryptic":"Cryptic","windspeed":"Windspeed"}.items() if dn == n), n)
        decs = probes_field.get(code, {})
        if not decs:
            axA.text(i, axA.get_ylim()[0] + 0.3, "(skip-probe)", ha="center",
                     fontsize=7, color="#888", style="italic")
    axA.axhline(0, color="black", lw=.8)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axA.set_ylabel("internal adj (mean ± std across seeds)")
    T = agg.get("totals") or {}
    if T.get("egnas") and T.get("p1"):
        Te, Tp = T["egnas"], T["p1"]
        ratio = Tp['std'] / Te['std'] if Te['std'] > 1e-9 else float('nan')
        axA.set_title(
            "A. EGNAS-full vs +P1 v7 per dataset — alphabetic order "
            "(black dots = individual seeds; Δ annotated above each pair)\n"
            f"Total (13 ds): EGNAS {Te['mean']:+.2f} ± {Te['std']:.2f}  →  "
            f"+P1 v7 {Tp['mean']:+.2f} ± {Tp['std']:.2f}   "
            f"(Δmean {Tp['mean']-Te['mean']:+.2f}, σ ×{ratio:.2f})",
            fontsize=11)
    axA.legend(loc="lower left", framealpha=.9)
    axA.grid(axis="y", alpha=.3)

    # B — σ change
    axB.bar(x - w / 2, es, w, label="EGNAS-full σ",
            color="#2c8a3a", edgecolor="black", linewidth=.5)
    axB.bar(x + w / 2, ps, w, label="+P1 v7 σ",
            color="#d97706", edgecolor="black", linewidth=.5)
    for i in range(len(names)):
        ratio = ps[i] / max(es[i], 1e-9)
        top = max(es[i], ps[i])
        axB.text(i, top + 0.12, f"×{ratio:.2f}", ha="center", fontsize=8,
                 fontweight="bold",
                 color="#b53030" if ratio > 1 else "#2c8a3a")
    axB.set_xticks(x)
    axB.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    axB.set_ylabel("σ of internal adj across seeds")
    axB.set_title("B. σ change per dataset — EGNAS-full σ → +P1 v7 σ "
                  "(×ratio annotated; red = P1 noisier than EG)", fontsize=11)
    axB.legend(loc="upper right", framealpha=.9)
    axB.grid(axis="y", alpha=.3)

    fig.suptitle("Tier-2a / P1 v7 (C-only + 9 ops + verify) — multi-seed × 13-dataset "
                 f"paired vs EGNAS-full   (seeds {agg.get('seeds')})",
                 fontsize=13, y=0.997)  # v6: s42+s1337 complete, s2024 partial
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def plot_tier2a_p1_v7_curves(out_path):
    """Per-dataset aggregated learning curves: EGNAS-full (green) vs +P1 v7 (orange).

    Mirrors plot_tier2bfull_curves but compares EGNAS-full baseline (the
    paired EG sweep dirs) against +P1 v7. 13 panels alphabetic; benchmark
    line per panel; thin lines = each seed, bold = across-seed mean, band
    = ±std.
    """
    base_eg = REPO_ROOT / "dev" / "logs" / "tier2bfull"
    base_p1 = REPO_ROOT / "dev" / "logs" / "tier2a_p1_v7"
    seeds = [42, 1337, 2024]
    s42_dirs = [base_eg / "t2bfull_s42", base_eg / "t2bfull_new_s42"]
    s42_dirs = [d for d in s42_dirs if d.exists()]
    p1_seed_dir = base_p1 / "p1v7_s42"
    if not p1_seed_dir.exists():
        print(f"  {p1_seed_dir.relative_to(REPO_ROOT)} not found.")
        return
    # Union codenames (EG sweep + P1 v7 sweep).
    codes = sorted({p.stem for d in s42_dirs for p in d.glob("*.csv")}
                   | {p.stem for p in p1_seed_dir.glob("*.csv")},
                   key=_ds_name)
    if not codes:
        print("  no per-codename CSVs found.")
        return

    def _aggregate(paths):
        per = []
        mx = 0
        for p in paths:
            if not p.exists():
                continue
            rs = load_csv(p)
            va = np.array([r["val_acc"] for r in rs]) * 100
            per.append(va)
            mx = max(mx, len(va))
        return per, mx

    def _eg_paths(code):
        return [base_eg / f"{tag}_s{s}" / f"{code}.csv"
                for s in seeds
                for tag in ("t2bfull", "t2bfull_new")]

    def _p1_paths(code):
        return [base_p1 / f"p1v7_s{s}" / f"{code}.csv" for s in seeds]

    cols = 4
    rows = (len(codes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                             squeeze=False)
    flat = axes.flatten()

    for ax, code in zip(flat, codes):
        ds = CODENAME_TO_DATASET.get(code, code)

        # benchmark reference (dashed red)
        try:
            md_p = REPO_ROOT / "datasets" / ds / "metadata"
            if md_p.exists():
                md = json.loads(md_p.read_text())
                bench = resolve_benchmark(code, float(md.get("benchmark", 0.0) or 0.0))
                if bench > 0.5:
                    ax.axhline(bench, color="#b53030", linestyle="--", lw=1.0,
                               alpha=0.75, label=f"benchmark {bench:.1f}%")
        except Exception:
            pass

        # EGNAS-full baseline (green, under)
        eg_per, eg_max = _aggregate(_eg_paths(code))
        if eg_per:
            mat = np.full((len(eg_per), eg_max), np.nan)
            for i, v in enumerate(eg_per):
                mat[i, : len(v)] = v
            m1, s1 = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(eg_max), m1 - s1, m1 + s1,
                            color="#2c8a3a", alpha=0.16, label="EGNAS-full ±std")
            ax.plot(np.arange(eg_max), m1, color="#2c8a3a", lw=1.4, alpha=0.9,
                    label="EGNAS-full mean")

        # +P1 v7 (orange, on top)
        p1_per, p1_max = _aggregate(_p1_paths(code))
        for v in p1_per:
            ax.plot(np.arange(len(v)), v, color="#d97706", alpha=0.30, lw=0.5)
        if p1_per:
            mat = np.full((len(p1_per), p1_max), np.nan)
            for i, v in enumerate(p1_per):
                mat[i, : len(v)] = v
            me, se = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(p1_max), me - se, me + se,
                            color="#d97706", alpha=0.24, label="+P1 v7 ±std")
            ax.plot(np.arange(p1_max), me, color="#d97706", lw=2.0,
                    label="+P1 v7 mean")

        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("val acc %", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", fontsize=6, framealpha=0.85)

    for ax in flat[len(codes):]:
        ax.axis("off")

    fig.suptitle("Tier-2a / P1 v7 vs EGNAS-full — per-dataset learning curves "
                 "(green = EGNAS-full mean ± std across seeds; "
                 "orange = +P1 v7 mean ± std; thin orange = each seed)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=sorted(TIER_CONFIG.keys()), default="tier1")
    p.add_argument("--ablation", action="store_true",
                   help="Also produce the Tier 1 ablation comparison chart.")
    p.add_argument("--multiseed", action="store_true",
                   help="Produce the multi-seed variance + aug-effect figure "
                        "(reads dev/logs/multiseed/_aggregate.json) and exit.")
    p.add_argument("--e1", action="store_true",
                   help="Produce the E1-vs-Tier1 variance-collapse figure "
                        "(reads dev/logs/e1val/_e1_aggregate.json) and exit.")
    p.add_argument("--tier2bfull", action="store_true",
                   help="Produce the Tier-2b FULL EGNAS vs Tier-1 figure "
                        "(reads dev/logs/tier2bfull/_tier2bfull_aggregate.json) "
                        "and exit.")
    p.add_argument("--tier2bfull-curves", action="store_true",
                   help="Produce per-dataset aggregated learning curves "
                        "for Tier-2b FULL EGNAS (3 seeds, mean ± std).")
    p.add_argument("--tier2a-p1-v4", action="store_true",
                   help="Produce the Tier-2a / P1 v4 vs EGNAS-full figure "
                        "(reads dev/logs/tier2a_p1_v4/_tier2a_p1_v4_aggregate.json) "
                        "and exit.")
    p.add_argument("--tier2a-p1-v4-curves", action="store_true",
                   help="Produce per-dataset aggregated learning curves for "
                        "Tier-2a / P1 v4 with EGNAS-full overlay (3 seeds).")
    p.add_argument("--tier2a-p1-v6", action="store_true",
                   help="Produce the Tier-2a / P1 v6 vs EGNAS-full figure "
                        "(reads dev/logs/tier2a_p1_v6/_tier2a_p1_v6_aggregate.json) "
                        "and exit.")
    p.add_argument("--tier2a-p1-v6-curves", action="store_true",
                   help="Produce per-dataset aggregated learning curves for "
                        "Tier-2a / P1 v6 with EGNAS-full overlay.")
    p.add_argument("--tier2a-p1-v7", action="store_true",
                   help="Tier-2a / P1 v7 vs EGNAS-full paired figure.")
    p.add_argument("--tier2a-p1-v7-curves", action="store_true",
                   help="Tier-2a / P1 v7 aggregated learning curves vs EGNAS-full.")
    args = p.parse_args()

    if args.multiseed:
        print("=== Multi-seed figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_multiseed(_img_out("multiseed.png"))
        return

    if args.e1:
        print("=== E1 figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_e1(_img_out("e1.png"))
        return

    if args.tier2bfull:
        print("=== Tier-2b FULL EGNAS figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2bfull(_img_out("tier2bfull.png"))
        return

    if args.tier2bfull_curves:
        print("=== Tier-2b FULL EGNAS — aggregated learning curves ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2bfull_curves(_img_out("tier2bfull_curves.png"))
        return

    if args.tier2a_p1_v4:
        print("=== Tier-2a / P1 v4 figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v4(_img_out("tier2a_p1_v4.png"))
        return

    if args.tier2a_p1_v4_curves:
        print("=== Tier-2a / P1 v4 — aggregated learning curves ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v4_curves(_img_out("tier2a_p1_v4_curves.png"))
        return

    if args.tier2a_p1_v6:
        print("=== Tier-2a / P1 v6 figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v6(_img_out("tier2a_p1_v6.png"))
        return

    if args.tier2a_p1_v6_curves:
        print("=== Tier-2a / P1 v6 — aggregated learning curves ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v6_curves(_img_out("tier2a_p1_v6_curves.png"))
        return

    if args.tier2a_p1_v7:
        print("=== Tier-2a / P1 v7 figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v7(_img_out("tier2a_p1_v7.png"))
        return

    if args.tier2a_p1_v7_curves:
        print("=== Tier-2a / P1 v7 — aggregated learning curves ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_tier2a_p1_v7_curves(_img_out("tier2a_p1_v7_curves.png"))
        return

    cfg = TIER_CONFIG[args.tier]
    csv_dir = cfg["csv_dir"]
    log_file = cfg["log_file"]
    out_dir = cfg["out_dir"]
    title_prefix = cfg["title"]

    print(f"=== {title_prefix} ===")
    log_paths = _as_paths(log_file)
    print("Reading scores from",
          ", ".join(str(p.relative_to(REPO_ROOT)) for p in log_paths))
    scores = parse_scores(log_paths)
    print(f"  parsed {len(scores)} datasets")

    csv_dirs = _as_paths(csv_dir)
    print("Reading CSVs from",
          ", ".join(str(d.relative_to(REPO_ROOT)) for d in csv_dirs))
    for d in csv_dirs:
        for path in sorted(d.glob("*.csv")):
            print(f"  {path.relative_to(REPO_ROOT)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not scores:
        # Partial run — the harness's `Scoring` section hasn't printed yet,
        # but per-dataset CSVs exist for datasets that already finished
        # training. Skip the score-bars chart (needs adj scores) and draw
        # only the learning-curves grid for whatever CSVs are on disk.
        seen = sorted({p.stem for d in csv_dirs for p in d.glob("*.csv")})
        if not seen:
            print("  no scores AND no CSVs — nothing to plot")
            return
        print(f"  no Scoring section yet; partial-run mode, drawing curves "
              f"only for {len(seen)} CSV-discovered codenames: {seen}")
        scores = {c: {"raw": 0.0, "benchmark": 0.0, "adj": 0.0} for c in seen}
        plot_learning_curves(scores, csv_dir,
                             _img_out(f"{args.tier}_curves.png"),
                             title_prefix + "  [PARTIAL — sweep still running]")
        return
    plot_score_bars(scores, _img_out(f"{args.tier}_scores.png"), title_prefix)
    plot_learning_curves(scores, csv_dir, _img_out(f"{args.tier}_curves.png"), title_prefix)

    if args.ablation:
        print("\n=== Tier 1 ablation chart ===")
        plot_ablation(_img_out("tier1_ablation.png"))


if __name__ == "__main__":
    main()
