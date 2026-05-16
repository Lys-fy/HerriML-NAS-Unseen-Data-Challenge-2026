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

REPO_ROOT = Path(__file__).resolve().parent.parent

# All plots live in the notebase so Obsidian can embed them with ![[name.png]].
IMPL_DIR = REPO_ROOT / "docs" / "notes" / "09 - Implementation" / "implementation"

# Machine-readable aggregate produced by dev/aggregate_multiseed.py.
MULTISEED_JSON = REPO_ROOT / "dev" / "logs" / "multiseed" / "_aggregate.json"

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
}

# Per-tier path resolution. Add new tiers here. All outputs land in IMPL_DIR
# (the Obsidian implementation/ folder) so notes can embed them.
TIER_CONFIG = {
    "tier0": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "training-2h",
        log_file=REPO_ROOT / "dev" / "logs" / "tier0-sweep-2h.log",
        out_dir=IMPL_DIR,
        title="Tier 0 — Hardcoded CNN",
    ),
    "tier1": dict(
        csv_dir=REPO_ROOT / "dev" / "logs" / "training-tier1-2h",
        log_file=REPO_ROOT / "dev" / "logs" / "tier1-sweep-2h.log",
        out_dir=IMPL_DIR,
        title="Tier 1 — Augmentation + Random Op Search",
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


def parse_scores(log_path: Path):
    text = log_path.read_text()
    scores = {}
    pat = re.compile(
        r"\s*(\w+):\s*raw=([\d.]+)%\s+benchmark=\s*([\d.]+)%\s+adj=\s*([+-]?[\d.]+)"
    )
    for m in pat.finditer(text):
        codename = m.group(1)
        scores[codename] = {
            "raw": float(m.group(2)),
            "benchmark": float(m.group(3)),
            "adj": float(m.group(4)),
        }
    return scores


def load_csv(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_score_bars(scores, out_path, title_prefix):
    items = sorted(scores.items(), key=lambda kv: -kv[1]["adj"])
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
    codenames_in_order = sorted(scores.keys(), key=lambda c: -scores[c]["adj"])
    n = len(codenames_in_order)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.0 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, codename in zip(axes_flat, codenames_in_order):
        csv_path = csv_dir / f"{codename}.csv"
        if not csv_path.exists():
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
    items = sorted(fbd.items(), key=lambda kv: -kv[1]["adj"]["mean"])
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=sorted(TIER_CONFIG.keys()), default="tier1")
    p.add_argument("--ablation", action="store_true",
                   help="Also produce the Tier 1 ablation comparison chart.")
    p.add_argument("--multiseed", action="store_true",
                   help="Produce the multi-seed variance + aug-effect figure "
                        "(reads dev/logs/multiseed/_aggregate.json) and exit.")
    args = p.parse_args()

    if args.multiseed:
        print("=== Multi-seed figure ===")
        IMPL_DIR.mkdir(parents=True, exist_ok=True)
        plot_multiseed(IMPL_DIR / "multiseed.png")
        return

    cfg = TIER_CONFIG[args.tier]
    csv_dir = cfg["csv_dir"]
    log_file = cfg["log_file"]
    out_dir = cfg["out_dir"]
    title_prefix = cfg["title"]

    print(f"=== {title_prefix} ===")
    print(f"Reading scores from {log_file.relative_to(REPO_ROOT)}")
    scores = parse_scores(log_file)
    print(f"  parsed {len(scores)} datasets")

    print(f"Reading CSVs from {csv_dir.relative_to(REPO_ROOT)}")
    for path in sorted(csv_dir.glob("*.csv")):
        print(f"  {path.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_score_bars(scores, out_dir / f"{args.tier}_scores.png", title_prefix)
    plot_learning_curves(scores, csv_dir, out_dir / f"{args.tier}_curves.png", title_prefix)

    if args.ablation:
        print("\n=== Tier 1 ablation chart ===")
        plot_ablation(out_dir / "tier1_ablation.png")


if __name__ == "__main__":
    main()
