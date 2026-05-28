#!/usr/bin/env python3
"""Plot the Language excursion + Language-across-all-methods comparison.

Two figures land in docs/notes/09 - Implementation/implementation/images/:
  lang_excursion.png        — 6 variations × 3 seeds, mean ± σ, seed dots,
                              benchmark/literature reference lines
  lang_across_methods.png   — horizontal bar chart: every Language measurement
                              we have, ordered by method, with literature
                              reference lines.

Sources read:
  dev/logs/lang_excursion/V{0..5}_*_s{42,1337,2024}.log
  dev/logs/multiseed/{full,augoff}_s{42,1337,2024}.log
  dev/logs/tier2bfull/t2bfull_s{42,1337,2024}.log
  dev/logs/tier2a_p1_v4/p1v4_s{42,1337,2024}.log
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "notes" / "09 - Implementation" / "implementation" / "images"
IMG.mkdir(parents=True, exist_ok=True)

SCORE = re.compile(
    r"\s*LaMelo:\s*raw=([\d.]+)%\s+benchmark=\s*([\d.]+)%\s+adj=\s*([+-]?[\d.]+)"
)

# Literature reference points (raw acc), from Geada et al. arXiv:2404.02189.
LIT = {
    "ResNeXt-50 (SOTA)": 93.97,
    "PC-DARTS / DARTS": 90.12,
    "Best competition submission": 89.71,
    "ResNet-18 (paper)": 87.00,
}
BENCH_HARNESS = 85.20  # the ResNet-18 baseline shipped in our metadata


def _scan(path: Path) -> list[tuple[float, float]] | None:
    """Return list of (raw_acc, adj) from a log; one entry per scoring line."""
    if not path.exists():
        return None
    out = []
    for m in SCORE.finditer(path.read_text()):
        out.append((float(m.group(1)), float(m.group(3))))
    return out


def _stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def collect_excursion() -> dict[str, dict]:
    """Return {V_name: {'raw_per_seed': [...], 'adj_per_seed': [...]}}."""
    var_order = ["V0_baseline", "V1_patience60", "V2_lr_low_warmup",
                 "V3_mixup", "V4_larger_model", "V5_combo"]
    base = REPO / "dev" / "logs" / "lang_excursion"
    out: dict[str, dict] = {}
    for v in var_order:
        raws, adjs = [], []
        for s in (42, 1337, 2024):
            xs = _scan(base / f"{v}_s{s}.log")
            if xs:
                raws.append(xs[-1][0])
                adjs.append(xs[-1][1])
        out[v] = {"raws": raws, "adjs": adjs}
    return out


def collect_other_methods() -> dict[str, dict]:
    """Return {method_label: {'raws': [...], 'adjs': [...]}} for non-excursion methods."""
    out: dict[str, dict] = {}
    for label, base, prefix in [
        ("Multi-seed Tier-1 (aug-on)", "multiseed", "full"),
        ("Multi-seed Tier-1 (aug-off)", "multiseed", "augoff"),
        ("Tier 2b — EGNAS-full", "tier2bfull", "t2bfull"),
        ("P1 v4 (probe-driven)", "tier2a_p1_v4", "p1v4"),
    ]:
        raws, adjs = [], []
        for s in (42, 1337, 2024):
            xs = _scan(REPO / "dev" / "logs" / base / f"{prefix}_s{s}.log")
            if xs:
                raws.append(xs[-1][0])
                adjs.append(xs[-1][1])
        out[label] = {"raws": raws, "adjs": adjs}
    return out


def plot_excursion(out_path: Path):
    data = collect_excursion()
    var_labels = {
        "V0_baseline": "V0\nbaseline\n(aug-off,\nTier-1 default)",
        "V1_patience60": "V1\npatience=60",
        "V2_lr_low_warmup": "V2\nlr=0.01\nwarmup=10",
        "V3_mixup": "V3\nmixup α=0.2",
        "V4_larger_model": "V4\nlarger model\n(W=72)",
        "V5_combo": "V5\ncombo\n(V1+V2+V3)",
    }
    names = list(data.keys())
    x = np.arange(len(names))
    raws = [data[n]["raws"] for n in names]
    means = [np.mean(r) if r else np.nan for r in raws]
    stds = [np.std(r, ddof=1) if len(r) > 1 else 0.0 for r in raws]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    # Color V3 / V5 differently to highlight winners.
    colors = ["#888"] * 6
    colors[3] = "#2c8a3a"   # V3
    colors[5] = "#d97706"   # V5

    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw=dict(ecolor="#222", lw=1.2),
                  label="variation mean ± σ across 3 seeds")
    # Seed dots
    for i, r in enumerate(raws):
        ax.scatter([i] * len(r), r, s=42, color="black",
                   alpha=0.7, zorder=5, edgecolor="white", linewidth=0.6)
    # Per-bar mean + Δ labels
    for i, n in enumerate(names):
        if means[i] and not np.isnan(means[i]):
            ax.text(i, means[i] + stds[i] + 0.55, f"{means[i]:.2f}%",
                    ha="center", fontsize=9.5, fontweight="bold")

    # Reference horizontal lines
    ax.axhline(BENCH_HARNESS, color="#b53030", linestyle="--", lw=1.4,
               label=f"Harness benchmark (= ResNet-18 stripped): {BENCH_HARNESS}%")
    ax.axhline(LIT["ResNet-18 (paper)"], color="#9c4dcc", linestyle=":", lw=1.0,
               label=f"ResNet-18 (paper): {LIT['ResNet-18 (paper)']}%")
    ax.axhline(LIT["Best competition submission"], color="#1f77b4",
               linestyle="-.", lw=1.0,
               label=f"Best competition submission: {LIT['Best competition submission']}%")
    ax.axhline(LIT["PC-DARTS / DARTS"], color="#0d80a8", linestyle=":", lw=1.0,
               label=f"PC-DARTS / DARTS: {LIT['PC-DARTS / DARTS']}%")
    ax.axhline(LIT["ResNeXt-50 (SOTA)"], color="#0a5a31", linestyle="-", lw=1.4,
               label=f"ResNeXt-50 public SOTA: {LIT['ResNeXt-50 (SOTA)']}%")

    ax.set_xticks(x)
    ax.set_xticklabels([var_labels[n] for n in names], fontsize=9)
    ax.set_ylabel("Language raw accuracy (%)", fontsize=11)
    ax.set_title("Language Excursion — 6 variations × 3 seeds (fixed Conv3 stack, "
                 "aug-off, no EGNAS)\n"
                 "All training-side tweaks plateau ~3-4 pp below harness benchmark; "
                 "Mixup is the only meaningful lever.",
                 fontsize=11)
    ax.set_ylim(74, 95)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO)}")


def plot_across_methods(out_path: Path):
    """Horizontal bar chart — every Language measurement we have, by method."""
    excursion = collect_excursion()
    others = collect_other_methods()

    # Order chronologically / by similarity
    rows = []
    rows.append(("Multi-seed Tier-1 (aug-on)", others["Multi-seed Tier-1 (aug-on)"]["raws"]))
    rows.append(("Tier 2b — EGNAS-full",          others["Tier 2b — EGNAS-full"]["raws"]))
    rows.append(("P1 v4 (probe-driven)",          others["P1 v4 (probe-driven)"]["raws"]))
    rows.append(("Multi-seed Tier-1 (aug-off)",   others["Multi-seed Tier-1 (aug-off)"]["raws"]))
    rows.append(("Excursion V0 (aug-off, fixed Conv3)", excursion["V0_baseline"]["raws"]))
    rows.append(("Excursion V1 (patience=60)",    excursion["V1_patience60"]["raws"]))
    rows.append(("Excursion V2 (lr=0.01, warmup=10)", excursion["V2_lr_low_warmup"]["raws"]))
    rows.append(("Excursion V4 (larger model W=72)",  excursion["V4_larger_model"]["raws"]))
    rows.append(("Excursion V3 (Mixup α=0.2)",        excursion["V3_mixup"]["raws"]))
    rows.append(("Excursion V5 (combo V1+V2+V3)",     excursion["V5_combo"]["raws"]))

    labels = [r[0] for r in rows]
    raws = [r[1] for r in rows]
    means = [np.mean(r) if r else np.nan for r in raws]
    stds = [np.std(r, ddof=1) if len(r) > 1 else 0.0 for r in raws]

    n = len(labels)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(13, max(7, 0.55 * n + 2)))

    # Color by "ours_best" vs ours-others vs literature
    # ours_best = V3 + V5 (the best training-side); slight emphasis
    colors = []
    for lab in labels:
        if "Mixup" in lab or "combo" in lab:
            colors.append("#2c8a3a")
        elif "EGNAS-full" in lab or "P1 v4" in lab:
            colors.append("#1f77b4")
        else:
            colors.append("#888")

    ax.barh(y, means, xerr=stds, capsize=5, height=0.65,
            color=colors, edgecolor="black", linewidth=0.6,
            error_kw=dict(ecolor="#222", lw=1.2))
    # Seed dots
    for yi, r in zip(y, raws):
        ax.scatter(r, [yi] * len(r), s=40, color="black",
                   alpha=0.7, zorder=5, edgecolor="white", linewidth=0.6)
    # Mean labels on the right of each bar
    for yi, m, s in zip(y, means, stds):
        if not np.isnan(m):
            ax.text(m + s + 0.4, yi, f"{m:.2f}% ± {s:.2f}",
                    va="center", fontsize=9)

    # Literature reference lines (vertical)
    bench_line = ax.axvline(BENCH_HARNESS, color="#b53030", linestyle="--",
                            lw=1.4, label=f"Harness benchmark: {BENCH_HARNESS}%")
    pcd_line = ax.axvline(LIT["PC-DARTS / DARTS"], color="#0d80a8",
                          linestyle=":", lw=1.2,
                          label=f"PC-DARTS / DARTS: {LIT['PC-DARTS / DARTS']}%")
    comp_line = ax.axvline(LIT["Best competition submission"],
                           color="#1f77b4", linestyle="-.", lw=1.0,
                           label=f"Best competition submission: {LIT['Best competition submission']}%")
    sota_line = ax.axvline(LIT["ResNeXt-50 (SOTA)"], color="#0a5a31",
                           linestyle="-", lw=1.6,
                           label=f"ResNeXt-50 public SOTA: {LIT['ResNeXt-50 (SOTA)']}%")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Language raw accuracy (%)", fontsize=11)
    ax.set_xlim(65, 96)
    ax.set_title("Language across every method we ran — vs published literature targets.\n"
                 "Training-side tricks plateau ~81-82 %; architectural change (ResNeXt-style "
                 "grouped convs) is the lever for 90+ %.",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    plot_excursion(IMG / "lang_excursion.png")
    plot_across_methods(IMG / "lang_across_methods.png")
