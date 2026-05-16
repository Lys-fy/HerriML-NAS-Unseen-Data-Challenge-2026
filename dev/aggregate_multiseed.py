"""Aggregate the 6-block multi-seed sweep into the variance + aug-effect finding.

Unit of analysis is **mean ± std across the 3 seeds per (config, dataset)** —
never a single number. Runs are non-reproducible even at a fixed seed
(`cudnn.benchmark=True`); see docs/notes Tier 1 - Results single-seed caveat.

Reads   dev/logs/multiseed/{full,augoff}_s{42,1337,2024}.log
Writes  dev/logs/multiseed/_aggregate.json  (machine-readable, for the plot step)
        + clean markdown tables to stdout (paste into Multi-Seed - Results.md).

Scoring: TWO lenses, always reported side by side —
  * shipped  — exactly what the grader computes (Sudoku/Sokoto benchmark=0.0,
               so its adj is just raw/10). Faithful to the eval server.
  * internal — honest tier-to-tier comparison: Sokoto benchmark overridden to
               70.95 via dev/internal_benchmarks.py. Used for the real story.

The full vs aug-off comparison is restricted to the 7 aug-firing datasets
(aug-off is a no-op on the other 4, so it was not re-run there) — the honest
comparison is per-dataset on the shared 7, NOT total-vs-total (different N).
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from internal_benchmarks import adj_score, adj_score_internal

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "dev" / "logs" / "multiseed"
OUT_JSON = LOG_DIR / "_aggregate.json"

SEEDS = [42, 1337, 2024]
CONFIGS = ["full", "augoff"]

# Mirrors dev/plot_tier_results.py:CODENAME_TO_DATASET (harness codenames are
# a fixed per-dataset mapping, stable across runs).
CODENAME_TO_DATASET = {
    "Adaline": "AddNIST", "Caitie": "CIFARTile", "Chester": "Chesseract",
    "conway": "GameOfLife", "Sadie": "GeoClassing", "Gutenberg": "Gutenberg",
    "LaMelo": "Language", "Mateo": "MultNIST", "Myopia": "Myofibre",
    "Sokoto": "Sudoku", "Volga": "Voxel",
}
# The 7 datasets where the metadata-only aug policy fires (run_multiseed.sh
# AUG_DATASETS). aug-off only differs from full on these.
AUG_CODENAMES = ["Adaline", "Mateo", "Caitie", "Sadie", "Gutenberg", "LaMelo", "Myopia"]

# Same regex as plot_tier_results.py:parse_scores.
_SCORE_RE = re.compile(
    r"\s*(\w+):\s*raw=([\d.]+)%\s+benchmark=\s*([\d.]+)%\s+adj=\s*([+-]?[\d.]+)"
)


def parse_log(path: Path):
    """codename -> {raw, benchmark(shipped), adj_logged}."""
    out = {}
    for m in _SCORE_RE.finditer(path.read_text()):
        out[m.group(1)] = {
            "raw": float(m.group(2)),
            "benchmark": float(m.group(3)),
            "adj_logged": float(m.group(4)),
        }
    return out


def mstats(xs):
    """mean / sample-std (ddof=1, n=3) / min / max / spread, + raw values."""
    xs = [round(x, 4) for x in xs]
    return {
        "mean": round(statistics.fmean(xs), 4),
        "std": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "min": min(xs), "max": max(xs),
        "spread": round(max(xs) - min(xs), 4),
        "n": len(xs), "vals": xs,
    }


def fmt(s):
    return f"{s['mean']:+.3f} ± {s['std']:.3f}"


def main():
    # records[config][seed][codename] = {raw, benchmark, adj_shipped, adj_internal}
    records = {c: {} for c in CONFIGS}
    missing = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            p = LOG_DIR / f"{cfg}_s{seed}.log"
            if not p.exists():
                missing.append(p.name)
                continue
            parsed = parse_log(p)
            seed_rec = {}
            for code, d in parsed.items():
                raw, bench = d["raw"], d["benchmark"]
                shipped = adj_score(raw, bench)
                internal = adj_score_internal(code, raw, bench)
                # Parse sanity: recomputed shipped must match the logged adj.
                if abs(shipped - d["adj_logged"]) > 0.02:
                    print(f"  ! parse mismatch {cfg}_s{seed} {code}: "
                          f"logged={d['adj_logged']:+.3f} recomputed={shipped:+.3f}")
                seed_rec[code] = {
                    "raw": raw, "benchmark": bench,
                    "adj_shipped": round(shipped, 4),
                    "adj_internal": round(internal, 4),
                }
            records[cfg][seed] = seed_rec

    if missing:
        print(f"WARNING — missing logs (sweep incomplete?): {missing}\n")

    # ---- A) per-config seed TOTALS — the variance headline -----------------
    totals = {}  # config -> {shipped: mstats, internal: mstats, full7_internal: mstats}
    for cfg in CONFIGS:
        sh, it, it7 = [], [], []
        for seed in SEEDS:
            sr = records[cfg].get(seed, {})
            if not sr:
                continue
            sh.append(sum(v["adj_shipped"] for v in sr.values()))
            it.append(sum(v["adj_internal"] for v in sr.values()))
            it7.append(sum(v["adj_internal"] for c, v in sr.items()
                           if c in AUG_CODENAMES))
        totals[cfg] = {
            "shipped": mstats(sh), "internal": mstats(it),
            "internal_aug7": mstats(it7),
            "n_datasets": len(records[cfg].get(SEEDS[0], {})),
        }

    # ---- B) full per-dataset mean±std (internal adj) ----------------------
    full_by_ds = {}
    for code in CODENAME_TO_DATASET:
        vals = [records["full"][s][code]["adj_internal"]
                for s in SEEDS if code in records["full"].get(s, {})]
        raws = [records["full"][s][code]["raw"]
                for s in SEEDS if code in records["full"].get(s, {})]
        if vals:
            full_by_ds[code] = {"adj": mstats(vals), "raw": mstats(raws)}

    # ---- C) aug effect: full vs aug-off on the shared 7 -------------------
    aug_effect = {}
    for code in AUG_CODENAMES:
        f = [records["full"][s][code]["adj_internal"]
             for s in SEEDS if code in records["full"].get(s, {})]
        o = [records["augoff"][s][code]["adj_internal"]
             for s in SEEDS if code in records["augoff"].get(s, {})]
        if f and o:
            delta = [fi - oi for fi, oi in zip(f, o)]
            aug_effect[code] = {
                "full": mstats(f), "augoff": mstats(o), "delta": mstats(delta),
            }

    # ===================== emit markdown ==================================
    L = []
    L.append("## A. Seed variance — per-config totals (THE headline)\n")
    L.append("| config | N ds | internal total (mean ± std) | min | max | spread | seeds |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for cfg in CONFIGS:
        t = totals[cfg]["internal"]
        L.append(f"| {cfg} | {totals[cfg]['n_datasets']} | **{fmt(t)}** | "
                 f"{t['min']:+.2f} | {t['max']:+.2f} | {t['spread']:.2f} | "
                 f"{', '.join(f'{v:+.2f}' for v in t['vals'])} |")
    L.append("\n*shipped-scoring totals (what the grader sees, Sokoto bench=0):*  ")
    for cfg in CONFIGS:
        s = totals[cfg]["shipped"]
        L.append(f"`{cfg}` = {fmt(s)}  (seeds {', '.join(f'{v:+.2f}' for v in s['vals'])})  ")

    L.append("\n## B. `full` per-dataset — internal adj, mean ± std over 3 seeds\n")
    L.append("| dataset | adj mean ± std | spread | raw% mean ± std | seeds (adj) |")
    L.append("|---|---:|---:|---:|---|")
    for code, d in sorted(full_by_ds.items(), key=lambda kv: -kv[1]["adj"]["mean"]):
        a, r = d["adj"], d["raw"]
        L.append(f"| {CODENAME_TO_DATASET[code]} | **{fmt(a)}** | {a['spread']:.2f} | "
                 f"{r['mean']:.2f} ± {r['std']:.2f} | "
                 f"{', '.join(f'{v:+.2f}' for v in a['vals'])} |")

    L.append("\n## C. Augmentation effect — `full` − `augoff`, shared 7 datasets, internal adj\n")
    L.append("| dataset | full | aug-off | Δ (full−augoff) mean ± std |")
    L.append("|---|---:|---:|---:|")
    for code in AUG_CODENAMES:
        if code not in aug_effect:
            continue
        e = aug_effect[code]
        L.append(f"| {CODENAME_TO_DATASET[code]} | {fmt(e['full'])} | "
                 f"{fmt(e['augoff'])} | **{fmt(e['delta'])}** |")
    f7 = totals["full"]["internal_aug7"]
    o7 = totals["augoff"]["internal"]  # augoff IS the 7
    L.append(f"| **TOTAL (7 ds)** | **{fmt(f7)}** | **{fmt(o7)}** | "
             f"**{f7['mean'] - o7['mean']:+.3f}** |")

    md = "\n".join(L)
    print(md)

    OUT_JSON.write_text(json.dumps({
        "seeds": SEEDS, "configs": CONFIGS,
        "codename_to_dataset": CODENAME_TO_DATASET,
        "aug_codenames": AUG_CODENAMES,
        "totals": totals, "full_by_dataset": full_by_ds,
        "aug_effect": aug_effect,
        "records": records,
    }, indent=2))
    print(f"\n[wrote {OUT_JSON.relative_to(REPO_ROOT)}]")


if __name__ == "__main__":
    main()
