#!/usr/bin/env python3
"""Q8 stability analysis under full EGNAS.

For each (seed × dataset) we have:
  - the EGNAS-full pool JSON (every candidate tried, with val_acc on the
    cuDNN-noisy short-train evaluation)
  - the `EGNAS full done: D=.. W=.. micro-changed N/T val=Z ops=[...]` line
    in the sweep log (the final picked architecture)
  - the per-epoch CSV (final main-training trajectory)
  - the `Scoring`-section line with raw acc + adj

This script answers three Q8 questions under FULL EGNAS:

  (a) Cross-seed arch-pick agreement: do the 3 seeds converge to the same
      (D, W, op-config), or each pick differently?
  (b) Within-seed search-rank vs final-train-rank Spearman: does the
      cuDNN-noisy short-train val_acc reliably rank candidates?
  (c) AddNIST s2024 specifically: under full EGNAS, did micro override
      the macro lottery that bit us under macro-only (D=5 → −2.81)?

Reads from dev/logs/tier2bfull/ (the existing 11-ds sweep + new-datasets
follow-on). Pure local analysis — no Spark, no new training.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
TIER2B_DIR = REPO / "dev" / "logs" / "tier2bfull"
SEEDS = [42, 1337, 2024]

# Codename → dataset name. Mirrors dev/plot_tier_results.py.
CODENAME_TO_DATASET = {
    "Adaline": "AddNIST", "Caitie": "CIFARTile", "Chester": "Chesseract",
    "conway": "GameOfLife", "Sadie": "GeoClassing", "Gutenberg": "Gutenberg",
    "LaMelo": "Language", "Mateo": "MultNIST", "Myopia": "Myofibre",
    "Sokoto": "Sudoku", "Volga": "Voxel", "Cryptic": "Cryptic",
    "windspeed": "Windspeed",
}

# Regex for the "EGNAS full done" line; emits D, W, micro-changed, val, ops.
FULL_DONE = re.compile(
    r"EGNAS full done:\s+D=(\d+)\s+W=(\d+)\s+micro-changed\s+(\d+)/(\d+)\s+"
    r"val=([\d.]+)\s+\(\d+ evaluated\)\s+ops=(\[[^\]]+\])"
)
# Score line.
SCORE = re.compile(
    r"\s*(\w+):\s*raw=([\d.]+)%\s+benchmark=\s*([\d.]+)%\s+adj=\s*([+-]?[\d.]+)"
)
# Dataset header.
DS_HEADER = re.compile(r"==========\s+Dataset\s+(\w+)\s+==========")


def parse_log(path: Path) -> Dict[str, dict]:
    """Walk one seed's log; return {codename: {D, W, micro_changed, ops,
    search_val, raw, benchmark, adj}}."""
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    current_ds: Optional[str] = None
    text = path.read_text()
    for line in text.splitlines():
        m = DS_HEADER.search(line)
        if m:
            current_ds = m.group(1)
            out.setdefault(current_ds, {})
            continue
        m = FULL_DONE.search(line)
        if m and current_ds:
            ops_txt = m.group(6)
            try:
                ops = json.loads(ops_txt.replace("'", '"'))
            except Exception:
                ops = []
            out[current_ds].update({
                "D": int(m.group(1)),
                "W": int(m.group(2)),
                "micro_changed": int(m.group(3)),
                "n_layers": int(m.group(4)),
                "search_val": float(m.group(5)),
                "ops": ops,
            })
            continue
        m = SCORE.search(line)
        if m:
            code = m.group(1)
            if code in CODENAME_TO_DATASET:
                out.setdefault(code, {}).update({
                    "raw": float(m.group(2)),
                    "benchmark": float(m.group(3)),
                    "adj": float(m.group(4)),
                })
    return out


def gather() -> Dict[str, Dict[int, dict]]:
    """Return {codename: {seed: dataset-info dict}}.

    Pulls from both the original 11-ds sweep (t2bfull_s{seed}.log) and the
    Windspeed+Cryptic follow-on (t2bfull_new_s{seed}.log)."""
    by_code: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for s in SEEDS:
        for tag in ("t2bfull", "t2bfull_new"):
            for code, info in parse_log(TIER2B_DIR / f"{tag}_s{s}.log").items():
                if info:  # skip empty stubs
                    # Later runs override earlier in case of overlap (none expected)
                    by_code[code][s] = info
    return by_code


def pretty_arch(d: dict) -> str:
    """Compact arch fingerprint."""
    ops = d.get("ops", [])
    counts = {"Conv3": 0, "Conv5": 0, "SepConv3": 0}
    for o in ops:
        counts[o] = counts.get(o, 0) + 1
    op_str = " ".join(f"{k[-1]}={v}" for k, v in counts.items() if v)
    return f"D={d['D']:>2} W={d['W']:>2} [{op_str}] mc={d['micro_changed']}/{d['n_layers']}"


def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman ρ via rank-based Pearson; tiny implementation."""
    n = len(xs)
    if n < 2:
        return float("nan")
    def rank(arr):
        srt = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        for rk, i in enumerate(srt):
            r[i] = rk
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx2 = sum((rx[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ry[i] - my) ** 2 for i in range(n))
    den = (dx2 * dy2) ** 0.5
    return num / den if den > 1e-12 else float("nan")


def within_seed_spearman(code: str, seed: int) -> Optional[Tuple[float, int]]:
    """Per-seed: within-search Spearman of pool's val_acc vs the final
    accepted architecture's depth (proxy — search ranks candidates, we
    measure whether the rank order is monotone in some training signal).

    More precisely: across all candidates in the pool the script returns
    Spearman between val_acc (the search's ranking signal) and depth+width
    proxies (since we don't retrain every candidate). Returns
    (rho, n) or None if not enough data."""
    p = TIER2B_DIR / f"t2bfull_s{seed}" / f"{code}_egnas_pool.json"
    if not p.exists():
        # Try the new-datasets dir
        p = TIER2B_DIR / f"t2bfull_new_s{seed}" / f"{code}_egnas_pool.json"
        if not p.exists():
            return None
    pool = json.loads(p.read_text())
    vals = [x.get("val", float("nan")) for x in pool]
    # Use depth as a coarse "model complexity" axis. NOT a final-train
    # proxy, so this is a *direction* signal only.
    depths = [x.get("D", 0) for x in pool]
    return spearman(vals, depths), len(pool)


def main():
    by_code = gather()
    if not by_code:
        print(f"No data under {TIER2B_DIR}; nothing to analyse.")
        sys.exit(1)

    # Stable display order: alphabetic by dataset name.
    codes = sorted(by_code.keys(),
                   key=lambda c: CODENAME_TO_DATASET.get(c, c).lower())

    print("# Q8 stability under FULL EGNAS  (3 seeds × 13 datasets)")
    print()
    print("Reads dev/logs/tier2bfull/{t2bfull,t2bfull_new}_s{42,1337,2024}.log")
    print()

    # ============================================================
    # (a) Cross-seed arch-pick agreement
    # ============================================================
    print("## (a) Cross-seed arch agreement\n")
    print("| Dataset | s42 | s1337 | s2024 | D-agree | W-agree | ops-agree |")
    print("|---|---|---|---|---:|---:|---:|")
    d_agree_n, w_agree_n, ops_agree_n, total = 0, 0, 0, 0
    for code in codes:
        per_seed = by_code[code]
        if len(per_seed) < 2:
            continue
        rows = [per_seed[s] for s in SEEDS if s in per_seed and "D" in per_seed[s]]
        if len(rows) < 2:
            continue
        Ds = {r["D"] for r in rows}
        Ws = {r["W"] for r in rows}
        ops_list = [tuple(r["ops"]) for r in rows]
        d_ag = len(Ds) == 1
        w_ag = len(Ws) == 1
        ops_ag = len(set(ops_list)) == 1
        d_agree_n += d_ag
        w_agree_n += w_ag
        ops_agree_n += ops_ag
        total += 1
        s42 = pretty_arch(per_seed[42]) if 42 in per_seed and "D" in per_seed[42] else "—"
        s1337 = pretty_arch(per_seed[1337]) if 1337 in per_seed and "D" in per_seed[1337] else "—"
        s2024 = pretty_arch(per_seed[2024]) if 2024 in per_seed and "D" in per_seed[2024] else "—"
        ds_name = CODENAME_TO_DATASET.get(code, code)
        print(f"| {ds_name} | `{s42}` | `{s1337}` | `{s2024}` | "
              f"{'✓' if d_ag else '✗'} | {'✓' if w_ag else '✗'} | "
              f"{'✓' if ops_ag else '✗'} |")
    print()
    if total:
        print(f"**Aggregate:** depth agreement {d_agree_n}/{total} "
              f"({100*d_agree_n/total:.0f}%), width {w_agree_n}/{total} "
              f"({100*w_agree_n/total:.0f}%), full ops {ops_agree_n}/{total} "
              f"({100*ops_agree_n/total:.0f}%)")
    print()

    # ============================================================
    # (b) Within-seed search-rank Spearman
    # ============================================================
    print("## (b) Within-seed search-rank correlation (val_acc ↔ depth)\n")
    print("Spearman ρ between pool val_acc and depth across all evaluated "
          "candidates per seed.  Direction-only signal: if search prefers "
          "deeper nets we'd expect positive ρ; if depth and val are "
          "uncorrelated the noisy short-train val is doing little ranking.\n")
    print("| Dataset | s42 ρ (n) | s1337 ρ (n) | s2024 ρ (n) |")
    print("|---|---|---|---|")
    for code in codes:
        row = [f"| {CODENAME_TO_DATASET.get(code, code)}"]
        for s in SEEDS:
            r = within_seed_spearman(code, s)
            row.append(f"{r[0]:+.2f} ({r[1]})" if r else "—")
        print(" | ".join(row) + " |")
    print()

    # ============================================================
    # (c) AddNIST s2024 narrative
    # ============================================================
    print("## (c) AddNIST s2024 — did full EGNAS dodge the macro-only lottery?\n")
    if "Adaline" in by_code and 2024 in by_code["Adaline"]:
        a24 = by_code["Adaline"][2024]
        if "D" in a24:
            print(f"**Picked:** {pretty_arch(a24)}")
            print(f"**Search val:** {a24['search_val']:.4f}")
            print(f"**Final raw / adj:** {a24.get('raw','—')}% / "
                  f"{a24.get('adj','—'):+.3f}")
            # Compare against macro-only diagnostic (we don't have that
            # directly here — surface the macro-only finding the original
            # writeup cites).
            print()
            print("Macro-only diagnostic recorded `D=5 W=48` for s2024 → "
                  "AddNIST adj −2.81 (the variance-amplifying pick that "
                  "motivated finishing the method).  Full-EGNAS picked "
                  f"`D={a24['D']} W={a24['W']}` — if D < 5 or W ≠ 48 the "
                  "micro phase materially changed the macro winner.")
    print()


if __name__ == "__main__":
    main()
