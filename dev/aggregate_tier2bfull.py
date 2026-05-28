"""Tier-2b FULL EGNAS — 3-seed × 11-dataset paired analysis vs Tier-1.

Compares `dev/logs/tier2bfull/t2bfull_s{42,1337,2024}.log` against the paired
Tier-1 baseline `dev/logs/multiseed/full_s{42,1337,2024}.log` — same 11
datasets, same seeds, same 2 h budget (multiseed `full` arm IS the Tier-1
recipe). Internal adj (Sokoto/Sudoku overridden via
`dev/internal_benchmarks.py`). Also parses Q8 arch-stability (best D/W +
micro-change count per dataset/seed). Reuses `aggregate_multiseed`'s
parser/stats (DRY).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from internal_benchmarks import adj_score_internal
from aggregate_multiseed import parse_log, mstats, fmt, CODENAME_TO_DATASET

REPO = Path(__file__).resolve().parent.parent
EG = REPO / "dev" / "logs" / "tier2bfull"
MS = REPO / "dev" / "logs" / "multiseed"
OUT = EG / "_tier2bfull_aggregate.json"

SEEDS = [42, 1337, 2024]
# run_tier2b_full.sh ALL_DATASETS order (== "EGNAS full done" line order).
DATASETS = ["AddNIST", "CIFARTile", "Chesseract", "GameOfLife", "GeoClassing",
            "Gutenberg", "Language", "MultNIST", "Myofibre", "Sudoku", "Voxel"]
DS2CODE = {v: k for k, v in CODENAME_TO_DATASET.items()}
CODES = [DS2CODE[d] for d in DATASETS]

_PICK_RE = re.compile(
    r"EGNAS full done: D=(\d+) W=(\d+) micro-changed (\d+)/(\d+)")


def internal_by_code(scores):
    return {c: round(adj_score_internal(c, d["raw"], d["benchmark"]), 4)
            for c, d in scores.items() if c in CODES}


def picks(path):
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            for m in _PICK_RE.finditer(path.read_text())]


def main():
    eg = {c: [] for c in CODES}
    t1 = {c: [] for c in CODES}
    picks_by_ds = {d: [] for d in DATASETS}
    for s in SEEDS:
        ep, mp = EG / f"t2bfull_s{s}.log", MS / f"full_s{s}.log"
        if ep.exists():
            ei = internal_by_code(parse_log(ep))
            for c in CODES:
                if c in ei:
                    eg[c].append(ei[c])
            ps = picks(ep)
            for ds, p in zip(DATASETS, ps):
                picks_by_ds[ds].append((s,) + p)
        if mp.exists():
            bi = internal_by_code(parse_log(mp))
            for c in CODES:
                if c in bi:
                    t1[c].append(bi[c])

    print("# Tier-2b FULL EGNAS — 3-seed × 11-dataset paired verdict "
          "(internal adj)\n")
    print("| dataset | Tier-1 | EGNAS-full | Δmean | σ T1→EG | EGNAS per-seed |")
    print("|---|---:|---:|---:|---:|---|")
    rows = {}
    for c in CODES:
        if not eg[c] or not t1[c]:
            continue
        ds = CODENAME_TO_DATASET[c]
        b, e = mstats(t1[c]), mstats(eg[c])
        rows[ds] = {"tier1": b, "egnas": e,
                    "d_mean": round(e["mean"] - b["mean"], 4)}
        print(f"| {ds} | {fmt(b)} | **{fmt(e)}** | "
              f"**{rows[ds]['d_mean']:+.3f}** | "
              f"{b['std']:.2f}→{e['std']:.2f} | "
              f"{', '.join(f'{v:+.2f}' for v in e['vals'])} |")

    bt = ([sum(t1[c][i] for c in CODES) for i in range(3)]
          if all(len(t1[c]) == 3 for c in CODES) else None)
    et = ([sum(eg[c][i] for c in CODES) for i in range(3)]
          if all(len(eg[c]) == 3 for c in CODES) else None)
    B = E = None
    if bt and et:
        B, E = mstats(bt), mstats(et)
        ratio = (E['std'] / B['std']) if B['std'] > 1e-9 else float('nan')
        print(f"\n**Total (11 ds, internal):** Tier-1 **{fmt(B)}** vs "
              f"EGNAS-full **{fmt(E)}**  →  Δmean **{E['mean']-B['mean']:+.2f}**,"
              f" σ collapse **{B['std']:.2f}→{E['std']:.2f} (×{ratio:.2f})**")
        print(f"  - per-seed totals: Tier-1 {B['vals']}; EGNAS {E['vals']}")

    print("\n## Architecture-selection stability (Q8) — EGNAS pick per seed\n")
    print("| dataset | s42 | s1337 | s2024 | micro-changed |")
    print("|---|---|---|---|---|")
    for ds in DATASETS:
        d = {s: p for (s, D, W, nch, t) in picks_by_ds[ds] for p in [(D, W, nch, t)]}
        cells = []
        chgs = []
        for s in SEEDS:
            if s in d:
                D, W, nch, t = d[s]
                cells.append(f"D{D}/W{W}")
                chgs.append(f"{nch}/{t}")
            else:
                cells.append("—"); chgs.append("—")
        print(f"| {ds} | {cells[0]} | {cells[1]} | {cells[2]} | "
              f"{', '.join(chgs)} |")

    print("\n## Verdict\n")
    if B and E:
        print(f"- **Total:** EGNAS-full {fmt(E)} vs Tier-1 {fmt(B)}  →  "
              f"Δmean **{E['mean']-B['mean']:+.2f}**, σ ×"
              f"{(E['std']/B['std']):.2f}")
    big = [ds for ds, r in rows.items() if r["d_mean"] > 1.0]
    reg = [ds for ds, r in rows.items() if r["d_mean"] < -0.25]
    sig_up = [ds for ds, r in rows.items()
              if r["egnas"]["std"] > r["tier1"]["std"] + 1e-6]
    print(f"- Big wins (Δmean > +1.0): {big if big else 'none'}")
    print(f"- Regressions (Δmean < −0.25): {reg if reg else 'NONE ✓'}")
    print(f"- σ INCREASED under EGNAS-full: {sig_up if sig_up else 'NONE ✓'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seeds": SEEDS, "datasets": DATASETS,
        "egnas_full": eg, "tier1": t1,
        "picks_by_dataset": picks_by_ds,
        "rows": {k: {"tier1": r["tier1"], "egnas": r["egnas"],
                     "d_mean": r["d_mean"]} for k, r in rows.items()},
        "totals": ({"tier1": B, "egnas": E} if (B and E) else None),
    }, indent=2, default=str))
    print(f"\n[wrote {OUT.relative_to(REPO)}]")


if __name__ == "__main__":
    main()
