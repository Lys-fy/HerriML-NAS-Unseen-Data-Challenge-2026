"""Tier-2b EGNAS-macro diagnostic — paired vs Tier-1, + arch-selection stability.

Compares the EGNAS-macro diagnostic (dev/logs/egnas_macro/egnas_s*.log) against
the paired Tier-1 baseline (multi-seed dev/logs/multiseed/full_s*.log — same 3
datasets × same seeds × same 2 h budget). Reuses aggregate_multiseed's parser
+ stats (DRY). Internal adj (== shipped for these 3; no Sokoto).

The make-or-break question: does multi-seed macro search beat Tier-1's
CIFARTile −0.515 ± 0.521 ceiling? Also: is the macro search seed-stable in
the architecture it picks (Q8), or a lottery?
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from internal_benchmarks import adj_score_internal
from aggregate_multiseed import parse_log, mstats, fmt

REPO_ROOT = Path(__file__).resolve().parent.parent
EG_DIR = REPO_ROOT / "dev" / "logs" / "egnas_macro"
MS_DIR = REPO_ROOT / "dev" / "logs" / "multiseed"
OUT_JSON = EG_DIR / "_egnas_aggregate.json"

SEEDS = [42, 1337, 2024]
# run_egnas_macro.sh DATASETS order (= "EGNAS macro done" line order).
DATASETS = ["CIFARTile", "AddNIST", "GeoClassing"]
CODES = ["Caitie", "Adaline", "Sadie"]          # same dataset order
CODE2DS = dict(zip(CODES, DATASETS))
TIER1 = {  # paired Tier-1 baseline, internal adj mean±std (from Multi-Seed)
    "CIFARTile": "−0.515 ± 0.521", "AddNIST": "−0.378 ± 5.446",
    "GeoClassing": "+7.678 ± 0.294",
}
_PICK_RE = re.compile(r"EGNAS macro done: best D=(\d+) W=(\d+) val=([\d.]+)")


def internal_by_code(scores):
    return {c: round(adj_score_internal(c, d["raw"], d["benchmark"]), 4)
            for c, d in scores.items() if c in CODES}


def picks(path: Path):
    """[(D,W,val)] in DATASETS order (one EGNAS-done line per dataset)."""
    return [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
            for m in _PICK_RE.finditer(path.read_text())]


def main():
    eg = {c: [] for c in CODES}
    t1 = {c: [] for c in CODES}
    picks_by_ds = {d: [] for d in DATASETS}  # dataset -> [(seed,D,W,val)]
    for s in SEEDS:
        ep, mp = EG_DIR / f"egnas_s{s}.log", MS_DIR / f"full_s{s}.log"
        if ep.exists():
            ei = internal_by_code(parse_log(ep))
            for c in CODES:
                if c in ei:
                    eg[c].append(ei[c])
            for ds, (D, W, v) in zip(DATASETS, picks(ep)):
                picks_by_ds[ds].append((s, D, W, round(v, 4)))
        if mp.exists():
            bi = internal_by_code(parse_log(mp))
            for c in CODES:
                if c in bi:
                    t1[c].append(bi[c])

    print("# Tier-2b EGNAS-macro — 3-seed paired verdict (internal adj)\n")
    print("| dataset | Tier-1 | EGNAS | Δmean | σ T1→EGNAS | EGNAS per-seed |")
    print("|---|---:|---:|---:|---:|---|")
    rows = {}
    for c in CODES:
        ds = CODE2DS[c]
        if not eg[c] or not t1[c]:
            continue
        b, e = mstats(t1[c]), mstats(eg[c])
        rows[ds] = {"tier1": b, "egnas": e,
                    "d_mean": round(e["mean"] - b["mean"], 4)}
        print(f"| {ds} | {fmt(b)} | **{fmt(e)}** | **{rows[ds]['d_mean']:+.3f}** "
              f"| {b['std']:.2f}→{e['std']:.2f} | "
              f"{', '.join(f'{v:+.2f}' for v in e['vals'])} |")

    bt = [sum(t1[c][i] for c in CODES) for i in range(3)] \
        if all(len(t1[c]) == 3 for c in CODES) else None
    et = [sum(eg[c][i] for c in CODES) for i in range(3)] \
        if all(len(eg[c]) == 3 for c in CODES) else None
    if bt and et:
        B, E = mstats(bt), mstats(et)
        print(f"\nTotal (3 ds): Tier-1 **{fmt(B)}** vs EGNAS **{fmt(E)}**  "
              f"(Δmean {E['mean']-B['mean']:+.2f}; σ {B['std']:.2f}→{E['std']:.2f})")

    print("\n## Architecture-selection stability (Q8) — EGNAS pick per seed\n")
    print("| dataset | s42 | s1337 | s2024 | stable? |")
    print("|---|---|---|---|---|")
    for ds in DATASETS:
        ps = {s: (D, W) for s, D, W, _ in picks_by_ds[ds]}
        cells = [f"D{ps[s][0]}/W{ps[s][1]}" if s in ps else "—" for s in SEEDS]
        Ds = {ps[s][0] for s in SEEDS if s in ps}
        Ws = {ps[s][1] for s in SEEDS if s in ps}
        verdict = ("depth stable, width " +
                   ("stable" if len(Ws) == 1 else "VARIES") if len(Ds) <= 2
                   else "DEPTH VARIES")
        print(f"| {ds} | {cells[0]} | {cells[1]} | {cells[2]} | {verdict} |")

    # CIFARTile make-or-break.
    print("\n## Verdict\n")
    if "CIFARTile" in rows:
        c = rows["CIFARTile"]
        beat = c["egnas"]["mean"] > 0 and c["egnas"]["mean"] > c["tier1"]["mean"]
        print(f"- **CIFARTile (make-or-break):** Tier-1 {fmt(c['tier1'])} → "
              f"EGNAS {fmt(c['egnas'])}  "
              f"{'✓ beats the ceiling, multi-seed' if beat else '✗ no real gain'}")
    for ds in ("AddNIST", "GeoClassing"):
        if ds in rows:
            r = rows[ds]
            tag = ("⚠ regressed" if r["d_mean"] < -0.25 else "ok")
            print(f"- {ds}: Δ {r['d_mean']:+.2f} ({tag}); "
                  f"σ {r['tier1']['std']:.2f}→{r['egnas']['std']:.2f}")

    EG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"datasets": DATASETS, "egnas": eg, "tier1": t1,
         "picks": {k: v for k, v in picks_by_ds.items()},
         "rows": {k: {kk: vv for kk, vv in r.items()} for k, r in rows.items()}},
        indent=2, default=str))
    print(f"\n[wrote {OUT_JSON.relative_to(REPO_ROOT)}]")


if __name__ == "__main__":
    main()
