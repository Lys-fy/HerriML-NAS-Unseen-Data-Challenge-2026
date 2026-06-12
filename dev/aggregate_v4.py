"""Tier-2a / P1 v4 — paired analysis: Tier-2b-full vs +P1 (aug-safety probe).

Same 11 + 2 datasets, same 3 seeds, same 2 h budget. The only delta in the
pipeline is the probe's aug on/off decision (set by `probe_aug_safety()`
in `baselines/v2/helpers.py`). Internal adj uses
`dev/internal_benchmarks.py` (Sokoto=70.95 override etc.).

Outputs:
  - Per-dataset Δmean & σ change vs EGNAS-full baseline.
  - Probe-decision table (which datasets had aug turned off, by seed).
  - Total adj (with σ collapse).
  - JSON dump for the plotter.

Reads:
  - dev/logs/v2/v2_s{42,1337,2024}.log  (P1 sweep)
  - dev/logs/egnas_full/{t2bfull,t2bfull_new}_s{...}.log  (paired baseline)
  - dev/logs/v2/v2_s{...}/<codename>_p1_probe.json (decisions)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from internal_benchmarks import adj_score_internal
from aggregate_multiseed import parse_log, mstats, fmt, CODENAME_TO_DATASET

REPO = Path(__file__).resolve().parent.parent
P1 = REPO / "dev" / "logs" / "v4"
EG = REPO / "dev" / "logs" / "egnas_full"
OUT = P1 / "_v4_aggregate.json"

SEEDS = [42, 1337, 2024]
DATASETS_11 = ["AddNIST", "CIFARTile", "Chesseract", "GameOfLife", "GeoClassing",
               "Gutenberg", "Language", "MultNIST", "Myofibre", "Sudoku", "Voxel"]
DATASETS_NEW = ["Cryptic", "Windspeed"]
DATASETS = DATASETS_11 + DATASETS_NEW

DS2CODE = {v: k for k, v in CODENAME_TO_DATASET.items()}
DS2CODE.update({"Cryptic": "Cryptic", "Windspeed": "windspeed"})
CODES = [DS2CODE[d] for d in DATASETS]


def internal_by_code(scores):
    return {c: round(adj_score_internal(c, d["raw"], d["benchmark"]), 4)
            for c, d in scores.items() if c in CODES}


def egnas_scores(seed):
    """Merge the 11-ds + new-ds EGNAS-full logs for a single seed."""
    out = {}
    for tag in ("t2bfull", "t2bfull_new"):
        p = EG / f"{tag}_s{seed}.log"
        if p.exists():
            out.update(internal_by_code(parse_log(p)))
    return out


def v2_scores(seed):
    p = P1 / f"v4_s{seed}.log"
    return internal_by_code(parse_log(p)) if p.exists() else {}


def probe_decisions(seed):
    """Return {codename: 'on'|'off'|'skipped'} from per-dataset probe JSONs."""
    out = {}
    pdir = P1 / f"v4_s{seed}"
    if not pdir.exists():
        return out
    for jp in pdir.glob("*_p1_probe.json"):
        code = jp.stem.replace("_p1_probe", "")
        try:
            d = json.loads(jp.read_text())
            out[code] = "skipped" if d.get("skipped") else d["decision"]
        except Exception:
            out[code] = "?"
    return out


def main():
    p1 = {c: [] for c in CODES}
    eg = {c: [] for c in CODES}
    probes_by_code = {c: {} for c in CODES}  # {code: {seed: decision}}

    for s in SEEDS:
        ps, es = v2_scores(s), egnas_scores(s)
        for c in CODES:
            if c in ps:
                p1[c].append(ps[c])
            if c in es:
                eg[c].append(es[c])
        for code, dec in probe_decisions(s).items():
            if code in probes_by_code:
                probes_by_code[code][s] = dec

    print("# Tier-2a / P1 v4 — 3-seed × 13-dataset paired verdict "
          "(internal adj, vs EGNAS-full)\n")
    print("| dataset | EGNAS-full | +P1 | Δmean | σ EG→P1 | probe decisions |")
    print("|---|---:|---:|---:|---:|---|")
    rows = {}
    for c in CODES:
        if not p1[c] or not eg[c]:
            continue
        ds = CODENAME_TO_DATASET.get(c, c)
        e, p = mstats(eg[c]), mstats(p1[c])
        decs = ", ".join(
            f"s{s}:{probes_by_code[c].get(s, '?')}" for s in SEEDS)
        rows[ds] = {"egnas": e, "p1": p,
                    "d_mean": round(p["mean"] - e["mean"], 4),
                    "probes": probes_by_code[c]}
        print(f"| {ds} | {fmt(e)} | **{fmt(p)}** | "
              f"**{rows[ds]['d_mean']:+.3f}** | "
              f"{e['std']:.2f}→{p['std']:.2f} | {decs} |")

    et = ([sum(eg[c][i] for c in CODES) for i in range(3)]
          if all(len(eg[c]) == 3 for c in CODES) else None)
    pt = ([sum(p1[c][i] for c in CODES) for i in range(3)]
          if all(len(p1[c]) == 3 for c in CODES) else None)
    E = P = None
    if et and pt:
        E, P = mstats(et), mstats(pt)
        ratio = (P['std'] / E['std']) if E['std'] > 1e-9 else float('nan')
        print(f"\n**Total (13 ds, internal):** EGNAS-full **{fmt(E)}** vs "
              f"+P1 **{fmt(P)}**  →  Δmean **{P['mean']-E['mean']:+.2f}**,"
              f" σ **{E['std']:.2f}→{P['std']:.2f} (×{ratio:.2f})**")
        print(f"  - per-seed totals: EGNAS {E['vals']}; P1 {P['vals']}")

    # Probe-call summary.
    print("\n## Probe-call summary (across seeds)\n")
    print("| dataset | aug-on calls | aug-off calls | skipped | consensus |")
    print("|---|---:|---:|---:|---|")
    for c in CODES:
        ds = CODENAME_TO_DATASET.get(c, c)
        decs = list(probes_by_code[c].values())
        on_n = decs.count("on")
        off_n = decs.count("off")
        sk_n = decs.count("skipped")
        consensus = ("on" if on_n == 3 else
                     "off" if off_n == 3 else
                     "skipped" if sk_n == 3 else
                     f"split ({on_n}on/{off_n}off/{sk_n}skip)")
        print(f"| {ds} | {on_n} | {off_n} | {sk_n} | {consensus} |")

    # Big wins / regressions.
    print("\n## Verdict\n")
    if E and P:
        print(f"- **Total:** +P1 {fmt(P)} vs EGNAS-full {fmt(E)}  →  "
              f"Δmean **{P['mean']-E['mean']:+.2f}**, σ ×"
              f"{(P['std']/E['std']):.2f}")
    big = [ds for ds, r in rows.items() if r["d_mean"] > 1.0]
    reg = [ds for ds, r in rows.items() if r["d_mean"] < -0.25]
    sig_up = [ds for ds, r in rows.items()
              if r["p1"]["std"] > r["egnas"]["std"] + 1e-6]
    print(f"- Big wins (Δmean > +1.0): {big if big else 'none'}")
    print(f"- Regressions (Δmean < −0.25): {reg if reg else 'NONE ✓'}")
    print(f"- σ INCREASED under P1: {sig_up if sig_up else 'NONE ✓'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seeds": SEEDS, "datasets": DATASETS,
        "egnas_full": eg, "p1": p1,
        "probe_decisions": probes_by_code,
        "rows": {k: {"egnas": r["egnas"], "p1": r["p1"],
                     "d_mean": r["d_mean"], "probes": r["probes"]}
                 for k, r in rows.items()},
        "totals": ({"egnas": E, "p1": P} if (E and P) else None),
    }, indent=2, default=str))
    print(f"\n[wrote {OUT.relative_to(REPO)}]")


if __name__ == "__main__":
    main()
