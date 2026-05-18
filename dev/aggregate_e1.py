"""E1 validation: did the budget-filling ensemble collapse the variance?

Compares the E1 sweep (`dev/logs/e1val/e1_s{42,1337,2024}.log`) against the
Tier-1 paired baseline — the SAME 5 datasets × same seeds × same 2 h budget,
reused from the multi-seed sweep (`dev/logs/multiseed/full_s*.log`), because
E1's member 0 *is* that Tier-1 model.

Reuses aggregate_multiseed's parser + stats (DRY, identical regex/scoring).
Safe to run before the sweep lands: it always prints the Tier-1 baseline
side and reports E1 as "pending" until the e1val logs exist.

Reads   dev/logs/e1val/e1_s*.log + dev/logs/multiseed/full_s*.log
Writes  dev/logs/e1val/_e1_aggregate.json + markdown to stdout.

Success (thesis Q4): per-dataset σ_adj shrinks substantially vs Tier 1
(target AddNIST 5.45 -> <2) AND mean >= Tier-1 mean AND no regression on
the stable datasets (GeoClassing, Language) and no new -10.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from internal_benchmarks import adj_score_internal
from aggregate_multiseed import parse_log, mstats, fmt  # shared, no matplotlib

REPO_ROOT = Path(__file__).resolve().parent.parent
E1_DIR = REPO_ROOT / "dev" / "logs" / "e1val"
MS_DIR = REPO_ROOT / "dev" / "logs" / "multiseed"
OUT_JSON = E1_DIR / "_e1_aggregate.json"

SEEDS = [42, 1337, 2024]
# run_e1_validation.sh DATASETS order (positional zip for the ensemble lines).
DATASETS = ["AddNIST", "CIFARTile", "GeoClassing", "Language", "MultNIST"]
CODES = ["Adaline", "Caitie", "Sadie", "LaMelo", "Mateo"]  # same order
CODE2DS = dict(zip(CODES, DATASETS))

# Per-dataset success thresholds (adj σ). AddNIST is the headline target.
SIGMA_TARGET = {"AddNIST": 2.0, "MultNIST": 1.5}
# Datasets that must NOT regress in mean (stable win / seed-stable regression).
NO_REGRESS = ["GeoClassing", "Language", "CIFARTile"]

_ENS_RE = re.compile(r"E1: ensemble of (\d+) members \[([^\]]*)\]")
_AVG_RE = re.compile(r"E1 predict: averaged (\d+)/(\d+) members")


def _internal_by_code(scores: dict) -> dict:
    """code -> internal adj, for the 5 E1 datasets present in `scores`."""
    out = {}
    for code in CODES:
        if code in scores:
            d = scores[code]
            out[code] = round(adj_score_internal(code, d["raw"], d["benchmark"]), 4)
    return out


def _parse_ensemble(path: Path) -> dict:
    """code -> {members, predict_used} via positional zip with DATASETS order
    (ensemble/averaged banners print once per dataset, in run order)."""
    text = path.read_text()
    ens = [(int(m.group(1)), m.group(2)) for m in _ENS_RE.finditer(text)]
    avg = [(int(m.group(1)), int(m.group(2))) for m in _AVG_RE.finditer(text)]
    out = {}
    for i, code in enumerate(CODES):
        rec = {}
        if i < len(ens):
            rec["members"] = ens[i][0]
            rec["member_tags"] = ens[i][1]
        if i < len(avg):
            rec["predict_used"] = avg[i][0]
            rec["predict_total"] = avg[i][1]
        if rec:
            out[code] = rec
    return out


def main():
    # ---- Tier-1 paired baseline (always available) ----------------------
    base = {c: [] for c in CODES}      # code -> [adj per seed]
    base_missing = []
    for s in SEEDS:
        p = MS_DIR / f"full_s{s}.log"
        if not p.exists():
            base_missing.append(p.name)
            continue
        bi = _internal_by_code(parse_log(p))
        for c in CODES:
            if c in bi:
                base[c].append(bi[c])

    # ---- E1 sweep (may be pending) --------------------------------------
    e1 = {c: [] for c in CODES}
    ens = {c: [] for c in CODES}
    e1_present = [s for s in SEEDS if (E1_DIR / f"e1_s{s}.log").exists()]
    for s in e1_present:
        p = E1_DIR / f"e1_s{s}.log"
        ei = _internal_by_code(parse_log(p))
        for c in CODES:
            if c in ei:
                e1[c].append(ei[c])
        ep = _parse_ensemble(p)
        for c, rec in ep.items():
            if "members" in rec:
                ens[c].append(rec["members"])

    n_e1 = len(e1_present)
    print(f"# E1 validation — {n_e1}/3 e1 seeds present "
          f"{tuple(e1_present)}; baseline full_s* "
          f"{'OK' if not base_missing else 'MISSING ' + str(base_missing)}\n")

    if n_e1 == 0:
        print("E1 sweep not landed yet — Tier-1 baseline preview "
              "(internal adj, mean ± std over 3 seeds):\n")
        print("| dataset | Tier-1 (mean ± std) | σ | per-seed |")
        print("|---|---:|---:|---|")
        for c in CODES:
            if base[c]:
                st = mstats(base[c])
                print(f"| {CODE2DS[c]} | {fmt(st)} | {st['std']:.2f} | "
                      f"{', '.join(f'{v:+.2f}' for v in st['vals'])} |")
        print("\nRe-run this script after the sweep lands "
              "(handoff doc has the live-status command).")
        return

    # ---- per-dataset comparison ----------------------------------------
    rows = []
    for c in CODES:
        if not base[c] or not e1[c]:
            continue
        b, e = mstats(base[c]), mstats(e1[c])
        sigma_ratio = (e["std"] / b["std"]) if b["std"] > 1e-9 else float("nan")
        d_mean = round(e["mean"] - b["mean"], 4)
        ds = CODE2DS[c]
        flags = []
        if ds in SIGMA_TARGET:
            flags.append("σ✓" if e["std"] < SIGMA_TARGET[ds] else "σ✗(target "
                         f"{SIGMA_TARGET[ds]})")
        flags.append("σ↓" if e["std"] < b["std"] - 1e-6 else "σ≈/↑")
        if ds in NO_REGRESS:
            flags.append("mean✓" if d_mean >= -0.25 else "mean✗(regressed)")
        rows.append({"ds": ds, "base": b, "e1": e,
                     "sigma_ratio": round(sigma_ratio, 3),
                     "d_mean": d_mean,
                     "members": mstats(ens[c]) if ens[c] else None,
                     "flags": " ".join(flags)})

    print("## Per-dataset — Tier-1 vs E1 (internal adj, mean ± std / 3 seeds)\n")
    print("| dataset | Tier-1 | E1 | Δmean | σ T1→E1 | σ ratio | members | verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        mem = (f"{r['members']['mean']:.1f}" if r["members"] else "—")
        print(f"| {r['ds']} | {fmt(r['base'])} | {fmt(r['e1'])} | "
              f"**{r['d_mean']:+.3f}** | {r['base']['std']:.2f}→"
              f"{r['e1']['std']:.2f} | {r['sigma_ratio']:.2f} | {mem} | "
              f"{r['flags']} |")

    # ---- totals over the 5 datasets ------------------------------------
    bt = [sum(base[c][i] for c in CODES if len(base[c]) == 3) for i in range(3)] \
        if all(len(base[c]) == 3 for c in CODES) else None
    et = [sum(e1[c][i] for c in CODES if len(e1[c]) == n_e1)
          for i in range(n_e1)] if all(len(e1[c]) == n_e1 for c in CODES) else None
    print("\n## Totals over the 5 datasets (internal adj)\n")
    if bt and et:
        B, E = mstats(bt), mstats(et)
        print(f"- Tier-1 : **{fmt(B)}**  (seeds {', '.join(f'{v:+.2f}' for v in B['vals'])}; σ {B['std']:.2f})")
        print(f"- E1     : **{fmt(E)}**  (seeds {', '.join(f'{v:+.2f}' for v in E['vals'])}; σ {E['std']:.2f})")
        print(f"- **Δmean = {E['mean'] - B['mean']:+.3f}**, "
              f"**total σ {B['std']:.2f} → {E['std']:.2f}** "
              f"(×{(E['std']/B['std']) if B['std']>1e-9 else float('nan'):.2f})")

    # ---- verdict --------------------------------------------------------
    print("\n## Verdict (thesis Q4)\n")
    add = next((r for r in rows if r["ds"] == "AddNIST"), None)
    if add:
        ok = add["e1"]["std"] < SIGMA_TARGET["AddNIST"]
        print(f"- Headline: AddNIST σ {add['base']['std']:.2f} → "
              f"{add['e1']['std']:.2f}  "
              f"{'✓ TARGET MET (<2)' if ok else '✗ target <2 not met'}")
    reg = [r["ds"] for r in rows if "mean✗" in r["flags"]]
    print(f"- Mean regressions on must-not-regress datasets: "
          f"{reg if reg else 'none ✓'}")
    neg10 = [CODE2DS[c] for c in CODES
             for v in e1[c] if v <= -9.999]
    print(f"- New -10 floors under E1: {sorted(set(neg10)) if neg10 else 'none ✓'}")
    sigma_up = [r["ds"] for r in rows if r["e1"]["std"] > r["base"]["std"] + 1e-6]
    print(f"- Datasets where σ INCREASED under E1: "
          f"{sigma_up if sigma_up else 'none ✓'}")
    total_collapsed = bt and et and mstats(et)["std"] < 0.5 * mstats(bt)["std"]
    clean = add and (add["e1"]["std"] < SIGMA_TARGET["AddNIST"]) and not reg \
        and not neg10 and not sigma_up and total_collapsed
    if clean:
        print("\n⇒ Clean win: E1 is the variance-control layer — promote it.")
    else:
        print("\n⇒ MIXED — NOT a clean win. Read per-row, don't trust the "
              "totals. Interpretation is in docs/notes [[E1 - Results]] "
              "(mechanism validated on AddNIST-class only; total σ not "
              "collapsed ⇒ residual variance is structural ⇒ argues for the "
              "probe/Tier-2 direction; member-quality gate is a Tier-3 note).")

    E1_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "seeds": SEEDS, "e1_seeds_present": e1_present,
        "datasets": DATASETS, "code2ds": CODE2DS,
        "tier1_by_code": base, "e1_by_code": e1,
        "ensemble_members_by_code": ens,
        "rows": [{k: v for k, v in r.items()} for r in rows],
    }, indent=2, default=str))
    print(f"\n[wrote {OUT_JSON.relative_to(REPO_ROOT)}]")


if __name__ == "__main__":
    main()
