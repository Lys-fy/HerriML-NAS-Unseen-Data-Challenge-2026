"""Step-0 analysis (thesis Q8): is the Tier 1 random op-search variance-amplifying?

The search re-seeds its RNG per `search()` call, so the *candidate pool* is
deterministic per BASELINE_SEED and identical across datasets within a seed.
What is NOT deterministic is which candidate wins — it's ranked on a
cuDNN-nondeterministic short-train val_acc. So the question is:

  For the SAME dataset, does the search pick a DIFFERENT architecture across
  the 3 seeds, and how thin is the margin it's deciding on?

If the high-variance datasets (AddNIST/MultNIST) show different picks across
seeds on a razor-thin search margin, the search is an architecture lottery
feeding the downstream accuracy variance → E1 should ensemble the decorrelated
*candidate architectures*, not just seeds. Reads the existing full_s* logs
only; no Spark, no compute.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "dev" / "logs" / "multiseed"
SEEDS = [42, 1337, 2024]

CODENAME_TO_DATASET = {
    "Adaline": "AddNIST", "Caitie": "CIFARTile", "Chester": "Chesseract",
    "conway": "GameOfLife", "Sadie": "GeoClassing", "Gutenberg": "Gutenberg",
    "LaMelo": "Language", "Mateo": "MultNIST", "Myopia": "Myofibre",
    "Sokoto": "Sudoku", "Volga": "Voxel",
}

_CAND_RE = re.compile(r"candidate\s+(\d+)\s+ops=(\[[^\]]*\])\s+val_acc=([\d.]+)")
_DONE_RE = re.compile(r"search done: best arch=(\[[^\]]*\])\s+val_acc=([\d.]+)")
_SCORE_RE = re.compile(r"\s*(\w+):\s*raw=([\d.]+)%\s+benchmark=\s*[\d.]+%\s+adj=\s*([+-]?[\d.]+)")


def parse_full_log(path: Path):
    """Parse two independent ordered lists and zip positionally.

    Search blocks appear interleaved through the log body in dataset-processing
    order; the scoring lines appear all together in a block at the *end*, in the
    SAME dataset-processing order. So zip(blocks, scores) by position — do NOT
    try to interleave-match them.

    Each result: {codename, dataset, best_arch, best_val, cand_vals(desc),
                  margin (best - 2nd best of the 8 short-train vals),
                  chose_bootstrap (all-Conv3 candidate 0 won), raw, adj}.
    """
    blocks, scores, cands = [], [], []
    for line in path.read_text().splitlines():
        m = _CAND_RE.search(line)
        if m:
            cands.append((int(m.group(1)), m.group(2), float(m.group(3))))
            continue
        m = _DONE_RE.search(line)
        if m:
            vals = sorted((v for _, _, v in cands), reverse=True)
            boot = next((v for i, _, v in cands if i == 0), None)
            best_val = float(m.group(2))
            blocks.append({
                "best_arch": m.group(1),
                "best_val": best_val,
                "cand_vals": vals,
                "margin": round(vals[0] - vals[1], 4) if len(vals) > 1 else None,
                "chose_bootstrap": (
                    boot is not None and abs(boot - best_val) < 1e-6),
            })
            cands = []
            continue
        m = _SCORE_RE.match(line)
        if m:
            scores.append((m.group(1), float(m.group(2)), float(m.group(3))))

    out = []
    for blk, (code, raw, adj) in zip(blocks, scores):
        out.append({**blk, "codename": code,
                    "dataset": CODENAME_TO_DATASET.get(code, code),
                    "raw": raw, "adj": adj})
    return out, len(blocks), len(scores)


def main():
    # per_ds[dataset][seed] = block
    per_ds = {}
    for seed in SEEDS:
        p = LOG_DIR / f"full_s{seed}.log"
        if not p.exists():
            print(f"  ! missing {p.name}")
            continue
        blocks, nb, ns = parse_full_log(p)
        if nb != ns:
            print(f"  ! {p.name}: {nb} search blocks vs {ns} score lines "
                  f"— positional zip may be misaligned")
        for b in blocks:
            per_ds.setdefault(b["dataset"], {})[seed] = b

    agg = json.loads((LOG_DIR / "_aggregate.json").read_text())
    adj_std = {agg["codename_to_dataset"][c]: d["adj"]["std"]
               for c, d in agg["full_by_dataset"].items()}

    rows = []
    for ds, by_seed in per_ds.items():
        archs = [by_seed[s]["best_arch"] for s in SEEDS if s in by_seed]
        margins = [by_seed[s]["margin"] for s in SEEDS if s in by_seed
                   and by_seed[s]["margin"] is not None]
        n_distinct = len(set(archs))
        rows.append({
            "ds": ds,
            "adj_std": adj_std.get(ds, float("nan")),
            "n_distinct_arch": n_distinct,
            "unstable": n_distinct > 1,
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
            "archs": archs,
        })

    rows.sort(key=lambda r: -r["adj_std"])

    print("Tier 1 random op-search stability across the 3 `full` seeds")
    print("(adj_std = downstream accuracy variance; n_arch = distinct best-archs "
          "picked over 3 seeds; margin = search-time val gap best vs 2nd)\n")
    print(f"{'dataset':<12} {'adj_std':>8} {'n_arch/3':>9} {'mean_margin':>12}  picked-arch-per-seed")
    print("-" * 96)
    for r in rows:
        flag = "  ← UNSTABLE" if r["unstable"] else ""
        mm = f"{r['mean_margin']:.4f}" if r["mean_margin"] is not None else "  n/a"
        print(f"{r['ds']:<12} {r['adj_std']:>8.2f} {r['n_distinct_arch']:>6}/3 "
              f"{mm:>12}  {[a[:38] for a in r['archs']]}{flag}")

    # Verdict ----------------------------------------------------------------
    hi = [r for r in rows if r["adj_std"] >= 2.0]
    hi_unstable = [r for r in hi if r["unstable"]]
    lo = [r for r in rows if r["adj_std"] < 1.0]
    lo_unstable = [r for r in lo if r["unstable"]]
    print("\n--- verdict ---")
    print(f"high-variance datasets (adj_std≥2): {[r['ds'] for r in hi]}")
    print(f"  ...of which the search picked ≠ archs across seeds: "
          f"{[r['ds'] for r in hi_unstable]}")
    print(f"low-variance datasets (adj_std<1): {[r['ds'] for r in lo]}")
    print(f"  ...of which search-unstable: {[r['ds'] for r in lo_unstable]}")
    thin = [r for r in rows if r["mean_margin"] is not None and r["mean_margin"] < 0.02]
    print(f"datasets deciding on a thin margin (<0.02 val gap): "
          f"{[(r['ds'], r['mean_margin']) for r in thin]}")
    print("\nIf high-variance ⊆ search-unstable & margins are thin → the search "
          "is an architecture lottery feeding the variance → E1 must ensemble "
          "the candidate architectures (decorrelated), not just seeds.")


if __name__ == "__main__":
    main()
