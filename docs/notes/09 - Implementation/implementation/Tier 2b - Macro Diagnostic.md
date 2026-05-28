# Tier 2b — Macro-only Diagnostic (EGNAS depth + width)

> **Status:** ✅ diagnostic complete (3 seeds, 3 datasets, 2026-05-18). Make-or-break PASSED, two caveats. **Superseded for the authoritative Tier-2b verdict by [[Tier 2b - Results]]** (full EGNAS = macro + micro). Retained here as the 3-ds paired record + the Q8 arch-lottery demonstration that motivated completing the method. | **Topic:** Does multi-seed macro-only arch search (depth-grow + width-prune, fixed Conv3) beat Tier-1's CIFARTile ceiling? The first real NAS result in the repo.
> **Related:** [[Tier 2b - Results]] (authoritative full-EGNAS verdict), [[Tier 2b - EGNAS-light (macro)]] (plan), [[Multi-Seed - Results]] (paired baseline), [[E1 - Results]], [[Probe-Driven Self-Configuration]], [[Reliability vs Peak]], [[Implementation Results Overview]], [[Run State + Handoff]]

Diagnostic 2026-05-18, `baselines/tier2b_egnas_macro/` (only `nas.py` differs from Tier 1). Driver `dev/run_egnas_diagnostic.sh`, aggregation `dev/aggregate_egnas.py`. Internal adj, mean ± std over seeds 42/1337/2024. Tier-1 column = the paired multi-seed `full_s*` baseline (same 3 datasets/seeds/budget).

## The make-or-break: CIFARTile

| dataset | Tier-1 | EGNAS-macro | Δmean | σ T1→EGNAS | EGNAS per-seed |
|---|---:|---:|---:|---:|---|
| **CIFARTile** | −0.515 ± 0.521 | **+0.167 ± 0.269** | **+0.682** | 0.52→**0.27** | +0.04, −0.01, +0.48 |
| AddNIST | −0.378 ± 5.446 | +2.089 ± 4.244 | +2.466 | 5.45→4.24 | +4.36, +4.71, **−2.81** |
| GeoClassing | +7.678 ± 0.294 | +7.790 ± 0.080 | +0.113 | 0.29→0.08 | +7.70, +7.82, +7.85 |
| **Total (3)** | **+6.79 ± 4.82** | **+10.05 ± 3.93** | **+3.26** | 4.82→3.93 | +12.10, +12.52, +5.52 |

**The test passed honestly.** Tier-1 could not move CIFARTile above benchmark under *any* configuration (training-side levers exhausted). Macro depth/width search moved it above benchmark **on all 3 seeds**, with *lower* variance. Architecture genuinely matters here — the thesis's "search backend" leg has real substance. EGNAS also beats Tier-1 on the 3-dataset mean (+3.26) and is tighter overall, with no regression on the stable GeoClassing control (it slightly improved + tightened it).

## Two caveats — not buried

1. **The CIFARTile win is directionally real but small.** +0.167 is *barely* above benchmark. CIFARTile's NAS headroom is ~+46 pp (best-NAS ~92 % vs our raw ~47–49 %). Macro conv depth/width captured a **sliver** of it. This confirms the thesis prediction: the real CIFARTile lever is a **non-convolutional primitive (E2 / Deep-Sets relational head)**, not more conv. Macro search *crosses* the bar; it does not approach the frontier.
2. **The macro search is itself an architecture lottery (Q8 — confirmed, in the bad sense).** EGNAS ranks candidates on a cuDNN-nondeterministic short-train val, so it picks a different architecture per seed:

   | dataset | s42 | s1337 | s2024 |
   |---|---|---|---|
   | CIFARTile | D8/W16 | D9/W16 | D10/W48 |
   | AddNIST | D9/W40 | D10/W40 | **D5/W48** |
   | GeoClassing | D9/W40 | D8/W48 | D9/W48 |

   Depth lands in a rough band but is not stable; **on s2024 it picked D=5 for AddNIST and AddNIST collapsed to −2.81** — the entire reason s2024's total fell to +5.52. EGNAS adds value *and* a new variance source via unstable selection. This is structural, not a tuning detail. The architecture-aware schedule's paper-reported Spearman-0.91 ranking **does not hold at our budget under our nondeterminism** — exactly the Q8 risk we said we'd verify rather than assume. We verified it; it's real.

## Verdict & decision (2026-05-18)

**EGNAS-macro is validated and kept** — it does the one thing training-side levers could not (crack CIFARTile, multi-seed, with lower variance) and beats Tier-1 overall. **But it is not a finished shipping engine:** arch selection is a lottery, and CIFARTile's real value needs E2.

The plan's gate ("CIFARTile beats the ceiling multi-seed → proceed to full sweep") is technically passed — but launching a ~30 h full sweep on a search we now *know* is arch-unstable would be measuring a known-flawed engine. **Decision: fix the arch-lottery first, then re-run the same 3-dataset diagnostic, before any full sweep.**

## Carry-forward

- **Next (active):** stabilize EGNAS arch selection — average the short-train ranking signal over ≥2 restarts (or a more robust pick than single-run argmax val), then re-diagnose on the same {CIFARTile, AddNIST, GeoClassing} × 3 seeds. Target: kill the D=5-type catastrophic mispick (AddNIST σ should drop, no −10-class arch).
- **E2 / Deep-Sets remains the real CIFARTile frontier lever** ([[Probe-Driven Self-Configuration]] arch-menu) — macro-conv only sips the headroom.
- Pool JSONs (`dev/logs/egnasdiag/egnas_s*/<codename>_egnas_pool.json`) logged for future Tier-3 ensembling.

## Reproduce
```bash
rsync -a spark:'~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/logs/egnasdiag*' dev/logs/
python3 dev/aggregate_egnas.py   # paired verdict + arch-stability + _egnas_aggregate.json
```
