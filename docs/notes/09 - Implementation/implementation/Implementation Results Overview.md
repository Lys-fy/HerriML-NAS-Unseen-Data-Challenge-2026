# Implementation Results Overview

> **Status:** current | **Topic:** Index of *actual* implementation runs, findings, and scores. The plans live in [[Implementation Plan Overview]]; this folder is what happened when we ran them.
> **Related:** [[Implementation Plan Overview]], [[Tier 0 - Results]], [[Tier 1 - Results]], [[Multi-Seed - Results]], [[E1 - Results]], [[Tier 2b - Results]], [[Reliability vs Peak]], [[Threats to Validity]]

This is the **results** half of section 09. The split:

- `09 - Implementation/plan/` — what we *intend* to build (tiers, branches, acceptance criteria).
- `09 - Implementation/implementation/` — what we *actually built and measured* (this folder), with embedded graphs.

## Scoreboard (all 11 public datasets, `--time-limit 2.0` on DGX Spark GB10)

| Run | Total adj (shipped) | Datasets ≥ benchmark | −10 floors | Note |
|---|---:|---:|---:|---|
| [[Tier 0 - Results\|Tier 0]] | **−0.355** | 6 / 11 | 2 (AddNIST, MultNIST) | hardcoded CNN, no search, overfits in ~25 epochs |
| [[Tier 1 - Results\|Tier 1]] | **+24.644** ¹ | 9 / 11 | 0 | regularization + EMA + op-search; ¹ **single-seed, a favourable tail draw** |
| [[Multi-Seed - Results\|Tier 1 ×3 seeds]] | **+11.27 ± 8.21** | varies | varies | the trustworthy figure — std ≈ the mean; one seed scored +0.02, another +19.6 |

The **Tier 1 ×3-seed** row supersedes the single-seed +24.644 as the honest number ([[Multi-Seed - Results]]) — seed variance is ≈100 % of the signal.

Sudoku ships `benchmark: 0.0` (placeholder). Internal-benchmark scoring (Sudoku = Tier 0's 70.95 %) is tracked in `dev/internal_benchmarks.py` and used in [[Tier 1 - Results]] for the honest delta.

## Per-note index (the actual progression, in order)

- [[Tier 0 - Results]] — Hardcoded-CNN baseline. The "what does zero search get us" measurement. Established that the binding constraint is **overfitting**, not search.
- [[Tier 1 - Results]] — Augmentation + regularization + random op-search. The (single-seed) +25 swing, the Language regression, and the single-feature ablation.
- [[Multi-Seed - Results]] — **the last completed test run**: Tier 1 ×3 seeds (`full`) **+ the `aug-off` arm** ×3 seeds (54 runs total). The authoritative numbers. Contains: the seed-variance headline (σ≈100 %), the per-dataset stability table, and **§C the augmentation-effect table (full vs aug-off)** — aug is worth +21.5 over the 7 aug-firing datasets; Language is the lone seed-stable reversal. *This is the "aug-off / ect" results the scoreboard's single rows don't show — they live here.*
- [[E1 - Results]] — ✅ **complete (3 seeds) — MIXED verdict**: the budget-filling ensemble's *mechanism* is validated on its target (AddNIST σ 5.45→1.00 ×0.18, mean ↑) but it is **not a blanket fix** — total σ only ×0.75, GeoClassing regressed −0.34, Language gain is noise-inflated. Carries the full reasoning chain (multi-seed → Step 0/Q8 → why E1) and the honest analysis. Conditional/probe-gated component, not a default wrapper.
- [[Tier 2b - Results]] — ✅ **EGNAS-macro diagnostic (3 seeds) — make-or-break PASSED, two caveats**. Macro arch search moved **CIFARTile −0.515 → +0.167 (all 3 seeds, σ↓)** — the ceiling Tier-1 couldn't move under any config; beats Tier-1 on the 3-ds mean (+3.26). Caveats: the CIFARTile gain is *small* (real frontier lever is E2/Deep-Sets) and the macro search is itself a **Q8 arch-lottery** (s2024 picked D=5 for AddNIST → −2.81). Validated & kept; *not* a finished engine — fix arch-selection before any full sweep.

## Headline findings so far

1. **Reliability-before-peak is empirically validated.** Tier 0 → Tier 1 went −0.355 → +24.644 (single-seed; **≈ +11 ± 8 over 3 seeds**, [[Multi-Seed - Results]]) almost entirely by *not catastrophically under/over-fitting*, exactly as [[Reliability vs Peak]] predicts. The jump is real; the precise total was lucky.
2. **Augmentation is strongly net-positive but sharply targeted** (multi-seed `aug-off` arm, [[Multi-Seed - Results]] §C — supersedes the old single-seed +11.86/−5.65). Over the 7 aug-firing datasets aug is worth **+21.5**; on AddNIST/MultNIST aug-off is a dead **−10 ± 0 floor** every seed; the *only* seed-stable reversal is **Language (aug-cost −6.82 ± 1.88)**. Metadata can't predict which → the runtime aug-safety probe (P1) has a precisely-localised job.
3. **Search instability is universal but not the variance driver** (Step 0 / Q8, `dev/analyze_search_stability.py`). The op-search picks a *different* arch every seed for 10/11 datasets, yet downstream variance is huge only on AddNIST/MultNIST and tiny elsewhere (GeoClassing equally unstable, σ 0.29). So the old single-seed "+2.9 vs all-Conv3" is unconfirmed under variance, and Tier 2's macro-search would inherit the same lottery → it must emit an ensemble-able pool, not a single argmax arch.
4. **Architectural ceilings are real — and architecture search *does* breach them** ([[Tier 2b - Results]]). CIFARTile stayed below benchmark under *every* Tier-1 config; EGNAS-macro depth/width search moved it above benchmark on all 3 seeds (−0.515 → +0.167, σ↓). Confirmed: architecture matters here. But macro-conv only *sips* CIFARTile's ~+46 pp NAS headroom → the real frontier lever is a **non-conv primitive (E2 / Deep-Sets)**, as the thesis predicted.
5. **Seed variance is ≈100 % of the signal** ([[Multi-Seed - Results]], the headline). The Tier 1 recipe at 3 seeds = +8.4 ± 8.4 internal; AddNIST/MultNIST swing ±5.4/±3.0 (never "fixed", just lucky once). Single-seed deltas are dead as evidence; the aug double-edge *direction* is the only seed-stable thing → probeable (P1).
6. **Ensembling (E1) is a partial, conditional answer to variance — not the answer** ([[E1 - Results]], 3 seeds). It collapses the *right kind* of variance brilliantly (AddNIST σ ×0.18 + mean ↑) but **does not collapse total submission variance** (×0.75) and can regress budget-saturating datasets (GeoClassing −0.34). The decisive consequence: the residual variance is **structural / dataset-level**, not per-run noise — you cannot ensemble it away, you must *characterize the dataset and configure* (the probe thesis). This is the empirical bridge into Tier 2. E1 stays as a probe-gated component for AddNIST-class datasets, not a default wrapper.
7. **Macro architecture search works *but is itself a variance source* — Q8 answered in the bad sense** ([[Tier 2b - Results]]). EGNAS-macro genuinely cracks the CIFARTile ceiling (the first real NAS win) and beats Tier-1 on the 3-ds mean with lower σ — *but* it ranks candidates on a cuDNN-noisy short-train val and picks a different arch per seed; on s2024 it chose D=5 for AddNIST and AddNIST collapsed to −2.81. The architecture-aware-schedule's paper Spearman-0.91 does **not** hold at our budget. Net: search adds value *and* a new instability → arch selection must be stabilised before EGNAS can ship (next action), and E2 remains CIFARTile's real lever.

## Graph index

All plots are generated by `dev/plot_tier_results.py` and embedded in the per-tier notes:

- `tier0_scores.png`, `tier0_curves.png` — Tier 0 per-dataset adj + learning curves
- `tier1_scores.png`, `tier1_curves.png` — Tier 1 per-dataset adj + learning curves
- `tier1_ablation.png` — Tier 1 full vs single-feature-disabled, the per-component value chart
- `multiseed.png` — 3-seed variance + aug-off-effect figure (in [[Multi-Seed - Results]])
- `e1.png` — E1 vs Tier-1 per-dataset means + σ-collapse (in [[E1 - Results]])

Regenerate any of them with:

```bash
python3 dev/plot_tier_results.py --tier tier0
python3 dev/plot_tier_results.py --tier tier1 --ablation
python3 dev/aggregate_multiseed.py && python3 dev/plot_tier_results.py --multiseed
python3 dev/aggregate_e1.py && python3 dev/plot_tier_results.py --e1   # after E1 lands
python3 dev/aggregate_egnas.py   # Tier 2b paired verdict + arch-stability (→ _egnas_aggregate.json; no plot yet)
```
