# Implementation Results Overview

> **Status:** current | **Topic:** Index of *actual* implementation runs, findings, and scores. The plans live in [[Implementation Plan Overview]]; this folder is what happened when we ran them.
> **Related:** **[[Key Learnings]]** (read first — distilled insight from every experiment), [[Implementation Plan Overview]], [[Tier 0 - Results]], [[Tier 1 - Results]], [[Multi-Seed - Results]], [[E1 - Results]], [[Tier 2b - Results]], [[Tier 2a - Results]], [[Tier 2a - Language Deep Dive]], [[Reliability vs Peak]], [[Threats to Validity]]

> **2026-06-02 — latest:** P1 **v6 swept → net regression** (−12.10 vs EGNAS-full, σ ×2.98; rich vocab poisoned Sudoku, macro-prune cratered GameOfLife; only Language aug-probe +4.23 survives). **LSTM Language probe → row-BiLSTM cracks Language** (85.76 %, first arch reliably > benchmark). **v7** = simplify pivot (EGNAS-full + 4 conv ops + macro-prune-verify + aug-probe), built + GPU-smoke-passed; recurrence deploys only as a fail-safe convergence A/B. Full synthesis → [[Key Learnings]].

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
- [[Tier 2b - Results]] — ✅ **the major project result** (Tier 2b full EGNAS, macro + micro; 3 seeds × 11 datasets, 2026-05-19→20). Full EGNAS beats Tier-1 by **+19.60 mean** and collapses total σ **8.37 → 3.28 (×0.39)** — the s2024-cratering pattern finally broke. **6 big wins** (AddNIST σ ×0.09 + mean ↑5.8; MultNIST ×0.13 + ↑4.1; GameOfLife +7.2; Sudoku/Voxel/Language gains), **2 honest regressions** (Myofibre −0.96, Chesseract −0.37), Language still aug-bound (P1 territory). Variance fixed *by architecture*, not ensembling — E1 superseded. Embeds the paired-bars/σ-collapse figure + the aggregated learning curves (mean ± std + Tier-1 baseline overlay across seeds). Windspeed+Cryptic 13-ds extension running.
- [[Tier 2b - Macro Diagnostic]] — the 3-ds macro-only diagnostic stage (2026-05-18) that motivated finishing the method. Retained as the paired record + Q8 arch-lottery demonstration. Superseded by [[Tier 2b - Results]] above for the authoritative read.
- [[Tier 2a - Results]] — ✅ **first actual probe of the thesis, validated through v2/v3/v4** (2026-05-20 → 2026-05-28). Tier 2b full EGNAS + a runtime P1 aug-safety probe; targets Language (aug-cost −6.82 ± 1.88) and Myofibre. Iteration arc: v1 (raw-val) wrong on AddNIST → v2 (gap-penalised) net −4.14 vs EG due to noise-band coin-flips → v3 (asymmetric −6pp) net −10.11, K=4 noise floor is ~±8pp → v4 (3-probe shadow, A=multi-K4 drives, B/C log) net −4.32; **shadow data definitively says K=8 is the lever, not multi-trial**. Language reliably recovered (+3.72 Δmean) but EGNAS-internal variance on skip-probe datasets eats it. v5 path: swap driver A→C (multi-trial K=8). Embeds paired bars, σ-collapse, aggregated curves, per-seed plots.
- [[Tier 2a - Language Deep Dive]] — ✅ **Language excursion + literature review** (2026-05-28). 6-variation isolation experiment on Language alone (V0 baseline / V1 patience60 / V2 lr_low / V3 mixup / V4 larger / V5 combo) × 3 seeds. **Verdict: training-side ceiling is ~81.5 % raw, ~3.6 pp below the 85.20 % harness benchmark.** Mixup is the only training-side lever (+1.6 adj); patience/LR/capacity tweaks give ±0.4-0.8. **Published literature (Geada et al. arXiv 2404.02189): ResNeXt-50 hits 93.97 % on Language, PC-DARTS 90.12 %, best competition entry 89.71 %.** The lever the literature points to is **grouped convolutions (ResNeXt-style cardinality)** — none of which are in our current OP_VOCAB. Aug-off + Mixup on Language is **novel in the published literature**. Next experiment proposal: add `GroupConv3` to OP_VOCAB + ship Mixup-on-OFF in v5 P1 pipeline.
- **v5 (aborted partial)** — P1 v5 sweep killed 2026-05-28. Driver A→C + Mixup-on-OFF + GroupConv3 + ResNeXt added to OP_VOCAB. Probe mechanism worked correctly on all 7 s42 calls (driver C robust). New ops were picked on the predicted datasets (Chesseract → GroupConv3, Cryptic → ResNeXt, GameOfLife → ResNeXt). **But** the GameOfLife ResNeXt-at-L0 pick was a noisy-short-train false positive — 90.5 % best_val vs EG-s42's 99.9 % = **−9.4 pp regression**. Triggered reading the actual EGNAS paper + code to understand why. Code preserved at `baselines/tier2a_p1_augsafety_v5/`. See [[Tier 2a - Results]] §v5.
- **v6 (in flight)** — P1 v6 sweep launching tonight 2026-05-28. Q8-stabilised methodology informed by reading actual EGNAS paper. Bundles: probe C only (drop A+B), OP_VOCAB to 9 swappable ops (+Conv7, +DilatedConv3, +SEConv3, +SpatialSelfAttn2D), `MAX_SHORT_EPOCHS=12`, verification-of-accepts at 2× K, clock-extend (NAS doesn't deduct from trainer budget), zero-shot proxies (NASWOT/Synflow/SNIP/ZenScore) as shadow signals for post-sweep analysis. Sweep ~70 h. See [[Tier 2a - Results]] §v6 + [[EGNAS]] §"Our port vs upstream".

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
