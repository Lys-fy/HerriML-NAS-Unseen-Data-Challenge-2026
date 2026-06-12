# EGNAS Full
> Formerly "Tier 2b - Results" (Tier-era naming). Naming map: [[README - Map and Lineage]].

> **Status:** ✅ complete (3-seed × 11-dataset, 2026-05-19 → 2026-05-20). **Verdict: the largest and σ-tightest mean lift the project has seen — Δmean +19.60 vs Tier-1, total σ collapses ×0.39 (8.37 → 3.28).** Two real per-dataset regressions (Myofibre, Chesseract); Language stays aug-bound (P1 territory). | **Topic:** The complete EGNAS-light method (macro depth/width + micro per-layer op hill-climb), 11-dataset × 3-seed paired vs the Tier-1 multi-seed baseline. The earlier macro-only diagnostic stage is summarised in §Why this run; this note is the single authoritative Tier-2b record.
> **Related:** [[EGNAS Macro Diagnostic]] (the 3-ds macro-only stage that motivated this), [[Tier 2b - EGNAS-light (macro)]] (plan), [[Multi-Seed Variance]] (Tier-1 baseline), [[Ensemble E1]] (variance attempt — superseded by this), [[Probe-Driven Self-Configuration]], [[Reliability vs Peak]], [[Implementation Results Overview]], [[Run State + Handoff]]

Run 2026-05-19 → 2026-05-20 on DGX Spark. Code: `baselines/egnas_full/` (only `nas.py` differs from Tier 1; `egnas_macro/` frozen as comparator). Driver `dev/run_egnas_full.sh`, aggregation `dev/aggregate_egnas_full.py`, plot `dev/plot_tier_results.py --egnas-full`. Internal adj (Sokoto/Sudoku → 70.95 via `dev/internal_benchmarks.py`); paired vs `dev/logs/multiseed/full_s*.log`.

![[egnas_full.png]]
*Panel A: per-dataset Tier-1 vs EGNAS-full means ± std, sorted by Δ desc; black dots = individual seeds. Panel B: σ collapse per dataset (×ratio annotated).*

![[egnas_full_curves.png]]
*Per-dataset learning curves (val acc %): thin grey = each of the 3 seeds; bold green = mean across seeds; shaded band = ±std. Captures both the convergence shape and the across-seed dispersion.*

## Why this run — the reasoning chain

1. **Macro-only diagnostic stage passed but small + lottery.** CIFARTile −0.515 → +0.167 (multi-seed, σ↓) — the ceiling no Tier-1 config could move — but the gain was *small* and the macro argmax-over-noisy-candidates was itself a Q8 arch-lottery (s2024 picked D=5 for AddNIST → −2.81). Full record in [[EGNAS Macro Diagnostic]]; plan rationale in [[Tier 2b - EGNAS-light (macro)]] §Update 2026-05-18.
2. **Honest pivot decision: judging a proven method by *half* of it is premature.** The fix wasn't a re-rank denoiser or a sideways jump to E2 — it was **finishing EGNAS** (add the micro phase). The micro hill-climb is *incumbent-kept*: it only accepts a per-layer op swap that beats the current best, so a single noisy short-train cannot hijack it — that is the **principled** fix for the macro lottery, not a bolted-on denoiser.
3. This run is the complete method, 11 datasets × 3 seeds, paired against the Tier-1 baseline we already had.

## Headline

| | Tier-1 (3-seed) | **EGNAS-full (3-seed)** | Δ |
|---|---:|---:|---:|
| **Total (11 ds, internal adj)** | **+8.39 ± 8.37** | **+27.99 ± 3.28** | **+19.60 mean, σ ×0.39** |
| Per-seed totals | (16.75, 8.40, 0.02) | **(29.44, 30.29, 24.23)** | s2024 *did not crater* — first time |

EGNAS-full beats Tier-1 on the mean by **+19.6 adj over 11 datasets** *and* the total σ collapses from 8.37 to 3.28 (×0.39). The "single run = lottery draw" problem the multi-seed sweep exposed is now *mostly solved at the submission level* — not by ensembling (E1 tried; partial) but by **better architectures themselves**. This is also the first run where the project's "seed-2024 craters" pattern broke (E1: +12.6/+12.5/**−0.0**; macro: +12.1/+12.5/**+5.5**; EGNAS-full: +29.4/+30.3/**+24.2**).

## Per-dataset analysis (internal adj, sorted by Δmean desc)

| dataset     |       Tier-1 |       EGNAS-full |     Δmean |                 σ T1→EG | EGNAS per-seed       | read                                                    |
| ----------- | -----------: | ---------------: | --------: | ----------------------: | -------------------- | ------------------------------------------------------- |
| GameOfLife  | +0.91 ± 0.50 | **+8.15 ± 2.28** | **+7.24** |     0.50 → 2.28 (×4.53) | +9.97, +8.88, +5.58  | near-saturation; σ-up is quality variance               |
| AddNIST     | −0.38 ± 5.45 | **+5.41 ± 0.50** | **+5.79** | 5.45 → 0.50 (**×0.09**) | +4.87, +5.86, +5.50  | the variance bomb tamed                                 |
| MultNIST    | −0.11 ± 3.00 | **+3.97 ± 0.38** | **+4.08** | 3.00 → 0.38 (**×0.13**) | +4.40, +3.70, +3.81  | sister-dataset, same story                              |
| Sudoku      | +5.94 ± 0.34 |     +7.07 ± 0.39 |     +1.13 |                   ×1.12 | +6.85, +7.51, +6.84  | solid lift                                              |
| Language    | −9.38 ± 1.08 |     −8.32 ± 1.56 |     +1.06 |                   ×1.45 | −6.93, −8.03, −10.00 | **still stuck — aug problem, not arch (P1)**            |
| Voxel       | +1.38 ± 0.32 |     +2.41 ± 0.05 |     +1.03 |     0.32 → 0.05 (×0.14) | +2.46, +2.37, +2.40  | clean + remarkably tight                                |
| CIFARTile   | −0.52 ± 0.52 |     +0.03 ± 0.34 |     +0.54 |                   ×0.65 | −0.31, +0.36, +0.04  | barely above benchmark — needs E2 for the real headroom |
| Gutenberg   | +0.92 ± 0.13 |     +1.01 ± 0.07 |     +0.09 |                   ×0.55 | +1.01, +0.94, +1.08  | ~tie, σ tightened                                       |
| GeoClassing | +7.68 ± 0.29 |     +7.65 ± 0.35 |     −0.03 |                   ×1.20 | +7.25, +7.93, +7.77  | ~tie                                                    |
| Chesseract  | +0.50 ± 0.19 | **+0.13 ± 0.01** | **−0.37** |     0.19 → 0.01 (×0.06) | +0.14, +0.12, +0.11  | regression, but σ collapsed and absolute drop tiny      |
| Myofibre    | +1.44 ± 0.55 | **+0.49 ± 0.69** | **−0.96** |                   ×1.25 | −0.27, +0.64, +1.09  | **genuine regression — same pattern as macro-only**     |

### What worked, with honest mechanism

- **AddNIST σ 5.45 → 0.50 (×0.09)** AND mean −0.38 → +5.41. The flagship variance bomb solved — *by architecture*, not ensembling. Similarly MultNIST σ ×0.13. These two datasets were the unbreakable σ floor we'd been failing on; full EGNAS dissolves them.
- **GameOfLife mean +0.91 → +8.15** (+7.24, the biggest lift). EGNAS found near-saturating architectures (e.g. raw 99.84 % on s42). σ went *up* (×4.53) but that's because seeds range "great → near-perfect" (+5.58 to +9.97); this is **quality variance, not failure variance** — every seed is a strong positive.
- **CIFARTile** crossed benchmark all 3 seeds (vs macro-only's +0.17 mean) — but only marginally. The thesis-predicted conv-ceiling holds: the real headroom (~+46 pp) needs the non-conv primitive (E2), not more conv search.
- **Voxel** is remarkable — σ collapsed to 0.05, mean +1.0 lift. Architecture genuinely matters on 3D-cosplaying-as-channels input.

### Honest regressions and stuck points

- **Myofibre −0.96 (persistent finding).** Macro-only also regressed Myofibre (−0.27 in the 3-ds diagnostic). It's a 128×128 medical RGB dataset — large, slow to train; EGNAS's search overhead eats training time on a dataset that just needs solid training. This is *the* signature failure mode: **architecture search isn't free; it can regress datasets that don't reward architecture search.** That argues directly for the **probe-driven thesis** (gate EGNAS by whether the dataset rewards it). The Dushatskiy 2022 *Heed the Noise* paper specifically studies small medical-image-segmentation datasets and finds variability is high there — supports the hypothesis.
- **Chesseract −0.37.** σ collapsed almost to zero (0.19 → 0.01) but mean dropped; a variance-for-mean trade on a dataset that was already near-neutral. Mild.
- **Language −8.32 (still stuck).** EGNAS doesn't touch augmentation — and Language's problem is that conservative aug destroys its encoded-character signal. This is **P1's job (Tier 2a)**, not EGNAS's. The 2026-05-15 multi-seed found aug-off recovers Language to ~−2.5; that lever remains the right one.

## σ collapse — the bigger methodological story

Total σ went **8.37 → 3.28 (×0.39)** with the seed-2024 cratering pattern broken — *without ensembling*. E1 ([[Ensemble E1]]) attempted to fix variance by averaging multiple models and only got ×0.75 at submission level (and regressed a dataset). **Full EGNAS collapses σ via a different mechanism: the search itself finds *more robust* architectures.** That this aligns with **Dushatskiy/Alderliesten/Bosman (2022) — *Heed the Noise in Performance Evaluations in NAS*** (arXiv 2202.02078; `docs/papers/Dushatskiy-2022_*.pdf`) is no coincidence: that paper documents the same per-architecture noise problem and shows that more reliable per-arch evaluation finds better architectures. Our EGNAS-full's micro phase is precisely such a more-reliable evaluation (incumbent-kept → robust to a single noisy short-train). The empirical confirmation here gives the methodology a citation backbone.

## Q8 arch-stability — present but downstream-dampened

| dataset | s42 D/W | s1337 D/W | s2024 D/W | micro-changed |
|---|---|---|---|---|
| AddNIST | 10/32 | 9/48 | 10/48 | 3/10, 3/9, 3/10 |
| CIFARTile | 7/32 | 10/40 | 8/24 | 0/7, 1/10, 1/8 |
| Chesseract | 4/48 | 5/48 | 3/24 | 0/4, 0/5, 0/3 |
| GameOfLife | 9/48 | 8/48 | 6/48 | 0/9, 1/8, 2/6 |
| GeoClassing | 7/32 | 10/48 | 9/48 | 1/7, 1/10, 1/9 |
| Gutenberg | 7/48 | 7/48 | 10/32 | 0/7, 0/7, 1/10 |
| Language | 5/24 | 8/48 | 5/16 | 2/5, 1/8, 0/5 |
| MultNIST | 12/48 | 9/48 | 11/40 | 3/12, 0/9, 2/11 |
| Myofibre | 5/24 | 6/32 | 11/48 | 0/5, 0/6, 0/11 |
| Sudoku | 11/24 | 11/32 | 7/48 | 3/11, 1/11, 3/7 |
| Voxel | 11/48 | 9/40 | 9/48 | 0/11, 0/9, 1/9 |

The macro lottery **didn't disappear** — depth picks still drift seed-to-seed (e.g. Myofibre D=5/6/**11**, MultNIST D=12/9/11). But the *downstream impact* of that lottery is now dramatically dampened: AddNIST/MultNIST/Voxel got σ ≤ 0.5 anyway, because full EGNAS finds *good* architectures across the range. The Q8 problem is partly solved by structure (micro's incumbent-kept hill-climb), partly absorbed by training (the surviving variance is small in adj). Worth a future targeted re-rank fix (the original proposal) if we ever need tighter Q8 — but not blocking now.

### Q8 stability under full EGNAS — full re-read (2026-05-20)

Pool-level analysis of all 13 datasets × 3 seeds (`dev/analyze_q8_under_egnas.py`, reads the `*_egnas_pool.json` files):

- **Cross-seed arch agreement: 0/13 depth, 1/13 width, 0/13 full ops.** Even with full EGNAS the 3 seeds pick a *different* (D, W, op-config) for **every** dataset. Macro-only's picks-drift story persists unchanged.
- **Therefore σ-collapse is not from arch convergence.** EGNAS-full doesn't make the 3 seeds *agree* on an architecture; it finds architectures where downstream training is **insensitive** to the macro/micro pick within a sensible region of the search space. The search space around the local optimum is **flat enough** that quite-different architectures train to similar final accuracy. That's the mechanism — implicit ensembling-by-search-flatness, not consensus.
- **Within-seed search-rank correlation (Spearman ρ between pool val_acc and candidate depth) is real but dataset-dependent:**
  - **Positive** (search prefers deeper, signal is doing work): AddNIST (+0.40 to +0.70), MultNIST (+0.27 to +0.70), Sudoku (+0.23 to +0.71), Voxel (+0.24 to +0.46), Windspeed (+0.22 to +0.50), CIFARTile (+0.28 to +0.40).
  - **Negative or near-zero** (search val is noise / smaller nets actually scored higher): Language (−0.22 to **−0.54**), Cryptic (−0.43 on s42), Myofibre (−0.02 to −0.31), Chesseract (mixed).
  - **The EGNAS paper's Spearman-0.91 (search-rank ↔ final-train-rank) does not hold at our budget.** Q8 answered "in the bad sense" stands; full EGNAS works *despite* the noisy ranker, by structure (incumbent-kept + search-space flatness), not because the ranker is good.
- **AddNIST s2024 — the macro lottery is broken at the macro level, not the micro level.** Macro-only diagnostic picked D=5 W=48 → AddNIST adj **−2.81**. Full EGNAS for the same seed picked **D=10 W=48** → adj **+5.498**. The micro phase doesn't change D/W (it only swaps per-layer ops); so the macro phase itself decided differently this time. Most likely cause: the macro budget allocation differs in full EGNAS (60 % of the search slice) vs the macro-only diagnostic (100 %), and the depth-grow stopping rule fired later. Net: full EGNAS dodged the lottery, but *not* via the principled fix we hypothesised (micro overriding macro).

**Implication for the thesis.** σ-collapse is a *flatness* phenomenon, not a *consensus* phenomenon. That reframes what E2 (non-conv primitives) might cost: if non-conv ops break the flat region near the optimum, the σ-collapse could degrade even while peak accuracy improves. Worth designing the E2 sweep to measure σ-on-baseline-datasets as a check.

## Comparison to E1 (which this supersedes for variance control)

| | Mechanism | Total σ collapse | Mean lift vs Tier-1 | Verdict |
|---|---|---:|---:|---|
| E1 ensemble | average K diverse models at predict | ×0.75 | +5–10 (3-ds) | mixed; conditional component, not a fix |
| **EGNAS-full** | search finds robust architectures | **×0.39** | **+19.60** (11-ds) | the actual fix |

E1 is **retired as the headline variance fix**. It remains a possibly-useful probe-gated component for residual-variance cases ([[Ensemble E1]] §carry-forward), but full EGNAS does the variance job better, on the right axis (architecture), without the GeoClassing-regression-from-under-trained-members flaw.

## Decision / carry-forward

1. **Full EGNAS is the shipping candidate** (subject to remaining checks). Pending the Windspeed+Cryptic follow-on (running) for the full 13-dataset picture.
2. **Probe-driven thesis directly vindicated.** Myofibre regression confirms architecture search isn't free — probe-gating ("only apply heavy search where the dataset rewards it") is the right framework. Tier 2a / P1 remains the natural next build.
3. **CIFARTile's real headroom needs E2** (non-conv / Deep-Sets / relational). Macro+micro conv search only gets +0.54.
4. **Language needs P1**, not EGNAS.
5. **Heed-the-Noise (Dushatskiy 2022) is now the methodology citation** — folded into the variance discipline references.

## Reproduce

```bash
rsync -a spark:'~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/logs/egnas_full*' dev/logs/
python3 dev/aggregate_egnas_full.py          # paired verdict + Q8 + _egnas_full_aggregate.json
python3 dev/plot_tier_results.py --egnas-full  # → egnas_full.png
```

## Windspeed + Cryptic (2025 datasets, just resolved) — ✅ complete

EGNAS-full ran for both 2025-cycle datasets across the same 3 seeds (`dev/run_egnas_newdatasets.sh`, finished 2026-05-20 15:18). Caveat: **no Tier-1 paired baseline** exists for these two (they weren't in the multi-seed sweep) — so the table below is absolute, not paired. The 13-dataset version of the curves figure above includes both panels (Cryptic top-right, Windspeed bottom-left).

| Dataset | benchmark | s42 | s1337 | s2024 | mean | std |
|---|---:|---:|---:|---:|---:|---:|
| **Windspeed** (raw acc %) | 13.49 (real) | 20.82 | 16.70 | 19.96 | **19.16** | 2.17 |
| Windspeed (adj) | — | +0.847 | +0.370 | +0.747 | **+0.655** | 0.252 |
| **Cryptic** (raw acc %) | 0.00 (placeholder) | 66.18 | 69.78 | 70.12 | **68.69** | 2.18 |
| Cryptic (adj vs 0%) | — | +6.618 | +6.978 | +7.012 | **+6.869** | 0.218 |
| 2-ds Total (shipped) | — | +7.464 | +7.348 | +7.759 | **+7.524** | 0.211 |

**Reading the numbers.**
- **Windspeed** — the real signal lives here (real ResNet-18 benchmark = 13.49% on 20-class wind-direction). EGNAS-full lands 5.7 pp above benchmark with σ 2.17 across seeds — modestly positive, no seed catastrophe; consistent with the rest of the σ-collapse story.
- **Cryptic** — binary task; raw 68.7% ± 2.18 is comfortably above chance (50%). The shipped `benchmark = 0.0` is a placeholder (mirrors Sudoku's known-broken metadata) → the +6.87 adj is *not* meaningful for comparison; treat raw as the truth pending an internal-benchmarks override (TBD). On a binary problem σ ≈ 2 pp is tight; no instability sign.
- **Stability.** σ on the 2-ds Total = 0.211 — the smallest of any tier so far, mirroring the σ ×0.39 collapse seen on the 11-ds main sweep. No s2024 cratering on these either.

**Open follow-ups (not blockers):** decide a Cryptic internal benchmark (likely Tier-0 raw as the analogue we used for Sudoku) so the adj column becomes interpretable; the existing aggregator (`dev/aggregate_egnas_full.py`) still runs the paired comparison on the 11 ds only — extending to 13 needs a "no-paired" branch but isn't urgent.
