# Aug + Op Search
> Formerly "Tier 1 - Results" (Tier-era naming). Naming map: [[README - Map and Lineage]].

> **Status:** current | **Topic:** Measured results of the [[Tier 1 - Augmentation + Op Search|Tier 1 plan]] + the single-feature ablation that quantifies each component. Total adj **+24.644** (Tier 0 was −0.355 → **+25.0 swing**). ⚠ **+24.644 is one seed and a favourable tail draw** — the trustworthy figure is ≈ **+11 ± 8 shipped / +8 ± 8 internal** ([[Multi-Seed Variance]], 3 seeds).
> **Related:** [[Multi-Seed Variance]], [[Tier 1 - Augmentation + Op Search]], [[Fixed CNN]], [[Implementation Results Overview]], [[Tier 2 - EGNAS-Light]], [[Threats to Validity]], [[Reliability vs Peak]]

Run 2026-05-14, DGX Spark (NVIDIA GB10, torch 2.11.0+cu128), `--time-limit 2.0`, all 11 public datasets. Code: `baselines/aug_op_search/`. Plan: [[Tier 1 - Augmentation + Op Search]].

## Full-sweep scores

![[aug_op_search_scores.png]]

| Dataset | T0 raw | **T1 raw** | Δ raw | T0 adj | **T1 adj** | Δ adj |
|---|---:|---:|---:|---:|---:|---:|
| AddNIST | 77.13 | **91.74** | +14.61 | −10.000 | **+1.862** | **+11.862** ★ |
| MultNIST | 79.06 | **90.82** | +11.76 | −10.000 | −0.055 | **+9.945** ★ |
| Myofibre | 85.40 | **90.46** | +5.06 | −2.097 | +2.097 | **+4.194** ★ |
| GeoClassing | 90.23 | **94.49** | +4.26 | +5.033 | +7.200 | +2.167 ★ |
| Sudoku ¹ | 70.95 | 87.96 | +17.01 | +7.095 | +8.796 | +1.701 |
| Chesseract | 57.18 | 61.05 | +3.87 | −0.154 | +0.764 | +0.918 |
| CIFARTile | 40.89 | 44.37 | +3.48 | −1.155 | −0.498 | +0.657 |
| Gutenberg | 45.52 | 46.52 | +1.00 | +0.769 | +0.938 | +0.169 |
| Voxel | 76.32 | 76.01 | −0.31 | +1.735 | +1.627 | −0.108 |
| GameOfLife | 96.53 | 92.42 | −4.11 | +9.339 | +8.555 | −0.784 |
| Language | 83.84 | **75.37** | **−8.47** | −0.919 | **−6.642** | **−5.723** ✗ |
| **Total** | | | | **−0.355** | **+24.644** | **+24.999** |

¹ Sudoku ships placeholder `benchmark=0`. **Internal-benchmark** scoring (Sudoku = Tier 0's 70.95 %, via `dev/internal_benchmarks.py`): Tier 0 internal total **−7.450**, Tier 1 internal total **+21.700** — a +29.2 internal swing. The shipped-number total (+24.644) is what the real grader logic produces.

9 of 11 datasets improved. The +25 swing decomposes into one mechanism (regularization stops memorization → val keeps climbing) on the high-benchmark datasets, minus one big regression (Language).

## Learning curves

![[aug_op_search_curves.png]]

Versus [[Fixed CNN|Tier 0's curves]]: orange train-loss now stays elevated (~0.5–1.0) instead of crashing to 0.01, and blue/purple val curves keep climbing for the full budget. Purple dashed = EMA val, which leads live val after ~10 epochs on most datasets. CIFARTile is the exception — still the early-overfit shape, confirming its architectural ceiling.

## The Language regression

raw 83.84 → 75.37 (−8.5 pp), the only large loss. Language is `1×24×24` *encoded character grids*; the label depends on character-at-position. From metadata it is **indistinguishable** from a 24×24 grayscale image-classification dataset, so the conservative-but-still-spatial policy gave it `crop_pad=2 + cutout=4` — both destroy the encoding. This is the empirical proof that **metadata-only heuristics cannot reliably classify unseen data** ([[Threats to Validity]] §2/§3). The conservatism trade-off bought +21.8 (AddNIST+MultNIST) and cost 5.7 (Language) — strongly net positive, but Language is the canary for hidden Phase-2 datasets of similar structure.

## Single-feature ablation

To attribute the +25 swing to specific components, each Tier 1 feature was disabled in isolation on the 3 representative datasets (AddNIST = high-benchmark winner, CIFARTile = architectural ceiling, Language = regression). Run 2026-05-14, same hardware/budget.

![[aug_op_search_ablation.png]]

| Dataset | Tier 1 full | search-off | ema-off | aug-off |
|---|---:|---:|---:|---:|
| AddNIST | **+1.862** | −1.074 | −0.768 | **−10.000** |
| CIFARTile | **−0.498** | −0.794 | −0.796 | −1.668 |
| Language | −6.642 | −9.304 | −6.939 | **−0.993** |
| **3-ds subtotal** | **−5.278** | −11.172 | −8.504 | −12.661 |
| **Δ vs full** | — | **−5.89** | **−3.23** | **−7.38** |

### Per-feature value (Tier 1 full minus ablated; + = the feature helped)

| Dataset | search | EMA | augmentation |
|---|---:|---:|---:|
| AddNIST | +2.94 | +2.63 | **+11.86** |
| CIFARTile | +0.30 | +0.30 | +1.17 |
| Language | +2.66 | +0.30 | **−5.65** |

**Random op-type search: +5.89 across the 3 datasets, all positive, no downside.** And search-off *trains longer* (no 30 % search budget — Adaline 1760 s vs 2189 s with search) yet still loses, so the searched architectures genuinely beat all-Conv3. Validates escalating to [[Tier 2 - EGNAS-Light|EGNAS macro-search]].

**EMA: +3.23, consistent positive, no downside.** Cheap (~10 lines). Keep unconditionally.

**Augmentation: double-edged, the central finding.** +11.86 on AddNIST (it's *the* thing fixing it — aug-off ≈ Tier 0's 77.13 %), but −5.65 on Language (aug-off ≈ Tier 0's 83.84 %). Augmentation is simultaneously the highest-value and highest-risk component, and the risk is dataset-dependent in a way metadata can't predict.

> **⚠ Single-seed caveat — now resolved by [[Multi-Seed Variance]] (3 seeds).** Every number in this note is **one seed each**. The 3-seed sweep confirms: (1) the **+24.644 total was a favourable tail draw** — honest Tier 1 ≈ +11 ± 8 shipped / +8 ± 8 internal (one seed of the *same recipe* scored +0.02); (2) AddNIST/MultNIST's positive adj here is **not seed-stable** (AddNIST −0.4 ± 5.4 — they were never "fixed", just lucky this run); (3) what **is** seed-stable is the aug double-edge *direction* (Language aug-cost −6.82 ± 1.88, negative every seed; aug-off a dead −10 floor on AddNIST/MultNIST) and the tight wins (GeoClassing +7.68 ± 0.29, Sudoku +5.94 ± 0.34). The qualitative Tier-0→Tier-1 story (escaping the −10 floors) holds; the precise per-feature deltas and the +4.9 oracle-aug / ~+30.3 projections below are single-seed and **superseded** — read them with the [[Multi-Seed Variance]] spread.

## What this tells us

1. **No dead code in Tier 1** — search, EMA, and aug all earn their place; the only problem is aug's per-dataset double-edge.

2. **The runtime aug-safety probe is empirically proven, not speculative.** An oracle picking aug-on/off per dataset would score: AddNIST on (+1.862), CIFARTile on (−0.498), Language off (−0.993) → subtotal **−0.371** vs Tier 1 full's −5.278 = **+4.9 adj on 3 datasets**, almost all from un-breaking Language. Projected across all 11: Tier 1's +24.644 → **~+30.3** if Language is the only badly-broken dataset (Gutenberg got cutout-only and may be mildly affected too).

3. **CIFARTile's ceiling is confirmed under every config** (best −0.498). Training-side levers are exhausted there; only macro architecture search moves it.

4. **Reliability-before-peak is validated** ([[Reliability vs Peak]]): the +25 came from not under/over-fitting, not from peak-chasing.

## Implications for [[Tier 2 - EGNAS-Light|Tier 2]]

Priority order, now data-backed:

1. **Runtime aug-safety probe** (#1) — at `NAS.search()` start, train two tiny models (aug-on vs aug-off) ~2 epochs, pick the higher val_acc, use that policy. The ablation proves this recovers ~+5 adj without sacrificing the AddNIST-class gains. Metadata-blind → transfers to hidden datasets.
2. **EGNAS macro-search** — replaces the random op-search (already worth +5.89) with depth-grow/width-prune; directly attacks the CIFARTile ceiling.
3. **Per-regime regularization** — GameOfLife regressed −0.78 from universal DropPath/label-smoothing on a memorization-friendly task; scale reg strength from `metadata['benchmark']` + `num_classes`.
4. **Keep search + EMA as-is.** Unconditionally positive; no further ablation needed.

Tier 2 is additive (probe + EGNAS macro + per-regime reg), not a rewrite. See [[Tier 2 - EGNAS-Light]] for the plan; this note is its empirical justification.
