<# Tier 0 — Results

> **Status:** current | **Topic:** Actual measured results of the [[Tier 0 - Hardcoded CNN|Tier 0 plan]]. Hardcoded CNN, zero search, full sweep at 2 h/dataset. Total adj **−0.355**.
> **Related:** [[Tier 0 - Hardcoded CNN]], [[Implementation Results Overview]], [[Tier 1 - Results]], [[Reliability vs Peak]], [[Score-Efficient Targeting]], [[CIFARTile]]

Run 2026-05-12, DGX Spark (NVIDIA GB10, torch 2.11.0+cu128), `--time-limit 2.0`, all 11 public datasets. Code: `baselines/tier0_fixed_cnn/`. Plan + design rationale: [[Tier 0 - Hardcoded CNN]].

## Scores

![[tier0_scores.png]]

| Dataset | Raw % | Benchmark % | **Adj** | Wall (s) | Budget % | Best epoch | Final train loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| GameOfLife | 96.53 | 47.53 | **+9.339** | 113 | 1.6 | 106 | 0.227 |
| Sudoku ¹ | 70.95 | 0.0 | +7.095 | 233 | 3.2 | 155 | 0.016 |
| GeoClassing | 90.23 | 80.33 | **+5.033** | 1100 | 15.3 | 91 | 0.001 |
| Voxel | 76.32 | 71.35 | +1.735 | 89 | 1.2 | 17 | 0.293 |
| Gutenberg | 45.52 | 40.98 | +0.769 | 96 | 1.3 | 26 | 0.417 |
| Chesseract | 57.18 | 57.83 | −0.154 | 28 | 0.4 | 3 | 0.094 |
| Language | 83.84 | 85.20 | −0.919 | 192 | 2.7 | 63 | 0.004 |
| CIFARTile | 40.89 | 47.01 | −1.155 | 348 | 4.8 | 2 | 0.077 |
| Myofibre | 85.40 | 87.93 | −2.097 | 945 | 13.1 | 4 | 0.010 |
| AddNIST | 77.13 | 89.85 | **−10.000** | 459 | 6.4 | 83 | 0.017 |
| MultNIST | 79.06 | 90.87 | **−10.000** | 143 | 2.0 | 18 | 0.091 |
| **Total** | | | **−0.355** | | **4.7** avg | | |

¹ Sudoku ships `benchmark: 0.0` (placeholder per [[Datasets Overview]]); the +7.095 is meaningless absolutely. Internal benchmark (= this 70.95 %) is the zero-point for [[Tier 1 - Results|Tier 1]] comparison.

## Learning curves

![[tier0_curves.png]]

The single most diagnostic feature is the **orange train-loss line collapsing to ~0.01** while blue val plateaus — visible on AddNIST, MultNIST, Language, CIFARTile, Myofibre. That's textbook memorization with zero regularization.

## What the data established

1. **The binding constraint is overfitting, not search.** With no augmentation / no label smoothing / no DropPath, the 300 K-param CNN memorizes the train set in 20–30 epochs. Train loss → ~0.01, val plateaus, patience early-stop fires correctly. ~95 % of the per-dataset budget goes unused on most datasets (4.7 % average utilization).

2. **The two −10s are not crashes.** AddNIST (77.13 %) and MultNIST (79.06 %) ran end-to-end and produced predictions — they just couldn't reach the ~90 % benchmark in the epochs they used. Pure regularization deficit. This is exactly what [[Tier 1 - Results|Tier 1]] fixed.

3. **CIFARTile is an architectural-ceiling case.** Best val_acc fired at **epoch 2** (a lucky-init artifact); the linear conv stack with global pooling can't learn the multi-instance "count distinct classes" task. No amount of training time helps — confirmed again under Tier 1. This is the canonical [[CIFARTile|+41 pp NAS-rewarding dataset]] and the motivation for [[Tier 2 - EGNAS-Light|Tier 2's macro search]].

4. **GeoClassing and Myofibre were the only budget-users** (15 % and 13 % of the 2 h clock). GeoClassing converted it to score (+5.0); Myofibre overfit at epoch 4 and the extra time was wasted oscillating.

## Comparison vs the 3-min smoke sweep

Total improved from −6.256 (3-min) → −0.355 (2 h), a +5.9 swing — almost entirely from **GeoClassing alone (+3.75)**, the only dataset where extra wall-clock translated to accuracy. Everywhere else the LR-schedule fix (cosine `T_max` projected from epoch 0 instead of hardcoded) gave small consistent gains.

## Why this motivated the Tier 1 re-scope

The plan's Tier 1 was originally "just random op search." Tier 0's data forced a re-scope ([[Tier 1 - Augmentation + Op Search]]): if every architecture overfits in 30 epochs, ranking them by short-train val_acc ranks "which memorizes fastest." The regularization had to come *first*. See [[Tier 1 - Results]] for what that produced.

## Acceptance criteria (from the plan)

| Criterion | Result |
|---|---|
| All 11 datasets complete, no `Failed=True` | ✓ |
| Adaptive downsampling correct on 9×9 … 128×128 | ✓ |
| Test-loader assertions pass | ✓ |
| Clock-aware early-exit verified | ✓ |
| Cross-verification on x86 eval-pinned env | ✗ pending (see [[docs/environment.md]]) |

Tier 0 is "done" as a validated baseline and as the in-process fallback later tiers return from `NAS.search()` on exception.
