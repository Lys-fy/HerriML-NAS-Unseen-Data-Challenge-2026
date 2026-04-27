# Overview

> **Status:** current | **Topic:** Decision tree across NAS method families and which to use under what constraint.
> **Related:** [[EGNAS]], [[DARTS Family]], [[Zero-Cost Proxies]], [[Multi-Fidelity]], [[Weight-Sharing One-Shot]], [[Predictor-Guided Evolutionary]], [[Search vs Train Split]], [[Budget Allocation]]

Six NAS method families are in scope for this competition. Picking between them is a function of (1) per-dataset wall-clock budget, (2) cost per candidate evaluation, (3) search-space size, (4) how unfamiliar the dataset is. The 2025 winner [[EGNAS]] picked greedy global search; the 2025 runners-up picked predictors. Both work — for different reasons.

## Family summary

| Family | Search style | Eval cost / cand. | Strength | Weakness |
|---|---|---|---|---|
| **Greedy global** ([[EGNAS]]) | sequential hill-climb over depth → width → ops | short real training | adapts macro shape, no ranking proxies | sequential, hard to parallelise |
| **Predictor-guided evolutionary** ([[Predictor-Guided Evolutionary]]) | EA over arch encodings, surrogate + top-k full train | ~free (predictor) + few full trains | scales, reuses across queries | predictor must transfer to new data |
| **DARTS family** ([[DARTS Family]]) | gradient over relaxed supercell | one supernet train | fast, well-known | brittle, often skip-connection collapse, cell-only |
| **Weight-sharing one-shot** ([[Weight-Sharing One-Shot]]) | train supernet, sample sub-nets | one supernet train + sampling | reuses weights, hardware-aware | rankings can mismatch standalone training |
| **Zero-cost proxies** ([[Zero-Cost Proxies]]) | rank archs by Synflow / NASWOT / Zen scores | seconds | filter huge spaces cheaply | data-oblivious or fragile on novel data |
| **Multi-fidelity** ([[Multi-Fidelity]]) | bracketed early-stopping (Hyperband / ASHA / BOHB) | progressive | great budget allocation | needs many parallel trials |

These are not mutually exclusive — most strong submissions combine them (e.g. zero-cost proxy → top-k → multi-fidelity training).

## Decision tree (for the [[Clock and Time Budget]])

```
Budget per dataset T:

T < 5 min        → don't search. Train a fixed strong CNN baseline.
                   (See [[Reliability vs Peak]].)

T ≈ 30 min       → zero-cost proxy filter + multi-fidelity (ASHA-style)
   small space      ranking of 5–10 candidates. Or, EGNAS macro-only with
                    aggressive E_min cuts.

T ≈ 30 min       → EGNAS (depth-grow only, skip micro phase) at small W_max.
   large space      Or supernet-OFA-style sample → top-k re-train, if a
                    pretrained supernet from sibling dataset is available.

T ≈ 1 h          → EGNAS full pipeline at small W_max ≤ 32.
   (Phase 2)        Or predictor-guided EA if a fitted predictor exists.

T ≥ 4 h          → EGNAS at W_max=64 (paper's tiny setting).
   (Phase 3?)       Or DARTS-family on a cell + EGNAS for depth/width.
```

## Per-dataset constraint

| Constraint | Method choice |
|---|---|
| Tiny dataset (Sudoku, GameOfLife) | EGNAS-tiny or fixed CNN — search overhead dominates |
| 4D images, novel structure (CIFARTile, Voxel) | EGNAS or predictor-EA — DARTS cells underperform here, see [[CIFARTile]] |
| Image classification, conventional (AddNIST, MultNIST) | DARTS or EGNAS work; Predictor-EA cheapest if cached |
| Medical / binary (Myofibre) | EGNAS — depth/width matters more than micro ops |
| Near-noise-floor (Gutenberg) | don't search — saturate at benchmark, see [[Gutenberg]] |
| Big input or rare task (GeoClassing satellite) | One-shot supernet sampling if pretrained, else EGNAS-mobile |

## Why [[EGNAS]] is the default for this team

- It does not depend on a working accuracy predictor (which the 2nd-place team's Cryptic gap suggests is the failure mode for unseen data).
- It searches macro shape, which CIFARTile-class problems need.
- The total candidate count is ~25 — every one is fully scheduled, so a single OOM or NaN is recoverable, see [[Failure Modes]].
- Public reference implementation, paper-validated defaults.
- Disadvantage: sequential. Can't trivially parallelise across GPUs the way DARTS or supernet sampling can.

## When NOT to use [[EGNAS]]

- A reliable per-dataset budget below ~10 minutes — depth-grow phase alone needs ≥ ~15 candidates × short train.
- A pretrained OFA/SPOS supernet for the dataset family already exists — switch to one-shot sampling, much cheaper at inference time.
- Phase 2 smoke fails on a dataset twice — fall back to a fixed ResNet baseline with ASHA over LR/WD; the failure cliff is −10, see [[Scoring]].

## Common combinations seen in literature

- **[[Zero-Cost Proxies]] → [[Multi-Fidelity]]**: prune 10⁴ candidates to 10² via Synflow, then ASHA on the 10² with progressive epochs. Common in lightweight-NAS papers.
- **[[Predictor-Guided Evolutionary]] + [[Weight-Sharing One-Shot]]**: supernet provides weights; predictor filters; EA mutates. The OFA recipe.
- **[[DARTS Family]] cell + manual macro**: classic 2018–2020 setup. Beaten by [[EGNAS]] on CIFAR-10 and dominated on macro-sensitive datasets.

For per-dataset selection rules tied to scoring, see [[Score-Efficient Targeting]]. For budget-vs-accuracy analysis, see [[Budget Allocation]].
