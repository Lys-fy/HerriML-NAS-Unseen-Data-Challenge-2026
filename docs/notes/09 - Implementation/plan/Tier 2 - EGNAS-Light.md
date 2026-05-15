# Tier 2 — EGNAS-Light

> **Status:** planned | **Topic:** Sequential macro-micro hill-climbing NAS — the [[2025 - EVAI|2025-winning method]] scaled to fit a Phase-2 ~1 h per-dataset clock. *This is the version that ships to Phase 2 unless something better surfaces.*
> **Related:** [[Implementation Plan Overview]], [[Tier 1 - Augmentation + Op Search]], [[Tier 3 - EGNAS + Reliability]], [[EGNAS]], [[2025 - EVAI]], [[Budget Allocation]]

## Goal

Implement the [[EGNAS]] algorithm — macro search (depth-grow then width-prune) followed by micro search (per-layer op + kernel hill-climb) — scaled down so the full sequence fits in a Phase-2 budget. Inherits [[Tier 1 - Augmentation + Op Search]]'s search-loop plumbing.

This is the **shipping candidate**. The 2025 winner ([[2025 - EVAI]], EGNAS by Siddiqui et al.) used precisely this algorithm; the [[Approaches Overview|decision tree]] picks it as the default for our setup.

## Why EGNAS specifically

From [[Approaches Overview]] and [[EGNAS]]:

1. **No predictor.** Predictor-guided methods ([[Predictor-Guided Evolutionary]]) leak priors from their training corpus and lose ~3 pp on novel data like [[Cryptic]] (the 2025 deciding dataset).
2. **No supernet.** Weight-sharing one-shot ([[Weight-Sharing One-Shot]]) suffers from search-vs-eval ranking mismatch and needs offline pretraining we haven't done.
3. **Macro-flexible.** [[DARTS Family|DARTS cells]] don't search depth/width, which is exactly what [[CIFARTile]]-shape datasets reward (+41 pp NAS-vs-CNN gain).
4. **Bounded candidate count (~25).** Each candidate gets a short real training; rankings aren't proxy-polluted.

## The algorithm (paraphrased from Siddiqui et al. 2025, Alg. 1)

```
Init:  D = D_min (e.g. 4), W = W_max (e.g. 64), O = SepConv3, downsample plan from Tier 0
MACRO
  1. Depth-grow:    while val_acc improves by ≥ τ_grow, append a block.
                    Train each candidate for E(D) epochs (architecture-aware schedule).
                    Stop on D_max or 3 consecutive non-improvements.
  2. Width-prune:   while val_acc retained, shrink width by W_step (e.g. -8 channels).
                    Train each candidate for E(D, prune) epochs.
                    Stop on W_min or 3 strikes.
MICRO  (re-baseline at E_min epochs with discovered D_f, W_f)
  3. Op-replace:    per layer, try O ∈ {Conv3, SepConv3}, equalising param count.
                    Accept if val_acc improves.
  4. Kernel-update: per layer, try K = 5. Same param-equalising trick.
Return (D_f, W_f, O_f, K_f)   → Trainer full-trains it
```

The **architecture-aware training schedule** is the load-bearing efficiency trick: bigger candidates train longer. From the [[EGNAS]] paper, ranking by params with ascending epochs gets Spearman 0.91 vs final accuracy, far better than fixed-epoch ranking (0.65). We inherit this exact schedule.

## Scaling for the Phase-2 budget

EGNAS's paper reports 0.24 GPU-day search cost on CIFAR-10 (~5.8 h on a Quadro RTX 8000). The Phase-2 budget is ~1 h per dataset. Cuts to fit:

| Knob | Paper | T2 default | Rationale |
|------|-------|-----------|-----------|
| `D_max` | 100 | 20 | Limits macro search to ~16 depth-grow candidates |
| `W_max` | 256 | 64 | "Tiny" setting in the paper, still strong on EMNIST/KMNIST |
| `E_min` (micro) | 2 | 2 | Unchanged |
| Drop op-replace? | no | optional | If clock tight, skip |
| Drop kernel-update? | no | optional | If clock tight, skip |
| Final retrain | 50 epochs | clock-bounded | Trainer.train() runs to safety margin, not fixed count |

[[Budget Allocation|Budget split]]: NAS gets 30–50 % depending on regime (see [[Search vs Train Split]]); Trainer gets the rest. The schedule auto-adapts via `clock.check()`.

## Adapting EGNAS to heterogeneous inputs

EGNAS assumes spatial input ≥ ~24 px with two downsamples at 1/3 and 2/3 of depth. Our [[Datasets Overview|inputs range 9×9 to 128×128]]. Reusing Tier 0's `plan_downsamples()` instead of EGNAS's fixed positions:

- Sudoku 9×9: no downsamples (max_ds=1, but placed late so the 4×4 GAP target works).
- AddNIST 28×28: 2 downsamples.
- CIFARTile / GeoClassing 64×64: 4 downsamples.
- Myofibre 128×128: 5 downsamples.

The depth-grow loop appends blocks but downsample positions are *recomputed* on each grow step so they stay correctly spread.

## Initialization (from paper Table 2)

Start `SepConv3` everywhere with `D_min` blocks at `W_max=64`. Paper finds this start dominates the other three init strategies.

## Search-loop hygiene (inherited from [[Tier 1 - Augmentation + Op Search]])

- `clock.check()` between every candidate. If next candidate's projected train time exceeds remaining search budget, terminate phase early with current-best.
- `try/except` around each candidate. OOM / NaN-loss → silently drop, continue.
- Best-arch-so-far recorded across phases (depth-grow's best should improve into width-prune's best, etc.).
- If the entire search budget exhausts before micro phase, return the current macro best — never fail.

## Acceptance criteria

- [ ] Macro + micro both terminate within their fraction of the search budget.
- [ ] Final architecture's depth is *not* always `D_min` (proves depth-grow runs).
- [ ] Final architecture's width is *not* always `W_max` (proves width-prune runs).
- [ ] On [[CIFARTile]] at `--time-limit 1.0`, scores ≥ +6.0 adj — this is the canonical NAS-rewarding dataset.
- [ ] On [[GameOfLife]] at `--time-limit 0.05`, scores ≥ +9.0 — should be near-saturated.
- [ ] An induced exception in macro phase doesn't prevent micro phase / final retrain.

## What Tier 2 deliberately doesn't do

- **No reliability layer.** Out-of-memory still raises; predictor failures still propagate. [[Tier 3 - EGNAS + Reliability]] wraps T2 in defensive code.
- **No per-regime budget classifier.** Uses the [[Budget Allocation|default 30/65/5 split]] regardless of `metadata['benchmark']`. T3 adds the classifier.
- **No ensembling / TTA.** Also T3.
- **No pre-filter via zero-cost proxies.** [[Branch B1 - Zero-Cost Proxies]] could plug in here later.

## Open questions

- Per-dataset `D_min` / `W_max` tuning. Should small inputs (Sudoku) start at smaller `W_max` to avoid an overfit ceiling? Paper doesn't ablate; we'd need our own.
- Whether the architecture-aware schedule transfers to non-image tasks ([[Voxel]], [[Myofibre]]). [[EGNAS]] §"Open questions" flags this.
- Whether to keep T1's random search as a "panic mode" fallback if EGNAS macro phase fails to find anything better than initialization within first 5 candidates.

## Repo layout

```
baselines/tier2_egnas_light/
  data_processor.py     # ≈ identical to Tier 1
  nas.py                # macro (depth-grow + width-prune) + micro (op/kernel hill-climb)
  trainer.py            # ≈ identical to Tier 0/1
  helpers.py            # T1 helpers + EGNAS-specific schedulers and stop criteria
```

## References

- [[EGNAS]] — full method note.
- [[2025 - EVAI]] — team that won 2025 with this algorithm.
- Public code: [siddikui/Efficient-Macro-Micro-NAS](https://github.com/siddikui/Efficient-Macro-Micro-NAS).
- Paper: Siddiqui, Kyrkou, Theocharides. *Efficient Global Neural Architecture Search*. SN Computer Science 6:282 (2025), [doi:10.1007/s42979-025-03790-z](https://doi.org/10.1007/s42979-025-03790-z).
