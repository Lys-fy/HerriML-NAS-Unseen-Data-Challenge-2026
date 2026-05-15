# Tier 0 — Hardcoded CNN

> **Status:** current — implemented in `baselines/tier0_fixed_cnn/` | **Topic:** Zero-search fixed CNN. Validates harness, dataloaders, clock, and shape adaptation across all 11 public datasets.
> **Related:** [[Implementation Plan Overview]], [[Tier 1 - Augmentation + Op Search]], [[Submission Contract]], [[Failure Modes]], [[Mock Harness]]

## Goal

The first tier whose job is **not to win on accuracy** but to prove the four-file submission contract runs end-to-end on every public dataset shape we have access to. Doubles as the in-process fallback that future tiers can return from `NAS.search()` on exception (per [[Failure Modes]] §"Defensive patterns").

## What's in it

Lives under `baselines/tier0_fixed_cnn/` as a self-contained 4-file package:

```
baselines/tier0_fixed_cnn/
  data_processor.py    # numpy → tensor → normalised DataLoaders
  nas.py               # returns a single fixed CNN, no search
  trainer.py           # SGD+cosine LR, clock-aware loop, best-val checkpoint
  helpers.py           # adaptive-downsample planner + ConvBlock + build_fixed_cnn
```

Runnable through the [[Mock Harness]] with the new `--submission-dir` flag:

```bash
python dev/mock_harness.py --submission-dir baselines/tier0_fixed_cnn Sudoku --time-limit 0.05 --score
```

## Model

```
Input (in_channels, H, W)
   ConvBlock(in_channels → 32, stride=1)           ← stem
   [6 × ConvBlock]                                 ← width and stride decided
                                                     by plan_downsamples()
   AdaptiveAvgPool2d(1) → Flatten → Linear(C → num_classes)
```

Each `ConvBlock` is `Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → ReLU(inplace)`. Width doubles on every downsampling block.

### Adaptive downsampling

The one piece of non-trivial logic. The competition's inputs span 9×9 ([[Sudoku]]) to 128×128 ([[Myofibre]]). `plan_downsamples(H, W, n_blocks, min_spatial=4)` places exactly `floor(log2(min(H,W) / min_spatial))` stride-2 ops evenly across the 6 blocks. So:

| Input | max_ds | Final spatial |
|---|---|---|
| 9×9 (Sudoku) | 1 | 4×4 |
| 28×28 (AddNIST) | 2 | 7×7 |
| 64×64 (CIFARTile) | 4 | 4×4 |
| 128×128 (Myofibre) | 5 | 4×4 |

GAP at the end always sees a non-trivial feature map; no spatial collapse anywhere.

### Shape robustness

`data_processor.py:_to_nchw` promotes 3D arrays to 4D (inserts a singleton channel axis). Discovered necessary on [[Sudoku]] — metadata advertises `(N, 1, 9, 9)` but the npy is actually `(N, 9, 9)`. Same shim covers any future dataset where the channel axis was elided.

`nas.py` accepts both 3-tuple and 4-tuple `metadata['input_shape']`. [[Voxel]]'s `(N, 20, 20, 20)` is interpreted as `in_channels=20` for a plain `Conv2d` — the strategy notes' preferred fallback (see [[Voxel]]).

## Training

`Trainer.train()` runs SGD (lr=0.025, momentum=0.9, wd=3e-4, nesterov) with a cosine LR schedule capped at `MAX_EPOCHS=100`. Reads `clock.check()` at every batch boundary and bails when remaining time drops below `SAFETY_MARGIN_SECONDS=60`. The 60-second margin reserves time for `predict()` plus grader I/O plus the 1-minute grace window from [[Clock and Time Budget]].

Best validation accuracy seen is checkpointed in-memory (`copy.deepcopy(model.state_dict())`) and restored before `predict()` — an aggressive early-stop still returns a usable model.

## What it deliberately doesn't do

- **No search.** Identical architecture across datasets (only dims change).
- **No augmentation.** Per-dataset augmentation policies belong to later tiers (cutout breaks [[CIFARTile]]'s tile boundary, flip may invalidate orientation-sensitive datasets like [[Myofibre]]).
- **No mixed precision.** `torch.amp` API has churned across PyTorch versions; the eval-server pin is unknown ([[Threats to Validity]] §1).
- **No dynamic batch size.** Fixed at 128. [[Tier 3 - EGNAS + Reliability]] adds OOM-safe halving.
- **No ensemble / TTA / SWA.** Also [[Tier 3 - EGNAS + Reliability]] / [[Tier 4 - Phase-2 Polish]].
- **No fallback inside `NAS.search()`.** Tier 0 *is* the fallback.

## Smoke result (April 2026)

`python dev/mock_harness.py --submission-dir baselines/tier0_fixed_cnn Sudoku --time-limit 0.05 --score`

```
Sokoto: raw=67.880%  benchmark= 0.000%  adj= +6.788  runtime=107.5s
```

100 epochs in 107s on the [[reference_dgx_spark|DGX Spark GB10]]. Hit `MAX_EPOCHS` not the clock — there's more training time available on a real Phase-2 budget. The `+6.788` is optimistic because [[Sudoku]] ships `benchmark: 0.0` which the [[Datasets Overview]] flags as a placeholder.

Full sweep across the 11 datasets at `--time-limit 0.05` recorded in `dev/logs/smoke-sweep.log`.

## Acceptance criteria

Tier 0 is "done" when:

- [x] All 11 public datasets complete without `Failed=True` in their stats pkl.
- [x] Adaptive downsampling produces non-empty downsample plans on inputs ≥ 16×16 and empty plans on inputs ≤ 9×9.
- [x] Test-loader assertions pass ([[Submission Contract]] §asserted constraints).
- [x] Clock-aware early-exit verified at `--time-limit 0.05`.
- [ ] *(pending)* Cross-verification on x86 with eval-pinned env (see [[docs/environment.md]]).

## Why this is the right T0

The strategy notes ([[Reliability vs Peak]]) argue the first 9 points of expected score come from "not crashing." Tier 0 is the minimum viable implementation of that property. Every later tier inherits its harness assumptions; if they fail end-to-end, T0 lets us bisect — does the failure repro on the hardcoded baseline? If yes, harness bug. If no, search/training-loop bug.

## What feeds back into T1+

The reusable pieces Tier 1+ will import (likely vendored into their own `helpers.py`):

- `plan_downsamples()` — adaptive downsampling logic.
- `ConvBlock` — the elementary op (T1 will add `SepConv`, kernel-5 variants, etc.).
- `build_fixed_cnn()` — the *structure*; T1 swaps the op-per-position decision.
- `per_channel_stats()` and `_to_nchw()` — data normalisation and shape coercion.
- Best-val checkpoint pattern and the safety-margin clock loop in `Trainer`.
