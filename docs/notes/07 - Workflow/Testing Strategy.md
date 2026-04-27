> **Status:** drafting | **Topic:** Integration tests over public datasets, stress tests with tight clocks, low-VRAM simulation.
> **Related:** [[Mock Harness]], [[Dev Loop]], [[Failure Modes]], [[Reliability vs Peak]], [[Threats to Validity]], [[Submission Packaging]]

A passing [[Dev Loop]] run shows the submission survived *one* dataset under *generous* conditions. That is necessary but very far from sufficient. Reliability ([[Reliability vs Peak]]) requires that the submission survives **every** plausible eval-environment configuration. This note enumerates the test matrix.

## Tier 1 — integration tests across public datasets

**Goal**: confirm the submission completes on every input shape we have access to.

```bash
# Full sweep across all 11 downloaded public datasets, default time limit
python dev/mock_harness.py --score 2>&1 | tee dev/logs/full-$(date +%Y%m%d-%H%M).log
```

Expected: 11/11 datasets produce predictions, no `Failed=True` stats. Total score logged. Treat any failure here as a release-blocker.

Sub-cases worth running explicitly because they probe different code paths:

| Dataset | Shape | What it tests |
|---|---|---|
| Sudoku | tiny, 0.02 GB | Smallest path, tightest sanity check |
| GameOfLife | tiny, 0.03 GB, simple | Should be trivially solvable; flags Trainer regressions |
| AddNIST | 3×28×28, 20 classes | Standard MNIST-shape; baseline correctness |
| CIFARTile | 3×64×64, 4 classes | NAS-rewarding ([[CIFARTile]]); flags NAS regressions |
| GeoClassing | 3×60×60, 10 classes | Real images; flags augmentation issues |
| Voxel | 3D-flavoured, 2.24 GB | Unusual shape; flags hardcoded-shape bugs |
| Myofibre | 9.71 GB | Largest; flags dataloader/IO scaling |

If we don't have time for all 11, this 7-dataset subset covers the diversity.

## Tier 2 — stress tests with tight clocks

**Goal**: confirm the submission **finishes** when the clock is much shorter than expected. The Phase 3 budget is "an unknown amount of runtime" ([[Phases and Timeline]]) — assume it could be 10× tighter than Phase 2's 1 h.

```bash
# 3-minute clock per dataset (0.05 h)
python dev/mock_harness.py Sudoku GameOfLife AddNIST CIFARTile --time-limit 0.05 --score
```

Pass criteria:

1. No timeouts (`Skipping dataset — out of time` is a fail).
2. Total score still ≥ 0 — i.e. clears benchmark on at least the easy datasets.
3. NAS and Trainer respect their share of the [[Budget Allocation|budget split]].

Repeat with `--time-limit 0.01` (36 seconds) — even more aggressive. The submission should still produce *some* predictions, even if poor: a hardcoded fallback model in [[Submission Contract]] is more valuable than a clever model that times out. See [[Reliability vs Peak]].

## Tier 3 — low-VRAM / low-RAM simulation

**Goal**: confirm the submission adapts to the unknown eval-environment memory ([[Threats to Validity]]).

PyTorch 2.x supports `torch.cuda.set_per_process_memory_fraction(0.25)` to cap available VRAM. Or: use a smaller card if available. Or: force CPU with `CUDA_VISIBLE_DEVICES=`.

```bash
# Force CPU-only — slow, but flags any code that crashes without CUDA
CUDA_VISIBLE_DEVICES= python dev/mock_harness.py Sudoku --time-limit 0.5

# Cap GPU memory (requires PyTorch >= 1.8, fraction is of total VRAM)
python -c "
import torch
torch.cuda.set_per_process_memory_fraction(0.25, 0)
import sys; sys.argv = ['dev/mock_harness.py', 'AddNIST', '--time-limit', '0.5']
exec(open('dev/mock_harness.py').read())
"
```

Pass criteria:

1. CPU-only path completes (slowly, but without crashing).
2. Constrained-VRAM path still completes — Trainer must dynamically reduce batch size on `CUDA out of memory`. See [[Failure Modes]].

## Tier 4 — randomness / variance check

**Goal**: confirm the score is reproducible enough to trust the leaderboard ([[Threats to Validity]] §4).

```bash
# Run the same submission 3 times on a high-leverage dataset
for i in 1 2 3; do
    python dev/mock_harness.py CIFARTile --score 2>&1 | tee dev/logs/repro-$i.log
done
grep "adj=" dev/logs/repro-*.log
```

Pass criteria: per-dataset adjusted score variance across runs ≤ ~0.5. Higher variance means we're submitting noise — set seeds (`torch.manual_seed`, `np.random.seed`, `torch.cuda.manual_seed_all`) explicitly and verify they take effect.

## Tier 5 — submission-package verification

**Goal**: confirm the *zipped* submission imports and runs identically to the in-tree code. This catches packaging bugs ([[Submission Packaging]]) — a class of bug that doesn't show up in [[Dev Loop]].

```bash
cd submission && zip -r ../submission.zip . -x '*.pyc' '__pycache__/*' '.*'
mkdir -p /tmp/verify-submission && unzip -o ~/repo/submission.zip -d /tmp/verify-submission/
# Run mock harness against the unpacked archive
DATASETS_DIR=~/local/nas-comp-datasets PYTHONPATH=/tmp/verify-submission python dev/mock_harness.py Sudoku --time-limit 0.1
```

Pass criteria: identical behaviour to in-tree run.

## Test cadence

| Trigger | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|:-:|:-:|:-:|:-:|:-:|
| After any change | reduced (3 datasets) | optional | — | — | — |
| Before merging to main | full | full | full | optional | — |
| Before [[Phase 2 Submission Strategy\|Phase 2 attempt]] | full | full | full | full | full |

Tier 5 is non-negotiable before every Phase 2 attempt. It's the cheapest test (60 seconds) and catches the highest-impact bug class (packaging mistakes producing −10 across the board).

## What we are *not* testing

We can't test on the actual hidden Phase 2 / 3 datasets — that's the entire point of the competition. We *can* approximate by leaving out one public dataset, training on the other 10, and pretending the held-out one is "hidden" — see [[Threats to Validity]] §3 for why this matters and §6 for limits of the approximation.
