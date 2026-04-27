> **Status:** drafting | **Topic:** Edit submission/, run mock_harness, score, iterate.
> **Related:** [[Mock Harness]], [[Testing Strategy]], [[Submission Packaging]], [[Submission Contract]], [[Failure Modes]]

The inner-loop workflow for any change to `submission/`. Mirrors what the organisers' grader does, so anything passing here has a high chance of passing on their servers ([[Mock Harness]]).

## One-time setup

```bash
# Datasets — idempotent figshare downloader, ~20 GB total
DATASETS_DIR=~/local/nas-comp-datasets ./scripts/download_datasets.sh

# Symlink into repo (datasets/ is gitignored — keep it out-of-tree to avoid Syncthing replication)
ln -s ~/local/nas-comp-datasets datasets

# Python env — match the starter kit pins where possible
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy scikit-learn
```

The downloader covers 11 of 13 public datasets. Cryptic and Windspeed have broken DOIs in the starter kit — see `scripts/download_datasets.sh` header and [[Cryptic]] / [[Windspeed]].

## The loop

```bash
# 1. Edit one of: submission/data_processor.py, submission/nas.py, submission/trainer.py, submission/helpers.py
$EDITOR submission/nas.py

# 2. Sanity-check on the smallest dataset (Sudoku, 0.02 GB) with a tight clock
python dev/mock_harness.py Sudoku --time-limit 0.05

# 3. If it passes, expand to 2-3 datasets covering different shapes
python dev/mock_harness.py Sudoku GameOfLife AddNIST --time-limit 0.1 --score

# 4. Full sweep before committing to a submission
python dev/mock_harness.py --score
```

Predictions go to `dev/predictions/<codename>.npy`; per-dataset stats (runtime, params, fail flag) go to `dev/predictions/<codename>_stats.pkl`. The `--score` flag computes per-dataset adjusted score using local `test_y.npy` (the real grader strips `test_y.npy` and scores via its own `evaluation/score.py`).

## What "passes" means

`mock_harness.py` mirrors `evaluation/main.py` ([[Pipeline Flow]]): it imports `submission/` via `sys.path`, instantiates `DataProcessor` → `NAS` → `Trainer`, asserts the test loader has `shuffle=False` and `drop_last=False`, and wraps everything in `try/except` that writes a `Failed=True` stats pkl on any exception. Pass = no exception, valid predictions written. See [[Failure Modes]] for the full list of crash paths.

A green run does **not** prove the submission is good — only that it doesn't crash and finishes in time. To prove it's good, look at `--score` output:

```
==========    Scoring   ==========
        Sudoku: raw=99.842%  benchmark=...  adj=+9.974   runtime=12.1s
   GameOfLife: raw=99.997%  benchmark=...  adj=+9.999   runtime=37.8s
   ...
  Total score: +XX.XXX
```

## Common failure signatures

| Symptom | Likely cause | Fix |
|---|---|---|
| `AssertionError: Test Dataloader is shuffling` | `DataLoader(shuffle=True)` on test loader | Set `shuffle=False`. See [[Submission Contract]] |
| `AssertionError: ... dropping last batch` | `drop_last=True` on test loader | Set `drop_last=False`. Predictions must be aligned to test_y |
| `Out of time but within 1-minute grace window` | `Trainer.train()` overshoots | Reduce epoch count, add `clock.check()` early-exit. See [[Clock and Time Budget]] |
| `Skipping dataset — out of time` | Hard timeout. −10 in real grader | Same as above, more aggressive |
| `CUDA out of memory` | Batch size too large for available VRAM | Read GPU memory at init, scale batch dynamically |
| `RuntimeError: shape mismatch` in DataProcessor | Hardcoded shape doesn't match dataset | Use `metadata['input_shape']`, never hardcode |

## Iteration cadence

- **<5 min between runs** during early development (small dataset, tight clock). Helps catch shape and contract bugs fast.
- **<30 min runs** for accuracy checks on AddNIST / MultNIST / GeoClassing. These are the canonical correctness-checks because they have known-achievable benchmarks (Geada Table 1, see [[Score-Efficient Targeting]]).
- **Full sweep (~6 h)** before any [[Phase 2 Submission Strategy|Phase 2 attempt]]. Run unattended overnight; check `dev/predictions/` in the morning.

See [[Testing Strategy]] for what to test beyond the loop itself (stress tests, low-VRAM simulation, randomness sweeps).

## Don't do this

- Don't edit `dev/mock_harness.py` to make a failing submission pass. The mock harness mirrors the grader; loosening it just hides bugs.
- Don't commit `dev/predictions/` — they're large and per-machine. (Should already be `.gitignore`d.)
- Don't put any imports in `submission/` that aren't in the starter kit's `requirements.txt`. The eval environment may not have them. See [[Submission Packaging]].
