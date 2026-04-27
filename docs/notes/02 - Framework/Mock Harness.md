> **Status:** drafting | **Topic:** What dev/mock_harness.py does and how to use it.
> **Related:** [[Pipeline Flow]], [[Dev Loop]], [[Testing Strategy]]

`dev/mock_harness.py` is a local mirror of the organisers' `evaluation/main.py`. Anything that completes successfully under it has a high chance of completing on the eval server.

## What it does

- Adds `submission/` to `sys.path` and imports `DataProcessor`, `NAS`, `Trainer` exactly the way the grader does (so absolute-import bugs surface locally).
- Constructs an identical `Clock(time_limit_in_hours)` per dataset.
- Calls `DataProcessor → NAS → Trainer.train → Trainer.predict` in the same order, with the same arguments.
- Asserts the same test-loader constraints (`not RandomSampler`, `not drop_last`) — see [[Submission Contract]].
- Wraps `run_submission` in `try/except`, writing a `Failed=True` stats pickle on any exception — same path as the grader's failure handler in [[Failure Modes]].
- Writes predictions to `dev/predictions/<codename>.npy` and stats to `dev/predictions/<codename>_stats.pkl`.

The `Clock`, the assertion text, and the call sequence are kept byte-for-byte equivalent to the grader so divergence cannot creep in.

## CLI

```bash
# Run all datasets present under datasets/
python dev/mock_harness.py

# Filter to specific datasets (folder names)
python dev/mock_harness.py Sudoku GameOfLife

# Override the per-dataset time limit (hours)
python dev/mock_harness.py --time-limit 0.05      # ~3 minutes — fast smoke test

# After running, also compute adjusted scores using local test_y.npy
python dev/mock_harness.py --score
```

Other knobs:

- `--datasets-dir PATH` (also `$DATASETS_DIR`) — defaults to `<repo>/datasets/` (which is a symlink to `~/local/nas-comp-datasets/`).
- `--predictions-dir PATH` — defaults to `<repo>/dev/predictions/`.

The `--score` flag uses **local** `test_y.npy` and the standard `(raw - benchmark) * 10 / (100 - benchmark)` formula (capped at `[-10, +10]`). See [[Scoring]]. The real grader runs scoring in a separate `score.py` after stripping `test_y.npy` from the package — locally we cheat by reading it.

## Differences from the real grader

Documented in the file header. Summary:

| | Real grader (`evaluation/main.py`) | Mock harness (`dev/mock_harness.py`) |
|---|---|---|
| `submission/` on `sys.path` | Already by directory layout | Inserted at runtime so the script can live outside the package |
| Datasets location | Their server | `$DATASETS_DIR` / `<repo>/datasets/` |
| Predictions output | Their predictions dir | `<repo>/dev/predictions/` |
| Scoring | `score.py`, no `test_y` available to submission | Optional `--score` flag, uses local `test_y.npy` |
| `Clock` class | Same | Same |
| Call sequence | Same | Same |
| Test-loader assertions | Same | Same |
| Outer `try/except` | Same | Same |
| Time check between stages | Same | Same |

**Identical** components: `Clock`, the call sequence, the test-loader assertions, the failure-write semantics. **Different** components: where datasets live, where predictions land, that scoring is optional and reads local labels.

## Workflow

Standard inner-loop iteration (see also [[Dev Loop]]):

```bash
# 1. Edit submission/{nas,trainer,data_processor}.py
# 2. Quick sanity check on a tiny dataset with a tiny budget:
python dev/mock_harness.py Sudoku --time-limit 0.05

# 3. End-to-end on the small datasets, with scoring:
python dev/mock_harness.py Sudoku GameOfLife AddNIST --score

# 4. Pre-flight before a Phase-2 submission: full Phase-2 budget:
python dev/mock_harness.py --time-limit 1.0 --score
```

If a run fails, the harness prints the traceback and marks the dataset failed in `dev/predictions/<codename>_stats.pkl` — *not* in your shell exit code. The harness exits 0 even on failures (matches the grader). When scripting CI-like checks, parse the stats files rather than relying on exit code.

## Caveats

- **Local test labels are inflated by their nature** — you've seen them and may have leaked information into hyperparameters. Treat local scores as a development signal, not a Phase-3 prediction. See [[Threats to Validity]].
- **Hardware mismatch.** The eval GPU is withheld; local hardware is not representative. A run completing in 25 min locally may take longer or shorter on their machine. Lean on `clock.check()` adaptivity, not local timing.
- **Stale starter-kit `requirements.txt`.** The starter pins `torch==1.10.1+cu113`. Your local Python likely runs a newer torch — code that works locally must also work on the grader's (unknown) torch version. Avoid version-locked APIs.
- **Two datasets unavailable**: [[Cryptic]] and [[Windspeed]] starter-kit DOIs are broken — they are not in `scripts/download_datasets.sh` and so won't appear locally. Develop on the other 11.
