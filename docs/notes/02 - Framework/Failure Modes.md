> **Status:** drafting | **Topic:** What causes a -10 (timeout, OOM, crash, wrong shape).
> **Related:** [[Submission Contract]], [[Reliability vs Peak]]

A failed dataset scores **`-10`**, regardless of why it failed. There is no partial credit. The grader catches all failures via two mechanisms: explicit time checks between stages, and a single outer `try/except` around the whole pipeline.

## The grader's failure path

From `evaluation/main.py`:

```python
def fail_dataset(metadata, predictions_dir):
    stats = {"Failed": True, "Runtime": -1, "Params": None}
    (predictions_dir / f"{metadata['codename']}_stats.pkl").write_bytes(pkl.dumps(stats))

# ...
try:
    run_submission(runclock, ds_path, predictions_dir)
except Exception as ex:
    print(f"Exception while running {ds_path.name}: {ex}", file=sys.stderr)
    fail_dataset(metadata, predictions_dir)
```

A `_stats.pkl` with `Failed=True` is what the scorer checks first. The scorer (`score.py`) hard-codes `-10` for any dataset whose stats file flags failure. See [[Scoring]].

## What can cause `Failed=True`

### 1. Timeout (`clock.check() < 0`)

- Detected at the **stage boundary checks** (before DataProcessor, NAS, Trainer.train, Trainer.predict). See [[Clock and Time Budget]].
- Grace window applies only between `train()` and `predict()` and is at most 60 seconds; even within grace, the organisers may still penalise.
- **Subtle case:** the clock check happens *between* stages, not *within* them. A `Trainer.train()` that loops without checking `clock.check()` will not be interrupted; it will simply return late, and the next stage's pre-check will fire.

### 2. Out-of-memory

- RAM or VRAM. Either raises (CUDA OOM is a `RuntimeError`); the outer `try/except` catches it.
- The starter kit and rules state submissions must "be adaptable to the amount of memory available." VRAM is unspecified. Therefore **try a small batch / small model first**, and grow only after confirming headroom.
- Useful guardrails: `torch.cuda.empty_cache()` between stages; deleting intermediate models from NAS; using gradient checkpointing for deep search candidates.

### 3. Uncaught Python exception

Anything raised inside any stage that you don't catch. Common cases:

- `ValueError` from a tensor reshape that assumes a different `input_shape` than `metadata` reports.
- `KeyError` from `metadata` (e.g. assuming `time_limit` is always present — it is, but be defensive).
- `RuntimeError` from a model that mismatches the data's number of channels.
- Pickling errors / multiprocessing errors from `DataLoader(num_workers=N)` on small datasets.
- `IndexError` / off-by-one in `Trainer.predict` if you assume specific batch dimensions.

The fix: when there's a clean fallback, catch the exception inside the relevant stage and switch to a **known-good baseline** (e.g. a small ResNet-18 trained for a few epochs). Returning early with a successful prediction beats raising for `-10`.

### 4. Wrong-shape test loader

The grader asserts:

```python
assert not isinstance(test_loader.sampler, RandomSampler), "Test Dataloader is shuffling, ..."
assert not test_loader.drop_last, "Test Dataloader is dropping last batch, ..."
```

`AssertionError` is an exception → outer `try/except` → `-10`. Even if accuracy on the loader would be good, predictions in the wrong order are worthless because the grader scores against fixed-order `test_y.npy`. See [[Submission Contract]].

### 5. Missing predictions file

If `predict()` does not write predictions (returns wrong type, raises silently in the wrapped grader code), the scorer sees a stats file but no predictions file → `-10`. Always return a list/array of integer class indices the same length as the test set.

### 6. Disqualifying behaviour (rare)

Attempting to fetch the held-out test datasets at runtime is **disqualifying** per the rules — see [[Rules]]. Effectively `-10` on every dataset and removal from the leaderboard.

## Defensive patterns

- **Outer guard** in `NAS.search`: if anything inside fails or runs out of time, return a hardcoded baseline architecture (e.g. simple CNN) sized to `metadata["input_shape"]` and `metadata["num_classes"]`.
- **Outer guard** in `Trainer.train`: if optimisation diverges (NaN loss) or fails, return the best-checkpoint model — never raise.
- **Time-bounded inner loops**: read `clock.check()` and break before the next batch if `time_remaining < safety_margin`.
- **Test-loader builder**: always construct `test_loader` with `shuffle=False, drop_last=False` and an explicit `SequentialSampler` if you want to be paranoid. Cover this in [[Testing Strategy]].
- **Mock-harness gating**: never push a Phase-2 submission that hasn't completed end-to-end on at least one local dataset via [[Mock Harness]]. The mock harness asserts the same constraints and uses the same Clock class, so it catches most of these failure modes locally.

## What does *not* fail

- Low accuracy (above or below benchmark) — that's just a low score, not a failure.
- Long-but-under-budget runs.
- Returning a model that's larger than the benchmark architecture (no size cap).
- Per-dataset failures are independent: a `-10` on dataset A does not affect scoring on dataset B.
