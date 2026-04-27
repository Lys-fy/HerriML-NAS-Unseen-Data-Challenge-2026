> **Status:** drafting | **Topic:** How clock.check() works, per-dataset budgets, grace window.
> **Related:** [[Failure Modes]], [[Budget Allocation]]

A `Clock` is constructed per dataset by the grader, threaded through `DataProcessor`, `NAS`, and `Trainer`, and is the **single mechanism** by which the grader tells our code "you're running out of time".

## The Clock object

From `evaluation/main.py` (and mirrored in [[Mock Harness]]):

```python
class Clock:
    def __init__(self, time_limit_in_hours: float):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + (time_limit_in_hours * 60 * 60)

    def check(self) -> float:
        return self.time_limit - time.perf_counter()
```

So:

- `clock.check()` returns **seconds remaining**, as a float.
- It is monotonic (uses `time.perf_counter`) and unaffected by system clock changes.
- **It can go negative.** A negative value means the budget is exhausted.
- The clock starts ticking at construction time, *before* `DataProcessor.__init__` runs. There is no warm-up grace.

## Per-dataset budget

- The grader constructs **a fresh `Clock` for each dataset**. Per-dataset budgets do not pool.
- Budget is `time_limit_in_hours = metadata.get("time_limit", 0.5)`. Default 0.5 h = **30 minutes**.
- Phase-2 datasets are configured with **~1 hour each**.
- Phase-3 budget is **unspecified** — read it from the clock at runtime, do not hard-code.

## How the grader uses the clock

Between stages, `evaluation/main.py` calls `is_out_of_time(clock, metadata, predictions_dir, grace=...)`:

```python
def is_out_of_time(clock, metadata, predictions_dir, grace=False) -> bool:
    if clock.check() < 0:
        if grace and clock.check() > -60:
            print("Out of time but within 1-minute grace window — predictions still ran "
                  "(organisers may apply penalty).")
        print("\n=== Skipping dataset — out of time ===")
        fail_dataset(metadata, predictions_dir)
        return True
    return False
```

Three places it's invoked:

1. **Before `DataProcessor.process()`** — `grace=False`. If you've already burned your budget (impossible at the start, possible if a custom DataProcessor blocks for ages on `__init__`), the dataset is failed.
2. **Before `NAS.search()`** — `grace=False`. Same.
3. **Before `Trainer.train()`** — `grace=False`. Same.
4. **After `Trainer.train()`, before `Trainer.predict()`** — `grace=True`. The grace window is the only place it matters.

`metadata["time_remaining"]` is also refreshed before each stage, so code that reads metadata (rather than the clock) sees a stale-but-pre-stage estimate. **Prefer `clock.check()` for inner-loop decisions.**

## The grace window — what is and isn't allowed

The 1-minute grace (`-60 < clock.check() < 0`) only applies between `train()` and `predict()`. The semantics in the printed message:

> "Out of time but within 1-minute grace window — predictions still ran (organisers may apply penalty)."

Read literally: predictions are still computed and recorded, but the organisers reserve the right to penalise. It is **not a free minute** — design as if you have zero seconds beyond `clock.check() == 0`. The grace exists to avoid spurious `-10` failures from the prediction pass itself sliding a few seconds over.

## Designing for the clock

- **Read it inside training loops.** Wrap epochs (or even batches, on big datasets) with `if clock.check() < SAFETY_MARGIN: break`. A reasonable `SAFETY_MARGIN` is the time to run one pass of `predict()` plus 10–30 seconds of overhead.
- **Account for per-stage entry checks.** The grader bails before NAS, before Trainer, before predict — so leaving 0.5 s on the clock at `NAS.search()`'s end means `Trainer.__init__` is still allowed to run, but `Trainer.train()` will be skipped. Aim to leave **more than the next stage's startup cost**.
- **Adapt to the budget at runtime.** `time_limit` varies between datasets and phases (30 min default, ~1 h Phase 2, unknown Phase 3). Drive epoch counts, search depth, and ensemble sizes from `clock.check()` rather than constants.
- **Estimate per-epoch cost early.** Time the first one or two epochs; project remaining epochs from `clock.check()` divided by per-epoch time. EGNAS (see [[EGNAS]]) effectively does this with its "architecture-aware training schedule".
- **Always reserve predict time.** A skipped `predict()` is `-10`. If you've trained the best model in the world but haven't written predictions, the score is `-10`. See [[Failure Modes]].

For the strategic side of allocation across datasets and stages, see [[Budget Allocation]].
