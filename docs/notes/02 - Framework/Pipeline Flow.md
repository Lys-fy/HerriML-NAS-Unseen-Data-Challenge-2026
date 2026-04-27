> **Status:** drafting | **Topic:** DataProcessor -> NAS -> Trainer.train -> Trainer.predict.
> **Related:** [[Submission Contract]], [[Clock and Time Budget]]

The grader runs four pipeline stages in fixed order, per dataset. The same `clock` flows through every stage and is checked between stages.

## Stage diagram

```
                       ┌────────────────────────────────────────┐
                       │  per dataset: Clock(time_limit_hours)  │
                       └────────────────────────────────────────┘
                                          │
            metadata + (train_x, train_y, valid_x, valid_y, test_x)
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ 1. DataProcessor(train_x, train_y, valid_x, valid_y,               │
   │                  test_x, metadata, clock)                          │
   │     .process() → (train_loader, valid_loader, test_loader)         │
   │                                                                    │
   │   asserts: test_loader has no RandomSampler, no drop_last          │
   └────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ 2. NAS(train_loader, valid_loader, metadata, clock)                │
   │     .search() → model                                              │
   └────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ 3. Trainer(model, device, train_loader, valid_loader,              │
   │            metadata, clock)                                        │
   │     .train() → trained model                                       │
   └────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ 4. Trainer.predict(test_loader) → list[int]                        │
   │     written to predictions/<codename>.npy                          │
   └────────────────────────────────────────────────────────────────────┘
```

## What the grader does between stages

`evaluation/main.py` (mirrored exactly in [[Mock Harness]]):

1. Loads `train_x.npy`, `train_y.npy`, `valid_x.npy`, `valid_y.npy`, `test_x.npy`, `metadata` for the dataset.
2. Constructs `Clock(time_limit)`. Snapshots an immutable copy for printing only.
3. Sets `metadata["time_remaining"] = clock.check()` and prints a banner.
4. **Pre-stage time check** before each of stages 1, 2, 3. If `clock.check() < 0`, marks the dataset failed (`-10`) and skips the rest.
5. After Stage 1, runs the test-loader assertions. Crash on violation.
6. Picks `device = cuda if available else cpu` *before* Stage 3 (so `NAS.search()` runs without a `device` arg — be careful with where you put models in `NAS`).
7. Between `train()` and `predict()`, applies the **grace check** (`grace=True`) — see [[Clock and Time Budget]]. Roughly: a 1-minute window where predictions still run but the organisers may apply a penalty.
8. Wraps the whole thing in a `try/except`. Any exception → fail (`-10`). See [[Failure Modes]].

## Where time goes — implicit budget split

The clock is a single per-dataset budget shared across all four stages. There is no enforced split, but a working baseline allocation looks like:

| Stage | Typical share | Notes |
|-------|---------------|-------|
| DataProcessor | < 5% | Tensor reshape / normalise / build loaders. Cheap unless you do something like in-Python augmentation pipelines. |
| NAS.search | 10 – 50% | Highly approach-dependent. Predictor-based and zero-cost are minutes; full retrain-per-candidate is the budget. See [[Approaches Overview]]. |
| Trainer.train | 40 – 80% | Final-architecture retrain. The dominant slice for most pipelines. |
| Trainer.predict | < 1% | Single forward pass over test set. |

See [[Search vs Train Split]] in section 06 for the trade-off analysis. A useful invariant: **`Trainer.train` should self-terminate against `clock.check()` rather than against an epoch count** — otherwise a slow-NAS run on a tough dataset eats predict time and times out.

## Single-clock implications

- Spending too long in `NAS.search` directly steals from `Trainer.train`. There is no second budget.
- Spending too long in `DataProcessor` is the silent killer because the grader doesn't bail out *during* a stage — it only checks before the next one. A 50-minute pre-processing pass on a 60-minute dataset will not be interrupted; it will simply leave 10 minutes for everything else.
- This is why threading `clock` through every stage matters: the inner loops of NAS and training must read it.

## Inputs that change per dataset

Only `metadata` and the data tensors change between dataset invocations. The submission code is the same module loaded once. Any global state must therefore be **safely re-initialised** between datasets — class instances are fresh per dataset, but module-level singletons (cached random states, lru_cache, file handles) persist.
