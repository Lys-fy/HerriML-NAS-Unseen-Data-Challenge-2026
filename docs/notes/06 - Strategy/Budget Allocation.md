> **Status:** drafting | **Topic:** How to split per-dataset time between NAS and Trainer phases.
> **Related:** [[Clock and Time Budget]], [[Search vs Train Split]], [[Reliability vs Peak]], [[Pipeline Flow]], [[EGNAS]], [[Score-Efficient Targeting]]

**Default split per dataset: 30 % NAS / 65 % Trainer / 5 % buffer.** Justified below; tune per-dataset using the heuristics from [[Search vs Train Split]].

## What the clock looks like

[[Clock and Time Budget]] establishes that `metadata['time_limit']` defaults to 0.5 h locally and ~1 h per dataset in Phase 2. The `clock` object is shared across [[Submission Contract]] → [[Submission Contract]] → [[Submission Contract]]. There is **one wall-clock per dataset**, and overrun anywhere triggers −10 (see [[Failure Modes]]).

Stages consuming time:

1. **DataProcessor.process()** — tensor conversion, dataloader construction, augmentation setup. Should be O(seconds) on the CPU side; treat as ~1 % of clock.
2. **NAS.search()** — architecture search. The interesting knob.
3. **Trainer.train()** — final retrain of the chosen architecture.
4. **Trainer.predict()** — inference on test set. O(seconds) but skipping it = −10.

## Why 30 / 65 / 5

- **65 % to Trainer**: even a perfect architecture is useless undertrained. The 2025 winners ([[2025 - EVAI]], [[2025 - Sichuan teams]]) all leaned heavily on the final retrain. [[EGNAS]]'s public defaults specify 50 final retrain epochs; on a 1-hour Phase-2 clock that requires ~40 minutes uninterrupted.
- **30 % to NAS**: enough for hill-climbing-style search (EGNAS-shaped) or for a few generations of predictor-guided evolution (see [[Predictor-Guided Evolutionary]]). Not enough for a from-scratch DARTS supernet — which is fine, those don't transfer well to unseen data anyway (see [[DARTS Family]]).
- **5 % buffer**: covers `process()`, `predict()`, dataloader spin-up, CUDA warm-up, and grader overhead. The mock harness also enforces a 1-minute grace window for prediction — that minute eats into this buffer.

## Per-dataset modulation

The 30/65 split is the prior. Adjust by [[Score-Efficient Targeting]] slope and [[Search vs Train Split]] reasoning:

| Dataset shape | NAS | Trainer | Rationale |
|---|---:|---:|---|
| Easy, near ceiling (e.g. AddNIST, MultNIST, GameOfLife) | 15 % | 80 % | NAS-vs-CNN gap is small; Trainer wins |
| NAS-rewarding (CIFARTile-style, see [[CIFARTile]]) | 50 % | 45 % | Geada Table 1: CNN best 51 % vs PC-DARTS 92 %. Architecture matters |
| Near-noise-floor (Gutenberg, Windspeed) | 10 % | 85 % | Ceiling caps both; spend less. See [[Threats to Validity]] |
| Unknown shape (typical Phase 2 hidden dataset) | 30 % | 65 % | Default — no information to deviate |

Hidden Phase 2 datasets are by design unknown — see [[Phases and Timeline]]. The default should be the *most* defensive split that still leaves room for [[Submission Contract]] to matter.

## Implementing the split

The `clock` object exposes `clock.check()` returning seconds remaining (see [[Clock and Time Budget]]). Each stage should:

1. Read `metadata['time_remaining']` once at init (the harness sets it from the immutable clock snapshot).
2. Compute its own deadline as `start_time + share * total_time`.
3. Check periodically and **return early gracefully** if past the deadline. Don't raise — the wrapper catches exceptions and fails the dataset.

Concretely in [[Submission Contract]]:

```python
nas_budget = metadata['time_remaining'] * 0.30
deadline = time.perf_counter() + nas_budget
while time.perf_counter() < deadline and not_converged:
    step()
```

[[Submission Contract]] uses the same pattern but checks against `clock.check()` directly so it absorbs any underspend from earlier stages. **Underspend by NAS should flow to Trainer**, not be wasted.

## Anti-patterns

- **Static epoch counts.** Don't hardcode "50 epochs" — on a tight Phase 3 clock that overruns. Always epoch-by-time, not epoch-by-count.
- **Symmetric split.** 50/50 is too NAS-heavy for a competition where reliability dominates ([[Reliability vs Peak]]).
- **Greedy NAS extension.** "Just one more iteration" inside `search()` — eats Trainer's budget. The hill-climbing schedule must hard-stop.

## Calibrating once we have data

After the first end-to-end mock run, look at runtimes per stage in `dev/predictions/<codename>_stats.pkl`. If Trainer spent <50 % of clock on most datasets, NAS is underbudgeted. If Trainer ran to deadline on most datasets, NAS is overbudgeted. Iterate the split per-dataset before committing to a [[Phase 2 Submission Strategy|Phase 2 submission]].
