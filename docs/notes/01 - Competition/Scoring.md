> **Status:** drafting | **Topic:** Per-dataset scoring formula, the +10 / -10 caps, slope by benchmark.
> **Related:** [[Reliability vs Peak]], [[Score-Efficient Targeting]]

Each dataset contributes an **adjusted score** in `[-10, +10]`. The total is the sum across datasets. The transformation is linear in raw accuracy but **slope-weighted by the dataset's public benchmark**.

## Formula

```
adj_score = (raw_acc_pct - benchmark_pct) * 10 / (100 - benchmark_pct)
adj_score = clip(adj_score, -10, +10)
```

Where:

- `raw_acc_pct` is your model's test accuracy as a percentage (0–100).
- `benchmark_pct` is a per-dataset constant — the organisers' published baseline accuracy (averaged off-the-shelf CNN performance, see [[Papers]] in section 08).

Anchors:

| `raw_acc` | `adj_score` |
|-----------|-------------|
| 100% | **+10** (cap) |
| `benchmark` | **0** |
| failure (timeout / OOM / crash / wrong shape) | **−10** (floor — see [[Failure Modes]]) |

A `raw_acc` *below* benchmark gives a negative adjusted score, but a successful run is bounded at `−10` regardless of how far below benchmark it lands. A failed run is forcibly `−10` no matter what.

## Slope by benchmark — why high-benchmark datasets matter more

Because the denominator is `(100 − benchmark)`, the points-per-percent slope grows as benchmark approaches 100:

| Benchmark | Points per +1% raw accuracy |
|-----------|-----------------------------|
| 50% | +0.20 |
| 70% | +0.33 |
| 90% | **+1.00** |
| 95% | **+2.00** |

Practical reading:

- On a **low-benchmark** dataset like [[Gutenberg]] (~50% benchmark, ~50% ceiling), grinding from 50% → 55% gives only +1.0 adjusted points. There is little upside.
- On a **high-benchmark** dataset like [[GeoClassing]] (~94% benchmark) or any dataset with a benchmark ≥90%, a +2% raw gain yields +2 adjusted points. Squeezing the last percent matters.
- The cap at `+10` means there is a **soft ceiling**: pushing `raw_acc` past `100%` is impossible, and the cap is hit at `raw_acc = 100%` only. Practical max-out happens slightly before that for any benchmark below 100.

See [[Score-Efficient Targeting]] for how this shapes per-dataset effort allocation.

## Total

```
total = sum(adj_score over all datasets in the phase)
```

There is no per-dataset weighting beyond what the formula already implies. With *N* datasets the achievable range is `[-10N, +10N]`. In **2025 the maximum was 30** (3 datasets — see [[2025 - EVAI]]) and the winner scored `12.192`. Only 3 of the 9 dataset-team scores in the top-3 cleared `+5`. The expected-value reading is that the formula *rewards completing every dataset above benchmark* far more than it rewards being best on any one.

## What this implies

- **Avoiding `-10` is the dominant first-order goal.** A single failed dataset wipes out roughly 5 percentage points of effort elsewhere. See [[Reliability vs Peak]].
- **Plateau-detection beats hill-climbing.** If `raw_acc` has plateaued near the benchmark on a dataset, additional time has near-zero expected return — **save the budget** for high-benchmark datasets where each minute moves more points. See [[Budget Allocation]].
- **Calibrate against benchmarks, not against absolute accuracy.** A 76% on Cryptic was the 2025 winner; a 99% on a low-benchmark dataset can be worth less.
