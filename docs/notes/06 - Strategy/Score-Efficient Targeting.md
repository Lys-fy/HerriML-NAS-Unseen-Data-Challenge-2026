> **Status:** drafting | **Topic:** Where points actually live. Slope formula `10 / (100 − benchmark)`.
> **Related:** [[Scoring]], [[Reliability vs Peak]], [[Search vs Train Split]], [[Budget Allocation]], [[CIFARTile]], [[Gutenberg]], [[GameOfLife]], [[Cryptic]], [[Windspeed]]

**Slope formula.** From [[Scoring]]: `adj = (raw − benchmark) * 10 / (100 − benchmark)`, capped at +10 / floored at −10. The slope `10 / (100 − benchmark)` is the points earned per **1 % absolute** raw-accuracy gain. High-benchmark datasets pay more per percentage point; low-benchmark datasets pay less.

| Benchmark | Slope (pts / 1 % raw gain) | Headroom to +10 cap |
|---:|---:|---:|
| 50 % | 0.20 | 50 % of accuracy |
| 60 % | 0.25 | 40 % |
| 70 % | 0.33 | 30 % |
| 80 % | 0.50 | 20 % |
| 90 % | 1.00 | 10 % |
| 95 % | 2.00 | 5 % |
| 99 % | 10.00 | 1 % |

## The two-axis problem

Each dataset has two competing properties for budget allocation:

- **Slope** — how many points per percent. High benchmark → steep slope.
- **Headroom** — how much room above benchmark exists before either (a) hitting the +10 cap or (b) hitting the dataset's actual achievable ceiling.

A dataset with steep slope and zero headroom is a trap: gigantic theoretical reward per percent, but you can't reach the percent. A dataset with shallow slope and large headroom rewards investment but slowly.

## Worked example — 2025 leaderboard

From [[Scoring]] / 2025 results (see [[2025 - EVAI]]):

| Dataset | Benchmark | Slope | Top-1 raw | Top-1 adj | What it tells us |
|---|---:|---:|---:|---:|---|
| GameOfLife | ~89.99 % (verify) | ~1.00 | 99.99 % | +9.998 | High slope + high headroom → free 10 points if solvable. All top-3 hit cap. |
| Cryptic | ~58.7 % (back-solved) | ~0.24 | 76.15 % | +1.745 | Shallow slope but big absolute gap (76 vs 58). Decided ranks 1–3. |
| Windspeed | ~9.4 % (back-solved) | ~0.11 | 17.4 % | +0.449 | Near-noise floor. Spending compute here is irrational — even +20 % raw is only +2.2 pts. |

Lesson: **GameOfLife was the easy money, Cryptic was the differentiator, Windspeed was a tar pit.** A team that ignored Windspeed entirely (and spent its compute on Cryptic) would not have lost much.

## Public-13 slope picture

Combining benchmark tables from [[PRIOR_FINDINGS|prior findings]] and the Geada CVPR 2024 paper (see [[Papers]]):

| Dataset | ResNet-18 | Best CNN | PC-DARTS NAS | Implied slope at benchmark |
|---|---:|---:|---:|---:|
| AddNIST | 92.1 % | 94.9 % | 97.9 % | ~1.27 |
| Language | 97.0 % | 97.0 % | 90.1 % | ~3.33 |
| MultNIST | 91.6 % | 94.0 % | 98.1 % | ~1.19 |
| CIFARTile | 45.6 % | 51.3 % | 92.3 % | ~0.18 |
| Gutenberg | 50.0 % | 50.0 % | 49.1 % | ~0.20 |
| GeoClassing | 80.3 % | 94.2 % | 96.0 % | ~0.51 |
| Chesseract | 57.8 % | 59.6 % | 60.8 % | ~0.24 |

Reading this:

- **Slope hotspots** are AddNIST, MultNIST, Language: every percent above benchmark is worth >+1 pt. Worth pushing to the cap.
- **Slope traps** are CIFARTile and Gutenberg: low benchmarks make each percent cheap. Don't be fooled by the absolute headroom — it takes a strong NAS-discovered architecture to convert any of it.
- **CIFARTile is special** ([[CIFARTile]]): the gap between best-CNN (51 %) and best-NAS (92 %) is the largest in the suite, so even at slope 0.18 it pays ~+8 pts to *anyone who can do real NAS*. This is the one place architecture search dominates training budget — see [[Search vs Train Split]].
- **Gutenberg is a noise floor** ([[Gutenberg]]): 50 % CNN, 49 % PC-DARTS. The Type-2 design (encoded text) caps the achievable accuracy near the benchmark. Spend minimal budget; accept ~0 score. See [[Threats to Validity]].

## Decision rule

For a dataset with benchmark `b` and estimated achievable ceiling `c`:

```
expected_max_points = (c - b) * 10 / (100 - b)
```

If `expected_max_points < 1.0`, treat as defensive (clear benchmark, don't time out — see [[Reliability vs Peak]]). If `expected_max_points > 5.0`, treat as offensive (push search and trainer hard). The bulk of strategic effort goes to datasets in the middle.

## What this means for hidden Phase 2 datasets

Phase 2 datasets are by design **unknown** ([[Phases and Timeline]]). We can't pre-compute slope. The submission must read `metadata['benchmark']` at runtime and adapt:

```python
slope = 10.0 / max(100.0 - metadata['benchmark'], 1e-6)
```

A high `slope` value (say >0.5) is a signal to push training quality. A low `slope` value (<0.2) is a signal to play defensively — clear benchmark, don't time out, don't burn budget chasing diminishing returns. See [[Search vs Train Split]] for the concrete adjustment.
