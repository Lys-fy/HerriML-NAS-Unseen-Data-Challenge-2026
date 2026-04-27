> **Status:** drafting | **Topic:** Per-dataset adjustment of NAS vs Trainer share based on benchmark and headroom.
> **Related:** [[Budget Allocation]], [[Score-Efficient Targeting]], [[CIFARTile]], [[Gutenberg]], [[EGNAS]], [[Predictor-Guided Evolutionary]], [[Reliability vs Peak]]

[[Budget Allocation]] sets the prior at **30 % NAS / 65 % Trainer / 5 % buffer**. This note explains how to *deviate* per dataset based on what `metadata['benchmark']` and the dataset's likely shape tell us.

## Why the split should not be uniform

Geada Table 1 (in [[Papers]] / referenced from [[Score-Efficient Targeting]]) shows the NAS-vs-CNN gap is dataset-specific:

| Dataset | Best CNN | PC-DARTS NAS | NAS gain |
|---|---:|---:|---:|
| AddNIST | 94.9 % | 97.9 % | +3.0 % |
| MultNIST | 94.0 % | 98.1 % | +4.1 % |
| Language | 97.0 % | 90.1 % | **−6.9 %** |
| CIFARTile | 51.3 % | 92.3 % | **+41.0 %** |
| Gutenberg | 50.0 % | 49.1 % | −0.9 % |
| GeoClassing | 94.2 % | 96.0 % | +1.8 % |
| Chesseract | 59.6 % | 60.8 % | +1.2 % |

NAS pays off **massively** on CIFARTile-style problems and **negatively** on Language. A uniform split therefore wastes either compute (on AddNIST: any decent CNN clears benchmark) or score (on CIFARTile: a generic CNN can't solve it).

## The four regimes

### Regime A — easy / near-ceiling

**Signature**: benchmark high (>85 %), best-CNN ≈ best-NAS, achievable ceiling near 100 %.
**Examples**: AddNIST, MultNIST, GameOfLife (see [[GameOfLife]]).
**Split**: 15 % NAS / 80 % Trainer / 5 % buffer.
**Why**: tiny architecture gap. The win is "train a competent ResNet-style net for many epochs". A few minutes of NAS to size depth/width is enough; hill-climbing kernels (the [[EGNAS]] inner loop) is overkill but harmless if cheap.

### Regime B — NAS-rewarding

**Signature**: benchmark low–mid (40–60 %), best-NAS dramatically above best-CNN, ceiling well above benchmark.
**Examples**: CIFARTile (see [[CIFARTile]]), and very likely a hidden Phase 2 dataset of similar shape.
**Split**: 50 % NAS / 45 % Trainer / 5 % buffer.
**Why**: this is where architecture search actually buys points. The +41 % NAS gain on CIFARTile is the largest score-per-compute investment in the suite. [[EGNAS]]-style depth/width search transfers; predictor methods may not (see [[Threats to Validity]]).

### Regime C — noise-floor / Type-2

**Signature**: best-CNN ≈ best-NAS ≈ benchmark, both near the dataset's achievable ceiling.
**Examples**: Gutenberg (see [[Gutenberg]]), Windspeed.
**Split**: 10 % NAS / 85 % Trainer / 5 % buffer — but mainly: **don't optimise**.
**Why**: from [[Score-Efficient Targeting]], the slope is shallow and the headroom is small. Any score above benchmark is bonus. The strategic move is to clear benchmark (≈0 adjusted) reliably and use the saved compute elsewhere. The Type-2 designation in the Geada paper means the task is "almost impossible for a human" — don't expect NAS to crack what spectrograms or encoded-text representations don't reveal.

### Regime D — unknown (Phase 2 hidden)

**Signature**: no prior — typical Phase 2 datasets ([[Phases and Timeline]]).
**Split**: 30 % NAS / 65 % Trainer / 5 % buffer (the [[Budget Allocation]] default).
**Why**: it's the maximum-entropy choice. Underspending NAS on a CIFARTile-shaped hidden dataset is recoverable (Trainer still hits benchmark). Underspending Trainer on an AddNIST-shaped hidden dataset is recoverable (NAS-discovered architecture is small and trains fast in remaining time). The default doesn't lose catastrophically in either direction.

## Runtime classifier

We have to choose the regime *at runtime*. Heuristic from `metadata['benchmark']` only (since input shape and num_classes are noisy signals):

```python
b = metadata['benchmark']
if b >= 85:
    nas_share, train_share = 0.15, 0.80   # Regime A
elif b >= 70:
    nas_share, train_share = 0.30, 0.65   # default
elif b >= 40:
    nas_share, train_share = 0.50, 0.45   # Regime B
else:
    nas_share, train_share = 0.10, 0.85   # Regime C
```

This is the simplest possible classifier. It's wrong on Language (which has b=97 but CNN > NAS — see [[Score-Efficient Targeting]]) but Language sits in Regime A anyway, so the practical loss is small.

## Phase-2-aware refinement

If we observe a Phase 2 hidden dataset behaving CIFARTile-shaped (low benchmark, slow CNN convergence after a few epochs), we *can't* re-tune within a single submission — the split is fixed at NAS init. But across the 7-attempt window of [[Phase 2 Submission Strategy]], we can iterate: submission *N* uses regime D, observes per-dataset stats from the leaderboard, and submission *N+1* uses a per-codename override.

## When to ignore this note

- Reliability is not yet at `p ≥ 0.95` — see [[Reliability vs Peak]]. Don't optimise the split until the pipeline doesn't crash.
- The submission's NAS doesn't actually exploit search budget (e.g. it returns a fixed architecture). Then any NAS share above ~5 % is wasted; use the freed budget for Trainer regardless of regime.
