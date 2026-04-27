> **Status:** drafting | **Topic:** What the NAS Unseen Data Challenge is and why it exists.
> **Related:** [[Phases and Timeline]], [[Scoring]], [[Rules]]

The **NAS Unseen-Data Challenge** is a recurring benchmark that asks teams to design **one fixed Neural Architecture Search pipeline** that performs well on image-classification datasets the team has never seen. The 2026 iteration is the **5th edition** and is hosted as a competition track at **AutoML 2026**. Site: <https://www.nascompetition.com/>.

## What makes it different from generic NAS benchmarks

- **No target dataset.** The submission is a *pipeline*, not a model. The same code is invoked on each unseen dataset, with only `metadata` (`num_classes`, `input_shape`, `benchmark`, `time_limit`) telling it what shape the data has.
- **Held-out evaluation.** Phase-2/3 datasets are kept on the organisers' servers; submitters never see them. Public datasets exist for development only.
- **Hard wall-clock budget per dataset.** A `clock` object (see [[Clock and Time Budget]]) is threaded through every stage. Time overruns score `-10`.
- **Heterogeneous problems.** Past datasets range from MNIST-derived arithmetic puzzles ([[AddNIST]], [[MultNIST]]) to satellite imagery ([[GeoClassing]]), tiled CIFAR ([[CIFARTile]]), chess-board state ([[Chesseract]]) and noise-floor data ([[Gutenberg]]). A submission must adapt to all of them with no hand-tuning per dataset.

## Organisers

Newcastle University (Magic Lab / NLAB):

- D. Towers, R. Geada, A. Atapour-Abarghouei, M. Forshaw, A. S. McGough.
- Contact: `nas-competition-contact@newcastle.ac.uk`.
- The organisers' own analysis paper, [Geada et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024W/NAS), is the canonical reference for what they consider important — see [[Papers]].

## High-level shape (2026 edition)

| | |
|---|---|
| Phase 1 (now) | Open-ended development on 13 public datasets. See [[Phases and Timeline]]. |
| Phase 2 (Jul 1 – Jul 31) | ≤7 timed submissions on the organisers' servers, hidden datasets, ~1 h per dataset. |
| Phase 3 (Aug 1+) | The last working Phase-2 submission is auto-promoted. No separate push. |

Per-dataset score is `(raw_acc - benchmark) * 10 / (100 - benchmark)`, capped at `[-10, +10]`. Total = sum across datasets. Full breakdown in [[Scoring]]. Submission limits and prohibitions in [[Rules]].

## Why care

The 2025 winners scored **12.192 / 30** — even the winner left two thirds of the available points on the table. The signal is that **reliability beats peak**: top teams avoided the `-10` cliff on every dataset rather than maximising one. See [[Reliability vs Peak]] and [[2025 - EVAI]].
