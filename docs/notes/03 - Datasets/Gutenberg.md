# Gutenberg

> **Status:** drafting | **Topic:** Three consecutive words from public-domain texts, encoded as a 27×18 letter-position grid. 6 authors.
> **Related:** [[Datasets Overview]], [[Language]], [[Threats to Validity]]

Concatenated 3-word sequences (each 3–6 letters, padded to 6) from English translations of works by 6 authors: Aquinas, Confucius, Hawthorne, Plato, Shakespeare, Tolstoy. Encoded as a binary 27×18 image (y-axis = letter A–Z + space; x-axis = position in the 18-character string).

| Field | Value |
|---|---|
| Codename | Gutenberg |
| Era | CVPR (2021–23) |
| Shape | 1×27×18 (channels-first) |
| Train / val / test | 45,000 / 15,000 / 6,000 (asymmetric — note the smaller test split) |
| Classes | 6 |
| Benchmark (ResNet-18) | 40.98% (`metadata`); Geada Tab. 1 reports 40.98% — random-chance is 16.7% |
| Best off-the-shelf CNN (Geada) | AlexNet 45.53% |
| Best NAS method (Geada Tab. 2) | PC-DARTS 49.12% |
| Best 2023-comp submission | **50.85%** (slightly beat NAS) |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/Gutenberg_Dataset/24574753 |

## Type and difficulty

**Type-2** (Geada): the research question is whether spatial patterns of letters carry enough information to identify an author. Answer from the data: weakly. The ceiling appears to be ~50%, which is **noise-floor territory** given:

- 3-word windows are extremely short — most word triples could plausibly come from any author.
- Translations of Confucius / Aquinas / Plato are by different translators, blurring author signal with translator signal.
- "Structure words" (Chapter, Scene, Prologue) were stripped, removing a usable shortcut.

## Why this matters for the competition

- **Highest scoring slope per percent on the low-benchmark side**: a benchmark of 41% means [[Scoring]] gives ~0.17 pt per 1% accuracy. Even a perfect 100% would only buy +10. The **floor risk is 100% of points** (any failure = −10) but the upside is small.
- This profile makes Gutenberg a **reliability dataset**, not a peak-chasing one. See [[Reliability vs Peak]].
- Competition submissions historically beat NAS here — suggesting the dataset rewards regularisation and good optimisation more than architecture. Don't burn search budget.

## Pathologies

- **Test set is 6,000, not 10,000** like most siblings. Easy to miss in a generic dataloader assumption.
- The 27×18 spatial shape is unusual; some standard image transforms (square-crop, ImageNet-normalisation) will silently misbehave.
- Class-balance is enforced post-hoc: the README says 11,000 per class evenly distributed, but the source corpora are unequal in size, so this is post-subsampling.

## Implications for our pipeline

- Treat as a **floor / reliability** target. Just don't fail. Don't optimise hard.
- Reuse the [[Language]] data pipeline (same 1-channel character-position encoding genus).
- Augmentation: standard image augmentations break the encoding. Stick to label-preserving ones (none, or very mild dropout-like masking).
