# Language

> **Status:** drafting | **Topic:** Four 6-letter words concatenated, encoded as a 24×24 binary letter-position grid. 10 languages.
> **Related:** [[Datasets Overview]], [[Gutenberg]]

A 24-character string (four six-letter words from one language) encoded as a binary 24×24 image: y-axis = string position, x-axis = letter (A–X, no Y, Z, or diacritic letters). Each filled pixel marks "letter X is at position Y". Label = source language.

| Field | Value |
|---|---|
| Codename | LaMelo |
| Era | CVPR (2021–23) |
| Shape | 1×24×24 (channels-first) |
| Train / val / test | 50,000 / 10,000 / 10,000 |
| Classes | 10 (English, Dutch, German, Spanish, French, Portuguese, Swahili, Zulu, Finnish, Swedish) |
| Benchmark (ResNet-18) | 85.20% (`metadata`); Geada Tab. 1 reports 87.00% (recorded as 97.00% in the [[PRIOR_FINDINGS]] table — that was a typo; the paper says 87.00%) |
| Best off-the-shelf CNN (Geada) | ResNeXt-50 93.97% |
| Best NAS method (Geada Tab. 2) | PC-DARTS 90.12% (random-DARTS tied at 90.12%) |
| Best 2023-comp submission | 89.71% |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/Language_Dataset/24574729 |

## Type and difficulty

**Type-2** (Geada): a human cannot solve this by inspection — neither the encoding nor the language-to-label mapping is known to participants. A model must learn the language signal from letter-frequency / character-cooccurrence patterns that survive the binary encoding.

Word-level data leakage is prevented: any given word appears in only one of train/val/test.

## Pathologies and what's notable

- **NAS underperforms here**. ResNeXt-50 (93.97%) beats every Geada-tested NAS method by ~4 pp. That is one of the few datasets where off-the-shelf wins outright.
  - Geada attributes ResNeXt's edge to its high cardinality, which seems to capture the "subtle patterns of character frequencies and arrangements" better than DARTS-cell architectures.
  - Implication: a fixed ResNeXt-50 backbone may outperform anything our NAS picks — see [[Search vs Train Split]].
- **Tiny channel dim**: 1-channel + small spatial extent. Some NAS implementations assume 3-channel input — Phase-1 sanity check that ours doesn't.
- **No spatial visual structure** in the conventional sense: nearest neighbours in pixel space don't share label-relevant features. Standard image augmentations (random crops, flips, colour jitter) **may hurt**, since they break the encoding contract. Verify before applying.

## Implications for our pipeline

- Score-relevance is moderate (benchmark ~85%, slope ~0.67 / 1%).
- Worth piloting a "skip search, use ResNeXt-style fixed backbone" path here — especially if the [[Clock and Time Budget]] is tight. See [[Score-Efficient Targeting]].
- Augmentation selection matters more than architecture: this is a "no spatial invariance" dataset.
