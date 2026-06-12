# Language

> **Status:** drafting | **Topic:** Four 6-letter words concatenated, encoded as a 24×24 binary letter-position grid. 10 languages.
> **Related:** [[Datasets Overview]], [[Gutenberg]], [[Language Deep Dive]] (our empirical findings), [[Pipeline v1-v7 - Aug Probe]] (probe history), [[Multi-Seed Variance]] (the aug-oracle), [[Reliability vs Peak]]

![[Language.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


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

## Empirical findings (2026-05-28) — what we have tried

Documented in detail in **[[Language Deep Dive]]**. Summary of what holds up under 3 seeds per configuration:

### What does NOT help (training-side, with current Conv-only OP_VOCAB)

| Intervention | Effect on Language raw acc | Verdict |
|---|---|---|
| **Default augmentation (`crop_pad=2, cutout=4`)** | drops raw to **68-73 %** (mean 70.16 % ± 2.62), often hits the −10 floor. | **Decisively harmful.** Confirmed across Tier-1, EGNAS-full, and P1 v4. The pixel-shift + cutout breaks the binary letter-position encoding. |
| Patience extension (30 → 60 epochs) | 79.10 % → 80.02 % (+0.6 adj) | small, not enough |
| Lower LR + longer warmup (0.025 → 0.01, 5 → 10) | 79.10 % → 79.68 % (+0.4 adj) | small, not enough |
| Larger model width (48 → 72) | 79.10 % → 80.32 % (+0.8 adj) | small, not enough |
| Combined patience + LR + Mixup | matches Mixup alone on mean | only stabilises variance |

**The training-side ceiling is ~81.5 % raw** on the Tier-1 fixed-Conv3 stack with aug-off. We have shown this with 6 variations × 3 seeds; mean spread of the 6 variations is only 2.5 pp (79.1 % → 81.6 %). No training-side intervention crosses the harness benchmark of 85.20 %.

### What DOES help (validated through 3-seed sweeps)

| Intervention | Effect on Language | Notes |
|---|---|---|
| **Turn augmentation off** | +5 to +10 pp raw vs default | The single biggest training-side lever. Implemented via the P1 aug-safety probe — see [[Pipeline v1-v7 - Aug Probe]]. |
| **Mixup α=0.2** (with spatial aug off) | +1.6 adj over aug-off baseline → **81.5 % raw** | Implicit regularisation via label-space mixing; perturbs the loss landscape so the model keeps learning past where it would memorise/saturate. **Novel in the published NAS-Unseen literature** — no past team has documented this on Language. |

### What the literature says is needed to cross the benchmark

Per **Geada et al. arXiv:2404.02189** (CVPR 2024), the architectural lever for Language is **grouped convolutions with high cardinality (ResNeXt-style)**:

- ResNeXt-50 (cardinality 32, 4-wide-per-group) **vanilla** hits **93.97 %** — 4 pp above any NAS method tried.
- PC-DARTS / DARTS hit **90.12 %** — cell-based NAS, doesn't include grouped convs natively.
- The best public competition submission only hit **89.71 %** — still 4 pp below vanilla ResNeXt.
- DrNAS **88.55 %**, Bonsai-Net **87.65 %**.

**Implication: an architectural change to add grouped convolutions is the data-backed lever, not more training tricks.** Our current `OP_VOCAB` (`Conv3 / Conv5 / SepConv3 / MaxPool / Identity`) has no grouped-conv option — `SepConv3` is the degenerate max-cardinality endpoint (depthwise) and `Conv3` is the zero-cardinality endpoint (full mixing). The middle ground (cardinality 8/16/32) is exactly where ResNeXt operates.

### Planned next experiment

Adding `GroupConv3` (`Conv2d(in, out, k=3, groups=min(in_ch, 32))`) as a new entry in `OP_VOCAB`. The existing EGNAS micro hill-climb can then pick it per-layer. Combined with the v5 P1 probe (driver C from v4 shadow data) + Mixup-on-OFF, the expected outcome is the first time any configuration we run crosses the 85.20 % Language benchmark. See [[Language Deep Dive]] §8 for the full design.

### Per-method raw acc summary table

| Method | Language raw acc (mean ± σ, 3 seeds) | Crosses 85.20% benchmark? |
|---|---:|---|
| Multi-seed Tier-1 (aug-on) | 70.16 % ± 2.62 | No (≤73%) |
| Tier 2b — EGNAS-full | 72.49 % ± 2.96 | No |
| P1 v4 (probe-driven) | 78.39 % ± 3.39 | No |
| Multi-seed Tier-1 (aug-off, with op-search) | 81.41 % ± 3.00 | No (best seed 83.47%) |
| Excursion V3 (Mixup α=0.2, fixed Conv3) | 81.49 % ± 0.60 | No |
| Excursion V5 (combo V1+V2+V3) | **81.58 % ± 0.31** ← our best | No |
| --- *(reference targets — Geada et al.)* --- | | |
| ResNet-18 (paper) | 87.00 % | yes (+1.8) |
| Best published competition submission | 89.71 % | yes (+4.5) |
| PC-DARTS / DARTS | 90.12 % | yes (+4.9) |
| **ResNeXt-50 — public SOTA** | **93.97 %** | yes (+8.8) |
