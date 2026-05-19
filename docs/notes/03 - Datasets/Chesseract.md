# Chesseract

> **Status:** drafting | **Topic:** Late-game chess board states one-hot-encoded as 12×8×8 tensors. 3 classes (W/D/B).
> **Related:** [[Datasets Overview]], [[Failure Modes]]

![[Chesseract.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


Late-game (final 15% of moves) board positions from games played by 8 grandmasters (Fischer, Kasparov, Carlsen, Anand, Nakamura, Karpov, Caruana, Tal). Each board is one-hot-encoded by piece type × colour, producing **12 channels** (6 piece types × 2 colours) on an 8×8 board. Label = game outcome (White wins, Draw, Black wins).

| Field | Value |
|---|---|
| Codename | Chester |
| Era | CVPR-2023 |
| Shape | 12×8×8 (channels-first, 12 channels) |
| Train / val / test | 49,998 / 10,001 / 10,001 (Geada says 9,999 / 9,999 — small discrepancy, trust local README) |
| Classes | 3 (White wins / Draw / Black wins) |
| Benchmark (ResNet-18) | 57.83% (`metadata`) — random is 33.3% |
| Best off-the-shelf CNN (Geada) | DenseNet 59.60% |
| Best NAS method (Geada Tab. 2) | random-Bonsai **68.83%** (Bonsai-Net 60.76%, the best directed method) |
| Best 2023-comp submission | 62.98% |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/Chesseract_Dataset/24118743 |

## The 12-channel pathology

Most NAS pipelines (and many off-the-shelf CNN architectures) **hard-code 1 or 3 input channels**. Chesseract's 12 channels broke many 2023-competition submissions immediately. This is in [[Failure Modes]] — every layer of our pipeline must be channel-agnostic from `metadata['input_shape'][1]`.

The 8×8 spatial extent is also unusually small. Many architectures' default downsampling will reduce 8×8 to 1×1 or smaller before reaching the head — preventing learning. Custom stride / pool tuning needed.

## Type and difficulty

**Type-1** (Geada): a chess expert can usually estimate who is winning from a late-game board state. NAS asks: can a network learn this signal from one-hot encodings?

The data has temporal leakage prevention: no single position from the same game appears in multiple splits.

## Why random search wins here

Geada Tab. 2 shows random search in **Bonsai search space** beats Bonsai-Net directed search by ~8 pp (68.83% vs 60.76%). PC-DARTS (57.20%) and DrNAS (58.24%) barely beat ResNet-18.

The most likely interpretation: directed NAS methods overfit to architectures resembling those that work on CIFAR/ImageNet, but Chesseract's 12-channel 8×8 input has no overlap with that distribution. Random sampling explores wider and finds atypical-but-good configurations. This generalises to a Strategy-section principle: **on weird-shape datasets, random search is a strong baseline** — see [[Predictor-Guided Evolutionary]] caveats.

## Implications for our pipeline

- **Channel-agnostic data flow**: read `metadata['input_shape']` and propagate, never assume 1/3 channels.
- **Spatial-aware downsampling**: 8×8 input cannot survive ResNet-18-style stride-2 conv + 4 stride-2 stages → 0×0 by stage 4. Either reduce strides or pad.
- Score slope at 57.83% is ~0.24 / 1%. Maximum reachable upside ~+10 pts (capped); a 70% raw accuracy nets ~+2.9. Modest reward, big risk if the channel issue is mishandled. Treat as a **reliability-first** target.
