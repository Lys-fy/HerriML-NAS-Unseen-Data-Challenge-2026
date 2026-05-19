# CIFARTile

> **Status:** drafting | **Topic:** Four CIFAR-10 images tiled in a 2×2 grid; label = (number of distinct CIFAR classes) − 1.
> **Related:** [[Datasets Overview]], [[Score-Efficient Targeting]], [[DARTS Family]]

![[CIFARTile.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


Four CIFAR-10 images placed in a 2×2 grid (so a 3×64×64 image). Label is the count of *distinct* CIFAR-10 classes among the four sub-images, minus one — so labels run 0–3.

| Field | Value |
|---|---|
| Codename | (no codename in metadata; filename is CIFARTile) |
| Era | CVPR (2021–23) |
| Shape | 3×64×64 (channels-first) |
| Train / val / test | 45,000 / 15,000 / 10,000 |
| Classes | 4 |
| Benchmark (ResNet-18) | 45.56% (Geada Tab. 1) — barely above random (25%) |
| Best off-the-shelf CNN (Geada) | DenseNet 51.28% |
| Best NAS method (Geada Tab. 2) | PC-DARTS **92.28%** |
| Best 2023-comp submission | 73.08% |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/CIFARTile_Dataset/24551539 |

## The headline NAS dataset

CIFARTile is the **canonical NAS-vs-CNN signal** in this benchmark suite. ResNet-18 is stuck near random; PC-DARTS jumps to 92%. **+46.7 pp** is by far the largest NAS reward on any of the 8 CVPR-era datasets.

The 2023 competition ceiling (73.08%) lags the offline NAS upper bound (92.28%) by ~19 pp — suggesting that even competition-grade NAS pipelines couldn't extract the full search-space gain in the time-constrained setting.

## Why CNNs fail and NAS wins here

The task requires **multi-instance reasoning**: identify four objects, then count distinct classes among them. Three things have to happen:

1. The model must classify each tile (or at least each tile's local features).
2. It must somehow compare across the four sub-images.
3. It must count the equivalence-classes.

Standard image classifiers (CIFAR-trained ResNet-18, etc.) don't have a built-in mechanism for step 2. NAS finds architectures with the right inductive biases (multiple-receptive-field paths, attention-like skip connections) for the comparison step. Geada notes VGG16 is the worst here (24.43%) — its rigid sequential architecture lacks the path multiplicity to capture cross-tile relationships.

## Pathologies

- **2D-shape sensitivity**: the 2×2 tile boundary is at exactly H/2, W/2 = 32. Any data augmentation that crosses that boundary (e.g. 8-pixel random crop in the centre) will scramble the inter-tile relationship.
- The four labels are not equally likely — geometric constraints make label-3 (4 distinct classes) the least common per random sampling, and the README does not state whether the dataset was rebalanced.
- The label encodes a *symmetric function* of the four tile classes — order shouldn't matter. CNNs that develop position-dependent features will overfit to the training tile-arrangement distribution. A NAS that prefers small kernels + wide receptive fields plausibly does better here.

## Implications for our pipeline

- This is **the** dataset to point a real NAS at. If our search produces nothing better than ResNet-18, we have ~40 pp on the table.
- Score slope at benchmark 45.56% is ~0.18 / 1%, so absolute gain is huge but per-point conversion is modest. Reaching 95% would still net ~+9 pts. Falling back to ResNet-18 nets 0.
- Recommended budget allocation: oversize the search budget for CIFARTile relative to a uniform split. See [[Budget Allocation]] and [[Search vs Train Split]].
- A NAS method that cannot grow receptive field beyond ~32 pixels likely underperforms here.
