# 2024 - Frequency-Adaptive Dilated Convolution

> **Status:** reference | **Topic:** Dilation rates adapted at runtime from frequency content of the input | **Found by:** SOTA scan June 2026 (search-space angle)
> **Source:** [arXiv:2403.05369](https://arxiv.org/abs/2403.05369) | **Authors:** Chen et al. (CVPR 2024 highlight)
> **Related:** [[Papers]], [[EGNAS]], [[2022 - RF-Next]]

## TL;DR
The closest published thing to "adapt dilation to the data at runtime": FADC predicts dilation rates spatially, per location, from local frequency components of the feature map, instead of using a fixed global rate. Plus two helpers: AdaKern (per-channel reweighting of low vs high frequency kernel components) and FreqSelect (suppress background high frequencies so the network dares to use larger dilations). Validated on semantic segmentation and object detection. Code: [github.com/Linwei-Chen/FADC](https://github.com/Linwei-Chen/FADC).

## Key idea
Dilation is a sampling rate: by Nyquist-style reasoning, high local frequency content demands small dilation (dense sampling), low frequency content tolerates large dilation (wide RF for free). So measure local frequency, then set dilation accordingly, per pixel, learned end to end. The underlying principle is the useful part for us: the frequency spectrum of the data tells you what receptive field you can afford.

## Results that matter for us
- Consistent mIoU gains on segmentation backbones and around 1 to 3% AP on small-object detection benchmarks. All dense-prediction tasks; no classification results.
- The mechanism is per-input and per-location, which adds a dilation-prediction branch and nontrivial overhead. Nobody, including this paper, publishes per-DATASET dilation adaptation at runtime; FADC is per-pixel, RF-Next ([[2022 - RF-Next]]) is per-task offline. The per-dataset gap in between is open.

## Relevance to our pipeline
Medium-low, honestly. The machinery (per-pixel dilation prediction, frequency decomposition of kernels) is built for segmentation and would be over-engineering for 1-hour blind classification; we should not port FADC. Its value to us is conceptual and as proposal ammunition: it legitimises frequency analysis as the signal for choosing dilation, and confirms that adapting dilation to data beats fixed rates when RF demand varies. Our CIFARTile-vs-MultNIST split is the dataset-level version of their foreground-vs-background split.

## Future work the authors flag
Extending frequency-adaptive design to more architectures and tasks; balancing high-frequency detail against receptive field remains open. Nothing about dataset-level or budget-constrained adaptation.

## Concrete idea to steal
A frequency probe, not the module: in the first minutes on a hidden dataset, FFT a few hundred training images (and feature maps of a tiny stem), measure where the spectral energy sits, and map that to our backbone's dilation schedule (high-frequency-dominated: cap dilation growth; low-frequency or large-layout structure: keep exponential growth). Costs seconds, needs no training, and turns our biggest blind runtime decision into a measurement.
