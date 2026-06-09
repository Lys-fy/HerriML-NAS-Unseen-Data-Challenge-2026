# 2022 - DASH Efficient Architecture Search for Diverse Tasks
> **Status:** reference | **Topic:** per-task search over kernel sizes and dilations inside a fixed CNN topology | **Found by:** SOTA scan June 2026 (dataset-fingerprint angle)
> **Source:** https://arxiv.org/abs/2204.07554 | **Authors:** Junhong Shen et al. (NeurIPS 2022)
> **Related:** [[Papers]], [[Probe-Driven Self-Configuration]]

## TL;DR
DASH keeps the network topology fixed (a standard CNN backbone) and searches only over kernel sizes and dilation rates per layer, using a differentiable mixture of operations made cheap via Fourier diagonalization of convolution. It got best-known automated performance on 7 of 10 diverse tasks (NAS-Bench-360: PDEs, protein folding, ECG, audio, vision) at roughly 2x the cost of training one backbone.

## Key idea
The authors' diagnosis matches ours exactly: diverse tasks mostly do not need new topologies, they need the right receptive field. Small kernels suit local visual features, large kernels and dilations suit long-range or sequence-like structure. Instead of measuring the data, they make kernel size and dilation differentiable: each conv layer is a weighted mixture over a grid of (kernel size k, dilation d) options, and three tricks (mixed weights, FFT-based convolution, Kronecker dilation) reduce the cost from O(n |K| |D|) to roughly O(n log n), about 10x faster than DARTS on the same space.

## Results that matter for us
- Receptive field is the dominant axis of architecture adaptation across modalities, validated on tasks spanning the same kind of diversity as our challenge (sequence-ish, image-ish, scientific data).
- The aggregate winner across diverse tasks is a dilated-conv-family backbone with per-task (k, d) settings, which independently confirms why our TCN-style backbone wins broadly.
- Combined search plus retrain stayed within about 2x of plain training on 6 of 10 tasks, so the approach is in principle budget-compatible.

## Relevance to our pipeline
High. This is the closest published version of "pick receptive field per dataset", and it validates finding (3): different families want different receptive fields more than different topologies. The gap: DASH still selects (k, d) by gradient-based search (a candidate-ranking mechanism, subject to our proxy wall during short training), not by a direct data measurement. Our pet idea is the measurement-based shortcut DASH does not take. Caveat: a differentiable mixture trained jointly with weights is not the same failure mode as ranking separately trained candidates, so a tiny DASH-style warmup may dodge the proxy wall.

## Future work the authors flag
The paper and the CMU blog post frame DASH as "an early step in AutoML for diverse tasks" and point to broader decoupling of topology from operations; no concrete limitation list. Implicitly: search is still gradient-based, the topology is fixed, and there is no data-driven prior on which (k, d) region to search.

## Concrete idea to steal
Replace our backbone bake-off for receptive field choice with a single 1-2 epoch run of one small CNN whose layers are mixtures over 3-4 (kernel, dilation) options, then read off the argmax alpha per layer and instantiate our real backbone with that dilation schedule. The alphas are trained jointly with weights (one model, one run), so this costs about one short training, not N.
