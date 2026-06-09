# 2025 - L2M Mutual Information Scaling Law
> **Status:** reference | **Topic:** measure long-range dependence (bipartite MI) in data, derive the minimum model state size needed to capture it | **Found by:** SOTA scan June 2026 (dataset-fingerprint angle)
> **Source:** https://arxiv.org/abs/2503.04725 | **Authors:** Zhuo Chen et al. (arXiv 2025, MIT)
> **Related:** [[Papers]], [[Probe-Driven Self-Configuration]]

## TL;DR
The paper measures how mutual information between the two halves of a sequence grows with sequence length in natural language (a power law, exponent beta around 0.8) and proves the L2M condition: a model's history state (KV cache, SSM state) must scale at least as fast as that measured MI, or it cannot model the data. This is the cleanest published instance of "measure long-range dependence in the data, derive an architecture requirement".

## Key idea
Two-point MI (correlation between two distant tokens) is the classic long-range-dependence statistic but it understates what a model must store. They instead use bipartite MI: I(first half; second half) of a window, estimated with a strong reference model (LLaMA 3.1 405B) via a direct estimator and a vCLUB upper bound, validated on PG19 and Wikipedia. Theorem 5.4: dim(history state) must grow at least as fast as I_bipartite(L), which grows as L^beta. Transformers satisfy this for free (KV cache grows linearly); fixed-state SSMs and RNNs fail beyond some length unless their state is scaled up.

## Results that matter for us
- Confirms that a measurable statistic of raw data (MI scaling) cleanly separates which architecture families can or cannot fit it. This is the theoretical backbone of our pet idea.
- Power-law vs exponential MI decay is exactly the text-like vs local-texture distinction we want to detect at runtime (our Gutenberg/Language datasets vs natural images).
- Important honest caveat: their estimator needs a huge reference model. We cannot run that. But cheap two-point proxies (pixel autocorrelation length, MI between distant patches via binning) capture the decay-shape signal, just not the full bipartite quantity.

## Relevance to our pipeline
High conceptually, low as a drop-in method. It answers "is measure long-range dependence to pick the backbone already published?" with: the principle yes (for sequence models, 2025), but as a forward derivation (data property implies architecture requirement), not as a runtime selection procedure, and not for images or for choosing conv dilation schedules. Nobody has turned this into a blind per-dataset selector, which is our niche. It also supports finding (3): our text-like datasets genuinely require a different dependency-handling family.

## Future work the authors flag
Limitations: only English text, only autoregressive LMs, relies on open-source reference models, addresses capacity not reasoning. Future work: extend to vision and diffusion models (they explicitly say vision is open), design synthetic datasets with controlled MI scaling, and two open questions: can architectures be designed to just meet the theoretical minimum, and can linear-complexity architectures satisfy the L2M condition.

## Concrete idea to steal
A poor-man's MI decay curve: flatten each input tensor in its natural raster order, estimate MI(x_i; x_{i+tau}) for tau in {1, 2, 4, ..., 512} by quantizing values into 16 bins over a few hundred samples (seconds of CPU). Fit exponential vs power-law decay; power-law tail or large correlation length votes for the dilated/TCN backbone or the 1D-sequence model, fast exponential decay votes for plain local conv. Same statistic computed row-wise vs column-wise also hints whether the tensor is really an image or an encoded sequence.
