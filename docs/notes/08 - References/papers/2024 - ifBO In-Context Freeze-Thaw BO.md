# 2024 - ifBO In-Context Freeze-Thaw BO

> **Status:** reference | **Topic:** FT-PFN surrogate extrapolates many partial curves jointly (conditioned on configs) for freeze-thaw style resource allocation | **Found by:** SOTA scan June 2026 (learning-curve angle)
> **Source:** [arXiv:2404.16795](https://arxiv.org/abs/2404.16795) | **Authors:** Rakotoarison et al. (ICML 2024)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[2023 - LC-PFN Bayesian Learning Curve Extrapolation]]

## TL;DR
The successor to LC-PFN. FT-PFN is a prior-data fitted transformer that takes ALL partial curves observed so far plus their hyperparameter configurations as context, and extrapolates any of them in a single forward pass. Combined with a randomized multi-fidelity acquisition (MFPI-random) it forms ifBO, a freeze-thaw Bayesian optimizer: at each step it asks "which paused candidate should get the next epoch(s)?" State of the art on LCBench, PD1, Taskset; predictions 10-100x faster than deep-GP (DyHPO) or deep-ensemble (DPL) surrogates, with far better log-likelihood (calibration). Code: github.com/automl/ifBO.

## Key idea
Treat candidate selection as dynamic resource allocation over partially trained candidates, not as a one-shot ranking. The surrogate is never refit online (pure in-context inference), removing the overhead, instability, and hyper-hyperparameters of online-trained surrogates. The synthetic prior generates task-correlated curve sets: a random neural network maps configurations to curve parameters of a combined basis model (power law, exponential, inverse log, Hill), so the PFN learns that similar configs have similar curves.

## Results that matter for us
- Calibration gap is large: log-likelihood about +2.1 (FT-PFN) vs -0.4 (DyHPO) vs -9.2 (DPL) on LCBench with 1,800 context samples. If we want uncertainty we can act on, this is the best published surrogate.
- 10-100x faster inference than the GP and ensemble alternatives; single forward pass, no refitting.
- Freeze-thaw framing matches our verification loop better than one-shot ranking: keep all backbones paused, repeatedly extend whichever curve the model is most uncertain-but-hopeful about, instead of committing at minute 3.
- Trained for budgets up to 1,000 steps, up to 10 hyperparameters, everything normalized to [0,1]: our 3-backbone, tens-of-epochs setting fits comfortably inside this envelope.

## Relevance to our pipeline
Medium-high. The surrogate itself is directly usable: encode our 3 backbones as 1-3 dummy "hyperparameters" and let FT-PFN extrapolate all three curves jointly, which should beat per-curve LC-PFN because evidence is shared across candidates on the same dataset. The full ifBO loop (acquisition over a big config space) is more machinery than our 3-way bake-off needs, and it inherits the parametric-prior blind spot around LR-schedule inflections (same family of basis curves as LC-PFN). It optimizes simple regret for finding one good config, not rank fidelity, so our bake-off metric (correct winner picked) still has to be validated separately.

## Future work the authors flag
- Scaling to modern deep learning (LLM pretraining scale) remains open; they plan larger models, more data, larger context sizes.
- In-context meta-learning from related tasks' curves, incorporating user priors, gradient statistics.
- Parallel/batch suggestion to reduce critical-path overhead.
- Stated constraints: metrics and configs normalized to [0,1], max 10 hyperparameters, max 1,000 budget steps.

## Concrete idea to steal
Replay our 13 bake-offs offline as freeze-thaw allocation: give FT-PFN the three backbone curves truncated at minute 3 equivalents, then simulate "spend the next epoch on the candidate whose posterior gives the highest probability of improving over the current leader", and measure (1) winner-pick accuracy vs our fixed-horizon 6/13 and 8/13 baselines, and (2) how many total epochs the allocator needs before its top-1 posterior probability exceeds, say, 0.8. That second number is a data-driven answer to "how long must the verification re-train be", per dataset, at zero GPU cost since the curves are already logged.
