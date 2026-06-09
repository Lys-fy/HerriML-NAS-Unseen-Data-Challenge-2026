# 2023 - LC-PFN Bayesian Learning Curve Extrapolation

> **Status:** reference | **Topic:** Transformer pre-trained on synthetic curves does Bayesian learning-curve extrapolation in one forward pass | **Found by:** SOTA scan June 2026 (learning-curve angle)
> **Source:** [arXiv:2310.20447](https://arxiv.org/abs/2310.20447) | **Authors:** Adriaensen et al. (NeurIPS 2023)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Zero-Cost Proxies]]

## TL;DR
LC-PFN is a prior-data fitted network (PFN): a transformer trained once, offline, on ~10M synthetic learning curves sampled from a parametric prior (the Domhan 2015 family of basis curves: power law, exponential, log, etc.). At runtime you feed it a partial curve (any cutoff, even 10% of the budget) and it returns the full posterior predictive distribution over future curve values in under 100 ms on CPU. It matches or beats the Domhan MCMC approach while being over 10,000x faster, and a simple early-stopping rule built on it gives 2-6x model-selection speedups on most benchmarks. Code and pretrained models: github.com/automl/lcpfn.

## Key idea
Amortize Bayesian curve inference. Instead of running MCMC over a parametric curve model per candidate (minutes per inference), train a transformer to approximate the posterior predictive distribution in-context. During PFN training, cutoffs T' are sampled uniformly, so the model learns to extrapolate from very short prefixes. Output is a discretized (Riemann) distribution, so you get calibrated quantiles, not just a point estimate.

## Results that matter for us
- Validated on 20,000 real curves from LCBench, NAS-Bench-201, Taskset, PD1 (MLPs, CNNs, RNNs, Transformers; tabular, image, text, protein). Beats or matches MCMC variants on log-likelihood and MSE at 10/20/40/80 percent cutoffs.
- Inference < 0.1 s on CPU: genuinely negligible overhead for a 1-hour budget.
- Early-stopping criterion: after each epoch, compute P(final accuracy will exceed best-so-far); stop the run if that probability falls below 1 - 0.95 = 0.05 (same rule as Domhan 2015). Gives 2-6x speedups on 45 of 53 tasks.
- Critical negative result: on NAS-Bench-201 (CIFAR-style CNN training with LR schedules) ALL termination criteria fail, including theirs and plain Patience. The curves have an inflection point (slow start, then a jump at LR decay) that the prior does not model. This is exactly the curve shape our backbones produce, so do not expect plug-and-play reliability.

## Relevance to our pipeline
High. This is the strongest published answer to our proxy wall in the "extrapolate instead of trust the short ranking" direction: per-candidate, needs only the curve itself (no cross-architecture training data), calibrated uncertainty, and effectively zero compute. The honest caveat is the NAS-Bench-201 failure mode: our cosine/step LR schedules create the same inflection shapes that broke their prior, and our bake-off logs (leaders crossing between minute 5 and 10) are exactly the symptom of inflection-driven rank churn. So LC-PFN must be validated on OUR curves before trusting it, which we can do for free.

## Future work the authors flag
- Better priors that model divergence, slow start, double descent, correlated heteroscedastic noise (explicitly motivated by the NAS-Bench-201 failure).
- Few-shot in-context meta-learning: condition on other curves from the same task or related tasks, plus hyperparameters and meta-features (later realized as FT-PFN, see [[2024 - ifBO In-Context Freeze-Thaw BO]]).
- Integrating LC-PFN as a termination criterion in standard DL libraries.
- PFN-specific limits: prior must be generative, high-entropy priors are hard.

## Concrete idea to steal
Run the pretrained LC-PFN (pip-installable, CPU-only) over our 13-dataset, 3-backbone learning-curve CSVs: at cutoffs 5, 10, 15, 20, 30 epochs, extrapolate each backbone's final accuracy and check (a) whether the predicted-mean ranking beats the raw short-train-accuracy ranking on the 13 bake-offs (current baseline: 6/13 correct at 3 min), and (b) whether the posterior intervals are calibrated on our curve shapes or break at the LR-decay inflection like on NAS-Bench-201. If (b) fails, the fix the authors suggest is a prior that knows about LR schedules; cheaper for us is to feed curves only after the first LR drop or to rank by predicted-final with an uncertainty gate.
