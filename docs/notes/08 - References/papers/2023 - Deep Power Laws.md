# 2023 - Deep Power Laws

> **Status:** reference | **Topic:** Multi-fidelity HPO that pauses/resumes candidates based on power-law extrapolation of learning curves, SOTA anytime results | **Found by:** SOTA scan June 2026 (budget-aware NAS angle)
> **Source:** https://arxiv.org/abs/2302.00441 | **Authors:** Arlind Kadra et al. (Kadra, Janowski, Wistuba, Grabocka; NeurIPS 2023, "Scaling Laws for Hyperparameter Optimization")
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Zero-Cost Proxies]]

## TL;DR
DPL is the 2023+ state of the art in racing-style candidate elimination. Instead of comparing raw short-budget validation scores (the mistake behind our proxy wall), it fits an ensemble of neural power-law models to each candidate's partial learning curve, extrapolates to the FULL budget, and uses Expected Improvement on the extrapolated final performance to decide which candidate to resume next (gray-box pause/resume, nothing is killed permanently). Best anytime results against 7 SOTA baselines over 59 tasks spanning tabular, image, and NLP.

## Key idea
Learning curves of deep nets follow power laws, so the right comparison between two candidates is not "who is better at epoch 3" but "whose fitted power law predicts a better epoch-50 value". The surrogate is an ensemble of small NNs constrained to power-law-shaped outputs, giving mean and variance of the predicted full-budget performance; EI over that posterior picks the next candidate to advance by one fidelity step. Elimination is implicit and reversible: bad candidates just never get scheduled again, which means a decision is only ever as final as the evidence behind it.

## Results that matter for us
- Best anytime regret on all three benchmark families (LCBench, TaskSet, PD1), beating ASHA, Hyperband, BOHB, DEHB, DyHPO and others; anytime performance is exactly the metric a 1-hour deadline cares about.
- Direct evidence against naive short-horizon ranking: methods that rank on raw low-fidelity scores get overtaken once curves cross; curve extrapolation handles the crossing-curves problem that broke our short-training candidate comparisons (finding 1).
- Tuning a small proxy model and transferring (their LLM experiment) found the oracle config of a large model in 3.6h, vs 21.5h for one full training, evidence that careful extrapolation beats brute force under tight budgets.

## Relevance to our pipeline
**Medium.** The full machinery (ensemble surrogate, EI loop, pause/resume scheduling) is overkill for our 30-ish candidate evaluations per hour, and it is HPO over fixed search spaces, not macro NAS. But the core insight transfers cheaply and lands directly on our proxy wall: our verification step currently re-trains a candidate at a longer horizon and compares raw scores; fitting a 3-parameter power law to both candidates' partial curves (least squares, milliseconds) and comparing extrapolated end-of-budget accuracy is strictly more information from the same compute. The reversible-elimination idea also fits our greedy accept/reject: an "uncertain" verdict could be parked rather than decided.

## Future work the authors flag
From the conclusion: (a) uncertainty estimates from deep ensembles turned out worse than classic BO surrogates such as Gaussian processes, contrary to common perception; (b) training an ensemble of power-law models adds real computational cost; (c) they plan to combine power laws with Gaussian processes and to investigate additional fidelity types beyond epochs.

## Concrete idea to steal
Replace raw-score comparison in our verification step with power-law extrapolation: fit `y(e) = a - b * e^(-c)` to the validation-accuracy curve of incumbent and challenger over the epochs we already train, extrapolate both to the planned train-to-budget horizon, and accept the challenger only if its extrapolated final accuracy wins by more than the fit residual. Cost is near zero, and it directly attacks the crossing-curves failure mode we measured.
