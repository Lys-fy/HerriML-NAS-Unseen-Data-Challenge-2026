# 2021 - How Powerful are Performance Predictors

> **Status:** reference | **Topic:** Large-scale study of 31 NAS performance predictors; which family wins at which budget | **Found by:** SOTA scan June 2026 (learning-curve angle)
> **Source:** [arXiv:2104.01177](https://arxiv.org/abs/2104.01177) | **Authors:** White et al. (NeurIPS 2021)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Zero-Cost Proxies]]

## TL;DR
The reference empirical study on the proxy wall. 31 predictors (learning-curve extrapolation, zero-cost proxies, model-based surrogates, weight sharing) compared on NAS-Bench-101/201/301 and an NLP space, across the two budget axes that matter: initialization time (one-off cost before any query) and query time (cost per candidate). Main message: there is no universal winner, the best predictor depends on your budget point, and families are complementary (combining them, their OMNI predictor, beats any single one by a wide margin). Code: github.com/automl/naslib.

## Key idea
Stop asking "what is the best proxy" and ask "what is the best proxy given (init budget, query budget)". Learning-curve methods need zero initialization but high query cost (you must partially train every candidate). Model-based predictors (GCN, XGBoost, NGBoost) need many fully trained architectures up front, which a 1-hour blind pipeline can never afford. Zero-cost proxies are nearly free on both axes but cap out early and do not scale to harder spaces.

## Results that matter for us
- In the zero-init, high-query regime (ours: no offline architecture corpus for hidden datasets, but we do train candidates), learning-curve methods dominate, and the best one is SoTL-E, the sum of training losses at the last epochs, identical to TSE-E ([[2021 - Training Speed Estimation (TSE)]]). Plain parametric curve fitting (Domhan-style LCE) and curve SVR rank below it.
- Learning-curve fidelities tested at 4, 12, 36, 108 epochs on NAS-Bench-101: predictive quality rises steeply with fidelity, which puts numbers on "how many epochs until the ranking stabilizes".
- Zero-cost proxies (SynFlow, Jacobian covariance) are strong on the small cell benchmarks but "did not yield high Kendall tau" on the larger DARTS space: consistent with our observation that zero-shot proxies degenerate to parameter counting on real spaces.
- Combining families helps a lot: OMNI feeds curve features plus zero-cost features into a model and gains about 30 percent Kendall tau over the next best in some settings. Even without a trained model, the lesson is to ensemble independent signals (val acc rank + SoTL rank + params) rather than trust one.

## Relevance to our pipeline
High as a map, medium as a method. It does not give a dataset-conditional rule for when a 3-minute ranking is trustworthy (nobody does), but it firmly answers which signal to rank on at our budget point (SoTL-E over early val accuracy) and warns off model-based predictors that need an init corpus we cannot collect blind. The 4-to-108-epoch fidelity sweep is the closest published analogue of our 3/5/10-minute bake-off instability.

## Future work the authors flag
- Only four search spaces and two domains tested (image classification, NLP); they call for real-world, non-tabular evaluations: our 13-dataset suite is exactly such a testbed.
- Better ensembling than feature concatenation, and quantifying the correlation structure among predictor families.

## Concrete idea to steal
Replicate their budget-regime table on our own data: for each of the 13 datasets and each cutoff (3/5/10 min), compute Kendall tau and winner-pick accuracy for (a) early val accuracy, (b) SoTL-E, (c) a parametric power-law fit's predicted final value, (d) the rank-average of (a)+(b). Their result predicts (b) > (c) > (a) and (d) best overall; if that ordering holds on our curves, we swap the bake-off decision rule with one line of code and no extra GPU time.
