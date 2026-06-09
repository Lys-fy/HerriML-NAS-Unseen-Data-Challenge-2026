# 2021 - Training Speed Estimation (TSE)

> **Status:** reference | **Topic:** Rank architectures by summed training losses instead of early validation accuracy | **Found by:** SOTA scan June 2026 (learning-curve angle)
> **Source:** [arXiv:2006.04492](https://arxiv.org/abs/2006.04492) | **Authors:** Ru et al. (NeurIPS 2021, spotlight)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Zero-Cost Proxies]]

## TL;DR
"Speedy Performance Estimation for Neural Architecture Search". Replace early-stopped validation accuracy with the sum of training losses over epochs as the ranking signal. Motivated by a PAC-Bayes connection between training speed and generalization. Model-free, zero extra compute (training losses are already logged), no surrogate, no hyperparameter tuning. At 10 of 200 epochs on NAS-Bench-201 it reaches Spearman rho around 0.85-0.90 with final test rank, while early-stopped validation accuracy sits around 0.40-0.50 and learning-curve SVR extrapolation around 0.60-0.70.

## Key idea
Early validation accuracy is noisy and biased by where each architecture is in its optimization trajectory; how FAST a network drives its training loss down is a more stable fingerprint of final quality. Variants:
- TSE: sum of mean training losses over all epochs so far.
- TSE-E: sum over only the last E epochs (default E=1, drops the noisy burn-in).
- TSE-EMA: exponentially weighted sum, gamma = 0.999, weights later epochs more.

## Results that matter for us
- NAS-Bench-201 at T=10 epochs: TSE-EMA rho ~0.85-0.90 (CIFAR-10/100) vs ~0.40-0.50 for early-stopped val accuracy. This is the single largest published gap between a curve-based signal and the naive signal we currently use in bake-offs.
- Holds on DARTS-space and ResNet/ResNeXt families, including ranking within the top 1 percent.
- Effective over a wide range of cutoffs, but degrades when training losses approach zero (overfitting regime, T > 0.5 of full budget on easy datasets): once everyone fits the train set, training speed stops discriminating.
- Assumes all candidates trained under the SAME hyperparameters; comparing across different optimizers or schedules breaks the signal.
- Independently confirmed in White et al. 2021 ([[2021 - How Powerful are Performance Predictors]]): SoTL-E (same quantity) is the best learning-curve method in their 31-predictor study.

## Relevance to our pipeline
High, and the cheapest option on the table. Our proxy wall is literally "short-train validation ranking lies"; TSE claims the fix is to rank on a different column of the same log. No uncertainty estimate though, it is a point score, so it cannot by itself implement an abstain rule; it can replace the signal being ranked, and significance gating must come from elsewhere. Caveats for us: our 3 backbones are very different families, so loss scales may not be comparable across them (TSE was validated within a search space with shared training hyperparameters); and label-noisy datasets like Gutenberg may hit the degenerate near-zero-loss regime late, which is when TSE goes blind.

## Future work the authors flag
Extending the estimator to compare candidates trained under DIFFERENT hyperparameters is explicitly left open (this is exactly our cross-backbone case, so treat cross-family TSE as unvalidated). No other limitations section; robustness to overfitting regimes is acknowledged but not solved.

## Concrete idea to steal
If our 13-dataset CSVs include per-epoch training loss (and if not, add the column, it is free): compute TSE-E and TSE-EMA at the 3/5/10-minute cutoffs for all 3 backbones on all 13 datasets and score winner-pick accuracy against the known full-budget winner. Baseline to beat: 6/13 (3 min) and 8/13 (5 and 10 min) from validation-accuracy ranking. Also test a rank-ensemble of TSE-EMA and validation accuracy. If cross-family loss scales are a problem, compare TSE within-family (for op-swap and width/depth decisions, where hyperparameters are shared) where it should shine most.
