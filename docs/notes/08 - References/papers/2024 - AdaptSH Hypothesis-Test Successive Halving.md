# 2024 - AdaptSH Hypothesis-Test Successive Halving

> **Status:** reference | **Topic:** Successive halving where elimination is a corrected hypothesis test instead of a fixed-fraction cut | **Found by:** SOTA scan June 2026 (learning-curve angle)
> **Source:** [doi:10.3390/e26020150](https://www.mdpi.com/1099-4300/26/2/150) | **Authors:** Chen and Ghosh (Entropy 2024)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[2021 - Training Speed Estimation (TSE)]]

## TL;DR
AdaptSH replaces successive halving's "drop the worst half every round" with a statistical decision: at each round, compare every candidate against the current best via a relative-similarity hypothesis test on an exponentially weighted performance statistic (MMD squared, since their candidates are generative models), correct for multiple testing (Benjamini-Yekutieli), and eliminate only candidates that are SIGNIFICANTLY worse. Candidates that are not significantly separated all survive and keep training. Beats vanilla successive halving across budgets on their (small, synthetic) benchmarks.

## Key idea
Fixed-fraction elimination kills slow starters on noise; significance-gated elimination adapts the cut to the actual evidence. The exponential weighting of intermediate scores is their nod to learning-curve dynamics: recent epochs count more. The lineage goes back to Hoeffding races (Maron and Moore, NeurIPS 1993: drop a model only when confidence intervals separate; the survivors are returned as a statistically indistinguishable set) and F-Race (Birattari et al. 2002, Friedman-test racing, the engine inside irace). AdaptSH is the recent DL-adjacent revival of that idea inside a multi-fidelity scheduler.

## Results that matter for us
- Direct published precedent for "keep training contenders until separation is significant": elimination happens only on statistical evidence, and non-separated candidates persist to the next fidelity. This is the formal version of our crude verification re-train.
- The racing family (Hoeffding races) also gives the abstain semantics we want: if the budget ends before separation, you get a SET of indistinguishable candidates, and a pipeline is free to resolve that set by a prior preference (our default backbone). Nobody we found publishes exactly "abstain and keep the incumbent default", but Hoeffding races plus an incumbent tie-break is that mechanism in two published pieces.
- Honest caveats: their test statistic is MMD between generated and real samples (generative models), not validation accuracy; the asymptotic-normality assumptions would need rechecking for accuracy-over-epochs; experiments are only two synthetic 2D/3D datasets. This is a mechanism citation, not an off-the-shelf tool.

## Relevance to our pipeline
Medium for the specific method, high for the framing. Our accept/reject problem ("is the challenger really better than the incumbent backbone, or is the 3-minute lead noise?") is exactly a race between two arms. The blocker for a textbook test is that we have ONE curve per candidate per dataset, not repeated samples; variance must come from somewhere (across-epoch residuals of a curve fit, bootstrap over validation batches, or across-dataset priors from our 13 public datasets). The multiple-testing correction matters for our per-layer op-swap loop, where we currently run many sequential accept/reject decisions and got poisoned by exactly the false positives such a correction controls.

## Future work the authors flag
- Extending the statistical theory to loss metrics other than MMD (they note new theory would be needed: accuracy-based racing for NAS is open).
- Relaxing the asymptotic normality assumptions.
- Broader, real-world experimental validation beyond the two synthetic datasets.

## Concrete idea to steal
Build a two-arm race rule and validate it offline on our CSVs: at each epoch, compute the challenger-minus-incumbent gap in smoothed validation accuracy, estimate its noise from the recent per-epoch fluctuations of both curves (or bootstrap over the final-window epochs), and accept the challenger only when the gap is significant at alpha after a Bonferroni-style correction over however many swaps the run will attempt; otherwise keep the incumbent (abstain = default). Sweep alpha and the smoothing window over the 13 datasets and measure two things: how often the rule accepts the wrong backbone (our current 6/13 failure mode) and how much budget it spends before deciding. The race framing converts our ad-hoc "verify longer" into a tunable false-accept rate.
