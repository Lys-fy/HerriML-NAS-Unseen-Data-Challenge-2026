# 2023 - PASHA

> **Status:** reference | **Topic:** ASHA variant that grows the maximum training fidelity only while candidate rankings are still unstable | **Found by:** SOTA scan June 2026 (budget-aware NAS angle)
> **Source:** https://arxiv.org/abs/2207.06940 | **Authors:** Ondrej Bohdal et al. (Bohdal, Balles, Wistuba, Ermis, Archambeau, Zappella; ICLR 2023)
> **Related:** [[Papers]], [[Multi-Fidelity]]

## TL;DR
PASHA (Progressive ASHA) attacks the question "how much fidelity does the search actually need?" ASHA needs a user-chosen maximum resource (epochs); PASHA instead starts with a tiny maximum and only unlocks the next rung when a measured signal says the current fidelity is not enough: it checks whether the ranking of configurations in the top two rungs is consistent. If rankings have stabilized, it stops the search early and returns the leader. Cuts tuning time to roughly one third of ASHA with no noticeable quality loss.

## Key idea
Two mechanisms, both cheap:

1. **Stop-growing rule:** increase the maximum resources only when the ranking of configurations between the two highest rungs becomes inconsistent. Stable ranking = the search has learned all it can at this fidelity = stop.
2. **Soft ranking with automatic noise estimation:** raw rankings flip due to training noise, so configurations whose performance differs by less than a threshold epsilon are treated as tied. Epsilon is estimated at runtime by measuring how often configurations swap ranks across epochs (frequent swappers define the noise floor). This makes the stability test statistically honest instead of reacting to noise.

This is, as far as our scan found, the closest published mechanism to "spend budget on search only while a measured signal says searching is still informative."

## Results that matter for us
- Roughly 3x reduction in tuning time vs ASHA on NAS-Bench-201 and similar benchmarks, with the same final configuration quality in most cases.
- On benchmarks with few rungs (LCBench, max 50 epochs) speedups are modest: PASHA needs a deep fidelity ladder to have room to stop early. Authors recommend max/min resource ratio of at least 100 with eta = 3, achievable by counting gradient updates instead of epochs.
- The soft-ranking noise estimate directly addresses our finding (1), the proxy wall: they confirm that near-tied candidates swap ranks under short training, and they quantify the tie threshold at runtime rather than pretending short-budget rankings are exact.

## Relevance to our pipeline
**High.** This is the strongest prior art for our "adaptive search budget" idea, so we must cite it and differentiate. What PASHA does: adaptively decides how much fidelity (and hence budget) the search consumes, driven by a runtime ranking-stability signal. What PASHA does NOT do: it never decides whether to search at all, it always evaluates its full sample of configurations at the lowest rung; and it frames savings as cost reduction, not as budget reallocated to training the winner longer (our train-to-budget would absorb the freed time). Caveat for us: with a 1-hour budget and expensive epochs our fidelity ladder is short, exactly the regime where PASHA's own limitations section says gains shrink.

## Future work the authors flag
From the conclusions and limitations: (a) speedups are small when the number of rungs is small, mitigate by using finer resource units (gradient updates); (b) investigate how rung and resource-level definitions affect decisions, for multi-fidelity methods in general, not just PASHA; (c) combine PASHA with Bayesian-optimization search strategies; (d) combine with transfer-learning multi-fidelity methods such as RUSH to cut time further.

## Concrete idea to steal
The soft-ranking stability test with runtime noise estimation as our search kill switch: during greedy search, track validation scores of the current top candidates across epochs, estimate the noise threshold epsilon from rank swaps, and when all surviving candidates are within epsilon of each other (a flat landscape, our finding 3), abort the search slice immediately and hand the remaining time to train-to-budget. This is a statistically grounded version of our landscape-steepness probe and costs nothing extra to compute.
