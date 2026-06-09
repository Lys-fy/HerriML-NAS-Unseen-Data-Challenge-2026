# 2024 - ATLAS Anytime NAS

> **Status:** reference | **Topic:** Anytime NAS via a budget-aware coordinator that splits time between zero-cost filtering and successive-halving refinement | **Found by:** SOTA scan June 2026 (budget-aware NAS angle)
> **Source:** https://arxiv.org/abs/2403.10318 | **Authors:** Naili Xing et al. (Xing, Cai, Luo, Ooi, Pei; arXiv 2024, latest revision 2026)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Zero-Cost Proxies]]

## TL;DR
First NAS method that is explicitly *anytime*: give it any wall-clock budget T_max and it returns the best architecture it can within that budget, and a better one if you give it more time. It runs a two-phase filter-and-refine scheme (zero-cost proxy scoring of M candidates, then successive halving over the top K) and a "budget-aware coordinator" that solves for M and K so the two phases exactly fill T_max. Built for tabular data with a new proxy (ExpressFlow) and a 160k-architecture benchmark (NAS-Bench-Tabular).

## Key idea
The coordinator treats budget allocation as constrained optimization. With t1 = measured time per proxy evaluation and t2 = measured time per training epoch:

- Filtering time: T1 = t1 * M
- Refinement time: T2 = K * U * t2 * floor(log_eta K), standard successive halving with U epochs per rung, keep top 1/eta each round
- Constraint: T1 + T2 <= T_max

They then fix the shape empirically: M/K of about 30 and U = 2 epochs worked across their datasets, so for any T_max the coordinator just solves for M and K. The anytime property falls out: small T_max gives small M, K; more budget grows both.

## Results that matter for us
- Up to 82.75x faster than training-based regularized-evolution NAS at matching the best architecture (Frappe), 69x on Criteo, only 1.75x on Diabetes, so gains are dataset dependent.
- Returns a usable architecture within seconds even at tiny budgets; baselines need minutes per single candidate.
- Their proxy ExpressFlow ranks first among nine zero-cost proxies on tabular data (average SRCC rank 1.0), but this is exactly the kind of claim our "proxy wall" finding makes us skeptical about for image-like data: their own related work concedes existing proxies transfer poorly across domains, which is why they had to build a new one.

## Relevance to our pipeline
**Medium-high.** This is the closest published frame to our problem: hard wall-clock budget, allocation between a cheap scouting phase and an expensive training phase, decided at runtime from measured per-unit costs (t1, t2), not from dataset-specific gates. The transferable part is the coordinator arithmetic, not the proxy: our finding (1), the proxy wall, says their filtering phase would degenerate to parameter counting on our hidden datasets, and the search space is 4-layer MLPs, nothing like our conv/dilated backbones. Note also: their allocation ratio (M/K = 30) is itself a tuned constant, they never adapt it per dataset, so they do NOT answer our "should we search at all" question, the coordinator always runs both phases.

## Future work the authors flag
The conclusion is mostly a summary. Implicit limitations: search space limited to fully connected DNNs up to 4 layers, evaluation on a handful of tabular datasets, ExpressFlow is ReLU-specific, and the coordinator constants (M/K ratio, U) are set empirically with no theory. They suggest extending to broader architecture families and grounding the coordinator parameters theoretically.

## Concrete idea to steal
Make our budget split a *computed* quantity instead of a fixed 30% slice: at runtime, measure seconds-per-epoch on the actual dataset (we already do a probe pass), then solve a tiny closed-form budget equation for "how many candidates at how many epochs fit in the slice, leaving the final train-to-budget horizon at least H epochs". If the solution gives fewer than 2 viable candidates, the equation itself says "do not search", which composes cleanly with our steepness-probe idea.
