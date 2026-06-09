# 2019 - HyperSched

> **Status:** reference | **Topic:** Deadline-aware scheduler that explicitly trades off exploring new configurations vs training the best one to a hard deadline | **Found by:** SOTA scan June 2026 (budget-aware NAS angle)
> **Source:** https://arxiv.org/abs/2001.02338 | **Authors:** Richard Liaw et al. (Liaw, Bhardwaj, Dunlap, Zou, Gonzalez, Stoica, Tumanov; ACM SoCC 2019)
> **Related:** [[Papers]], [[Multi-Fidelity]]

## TL;DR
HyperSched is the canonical "maximize accuracy by a fixed deadline" paper. It extends ASHA with a deadline-aware *entrance policy* (stop admitting new candidates when remaining time makes exploring pointless) and dynamic resource reallocation (concentrate freed GPUs on the surviving leaders so the final model is fully trained when the clock runs out). The goal is exactly ours: one trained model at the deadline, not a leaderboard of half-trained ones.

## Key idea
The search has to morph from exploration into exploitation before the deadline, and the switch point is computed, not scheduled. The entrance policy admits a new trial only if

`min(R * Ta, t_f * eta) < Tn`

where R = max epochs, Ta = measured time per epoch, t_f = training time of the furthest-progressed trial, eta = halving rate, Tn = remaining time to deadline. In words: only start a new candidate if there is still enough wall-clock left for it to plausibly catch up and complete. After admission closes, freed resources flow to the top trials (sorted round-robin allocation), so the best candidate ends fully trained. Decisions use measured quantities (epoch time, resize overhead) collected during the run, no prior knowledge of the dataset, which matches our no-dataset-specific-gates rule.

## Results that matter for us
- Up to 10% absolute accuracy improvement under time constraints vs ASHA and other searchers across deep learning workloads.
- Ablation: their adaptive entrance policy beats fixed "exploration fraction" policies (the fixed 30%-style split we currently use) across multiple deadlines and cluster sizes, robust though not optimal; in simulation the gap is up to 50%.
- Confirms our finding (3) from the systems side: a fixed search slice wastes budget whenever the deadline-and-cost arithmetic says exploration can no longer pay off.

## Relevance to our pipeline
**High.** This is the published answer to "how do you split a hard wall-clock budget between searching and training the winner", which is our research question 2 in its allocation form. Differences: HyperSched assumes a multi-GPU cluster and gets much of its win from reallocating parallel resources, while we have one device, so for us only the entrance-policy arithmetic transfers, not the placement machinery. Also the admission signal is pure time arithmetic; it never looks at whether the candidates actually differ in quality (landscape flatness), so it would still explore on a dataset where search is useless, as long as time permits. Our probe idea is complementary, not subsumed.

## Future work the authors flag
From the discussion section: (a) the entrance policy estimates a candidate's epoch time assuming a single resource atom, which is overly pessimistic, a better policy would assume promising jobs get more resources; (b) reduce distributed inefficiency by colocating data-parallel workers; (c) make the exploration policy aware of cluster fragmentation and optimize placement of allocations. All cluster-oriented, none about single-device budgets or about deciding whether search is worthwhile at all.

## Concrete idea to steal
The admission inequality, ported to one GPU: before each new greedy-search candidate, check `time_to_train_candidate_to_decision + minimum_final_train_horizon < remaining_budget` using the measured seconds-per-epoch from the run so far. The first time it fails, close the search and start train-to-budget immediately. This replaces our fixed 30% search slice with a per-dataset computed cutoff and costs one comparison.
