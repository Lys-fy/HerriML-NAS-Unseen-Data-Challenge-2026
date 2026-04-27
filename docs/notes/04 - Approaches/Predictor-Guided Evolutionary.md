# Predictor-Guided Evolutionary

> **Status:** current | **Topic:** Surrogate accuracy predictor + evolutionary search; the inferred approach of the 2nd and 3rd place 2025 teams.
> **Related:** [[Approaches Overview]], [[2025 - Sichuan teams]], [[EGNAS]], [[Weight-Sharing One-Shot]], [[Zero-Cost Proxies]]

A two-loop NAS strategy: an evolutionary algorithm proposes architectures; a learned surrogate scores them cheaply; only the top-k get fully trained. Inferred 2025 approach for both [[2025 - Sichuan teams]] entries (Yanan Sun's group, Sichuan University).

## Pattern

```
1. Encode arch → vector (graph adjacency / OP sequence / embedding).
2. Train predictor f(arch) → val_acc on a small set of fully-trained pairs.
3. EA loop:
     mutate / crossover candidates
     score with predictor (no training)
     select top-k by predictor score
4. Fully train top-k, return the best.
5. Optionally refit predictor on new (arch, acc) data and re-loop.
```

Mutations are typical EA primitives: swap an op, change kernel, add/remove a layer. Selection is tournament or rank-based.

## Predictor families

| Predictor type | Inputs | Examples |
|---|---|---|
| **MLP / GBM on hand-features** | depth, width, FLOPs, params, op counts | NAS-RL ranker, early predictor papers |
| **GNN on arch graph** | adjacency + node ops | NPENAS, BANANAS-GNN |
| **Listwise rankers** | pairs of arches | Sun group's "Listwise Ranking Predictor" (Li et al., Swarm and Evol. Comp., 2025) |
| **Domain-adaptive** | arches + dataset features | Sun group's "DAP: Domain Adaptive Performance Predictor" (Song et al., IEEE TC, 2026) |
| **Online / active** | growing labelled set during search | Sun group's "Online Performance Predictor" (Song et al., IEEE TSMC, 2025) |

The Sun group's published direction (see [[2025 - Sichuan teams]]) is exactly this family — DAP and the online predictor are the load-bearing techniques behind their submissions.

## Why predictors

- **Throughput**: a forward pass on a predictor is sub-millisecond. EA can evaluate 10⁵ candidates where full training caps at 10¹.
- **Reuse**: a predictor fitted on one dataset can sometimes transfer (DAP explicitly).
- **Decoupling**: search step is cheap, training step is expensive — match each to its right hardware budget.

## Why predictors break on the 2025 task

The 2025 leaderboard (see [[2025 - EVAI]] and [[2025 - Sichuan teams]]) is consistent with predictor calibration being the bottleneck on novel data:

| Dataset | EVAI ([[EGNAS]]) | Sichuan #1 (predictor) | Gap |
|---|---|---|---|
| GameOfLife | 99.99% | 99.66% | 0.3% |
| Windspeed | 17.4% | 15.3% | unsearchable for both |
| Cryptic | **76.1%** | 73.2% | **2.9%** |

GameOfLife is conventional → predictor priors transfer. Cryptic is novel → predictor's learned mapping `arch → val_acc` doesn't extrapolate; EGNAS's "actually train short candidates" approach pays off there. This is the canonical "predictor distribution shift" problem and is the *raison d'être* of DAP-style domain adaptation. With more dataset diversity in the predictor's training corpus, the gap could close.

## Practical recipe (if we choose to implement)

- **Bootstrap set**: 30–100 fully trained `(arch, val_acc)` pairs on the target dataset. Cheap: small archs at small `E`. This is the warm-start cost.
- **Predictor**: GNN works but a gradient-boosted tree on (depth, width, op counts, FLOPs, params) is competitive and faster to fit on tiny data — see [Zero-Cost Proxies for Lightweight NAS](https://arxiv.org/abs/2101.08134) Table 8.
- **Outer EA**: tournament selection, mutation rate ~0.2 per gene, population 50, 20 generations.
- **Refit cycle**: every 2 generations, fully train the top-3 and add to predictor train set ("online predictor").
- **Spend split**: 60% search budget on bootstrap full-trainings, 30% on predictor refits during search, 10% on final winner re-train. Tune by [[Budget Allocation]].

## When this beats [[EGNAS]]

- Stable, well-known dataset family (CIFAR-shaped, MNIST-shaped) where a pretrained predictor or transferable embeddings exist.
- Large per-dataset budget (≥ 6 h) that lets the bootstrap set get to ≥ 100 pairs.
- Parallel hardware: predictor evaluation parallelises trivially; EGNAS sequence does not.

## When this loses to [[EGNAS]]

- Truly novel data (Cryptic-class). The predictor's training distribution doesn't cover it.
- Tight per-dataset budget where the bootstrap cost exceeds the available time.
- Single-GPU eval environment (the [[Phases and Timeline]] eval HW is unspecified — assume single GPU).

## Open questions

- Whether DAP (the 2026 IEEE TC paper from Song et al.) is what's actually deployed in the Sichuan submissions, or just published in parallel.
- Whether listwise ranking predictors are robust to label noise from short-budget bootstrap trainings — relevant to the Phase 2 budget.
- How much of the Cryptic gap is predictor calibration vs search-space coverage (does Sichuan's space lack the right depth?). No public ablation.

## See also

- [[2025 - Sichuan teams]] for who-did-what.
- [[EGNAS]] for the alternative that won (and the methodological contrast).
- [[Zero-Cost Proxies]] — orthogonal cheap ranking, sometimes used inside the EA loop instead of a learned predictor.
- [[Threats to Validity]] — distribution shift is the headline failure mode of this whole family.
- [[Cryptic]] — the dataset where the predictor-EGNAS gap was decisive.
