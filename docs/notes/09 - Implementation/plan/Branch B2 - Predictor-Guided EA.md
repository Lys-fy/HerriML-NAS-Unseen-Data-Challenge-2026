# Branch B2 — Predictor-Guided Evolutionary

> **Status:** planned (experimental — comparator against the shipping path) | **Topic:** GBM/GNN surrogate accuracy predictor + evolutionary search. The approach [[2025 - Sichuan teams|Yanan Sun's group]] used to take 2nd and 3rd in 2025. *Likely loses to [[Tier 2 - EGNAS-Light|EGNAS]] on novel data; useful as a sanity-check.*
> **Related:** [[Implementation Plan Overview]], [[Predictor-Guided Evolutionary]], [[2025 - Sichuan teams]], [[Threats to Validity]], [[Tier 2 - EGNAS-Light]], [[Cryptic]]

## Goal

Implement the [[Predictor-Guided Evolutionary]] pattern as a full submission variant. Two reasons:

1. **Comparator**: confirm [[2025 - EVAI|EGNAS]] really does beat predictor-EA on our hardware and our public 13. If it doesn't, our shipping decision is wrong.
2. **Hedge**: if a Phase-2 dataset turns out to be in the predictor's "comfort zone" (CIFAR-shape, conventional macro), predictor-EA might outscore EGNAS by 0.5–1.0 adj_score.

Lives at `baselines/b2_predictor_ea/`. Same 4-file shape as the shipping tiers — independently mock-harness-runnable.

## Recipe

```
1. Bootstrap set:
     Sample 30–50 architectures (uniformly or via [[Branch B1 - Zero-Cost Proxies]] for coverage).
     Fully (or short-) train each. Record (arch_features, val_acc) pairs.

2. Predictor:
     GBM (XGBoost / LightGBM) on hand features:
       depth, base_width, sum_of_kernel_sizes, count(SepConv), count(Conv5),
       FLOPs, param_count, downsample_count
     Cross-validate, target RMSE < 0.05 on log-loss.

3. EA loop (~20 generations, population 50):
     for gen in range(20):
       parents = tournament_select(population, k=3)
       children = [mutate(p) or crossover(p1, p2) for ...]
       all_scores = predictor.predict([encode(c) for c in candidates])
       population = top_k(candidates ∪ population, k=50, by=all_scores)

4. Refit cycle (online predictor variant — Song et al. IEEE TSMC 2025):
     every 5 generations: full-train top 3, add to predictor train set, refit.

5. Final: full-train the top-3 EA-ranked, return the best to Trainer.
```

## Encoding the architecture

For a 6-block linear stack with ops in `{Conv3, Conv5, SepConv3, MaxPool, Identity}`, a flat encoding:

```python
def encode(arch_spec):
    return np.array([
        arch_spec.depth,
        arch_spec.base_width,
        *one_hot(arch_spec.ops),      # 6 * 5 = 30 dims
        arch_spec.flops,
        arch_spec.params,
        sum(arch_spec.downsample_plan),
    ])
```

GBM handles 30+ features without issue. Could swap to a small GNN if the search space later becomes branched (cell-based, residual), but for a linear stack a feature-vector predictor is fine.

## Why this is expected to lose to EGNAS

From the [[2025 - Sichuan teams]] result:

| Dataset | EGNAS (EVAI) | Predictor-EA (Sichuan #1) | Gap |
|---|---|---|---|
| GameOfLife | 99.99 % | 99.66 % | 0.3 % |
| Windspeed | 17.4 % | 15.3 % | unsearchable for both |
| Cryptic | **76.1 %** | 73.2 % | **2.9 %** |

The Cryptic gap is the diagnostic. Predictor-EA's predictor was trained on (presumably) a CIFAR-style dataset corpus; Cryptic is novel; the predictor mis-ranks. EGNAS's "actually short-train candidates" sidesteps this entirely.

Our public 13 is a small training corpus by predictor-NAS standards. The predictor will be worse than Sun's because we have less data to fit it. The Cryptic-gap effect should be more pronounced for us.

## Why we still build it

- **Public-13 evaluation**: if our predictor outperforms EGNAS on the public 13, that's evidence the predictor approach scales differently than we think — important to know before Phase 2.
- **Risk hedge**: hidden datasets are unknown. If one turns out CIFAR-shaped (e.g. a follow-on to AddNIST), predictor-EA could match or beat EGNAS — and the marginal cost of running both during Phase 1 is just dev time.
- **Sun group continues to publish**. The DAP (IEEE TC 2026), Online Predictor (IEEE TSMC 2025), and Listwise Ranking (Swarm and Evol. Comp. 2025) papers are direct upgrades. If their 2026 entry uses DAP, we want our own version to compare.

## Implementation notes

- **GBM, not GNN**, for bootstrap-set sizes ≤ 100. From [[Predictor-Guided Evolutionary]] §practical recipe: GBM on hand features is competitive with GNNs at this scale.
- **Bootstrap budget**: 60 % of NAS budget. Per [[Predictor-Guided Evolutionary]] recommended split.
- **EA budget**: 30 %. Predictor-scored EA is sub-second; this is mostly bookkeeping.
- **Final retrain**: 10 %. Trainer handles the full retrain.
- **Mutation rate**: 0.2 per gene. Tournament size 3. Population 50. Generations 20. Per [[Predictor-Guided Evolutionary]] defaults.

The bootstrap cost is the bottleneck: 50 arches × 2 epochs each × ~30 s = 50 min. Doesn't fit a 1 h Phase-2 budget. Workarounds:

- Shrink bootstrap to 20 candidates with very-short trains (1 epoch).
- Use [[Branch B1 - Zero-Cost Proxies]] to skip the bootstrap and seed the predictor with proxy scores. Less accurate, but cheap.

## Acceptance criteria

- [ ] Bootstrap + EA + final retrain complete within Phase-2 budget on all 11 public datasets.
- [ ] Predictor cross-val RMSE on log-loss < 0.05 on the public 13.
- [ ] At least matches Tier 0 on every dataset (predictor never *hurts*).
- [ ] On AddNIST / MultNIST (CIFAR-shape, high benchmark), scores ≥ Tier 2.
- [ ] On CIFARTile / Voxel (novel / search-rewarding), scores ≥ Tier 0 (it's OK to lose to T2 here).

## When to merge B2 into the shipping path

Almost never directly. The realistic merge path:

- B2's *predictor* (not its EA) used inside [[Tier 2 - EGNAS-Light]] as a "probably skip this candidate" filter on the depth-grow phase.
- B2's *EA* used as a second-stage refinement after EGNAS macro completes (instead of EGNAS micro).

Neither is committed to. Both stay experimental until Phase-2 data shows one of them wins.

## Repo layout

```
baselines/b2_predictor_ea/
  data_processor.py     # identical to T0/T1
  nas.py                # bootstrap + predictor fit + EA
  trainer.py            # identical to T0/T1
  helpers.py            # arch encoding, GBM wrapper, EA primitives
```

## Open questions

- Whether a *listwise* loss (Li et al. 2025) beats absolute-acc regression for our bootstrap-set sizes.
- Whether DAP-style domain adaptation (Song et al. IEEE TC 2026) is implementable from public information. Paper hasn't shipped code yet — track [[Researchers and Groups|the Sun group page]].
- Whether the predictor can be fit *across* the public 13 datasets (one predictor, multi-task) — closer to the DAP idea. Higher upside, more dev time.

## References

- [[Predictor-Guided Evolutionary]] — full method note.
- [[2025 - Sichuan teams]] — team and approach context.
- Song, X. et al. *DAP: Domain Adaptive Performance Predictor for Efficient NAS.* IEEE Trans. Computers, 2026.
- Song, X. et al. *Online Performance Predictor for Evolutionary NAS.* IEEE TSMC, 2025.
- Li, N. et al. *Listwise Ranking Predictor for Evolutionary NAS.* Swarm and Evol. Comp., 2025.
