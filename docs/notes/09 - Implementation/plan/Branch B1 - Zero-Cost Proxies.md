# Branch B1 — Zero-Cost Proxies

> **Status:** planned (experimental, optional merge into [[Tier 2 - EGNAS-Light|T2]] / [[Tier 3 - EGNAS + Reliability|T3]]) | **Topic:** Synflow + NASWOT + params 3-way vote as a cheap pre-filter to rank candidate architectures without training. Only valuable once the search space grows past ~50 candidates.
> **Related:** [[Implementation Plan Overview]], [[Zero-Cost Proxies]], [[Tier 2 - EGNAS-Light]], [[Approaches Overview]], [[Multi-Fidelity]]

## Goal

Use [[Zero-Cost Proxies]] (Synflow, NASWOT, parameter count) to rank candidate architectures in seconds instead of minutes-of-training. Becomes a pre-filter in front of any search loop that has more candidates than budget.

This is an **experimental branch** — it doesn't replace the shipping path, but plugs in as an optional component if it ranks well on the public 13.

## Recipe

```python
def score_arch(model, sample_batch):
    # 3-way vote over normalised ranks.
    s1 = synflow(model)             # |w * dL/dw| sum, all-ones input
    s2 = naswot(model, sample_batch)  # log|K| of ReLU activation similarity
    s3 = -count_params(model)        # invert so smaller = lower score (corrects bias)
    return mean_rank([s1, s2, s3])
```

Used as:

```python
# inside an EGNAS-light or random-search loop
candidates = generate_K_arches(K=50)            # cheap to enumerate
ranked = sorted(candidates, key=lambda a: score_arch(build(a), sample))
top_k = ranked[:10]                              # 10 → full evaluation
# proceed with normal short-train ranking on top_k
```

## Implementation notes

From [[Zero-Cost Proxies]] §calibration concerns:

1. **Always normalise by params**. Synflow alone biases toward bigger nets. The `-count_params` term in the vote partially compensates.
2. **Fixed seed** for NAS sampling and proxy evaluation. Per [[Threats to Validity]] §4.
3. **Average over 3 random inits** per arch (proxy depends on init).
4. **Validate on public 13 before trusting** — Synflow on macro spaces is documented as unreliable in [Ning et al. 2020](https://arxiv.org/abs/2008.09777). We need our own correlation check.

Library: [SamsungLabs/zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas) has implementations. Vendor a minimal subset into `helpers.py` — don't depend on the package at submission time (per [[Submission Packaging]] "no surprise imports").

## When this beats raw search

| Search space size | Pay-off |
|---|---|
| ≤ 25 candidates (T2 EGNAS) | none — full-train all 25 is already cheap |
| 50–200 candidates | maybe — proxy filter saves 60–80% of evaluation cost |
| 1000+ candidates | yes — proxy is the only way to keep up |

[[Tier 2 - EGNAS-Light]] is at the bottom of this table — proxies don't help unless we expand the search space (e.g. by adding kernel sizes 7, 9 and op variants, growing to 200+ candidates).

Where it *does* help: feeding [[Branch B2 - Predictor-Guided EA]]'s bootstrap set. Instead of 30 randomly-chosen architectures, pick 30 spread evenly across the proxy ranking — better coverage of the accuracy spectrum.

## When this loses

[[Zero-Cost Proxies]] §"Where this is not useful":

- Sole ranker for the final winner (top-10% correlation drops to ~0.42).
- [[Gutenberg]] / [[Windspeed]] — noise-floor datasets where every arch scores similarly.
- Truly novel inputs ([[Voxel]] 3-D, [[Myofibre]] medical) — proxies validated on CIFAR-style cells.

## Validation gate

Before merging B1 into the shipping path, run on the public 13:

```python
# For each dataset:
candidates = generate_K(K=30)
proxy_ranking = score_all_by_proxy(candidates)
actual_ranking = train_to_completion_and_rank(candidates)   # ground truth
spearman = scipy.stats.spearmanr(proxy_ranking, actual_ranking)
```

Acceptance: Spearman ≥ 0.5 on at least 8/11 public datasets. Below that, the proxy is too noisy to trust and B1 stays an experimental note, not in the shipping path.

## What B1 does *not* add

- It is not a complete NAS algorithm. It is a *ranking sub-component* of another algorithm.
- Does not address the search-vs-train gap of weight-sharing methods ([[Weight-Sharing One-Shot]]).
- Does not transfer across architecture families — re-validate per search space.

## Repo layout

If/when shipped:

```
helpers.py inside the shipping-tier package gets:
  +  synflow(model) -> float
  +  naswot(model, batch) -> float
  +  rank_candidates(arches, sample_batch) -> sorted list
```

No separate baseline directory — B1 is a function library, not a full pipeline variant.

## Open questions

- How well do proxies transfer between depth-grow candidates in [[Tier 2 - EGNAS-Light|EGNAS]]? Sequential dependency means proxy-ranking each depth-grow step might double-count noise.
- Whether the proxy can serve as a *budget allocator* (give larger short-train budget to higher-proxy candidates) rather than just a filter. Hybrid Multi-Fidelity + proxy.

## References

- [[Zero-Cost Proxies]] — full method note.
- [Abdelfattah et al., *Zero-Cost Proxies for Lightweight NAS*, ICLR 2021](https://arxiv.org/abs/2101.08134).
- [Mellor et al., *Neural Architecture Search without Training*, ICML 2021](https://arxiv.org/abs/2006.04647).
- Code: [SamsungLabs/zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas).
