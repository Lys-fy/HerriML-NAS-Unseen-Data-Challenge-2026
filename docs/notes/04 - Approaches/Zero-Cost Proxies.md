# Zero-Cost Proxies

> **Status:** current | **Topic:** Synflow, NASWOT, Zen-NAS, Jacov — score architectures without training.
> **Related:** [[Approaches Overview]], [[Multi-Fidelity]], [[Predictor-Guided Evolutionary]], [[EGNAS]]

Rank candidate architectures using a single forward (and sometimes backward) pass on a random batch of data, no training needed. Cost: seconds per arch. Used to prune large search spaces cheaply before more expensive ranking ([[Multi-Fidelity]] / full training).

## The proxies

| Proxy | Signal | Data needed | Cost | Reference |
|---|---|---|---|---|
| **Synflow** | sum of `\|param × ∂L/∂param\|` with all-ones input | none (data-free) | 1 forward + 1 backward | [Tanaka et al. 2020](https://arxiv.org/abs/2006.05467); adapted for NAS by [Abdelfattah et al., ICLR 2021](https://arxiv.org/abs/2101.08134) |
| **Jacov / NASWOT** | log-det of similarity matrix of ReLU activation patterns | one minibatch | 1 forward | [Mellor et al., ICML 2021](https://arxiv.org/abs/2006.04647) |
| **Zen-Score** | gradient of perturbation of Gaussian input through the net | one Gaussian batch | several forwards | [Lin et al., ICCV 2021](https://arxiv.org/abs/2102.01063) |
| **Grad-Norm / SNIP / GraSP / Fisher** | various pruning-at-init scores | one minibatch | 1 fwd + 1 bwd | Abdelfattah et al. 2021 |
| **Params / FLOPs** | static count | none | free | trivial baseline |

The Abdelfattah et al. paper unified these — code at [mohsaied/zero-cost-nas](https://github.com/mohsaied/zero-cost-nas).

## How it works (Synflow)

```
1. Make all weights positive (abs).
2. Feed an all-ones tensor through the net.
3. L = sum of all outputs.
4. Backprop. Score = sum_w |w * dL/dw|.
```

Higher = "more synaptic flow" = better. Data-free, so it ranks purely by topology + parameter magnitude.

## How it works (NASWOT / Jacov)

```
For a minibatch of N images:
1. Forward through net. Record binary ReLU activation pattern per image.
2. K = N×N kernel of code-overlap (Hamming-style).
3. Score = log |K|.  (Higher = more linearly separable inputs at init.)
```

Data-aware. Sensitive to batch composition, so use a fixed seed.

## How it works (Zen-NAS)

```
1. Sample two Gaussian inputs x, x+δ.
2. Forward through randomly-initialised net.
3. Score = E[ ||f(x+δ) - f(x)|| / ||δ|| ] × √(layer-wise BN stds).
```

Captures expressivity (how the net amplifies input perturbations) and is the strongest of the family on ImageNet — reportedly matches EfficientNet-B5.

## Reported correlations

On NAS-Bench-201 (CIFAR-10/100/ImageNet16-120, see [[Approaches Overview]]'s NATS-Bench reference):

| Proxy | Spearman ρ vs final acc (NAS-Bench-201, CIFAR-10) |
|---|---|
| Synflow | 0.74–0.76 |
| Jacov | 0.73 |
| NASWOT | 0.77 |
| Params | 0.75 |
| FLOPs | 0.74 |
| Grad-norm | 0.58 |
| GraSP | 0.62 |

Important caveat: **on the top-10% of architectures** (where you actually want resolution), Synflow's correlation drops to 0.42. Translation: zero-cost proxies are good filters, weak final-rankers.

## Calibration concerns

1. **Proxies prefer big nets.** Synflow's saliency sum scales with parameter count. On a fixed-FLOPs space the bias collapses, but on the open spaces in this competition, naive use rewards bloat. Normalise by params or FLOPs.
2. **Data-oblivious proxies don't generalise across tasks.** Synflow gives one universal ranking that cannot adapt to dataset peculiarities. NASWOT and Zen are data-aware but still poor on novel input distributions.
3. **Search-space dependence.** [Ning et al.](https://arxiv.org/abs/2008.09777) and others show proxies developed on cell spaces (DARTS / NAS-Bench-201) do not transfer to global / macro spaces — relevant since [[EGNAS]] uses a macro space.
4. **Initialisation noise.** All proxies depend on the random init. Average ≥3 seeds before ranking.
5. **Score-vs-FLOPs correlation can dominate.** If your space has a wide param-count range, the proxy ranking ≈ "bigger is better". A simple `params` baseline often matches or beats fancy proxies in that regime.

## Where this is useful in our pipeline

- **Pre-filter** for [[Predictor-Guided Evolutionary]]: prune 10⁴ candidates to 10² for predictor training. Cheap fan-out.
- **Initial seeding** for [[EGNAS]]: pick the starting `(D_min, W_max, Sep, 3×3)` from a Synflow ranking instead of EGNAS's manual baseline. Marginal gain at best.
- **Tiebreaker** when two architectures look indistinguishable on short-budget val_acc.
- **Sanity check**: a proxy ranking that disagrees wildly with short-budget val_acc points at a buggy training loop or noisy data, see [[Failure Modes]].

## Where this is not useful

- As a sole ranker for the final winner. The top-10% correlation drop bites here.
- On [[Gutenberg]] or other near-noise-floor datasets — every arch scores near the same, the proxy's signal is below noise.
- On novel structure (Voxel, Myofibre) — proxies were validated on natural-image cell spaces.

## When [[EGNAS]] obsoletes proxies for us

EGNAS evaluates ~25 candidates with short real training. That's already cheap. A zero-cost pre-filter doesn't shrink ~25 meaningfully. Proxies pay off when candidate count is in the hundreds — i.e. inside [[Predictor-Guided Evolutionary]] or [[Weight-Sharing One-Shot]] sub-net sampling.

## Defaults if implemented

- Use **Synflow + NASWOT + params** as a 3-proxy vote (Abdelfattah et al. recipe). Diversifies failure modes.
- Batch size 32, fixed seed, average over 3 random inits per arch.
- Don't extrapolate to a different architecture family without re-validating — proxies are not portable across search spaces.

## See also

- [[Approaches Overview]] for cross-family decision tree.
- [[Multi-Fidelity]] — natural downstream after a proxy filter.
- [[Predictor-Guided Evolutionary]] — sometimes uses proxies in the inner loop.
- [Zero-Cost Proxies for Lightweight NAS](https://arxiv.org/abs/2101.08134) is the canonical reference; [an evaluation paper](https://link.springer.com/article/10.1007/s11263-024-02265-7) discusses generalisability.
