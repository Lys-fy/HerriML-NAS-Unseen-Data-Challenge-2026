# Weight-Sharing One-Shot

> **Status:** current | **Topic:** Supernet methods — train one over-parameterised net, sample sub-architectures from it.
> **Related:** [[Approaches Overview]], [[DARTS Family]], [[EGNAS]], [[Zero-Cost Proxies]]

Train a single over-parameterised supernet whose subgraphs correspond to architectures in the search space. After training, sample / search sub-nets, score them using the inherited supernet weights, and take the top one. Three canonical instances: **SPOS** (uniform sampling), **ProxylessNAS** (binarised gating), **OFA** (progressive shrinking). [[DARTS Family]] is also weight-sharing but with continuous relaxation rather than discrete sampling — it's a sibling, not a parent.

## The recipe

```
1. Build a supernet covering all candidate ops at all positions.
2. Train it with uniform / weighted sampling of sub-paths.
3. Search sub-architectures (EA / RL / random) using inherited weights.
4. Pick top-k, retrain from scratch (or fine-tune).
```

The point: amortise architecture search cost across one supernet train. Steps 3–4 are essentially free at inference time.

## SPOS — Single Path One-Shot (Guo et al. 2019, [arXiv:1904.00420](https://arxiv.org/abs/1904.00420))

- Each iteration samples one path uniformly from the supernet, trains only that path's weights.
- All paths get equal training expectation → reduces co-adaptation.
- Search is post-hoc EA over the trained supernet, scoring sub-paths in milliseconds.
- Constraints (FLOPs, latency) handled by rejecting infeasible candidates.

Simplest of the family, robust, the right starting point if going one-shot.

## ProxylessNAS (Cai et al. 2018)

- Binarised architecture gates: per-edge, only one op is "on" at a time, sampled stochastically.
- Memory drops vs DARTS — only one op runs per edge per forward pass.
- Hardware-aware: latency loss term added to update arch params.
- Searches directly on ImageNet (the "no proxy task" claim).

## OFA — Once-For-All (Cai et al. 2020, [ICLR 2020](https://openreview.net/forum?id=HylxE1HKwS))

The most ambitious. One supernet covers a 4D space (depth, width, kernel, expand-ratio) over MobileNetV3-style blocks. Train **once** then specialise to many target devices without re-training:

- **Progressive shrinking**: start with full-size training, then enable training of progressively smaller sub-nets — large net teaches small net via shared weights. Without progressive shrinking, sub-net accuracy drops by ~2 percentage points (per the OFA paper's ablation).
- After supernet training: a small accuracy-and-latency predictor lets you pick the best sub-net for any target hardware in seconds.
- Reported: 14× faster training than ProxylessNAS, 16× faster than FBNet, 1142× faster than MnasNet across 40 deployment scenarios.

OFA's payoff is amortising one expensive train across many deployments. In our setting (unknown eval GPU, 12+ datasets, single target per dataset) the amortisation story is weaker — but a OFA-trained-on-CIFAR-shaped-data supernet could still serve as a transferable backbone.

## Pros vs DARTS

| | DARTS | One-shot (SPOS / OFA) |
|---|---|---|
| Memory at search | every op runs every fwd | one op per edge |
| Discretisation gap | severe (skip collapse) | smaller (sub-paths trained as themselves) |
| Search after training | not separable | separable, EA / random / pred |
| Macro shape search | no | yes (OFA: depth, width) |
| Wall-clock for first arch | low (one supernet train) | low (one supernet train) |
| Wall-clock for k archs | k × supernet train | 1 × supernet train + k × cheap retrain |

OFA is the closest one-shot match for the macro-flexibility that [[EGNAS]] gives us — it actually searches depth and width.

## Cons (the ranking-fidelity problem)

The headline weakness, applies to the whole family:

> Sub-net accuracy under inherited supernet weights ≠ sub-net accuracy after standalone training.

Reasons:
- **Co-adaptation**: weights co-trained as part of the supernet may rely on neighbouring ops that the discretised sub-net won't have.
- **Sampling bias**: paths sampled less during training are under-trained → unfairly low scores.
- **Capacity mismatch**: the supernet's BN stats and weight scale are tuned for the full supernet, not a single sub-path.

Result: ranking from supernet ≠ ranking after retrain. SPOS's uniform sampling reduces this; progressive shrinking (OFA) reduces it further; DARTS doesn't address it at all (related: [[DARTS Family]]).

## When this beats [[EGNAS]] for us

- **Pretrained supernet exists** for CIFAR-shaped or ImageNet-shaped data and we can adopt it. Not the case as-is, but if a teammate trained one on AddNIST/MultNIST it could transfer to similar datasets cheaply.
- **Many datasets share input shape and class structure.** Train one supernet on a meta-dataset, deploy specialised sub-nets per actual dataset. Risky given Cryptic/Windspeed/Voxel are not standard image classification.
- **Per-dataset budget is too small to fit even an EGNAS pass** but supernet training was paid offline.

## When [[EGNAS]] is better

- **Heterogeneous datasets.** A supernet trained on one input shape cannot serve datasets with different `(C, H, W)`. The competition gives mixed shapes (see [[CIFARTile]], [[Voxel]], [[Myofibre]]); EGNAS just adapts.
- **Novel datasets where ranking fidelity is fragile.** Same root-cause problem as [[Predictor-Guided Evolutionary]] on Cryptic — inherited rankings don't transfer.
- **Tight Phase 2 budget.** Building a supernet during the per-dataset clock is hopeless; reusing one assumes offline pretraining we haven't done.

## Practical defaults if implemented

- Start with **SPOS** before OFA. Simpler, fewer moving parts, the OFA progressive-shrinking schedule is finicky.
- Supernet training at least 100 epochs with uniform path sampling.
- Post-supernet, search with a simple GA (population 50, generations 20) using val accuracy from inherited weights.
- **Always retrain the top-k from scratch** — never trust supernet-inherited val acc as the final ranking.

## Open questions

- Whether a supernet pretrained on the public 13 datasets (see [[AddNIST]], [[CIFARTile]], etc.) transfers to the hidden Phase 2/3 datasets. Bet against it for novel-input datasets.
- Whether OFA's progressive-shrinking schedule is feasible inside the per-dataset clock — almost certainly no, only as offline preprocessing.

## See also

- [[Approaches Overview]] cross-family decision tree.
- [[DARTS Family]] — sibling that uses continuous gates instead of discrete sampling; shares ranking-fidelity issues.
- [[EGNAS]] — what won 2025 instead of one-shot.
- Code: [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all), [megvii-model/SinglePathOneShot](https://github.com/megvii-model/SinglePathOneShot).
