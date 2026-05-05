# DARTS Family

> **Status:** current | **Topic:** DARTS, PC-DARTS, DrNAS — gradient-based weight-sharing cell search.
> **Related:** [[Approaches Overview]], [[Weight-Sharing One-Shot]], [[EGNAS]], [[Zero-Cost Proxies]]

Differentiable architecture search: relax the discrete arch choice to a continuous mixture, optimise mixture weights and network weights jointly with SGD, then discretise. Fast, fragile, and the dominant pre-2022 approach. The organisers' own benchmark uses PC-DARTS as their NAS reference (see the per-dataset benchmark table in [[CIFARTile]] / [[AddNIST]]).

## Vanilla DARTS (Liu et al. 2018, [arXiv:1806.09055](https://arxiv.org/abs/1806.09055))

A cell is a DAG of nodes. Each edge between nodes has a *mixed* op:

```
o_mixed(x) = Σ_i softmax(α_i) · op_i(x)
```

α (architecture parameters) and w (weights) are optimised in a bilevel formulation:

```
min_α   L_val(w*(α), α)
s.t.    w*(α) = argmin_w L_train(w, α)
```

Approximated by a single SGD step on w before each α update. After search, take `argmax_i α_i` per edge.

## PC-DARTS (Xu et al. 2019, [arXiv:1907.05737](https://arxiv.org/abs/1907.05737))

Vanilla DARTS uses a memory-heavy supernet — every op runs at every edge each forward pass. PC-DARTS samples a subset of channels (1/K) for ops and shortcut the rest. Memory drops by K, batch size grows, training stabilises.

Reported: CIFAR-10 2.57% err in 0.1 GPU-d; ImageNet (mobile) 24.2% top-1 in 3.8 GPU-d. This is the method whose results appear as "PC-DARTS (NAS)" in the organisers' benchmark tables (see [[CIFARTile]] for the row where it shines, [[Language]] for where it loses).

## DrNAS (Chen et al. 2021, [arXiv:2006.10355](https://arxiv.org/abs/2006.10355))

Replace softmax over α with a Dirichlet distribution. α parameterises Dirichlet concentration; sample architectures via pathwise gradients. Adds:
- Stochasticity → encourages exploration, mitigates the early-collapse failure mode.
- Progressive learning: shrink the supernet during search to bridge search/eval gap.

Reported: CIFAR-10 2.46% err; ImageNet (mobile) 23.7%; SOTA on NAS-Bench-201 (see [[Approaches Overview]] for the benchmark context).

## Variations for Better Generalization

- **SDARTS / Smooth DARTS** (Chen & Hsieh 2020): Applies perturbation to architecture parameters to smooth the loss landscape, directly addressing the generalization gap and high-curvature issues.
- **Beta-DARTS** (Ye et al. 2022, [arXiv:2203.01665](https://arxiv.org/abs/2203.01665)): Introduces Beta-decay regularization to architecture parameters to prevent the "skip-connection dominance" and improve stability.
- **RobustDARTS** (Zela et al. 2020): Proposes early stopping based on the Hessian's largest eigenvalue to avoid sharp minima that generalize poorly.


## How it works (cell encoding)

DARTS searches a *cell* (typically two: normal + reduction), then a hand-set macro stacks 8–20 cells. Op set: 3×3 / 5×5 sep-conv, 3×3 / 5×5 dilated sep-conv, 3×3 max-pool, 3×3 avg-pool, identity, zero. Fixed channel-doubling at predefined reduction cells. **The macro shape is not searched** — this is the family's load-bearing limitation for [[CIFARTile]]-style problems where macro shape matters.

## Failure modes

The reason DARTS-family work is unreliable on novel data, in priority order:

1. **Skip-connection collapse / parameter-free dominance.** α drifts toward identity / zero / pool. The discretised cell ends up parameter-light and underperforms a random cell. Documented in [Zela et al., RobustDARTS](https://openreview.net/forum?id=H1gDNyrKDS) (ICLR 2020) — DARTS yields degenerate architectures across multiple search spaces, sometimes no better than random search. PC-DARTS partially fixes this; DrNAS fixes it further.
2. **Gradient mismatch.** The bilevel approximation (one SGD step on w, then a step on α) has high variance when train and val curvatures differ. High-curvature solutions in α-space generalise poorly.
3. **Search-eval gap.** The supernet trained during search is not the same network as the discretised final arch. Performance ranking from supernet activations does not match standalone training. (Same root cause as in [[Weight-Sharing One-Shot]].)
4. **Hyperparameter brittleness.** DARTS results swing by ≥1% with seed; LR, weight decay, and unrolled-vs-first-order all matter. RobustDARTS adds early-stopping on Hessian eigenvalue to mitigate.
5. **Cell-only search ignores macro.** When the right answer is a 60-layer narrow net (e.g. EGNAS's discovered EMNIST arch), DARTS can't find it — the macro is hand-fixed.

## When this works

- Image classification on CIFAR-shaped data with conventional macro (depth ~20, width 36–64). On the organisers' benchmark, PC-DARTS dominates ResNet-18 on [[AddNIST]], [[MultNIST]], and notably [[CIFARTile]] (45% → 92%). The macro is correct *for those datasets*.
- Plenty of GPU memory (vanilla DARTS), or PC-DARTS for tighter budgets.
- A NAS benchmark you can sanity-check on first.

## When this breaks (relevant for this competition)

- [[Language]] dataset: PC-DARTS scores 90.1%, below ResNet-18's 97.0% per the organisers' table. The cell space doesn't match what Language needs.
- Novel structure datasets without obvious cell repetition (Voxel, Myofibre).
- Tight wall-clock — vanilla DARTS supernet train alone is ~1 GPU-day; PC-DARTS ~0.1 GPU-d for CIFAR but still painful per-dataset under [[Clock and Time Budget]].

## Practical defaults if used

- Prefer PC-DARTS or DrNAS over vanilla DARTS — same idea, fewer pitfalls.
- Use the standard 8-op DARTS search space; don't extend without ablation.
- Add early stopping on Hessian eigenvalue (RobustDARTS) or use DrNAS's progressive shrinking.
- Re-train the discretised cell from scratch — never use supernet weights as init.

## Why we de-prioritise DARTS for this team

[[EGNAS]] beats DARTS on the same datasets the paper compares (CIFAR-10/100, FashionMNIST), is more macro-flexible, and has saner failure modes (a bad candidate just gets a low val_acc, not skip-collapse). DARTS is in scope as a backup and as a literature anchor — the organisers benchmark against PC-DARTS, so we must understand it.

## See also

- [[Approaches Overview]] for cross-family decision tree.
- [[Weight-Sharing One-Shot]] — DARTS is a degenerate one-shot method; the failure modes overlap.
- [[Zero-Cost Proxies]] — sometimes used to filter the DARTS cell space before relaxation.
- Code: [yuhuixu1993/PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS), [xiangning-chen/DrNAS](https://github.com/xiangning-chen/DrNAS).
