# Branch B3 — DARTS Comparator

> **Status:** planned (experimental — comparator only) | **Topic:** PC-DARTS / DrNAS implementation as the organisers' published reference and a literature anchor. Not expected to ship — the [[DARTS Family|family's failure modes]] make it brittle for unseen data.
> **Related:** [[Implementation Plan Overview]], [[DARTS Family]], [[Weight-Sharing One-Shot]], [[Approaches Overview]], [[Threats to Validity]], [[CIFARTile]], [[Language]]

## Goal

Build a [[DARTS Family|PC-DARTS]] or DrNAS variant as a comparator. Two reasons:

1. **The organisers benchmark against PC-DARTS** ([[Papers|Geada et al. CVPR 2024]]). Their per-dataset NAS column in Tab. 2 is what every other team is implicitly trying to beat. If we can't reproduce their PC-DARTS numbers locally, our implementation has a bug.
2. **Cell-based search is a recognisable alternative**. If a hidden dataset is conventional image classification, DARTS-discovered cells may transfer well — possibly better than EGNAS's macro search.

Realistic outcome: B3 confirms PC-DARTS is brittle on macro-sensitive / novel data and reinforces that [[Tier 2 - EGNAS-Light]] is the right shipping choice. **Not expected to be a Phase-2 candidate.**

## Recipe (PC-DARTS, abbreviated)

```
Define a cell DAG:
  N nodes, edges with mixed ops
  ops = {Conv3x3 sep, Conv5x5 sep, dilconv3, dilconv5, maxpool, avgpool, identity, zero}

Supernet:
  Stack 8 cells (normal + reduction interleaved)
  PC-DARTS partial-channel sampling: 1/K of channels go through mixed-op, rest pass through

Search (~50 epochs, supernet train):
  Alternate:
    SGD on weights w using L_train(w, α)
    Adam on arch params α using L_val(w, α)

Discretise:
  For each edge: argmax_op = ops[argmax α]
  Result: a discrete normal cell + reduction cell

Final retrain (clock-bounded):
  Stack the discovered cells, full-train from scratch on the dataset
```

Reference: [Xu et al. 2019, *PC-DARTS*, ICLR 2020](https://arxiv.org/abs/1907.05737). Code: [yuhuixu1993/PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Adapting to the competition

PC-DARTS's macro is **fixed at 8 cells** with reduction cells at positions 1/3 and 2/3. This is what makes it incompatible with most of our dataset shapes — see [[DARTS Family]] §"When this breaks":

- [[Sudoku]] 9×9: 8 cells with 2 reductions → spatial 9 → 4 → 2. Borderline.
- [[GameOfLife]] 10×10: same. Borderline.
- [[CIFARTile]] 64×64: standard. Should work — and is where PC-DARTS won +41 pp over CNN per Geada.
- [[Myofibre]] 128×128: 8 cells with 2 reductions ends at 32×32. Way more cells needed to use the spatial info. Underperforms.

Choices for B3:

- **Adaptive cell count**: pick number of cells from `log2(min(H, W) / 4)` like our `plan_downsamples`. Deviates from the paper but matches our shape mix.
- **Keep paper-fixed 8 cells**: only matches a few dataset shapes. Cleaner comparator but useless on Myofibre / Sudoku.

Recommendation: implement adaptive first; if it disagrees badly with paper-fixed, document both.

## Failure modes to expect

From [[DARTS Family]] §"Failure modes":

1. **Skip-connection collapse** ([Zela et al. RobustDARTS, ICLR 2020](https://openreview.net/forum?id=H1gDNyrKDS)). PC-DARTS partially mitigates but doesn't eliminate. Watch for an α distribution where Identity dominates everywhere.
2. **High-curvature solutions**. Need [[DARTS Family|RobustDARTS early-stop on Hessian eigenvalue]] or use DrNAS's progressive shrinking.
3. **Search-vs-eval gap**. Always retrain the discovered cell from scratch, never use supernet weights as init.

These will probably surface on at least 2 of the 11 public datasets — exactly the brittleness the strategy notes use to *not* pick DARTS for the shipping path.

## Acceptance criteria

- [ ] All 11 public datasets complete without `Failed=True`.
- [ ] On [[CIFARTile]]: scores ≥ +6.0 adj at `--time-limit 1.0` — this is the dataset PC-DARTS dominates.
- [ ] On [[Language]] (where PC-DARTS *loses* to ResNet-18 per [[Score-Efficient Targeting]]): no worse than 80 % adj_score lower than Tier 0.
- [ ] Discovered cells don't collapse to skip-only on more than 2/11 datasets.
- [ ] Total score across the public 13 is comparable to Geada's Tab. 2 PC-DARTS column.

The last point is the *real* acceptance criterion. If our PC-DARTS can't match the organisers' published PC-DARTS, the implementation is broken — not our hardware or schedule.

## When B3 would graduate to the shipping path

Only if it scores ≥ [[Tier 2 - EGNAS-Light]] on at least 8/11 public datasets. Realistic expectation: 4/11 (CIFAR-shape datasets), losing badly on novel shapes. Stays a comparator.

## Repo layout

```
baselines/b3_pc_darts/
  data_processor.py     # identical to T0/T1
  nas.py                # supernet construction + bilevel optimisation + discretisation
  trainer.py            # final-cell retrain (similar to T0/T1)
  helpers.py            # cell DAG, mixed-op, PC partial-channel logic
```

PC-DARTS is the most code of any tier or branch by a wide margin. ~600 LOC for the search loop alone. Vendoring from the public repo with shape/clock adaptations is the practical path.

## Why DrNAS might be a better B3 than PC-DARTS

[Chen et al. 2021](https://arxiv.org/abs/2006.10355). Replaces softmax over α with a Dirichlet. Fixes:

- Skip-collapse: stochastic sampling can't dominate any one edge.
- Search-eval gap: progressive shrinking explicitly brings supernet closer to discrete arch.

Reported NAS-Bench-201 SOTA. Trade-off: extra Dirichlet bookkeeping, slightly more memory.

If we're going to build *one* DARTS variant, DrNAS is the better-conditioned choice. PC-DARTS is recommended only because it's the published organiser benchmark.

## Open questions

- Whether to use the original DARTS op set (7 ops + zero) or a reduced set (4 ops). Reduced is faster, may underperform on macro-sensitive datasets.
- Whether to skip the alternating bilevel optimization and use a one-shot single-level approximation (faster, slightly worse).
- Whether to share the discovered cell across datasets (search once on, say, AddNIST, deploy on all). This is what most NAS competitions assume but the unseen-data setting explicitly forbids — see [[Threats to Validity]] §2.

## References

- [[DARTS Family]] — full method note with failure modes.
- [Xu et al., *PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search*, ICLR 2020](https://arxiv.org/abs/1907.05737).
- [Chen et al., *DrNAS: Dirichlet Neural Architecture Search*, ICLR 2021](https://arxiv.org/abs/2006.10355).
- [Liu et al., *DARTS: Differentiable Architecture Search*, ICLR 2019](https://arxiv.org/abs/1806.09055).
- Reference code: [yuhuixu1993/PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS), [xiangning-chen/DrNAS](https://github.com/xiangning-chen/DrNAS).
