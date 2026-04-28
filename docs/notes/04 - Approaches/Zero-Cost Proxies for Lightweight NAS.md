# Zero-Cost Proxies for Lightweight NAS

> **Status:** Reference | **Topic:** Unified ZCP Framework, Pruning-at-init, Benchmarking.
> **Related:** [[Zero-Cost Proxies]], [[NAS-WOT]], [[Synflow]]

This paper (by Abdelfattah et al.) provides the first comprehensive evaluation of multiple zero-cost proxies (Synflow, NAS-WOT, SNIP, Fisher, etc.) across 9 different search spaces and diverse tasks.

## Main Advantage
**Unified Framework:** It identifies which proxies work best for which types of search spaces. It specifically warns that **Params** and **FLOPs** are often stronger baselines than complex proxies when the search space has a wide range of model sizes—a critical insight for our **unconstrained** competition search space.

## Core Idea
The authors treat NAS as a "Pruning-at-Initialization" problem. They evaluate proxies based on three categories:
1.  **Gradient-based:** (SNIP, GraSP, Fisher) — require 1 forward + 1 backward pass.
2.  **Saliency-based:** (Synflow) — data-agnostic, measures parameter importance.
3.  **Activation-based:** (NAS-WOT) — measures input separation.

## Key Insights for our Pipeline
- **Don't use just one:** Combining `Synflow` (topology) with `NAS-WOT` (data-awareness) yields the most robust ranking.
- **The "Top-10% Collapse":** ZC-proxies are great for finding the top 10% of a space, but they are very poor at ranking within that top 10%. Use them for *filtering*, not for *final selection*.

## Reference
- [Abdelfattah et al., ICLR 2021](https://arxiv.org/abs/2101.08134)
