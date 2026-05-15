# Tiers Overview

> **Status:** drafting | **Topic:** Staged build plan — shipping path (Tiers 0–4) plus experimental branches (B1–B4). Each tier evaluated before the next is started.
> **Related:** [[Approaches Overview]], [[Reliability vs Peak]], [[Phase 2 Submission Strategy]], [[Budget Allocation]], [[Failure Modes]]

The team's plan for the [[Submission Contract|submission code]] is built in **tiers** rather than written all at once. Each tier introduces one well-defined slice of complexity on top of the previous, can be **mock-harness-evaluated independently**, and remains in the repo as a fallback / comparator after the next tier supersedes it.

The whole structure is driven by the strategy notes' central claim: [[Reliability vs Peak|reliability dominates peak]]. Tiers exist so the *first* shippable version is a confirmed-working pipeline (Tier 0), and every later tier inherits its working harness. Going straight to EGNAS would mean debugging architecture search against an unverified harness — the failure modes overlap and become hard to bisect.

## Two axes

Tiers split into a **shipping path** (every one could plausibly become the Phase-2 submission) and **experimental branches** (built to A/B against the shipping path, may be merged in later if they win):

```
Shipping path:
  T0 (hardcoded CNN) ─► T1 (random op search) ─► T2 (EGNAS-light) ─► T3 (+ reliability) ─► T4 (Phase-2 polish)
                              ▲
                              │ at this point, B1 (zero-cost proxy filter) can be
                              │ merged into T2/T3 if it ranks well on the public 13.

Experimental branches:
  B1 (zero-cost proxies)
  B2 (predictor-guided EA)   ← what [[2025 - Sichuan teams]] did
  B3 (DARTS / PC-DARTS)      ← organisers' reference benchmark
  B4 (weight-sharing supernet)
```

## The shipping path

| Tier | Goal | Status |
|------|------|--------|
| [[Tier 0 - Hardcoded CNN]] | Validate harness, clock, shape adaptation. No search. | ✓ implemented |
| [[Tier 1 - Augmentation + Op Search]] | **Augmentation + label smoothing + Mixup + DropPath** to make the budget productive (Tier 0 uses only ~6 % of clock), *plus* a first random op-type search layer. Validates search-loop plumbing on a model that actually keeps learning past epoch 30. | planned |
| [[Tier 2 - EGNAS-Light]] | Sequential macro-micro hill-climb (Siddiqui et al. 2025, scaled). Macro depth/width + per-layer op/kernel. *This is the version that ships to Phase 2 unless something better surfaces.* | planned |
| [[Tier 3 - EGNAS + Reliability]] | T2 + per-regime budget classifier + OOM-safe batch sizing + in-search fallback + top-k ensemble + TTA. | planned |
| [[Tier 4 - Phase-2 Polish]] | Iterate from Phase-2 leaderboard signal. Per-codename overrides, mixed precision (if env-safe), SAM/SWA on high-leverage datasets. | planned |

## The experimental branches

| Branch | Goal | When it wins |
|--------|------|--------------|
| [[Branch B1 - Zero-Cost Proxies]] | Synflow + NASWOT + params 3-way vote, used as cheap pre-filter | T2/T3 search space grows to hundreds of candidates |
| [[Branch B2 - Predictor-Guided EA]] | GBM/GNN surrogate + evolutionary search over op-type vectors | Dataset family has strong priors — [[Cryptic]]-like novel data is where this *loses* |
| [[Branch B3 - DARTS Comparator]] | PC-DARTS / DrNAS implementation as literature anchor and organisers' reference | Standard image-classification datasets with conventional cells |
| [[Branch B4 - Weight-Sharing Supernet]] | SPOS / OFA-lite trained offline on public 13, sample sub-nets per Phase-2 dataset | Hidden datasets share shape & class structure with the public 13 — *unlikely per [[Threats to Validity]]* |

## How tiers are evaluated

Each tier must pass the [[Testing Strategy|full test matrix]] before the next is started:

1. **Tier 1 sweep** — all 11 public datasets at default time-limit, score logged.
2. **Tier 2 stress** — same datasets at `--time-limit 0.05` (3 min). The submission must finish and clear benchmark on at least the easy datasets.
3. **Tier 3 low-VRAM** — CPU-only path or capped VRAM, completes without crashing.
4. **Tier 5 packaging** — zipped archive imports cleanly in a fresh tmp dir.

Recorded in `dev/logs/tier-N-results.md` per tier so the score progression across tiers is bisectable.

## Why this order

- **T0 → T1**: T1 inherits T0's harness, dataloaders, and Trainer. If T1 fails, it's the search loop's fault — not the surrounding plumbing. Without T0 as scaffolding, T1 failures conflate two bugs.
- **T1 → T2**: T2 inherits T1's per-candidate timing infrastructure + `best so far` tracking. EGNAS's macro/micro hill-climb is structurally similar to T1's random-K-and-pick-best, just with a smarter generator.
- **T2 → T3**: T3 wraps T2 in defensive code without changing the search algorithm. Splitting these means a bug introduced by the reliability layer doesn't masquerade as a search-quality regression.
- **T3 → T4**: T4 is data-driven. We can't write T4 in advance — we need a Phase-2 attempt to see which datasets behaved unexpectedly.

## Why branches exist separately

The branches are **risk hedges**. If T2's EGNAS-light underperforms on a dataset where, say, PC-DARTS is the organisers' baseline (B3) — we want to know that *before* shipping. They run against the same mock harness, produce the same `dev/predictions/<codename>_stats.pkl`, and can be score-compared dataset-by-dataset.

[[2025 - EVAI]]'s win is the empirical evidence that the shipping path's choice (EGNAS) is correct in expectation. Branches are insurance against that evidence not transferring to 2026's hidden datasets.

## What's *not* on this roadmap

- Per-dataset hardcoded special cases (e.g. a Conway-aware 3×3 conv for [[GameOfLife]]). Tempting but [[Threats to Validity|overfits to public 13]] and adds zero value on hidden datasets.
- Internet-fetched pretrained weights — explicitly forbidden by [[Rules]].
- Submission-time NAS architecture caching across datasets. The mock harness re-instantiates everything per dataset; cross-dataset state would have to live on disk and is fragile.
- Anything that can't survive a `--time-limit 0.05` stress test ([[Testing Strategy]] Tier 2). Robustness is non-negotiable for the Phase-3 unknown-runtime risk.
