# Implementation Plan Overview

> **Status:** current | **Topic:** Staged build plan — shipping path (Tiers 0–4) plus experimental branches (B1–B4). Reframed 2026-05-15 around the [[Probe-Driven Self-Configuration]] thesis after the [[Aug + Op Search|Tier 1 ablation]].
> **Related:** [[Probe-Driven Self-Configuration]], [[Approaches Overview]], [[Reliability vs Peak]], [[Phase 2 Submission Strategy]], [[Aug + Op Search|Tier 1 - Results]], [[Failure Modes]]

> **Read [[Probe-Driven Self-Configuration]] first.** It is the submission's central thesis: the pipeline does not commit to a prior winner's NAS method — it runs cheap runtime probes to characterize the unseen dataset, then configures itself. EGNAS / predictor-EA / DARTS become the *menu the probe selects from*, not the submission. Everything below is the staged build that gets us there.

The [[Submission Contract|submission code]] is built in **tiers** rather than written all at once. Each tier introduces one well-defined slice of complexity on top of the previous, can be **mock-harness-evaluated independently**, and remains in the repo as a fallback / comparator after the next tier supersedes it.

The whole structure is driven by the strategy notes' central claim: [[Reliability vs Peak|reliability dominates peak]]. Tiers exist so the *first* shippable version is a confirmed-working pipeline (Tier 0), and every later tier inherits its working harness. Going straight to EGNAS would mean debugging architecture search against an unverified harness — the failure modes overlap and become hard to bisect.

**Status as of 2026-05-17:** Tier 0 (−0.355) and Tier 1 implemented & measured; the multi-seed sweep ([[Multi-Seed Variance|Multi-Seed - Results]]) is the authoritative read — Tier 1 = **+11.3 ± 8.2 shipped / +8.4 ± 8.4 internal** (the single-seed +24.644 was a tail draw), seed variance ≈100 % of the signal, plus the `aug-off` arm quantifying augmentation (+21.5 over 7 ds; Language the lone seed-stable reversal). That finding promoted **E1 (budget-filling ensemble)** from optional to necessary — its validation sweep is **running now** ([[Ensemble E1|E1 - Results]]). Tier 2+ decision is gated on the E1 verdict. See [[Run State + Handoff]] for live operational state.

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
| [[Tier 0 - Hardcoded CNN]] | Validate harness, clock, shape adaptation. No search. | ✓ done — −0.355, see [[Fixed CNN|Tier 0 - Results]] |
| [[Tier 1 - Augmentation + Op Search]] | Regularization stack + residual blocks + random op-search. Made the budget productive; +24.644. Its ablation motivates the probe thesis. | ✓ done — +24.644, see [[Aug + Op Search|Tier 1 - Results]] |
| [[Tier 2b - EGNAS-light (macro)]] | **Macro architecture search** (depth-grow + width-prune), the proven 2025 core. Primary job: crack the **CIFARTile** ceiling Tier 1 can't move. Macro-only v1, single-best + log pool, ≥3-seed paired. Design locked 2026-05-18. | **← ACTIVE HEAD** |
| [[Tier 2a - Probe v1]] | P1 aug-safety probe only (design locked 2026-05-18). **Deferred to after Tier 2b** — pivot 2026-05-18: confront the NAS core (EGNAS/CIFARTile) before the orthogonal aug-probe side-quest. | planned (after 2b) |
| [[Tier 3 - EGNAS + Reliability]] | Full probe suite (add P2 invariance) + unified controller + OOM-safe / fallback + **Neural Ensemble Search** (Zaidi 2021 — robustness under dataset shift, hedges the −10 clip). The complete self-configuring pipeline. | planned |
| [[Tier 4 - Phase-2 Polish]] | Iterate from Phase-2 leaderboard signal. Per-regime tuning of probe thresholds, mixed precision (if env-safe). | planned |

## The experimental branches — now the controller's menu

Under the [[Probe-Driven Self-Configuration]] thesis these are no longer "alternatives to A/B against the shipping path." They are the **backends the probe-driven controller dispatches to** (plus honest baselines proving the controller beats any single fixed method).

| Branch | Role under the thesis | When the probe selects it |
|--------|------|--------------|
| [[Branch B1 - Zero-Cost Proxies]] | candidate pre-filter inside a triggered search | P4 macro-sensitive + large candidate pool |
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
