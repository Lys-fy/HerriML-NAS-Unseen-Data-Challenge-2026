# 2022 - BLOX: Macro Neural Architecture Search Benchmark and Algorithms

> **Source:** Chau, Dudziak, Wen, Lane, Abdelfattah (Samsung AI Center), NeurIPS 2022 Datasets & Benchmarks. arXiv:2210.07271. github.com/SamsungLabs/blox
> **Why it matters here:** external confirmation of our proxy-reliability wall, and a headroom argument for a real macro (heterogeneous-block) search. Read 2026-06-15.
> **Related:** [[SOTA Scan - Novel Directions]], [[Key Learnings]], [[Pipeline v10 - Design + Idea Ledger]], [[EGNAS Full]]

## What it is
A benchmark (91,125 CNNs trained on CIFAR-100, + latency on several devices) for **macro** NAS, where "macro" means **each block/stage of the network may have a DIFFERENT architecture** (heterogeneous, A->B->C), as opposed to **cell-based** NAS which searches one block and repeats it (A->A->A). Note this is a DIFFERENT sense of "macro" than EGNAS's (where macro = depth + width skeleton). 3 stages x 45 block options = 45^3 models.

## The two findings that matter for us

1. **Heterogeneous (different-block) models dominate the Pareto frontier** over uniform-block models (Fig 4). I.e. there is real accuracy headroom in letting blocks differ - which a uniform stack (what v10's main path builds) cannot reach.

2. **Proxies / block signatures for "is this block good" are unreliable** - independently reproducing our proxy-reliability wall:
   - Block signatures (DONNA distillation-loss, LANA val-acc-change) are NOT consistently indicative of block performance; correlation with the oracle ranking is weak and **worse in later stages** (Fig 9).
   - Fine-tuned / distilled ranking does NOT match train-from-scratch ranking (Fig 8, Spearman 0.25-0.31).
   - Authors' conclusion: do NOT search with a cheap proxy and then rely on it; **directly search for a model that performs well when fully trained.** This is exactly our train-to-budget + verify-before-adopt philosophy.

Other notes: among conventional methods BRP-NAS (predictor-based) and regularised evolution did best; their efficient recipe is reduced fine-tune then full fine-tune (FT10+FT200); the *teacher/training methodology* mattered more than the student architecture.

## Implication for our project
- Confirms (from an independent macro-NAS study) that the short-horizon proxy rankings we abandoned for op-search are genuinely unreliable - the wall is real, not an artefact of our budget.
- Argues there IS headroom in heterogeneous architectures, which v10 only touches via the gated random arm (random per-layer ops on encoded data). A *real* macro search for us would have to be **verify-heavy** (the same machinery that fixed depth / width / random-arm), because the naive proxy route is exactly what BLOX shows fails. Strong v11+ direction: a systematic, verified heterogeneous-block search.
