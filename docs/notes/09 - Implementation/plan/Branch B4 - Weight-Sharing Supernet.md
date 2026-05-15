# Branch B4 — Weight-Sharing Supernet

> **Status:** planned (experimental — high-risk hedge, only viable with offline pretraining) | **Topic:** SPOS / OFA-lite trained offline on the public 13, sample sub-nets per Phase-2 dataset. *Probably fails because hidden datasets won't share shape with the supernet's training corpus.*
> **Related:** [[Implementation Plan Overview]], [[Weight-Sharing One-Shot]], [[DARTS Family]], [[Approaches Overview]], [[Threats to Validity]], [[Datasets Overview]]

## Goal

Train **one** supernet on multiple public datasets *before Phase 2*, then at Phase-2 inference time sample a small sub-architecture per hidden dataset from inherited supernet weights — paying ~zero per-dataset search cost. The recipe behind [[Weight-Sharing One-Shot|OFA]] (Once-For-All).

This is the highest-risk experimental branch. The strategy notes ([[Threats to Validity]] §3) explicitly warn it overfits to the public 13.

## Recipe (SPOS-lite)

```
Offline (before Phase 1 ends, on the DGX Spark):
  1. Build a supernet:
       N candidate ops per edge, fixed depth and width.
       Match the architecture family of [[Tier 2 - EGNAS-Light]] for transferability.
  2. Train (~100 epochs) by sampling one path uniformly per step.
     Use a mixed loader: union of (AddNIST + MultNIST + GeoClassing + CIFARTile + ...)
     with each dataset normalised to its own (C, H, W).
     Either rescale all to one shape (lossy) or train separate adapters per shape.
  3. Save the supernet state dict + adapter weights. Bundle in submission/.

Phase 2 (per dataset, online):
  1. Detect the input shape and pick the adapter (or downsample to the supernet's resolution).
  2. Sample 100 sub-paths from the supernet uniformly.
  3. Score each using the inherited weights (one forward pass on val_loader — seconds).
  4. Pick the top sub-path; full-train it from scratch on this dataset.
```

The "full-train from scratch" step is non-negotiable per [[Weight-Sharing One-Shot]] §"Cons": sub-net accuracy under supernet weights != accuracy after standalone training.

## Why this is high-risk

1. **Shape mismatch is the dominant failure mode.** The supernet is trained at one input shape (or a fixed set). Hidden datasets have unknown shapes. If Phase-2 hands us 5×5 inputs, no public-13 dataset matches and the supernet is useless.

2. **Co-adaptation in supernet weights doesn't transfer to discretised sub-nets.** [[Weight-Sharing One-Shot]] §"the ranking-fidelity problem" — sub-net rankings from supernet val acc don't match standalone-trained rankings.

3. **Hidden datasets are by design "unlike" the public ones.** The competition's premise. The 2025 Cryptic example: no public-13 dataset has Cryptic's structure; a supernet trained on 12 of them couldn't predict its 13th sibling, let alone Cryptic.

4. **Offline training cost is large.** ~100 epochs on a mixed-dataset supernet is a multi-day Spark job. That's research time we don't have if it doesn't pan out.

## Why we might still build it

- **B4's offline pretraining is sunk cost** — done once before Phase 2, reused across attempts.
- **If a hidden dataset *does* share shape with the public 13** (e.g. AddNIST sibling), B4 needs ~30 s of search vs Tier 2's 30 min. The Trainer phase then gets 2× the budget.
- **It's the canonical "one-shot" baseline** for AutoML competitions. Worth understanding empirically how badly it fails on novel data — informs future entries.
- **OFA / progressive shrinking is publishable** in our own right. If the team has academic incentives (this is graded coursework), an OFA-style ablation is a strong contribution.

## SPOS vs OFA — which lite version?

**SPOS** (Guo et al. 2019): uniform sampling per step, post-hoc EA for sub-net search. Simpler, more robust. Recommended for B4.

**OFA** (Cai et al. ICLR 2020): progressive shrinking schedule, latency-aware accuracy predictor. Finicky to implement, more research-novel. Probably overkill for our purposes.

**ProxylessNAS**: binarised gates during training, hardware-aware. Closer to [[DARTS Family]] than to SPOS. Skip.

Pick SPOS. Code reference: [megvii-model/SinglePathOneShot](https://github.com/megvii-model/SinglePathOneShot).

## The shape problem — how to handle

Three options:

| Approach | Pros | Cons |
|---|---|---|
| **Resize all to 32×32** at supernet train time | Single supernet, simple | Information loss on Myofibre / GeoClassing; can't recover at inference |
| **Multi-resolution supernet** with adapters per H×W | Each shape gets its native resolution | Multiple supernets to ship; total weight size grows |
| **Channel-only supernet** at fixed 32×32 | Resolution loss but supernet is tiny | Same info loss issue; fails on novel shapes |

The pragmatic answer: train at the *most common* shape (3×28×28 for AddNIST/MultNIST) and accept that Sudoku / GeoClassing get a separate Tier-3 fallback in B4's submission code.

## Acceptance criteria

- [ ] Supernet trains to a stable val acc on at least 4 of the public 13 (proves the supernet learns *something*).
- [ ] Sub-net sampling at Phase-2 time runs in ≤ 30 s.
- [ ] On AddNIST / MultNIST (shapes most likely covered by the supernet), scores ≥ Tier 1.
- [ ] On Voxel / Myofibre / GeoClassing (likely shape-mismatched), submission *falls back* to Tier 3 baseline without crashing.
- [ ] Total submission archive < 200 MB ([[Submission Packaging]] norm).

## When B4 graduates

Almost certainly never to the *exclusive* shipping path. The realistic role:

- Bundle the supernet weights in [[Tier 3 - EGNAS + Reliability|T3]]'s submission as an optional warm-start. If the hidden dataset shape matches, use it; if not, ignore.
- Use the supernet as a *trainer initializer* (not a NAS) — load pretrained weights for a similar-shape sub-net, fine-tune. Closer to transfer learning than to NAS.

Both are downstream uses; B4 as a standalone search algorithm doesn't ship.

## Repo layout

```
baselines/b4_supernet_spos/
  data_processor.py     # plus shape-resize adapters
  nas.py                # sub-net sampler + scorer
  trainer.py            # sub-net retrain from scratch
  helpers.py            # supernet load + path sampling
  supernet.pth          # ~50-150 MB (bundled in zip if B4 ships)
  supernet_train.py     # offline-only — not zipped
```

The `supernet_train.py` is the multi-day-Spark-job script. Lives in the baseline dir for reproducibility but is not in the Phase-2 zip.

## Open questions

- Whether to amortise supernet training across multiple competitions (or hide-future-edition datasets). Reuse value depends on how stable the competition's dataset family is — based on 2021–2025 history, fairly stable for CIFAR-shape datasets but rotating for the rest.
- Whether to use the supernet inside [[Tier 2 - EGNAS-Light|EGNAS]]'s depth-grow phase as a "warm-up" so each new depth candidate starts with supernet-inherited weights. Hybrid approach. Untested.
- Whether the team has the dev-time budget for this. *Probably not* — the strategy notes ([[Phase 2 Submission Strategy]]) implicitly assume the shipping path absorbs most of the work, with branches kept light.

## References

- [[Weight-Sharing One-Shot]] — full method note.
- [Guo et al., *Single Path One-Shot Neural Architecture Search with Uniform Sampling*, ECCV 2020](https://arxiv.org/abs/1904.00420).
- [Cai et al., *Once-For-All: Train One Network and Specialize it for Efficient Deployment*, ICLR 2020](https://openreview.net/forum?id=HylxE1HKwS).
- Code: [megvii-model/SinglePathOneShot](https://github.com/megvii-model/SinglePathOneShot), [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all).
