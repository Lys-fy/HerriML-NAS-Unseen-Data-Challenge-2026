# EGNAS

> **Status:** current | **Topic:** Efficient Global NAS — sequential macro-micro hill-climbing search.
> **Related:** [[Approaches Overview]], [[2025 - EVAI]], [[DARTS Family]], [[Predictor-Guided Evolutionary]], [[Zero-Cost Proxies]], [[Search vs Train Split]]

The 2025 winning method, by Siddiqui, Kyrkou, Theocharides (U. Cyprus + KIOS). Paper: [SN Computer Science 2025, 6:282](https://doi.org/10.1007/s42979-025-03790-z) (also [arXiv:2502.03553](https://arxiv.org/abs/2502.03553)). Public code: [siddikui/Efficient-Macro-Micro-NAS](https://github.com/siddikui/Efficient-Macro-Micro-NAS). Read the [[2025 - EVAI]] note for team context.

## Why it won

EGNAS sidesteps the two main NAS pain points in one shot:
1. **Modular ([[DARTS Family]]) search spaces ignore depth/width.** They search a cell, then stack manually — a competition with mixed datasets gets the macro shape wrong by default.
2. **Predictor / weight-sharing rankings are unreliable on novel data** ([[Predictor-Guided Evolutionary]] vs [[Weight-Sharing One-Shot]]) — the inductive priors don't transfer to "weird" datasets like CIFARTile or Cryptic.

Greedy global search avoids both: it adapts the full architecture per-dataset, and trains real candidates (just with short, architecture-aware schedules) so the rankings are not polluted by proxy or predictor noise. The cost (under 1 GPU-day on CIFAR-10) fits in the per-dataset budget.

## Search space (macro + micro)

End-to-end FCN — no FC, no pooling, no skip connections inside the search space; ResNet-style downsampling baked in at 1/3 and 2/3 of depth (channels double, resolution halves). Variables: depth `D ∈ [10, 100]`, base width `W ∈ [16, 128]` (paper) or `[16, 256]` (public code), per-layer op `O ∈ {Conv, SepConv}`, per-layer kernel `K ∈ {3, 5}` in code (`{3, 5, 7}` in paper). Total ≈ `6.16 × 10^18` candidates — same order as DARTS but with much wider accuracy range.

## The algorithm (Algorithm 1 in the paper)

Sequential, **not** weight-sharing, **not** predictor-based, **not** gradient.

```
Init: D = D_min (10), W = W_max (256), O = SepConv, K = 3
MACRO
  1. Grow: L ← L+1, train +1 epoch. Continue while val_acc improves by ≥0.10.
     Tolerate up to 3 drops < 0.05. Stop on max depth or 3 strikes.
  2. Prune: C ← C - W_res (2), train +2 epochs. Continue while val_acc retained.
     Stop on min width or 3 strikes.
MICRO  (re-baseline at E_min epochs with discovered D_f, W_f)
  3. Replace: per layer, try O ∈ {Conv, SepConv}. Adjust width to keep
     param count ≈ baseline. Accept if val_acc improves.
  4. Update: per layer, try K = 5. Same parameter-equalising trick.
Return (D_f, W_f, O_f, K_f)
```

Total candidates evaluated ≈ `2·D_f + D' + W'` (paper claims ~25), versus `O(10^4)` for early evolutionary NAS.

### Architecture-aware ("dynamic") training schedule

The crux. Section 4 of the paper shows: ranking 240 networks by 1-epoch val-acc has Spearman 0.65 with their final acc; ranking by params has 0.49. **But** sorting by params and training each network for 1, 2, 3, … epochs in ascending param order yields 0.91 correlation with final acc. Translation: bigger nets need more epochs to be ranked fairly — train them more.

EGNAS bakes this in: depth-grow phase trains each new candidate for one extra epoch beyond its predecessor (`E ← E + 1`); width-prune phase adds two extra epochs per pruned variant (since shrinking hurts early); operations and kernels micro-search uses a fixed `E_min = 2` per evaluation since the param count is held constant.

## Defaults from `train.py`

| Knob | Value |
|---|---|
| Optimizer | SGD, momentum 0.9 |
| LR | 0.025, cosine annealed to 0 |
| Weight decay | 3e-4 |
| Batch size | 64 |
| Final retrain epochs | 50 |
| Search inner epochs | start 10 (macro) / 2 (micro), grow with arch |
| Augmentation | Cutout length 16, standard data_transforms |
| Gradient clip | 5 (clip_grad_norm_) |
| Label smoothing / drop_path | not used |

No specialised tricks (no Mixup, no RandAugment, no SWA) — Section 5.2 explicitly avoids "enhanced training protocols that hide the contribution of the search strategy".

## Reported results

| Setting | Dataset | Test err / acc | Params | Search cost |
|---|---|---|---|---|
| Tiny (W_max=64) | CIFAR-10 | 4.09% err | 0.46 M | 0.24 GPU-d |
| Mobile (W_max=128) | CIFAR-10 | 3.17% err | 2.49 M | 0.43 GPU-d |
| Tiny | CIFAR-100 | 23.64% err | 0.49 M | 0.17 GPU-d |
| Tiny | EMNIST | **91.20%** acc (SOTA) | 0.40 M | 0.36 GPU-d |
| Tiny | KMNIST | **99.35%** acc (SOTA) | 0.42 M | 0.36 GPU-d |
| Tiny | FashionMNIST | 95.90% acc | 0.50 M | 0.11 GPU-d |

2× to 4× faster than fastest prior global NAS (Macro-NAS [24]); 18× smaller and faster than FT-DARTS on FashionMNIST.

## Adapting EGNAS to the competition

- **Per-dataset budget**: search cost on CIFAR-10 was 0.24 GPU-d (~5.8 h) — too long for the Phase 2 ~1 h smoke test. Cut `D_max`, drop the operations sub-phase, use smaller `E_min`, or train at lower resolution to fit the [[Clock and Time Budget]].
- **Channel doubling at 1/3 and 2/3** assumes spatial input ≥ ~24 px. For tiny inputs (Sudoku 9×9, GameOfLife 16×16 region) the second doubling will collapse spatial dims — clamp downsampling locations or skip the second.
- **Initialization** matters (Table 2): start `Sep + 3×3` always — it dominates the other three init strategies.
- **The strikes constants** (3 fails for grow, 3 for prune) are aggressive on small datasets where val accuracy is noisy. Average 2–3 seeds before deciding to stop.

## Our port vs upstream — discrepancies found by reading the actual code (2026-05-28)

Source: github.com/siddikui/Efficient-Macro-Micro-NAS (matched against arXiv:2502.03553). Comparison was triggered by v5's GameOfLife regression — the new ops we added to OP_VOCAB (GroupConv3, ResNeXt) produced noisy short-train picks that didn't hold up at full training. The mismatch with the original paper's methodology explains why.

| Aspect | Original (paper + code) | Our port (v5) | Note |
|---|---|---|---|
| OP_VOCAB size | **4** (`{plain_conv, sep_conv} × {k=3, k=5}`) | 7 (v5), 9 swappable (v6) | We've radically expanded the search space |
| Micro per-candidate K | hard-coded `args.epochs = 2` (`search.py:363`) | 2-6 depth-scaled (v5) | We're MORE generous than upstream |
| Macro initial K | `args.epochs = 10`, grows +1 per depth step | 2-6 depth-scaled | We're under-budgeting the macro phase by 4-5 × |
| Acceptance rule | `next_acc >= curr - 0.0` (ties accept) | `> curr + 0.003` (stricter) | Ours is more conservative — fewer false accepts but also fewer correct accepts |
| Verification of accepts | **none** | none (v5) → **2× K re-train, revert if fails** (v6) | Original commits noise; their 600-epoch final train washes it out. We can't afford that at 2 h, hence v6 verification |
| Final training | **600 epochs** (paper §4.1.2) | ~70 epochs at 2h budget | Their methodology's noise tolerance is built on a 10 × longer final train |
| Param-fairness in candidate eval | YES (`while candidate_params > baseline + 0.01: channels -= 1`, `search.py:394-398`) | **NO** | We compare arches with very different param counts (Conv3 10K vs GroupConv3 1.4K vs ResNeXt 6.3K) — uncontrolled confound |
| Search-time ranker | TEST acc (`valid_size=0.0`) | proper VAL acc | Ours is legal-cleaner but slightly noisier; not a fundamental difference |
| Search vs training time | wall-clock unbounded for both (no `--time-limit`) | 30 % search / 70 % training | Competition imposed; original didn't have this constraint |
| Search range | D ∈ [10, 100], W ∈ [16, 256] | D ∈ [3, 12], W ∈ [16, 48] | Massively smaller search range in our port — narrow vocab × wide range vs wide vocab × narrow range |

### What the misread "Spearman 0.91" claim actually measured

The paper's headline "Spearman 0.91 between rank at incrementally-trained budget and full-trained" is **NOT** "2-epoch val_acc on Unseen X predicts 600-epoch test acc on Unseen X". It is:
> "we sample 50 networks from 240 and sort w.r.t. their parameters in ascending order... the smallest network trained only for 1 epoch, the next one for 2 epochs, and so on. We observe a high correlation value of 0.91"

i.e., correlation of **(param count, sorted) ↔ (val_acc when each net is trained for monotonically-increasing epochs)** on **CIFAR-10 only**, aggregate over 50 sampled nets. Our Q8 analysis (Spearman ranges +0.7 to −0.5 across our datasets) was measuring a different thing — **we haven't been failing to reproduce anything the paper claimed**.

### Zero-shot proxies

Standard NAS technique that bypasses short-train val_acc entirely. Four well-validated proxies (NASWOT, Synflow, SNIP, ZenScore) are mathematical functions of an *untrained* random-init network — Kendall's τ vs full-train rankings is typically 0.4-0.7 on NAS-Bench-201 (Abdelfattah et al. ICLR 2021).

**Legal under competition rules.** They're not a trained model, not pre-trained weights, no external data — just architecture analysis on the random initialization. Same legal category as "we compute FLOPs".

In v6 we log all 4 per EGNAS candidate as **shadow signals** (no decision impact). Post-sweep Spearman analysis tells us if any of them transfer to the Unseen suite. If τ ≥ 0.5 for any proxy, v7 could replace the noisy short-train ranker with the proxy → 10-1000 × faster search.

References:
- Mellor et al. ICML 2021 — NASWOT
- Tanaka et al. NeurIPS 2020 — Synflow
- Lee et al. ICLR 2019 — SNIP
- Lin et al. ICCV 2021 — Zen-NAS / ZenScore
- Abdelfattah et al. ICLR 2021 — comprehensive benchmark

## Open questions

- Per-dataset search vs single global architecture: paper searches once per dataset. With a 30-min budget per dataset, can we cache discovered shapes across competition datasets and skip the depth-grow phase on similar ones? See [[Search vs Train Split]].
- The paper's reported costs use a single Quadro RTX 8000. The eval GPU is unknown ([[Phases and Timeline]]) — pessimistically assume slower. Trim accordingly.
- Whether the architecture-aware schedule transfers verbatim to non-image-classification-flavoured tasks like Voxel or Myofibre.
