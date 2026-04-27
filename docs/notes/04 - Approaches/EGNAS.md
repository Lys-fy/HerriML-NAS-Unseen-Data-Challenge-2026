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

## Open questions

- Per-dataset search vs single global architecture: paper searches once per dataset. With a 30-min budget per dataset, can we cache discovered shapes across competition datasets and skip the depth-grow phase on similar ones? See [[Search vs Train Split]].
- The paper's reported costs use a single Quadro RTX 8000. The eval GPU is unknown ([[Phases and Timeline]]) — pessimistically assume slower. Trim accordingly.
- Whether the architecture-aware schedule transfers verbatim to non-image-classification-flavoured tasks like Voxel or Myofibre.
