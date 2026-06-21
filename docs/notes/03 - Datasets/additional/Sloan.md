# Sloan (photometric source classification)

> **Status:** built 2026-06-17, spec frozen | **Topic:** classify GALAXY / QSO / STAR from raw photometry alone (the giveaway columns are dropped on purpose). Built from Kaggle Playground S6E6 (SDSS), blind to the pipeline, for the RQ2 generalisation test.
> **Related:** [[Datasets Overview]], [[Dataset Ideas - Backlog]], [[Sedol]], [[Cauchy]], [[Quincy]], [[GeoClassing]], [[Pipeline v10 - Design + Idea Ledger]]

![[Sloan.png]]
> *One row per class, the two 5x5 encoding channels (colour matrix b_i-b_j and sum matrix b_i+b_j). The class signal is in the colour structure. Regenerate: `python3 dev/plot_sloan_examples.py`.*

The raw Kaggle table is nearly trivial: `redshift` alone separates most classes and `spectral_type` / `galaxy_population` are leaked labels. We **drop all three** and keep only the raw photometry u,g,r,i,z plus sky position. Predicting GALAXY / QSO / STAR from photometry alone is the real historical astronomy problem (it is why spectroscopic surveys exist), so this is a fair, genuinely hard task rather than a solved one.

| Field                  | Value                                                                       |
| ---------------------- | --------------------------------------------------------------------------- |
| Codename               | Sloan (Sloan Digital Sky Survey)                                            |
| Shape                  | 2x5x5 (colour matrix / sum matrix, float32)                                 |
| corr1 / route          | **+0.83 / image-like** (smooth adjacent-wavelength colour gradients)        |
| Train / val / test     | 60,000 / 15,000 / 15,000 (balanced 3-way, 20k/5k/5k per class)              |
| Classes                | 3 (GALAXY, QSO, STAR); chance 33.3%                                         |
| Source                 | Kaggle Playground S6E6 (SDSS), `train.csv` 577,347 rows                     |
| Benchmark (`metadata`) | **84.17%** (ResNet-18 zero-point, 2026-06-18)                               |
| Generator              | `dev/make_sloan_dataset.py` (seed 11)                                       |
| Location               | `dev/datasets/Sloan` (gitignored) + spark `~/local/nas-comp-datasets/Sloan` |

## How it was generated

1. **Strip the giveaways.** Keep only u,g,r,i,z (photometric magnitudes). Drop redshift, spectral_type, galaxy_population (the leaks), and the sky position alpha,delta (weak, possibly survey-leaking, and a fingerprint-inflating constant plane).
2. **Robust clean + standardise.** Each band is clipped to its 0.1-99.9 percentile (guards SDSS sentinels), then z-scored using TRAIN statistics only.
3. **Physics encoding.** The five bands are a coarse SED ordered by wavelength u<g<r<i<z. Per source we build a 2x5x5 tensor: a colour matrix `C[i,j] = b_i - b_j` (all 10 colour indices) and a sum matrix `S[i,j] = b_i + b_j` (absolute brightness scale). Adjacent cells are adjacent wavelengths, so the matrices are smooth (hence image-like routing).
4. **Balance + package.** Equal per class (chance is exactly 1/3, so accuracy is meaningful). float32 `*_x.npy`, int64 `*_y.npy`, JSON `metadata`.

## Results (2026-06-18)

![[newdata_curves_grid_4x5.png]]
> *Custom-dataset grid (bottom three rows are the new sets): v10 best-val (green, 3 seeds + aggregate) over the ResNet-18 zero-point (blue).*

ResNet-18 benchmark **84.17%**. v10 3-seed test accuracy **84.60%** (s42 84.63, s1337 84.55, s2024 84.61), so **adjusted +0.27** (essentially tied with ResNet). Image-like and near-ceiling: both models saturate at ~84% within a few epochs, so there is little architectural headroom here. The task is moderately hard (well above the 33% floor) but low-dimensional, so it is a weak architecture discriminator.

## Measured properties (before any pipeline run)

- **The giveaway-drop is the whole design.** With redshift in, logistic regression is ~95%+; this set deliberately removes that shortcut so the classifier must use colour structure.
- **Signal is real but hard.** QSOs have a UV excess and distinctive colours, stars and galaxies overlap; expect a moderate benchmark, well above the 33.3% floor but far from solved. A logistic-regression sanity check on the 7 standardised features is owed (sklearn lives on spark).
- **Routes image-like (corr1 = +0.83).** The colour and sum matrices are smooth across adjacent wavelengths, so the fingerprint sees strong local correlation and uses the 2-D backbone, not the encoded route. Despite being tabular in origin, the physics encoding makes it an image-like task. (Dropping the two constant position planes lowered corr1 only from +0.92 to +0.83, confirming the smooth colour matrices, not the planes, drive the routing.)

## Open items

- ResNet-18 benchmark (84.17%) + v10 3-seed (84.60%): DONE (see Results).
- First v10 run; check the backbone gate and whether the 4-channel small (5x5) input over-pools the conv stack (if so, pad to 8x8 and regenerate).
- The 5-band SED could alternatively feed the 1-D TCN family directly; a 1-D variant is a cheap follow-up if the 2-D encoding underperforms.
