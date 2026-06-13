# Overview

> **Status:** drafting | **Topic:** All 14 known competition datasets in one comparison table; era split (CVPR 2021–23 vs AutoML era 2024+); characteristic groupings.
> **Related:** [[AddNIST]], [[Language]], [[MultNIST]], [[CIFARTile]], [[Gutenberg]], [[Isabella]], [[GeoClassing]], [[Chesseract]], [[Sudoku]], [[Voxel]], [[Myofibre]], [[GameOfLife]], [[Cryptic]], [[Windspeed]], [[Score-Efficient Targeting]], [[Reliability vs Peak]]

The competition pool spans **14 named datasets** across two eras. **All but [[Isabella]] are now publicly downloadable** (Isabella held back over museum-recording licensing). The two 2025 datasets ([[Cryptic]], [[Windspeed]]) — DOIs `10.7488/ds/8054` / `10.7488/ds/8053` — were *not broken*, just **published late (2025-12-05)**; they now resolve correctly to Edinburgh DataShare (`datashare.ed.ac.uk/handle/10283/9132` and `/9131`), CC-BY-4.0. Specs confirmed (2026-05-19) — see those notes.

All datasets share the same on-disk shape: six `.npy` files (`{train,valid,test}_{x,y}`) plus a `metadata` JSON (`num_classes`, `codename`, `input_shape`, `benchmark`) and a README. Pixels are channels-first 4-D `(N, C, H, W)`. The `benchmark` field is the ResNet-18 accuracy used as the zero-point in the [[Scoring]] formula.

## Comparison table

| Dataset | Codename | Era | Shape (C×H×W) | Classes | Benchmark | Best NAS (Geada) | Type |
|---|---|---|---|---:|---:|---:|---|
| [[AddNIST]] | Adaline | CVPR | 3×28×28 | 20 | 89.85% | 97.91% (Bonsai) | image-y, T1 |
| [[MultNIST]] | Mateo | CVPR | 3×28×28 | 10 | 90.87% | 98.10% (DrNAS) | image-y, T1 |
| [[Language]] | LaMelo | CVPR | 1×24×24 | 10 | 85.20% | 90.12% (PC-DARTS) | encoded, T2 |
| [[Gutenberg]] | Gutenberg | CVPR | 1×27×18 | 6 | 40.98% | 49.12% (PC-DARTS) | encoded, T2 |
| [[CIFARTile]] | — | CVPR | 3×64×64 | 4 | 45.56% | 92.28% (PC-DARTS) | image-y, T1 |
| [[Isabella]] | — | CVPR-2023 | 1×64×128 | 4 | 62.02% | 66.35% (rand-DARTS) | encoded, T2 |
| [[GeoClassing]] | Sadie | CVPR | 3×64×64 | 10 | 80.33% | 96.03% (DrNAS) | image-y, T2 |
| [[Chesseract]] | Chester | CVPR-2023 | 12×8×8 | 3 | 57.83% | 68.83% (rand-Bonsai) | encoded, T1 |
| [[Sudoku]] | Sokoto | 2024 | 1×9×9 | 9 | 0.0%¹ | — | encoded, T1 |
| [[Voxel]] | Volga | 2024 | 20×20×20² | 7 | 71.35% | — | 3D-encoded |
| [[Myofibre]] | — | 2024 | 3×128×128 | 2 | unconfirmed | — | image-y, medical |
| [[GameOfLife]] | conway | 2025 | 1×10×10 | 25 | 47.53% | — | encoded, T1 |
| [[Cryptic]] | unknown | 2025 | 6×768 (6 words, BERT-encoded) | unknown | ~73–76% (lb-inferred) | — | encoded text (cryptic-crossword clues) |
| [[Windspeed]] | unknown | 2025 | 12×32×32 (6h×{temp,pres}, ERA5) | ~21 (Beaufort bins 20–40) | ~13–14% (lb-inferred) | — | regression→ordinal classes; temporal shift (tr20–22/va23/te24) |

¹ Sudoku metadata ships `benchmark: 0.0` — almost certainly a placeholder, not a real ResNet-18 result. Treat as unknown.
² Voxel uses the channels dimension as a third *spatial* axis (not RGB) — see [[Voxel]].

## Era split

**CVPR era (2021–2023, 8 datasets)**: AddNIST, MultNIST, Language, Gutenberg, CIFARTile, Isabella, GeoClassing, Chesseract. Documented in [Geada et al. 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVNAS/html/Geada_Insights_From_The_Use_of_Previously_Unseen_Neural_Architecture_Search_CVPRW_2024_paper.html). All have published ResNet-18 / off-the-shelf-CNN / NAS-method baselines (Tab. 1 and 2 of that paper).

**AutoML era (2024+, 6 datasets)**: Sudoku, Voxel, Myofibre, GameOfLife, Cryptic, Windspeed. No published method comparisons — the README+metadata is currently the only source. Phase 2 / Phase 3 datasets are drawn from this pool plus possibly fresh ones.

## Characteristic groupings

**Classic image classification** (works fine with off-the-shelf CNNs out of the box):
- [[GeoClassing]] (satellite RGB), [[Myofibre]] (medical RGB).

**Image-y but with a hidden compositional task** (Type-1 in Geada's taxonomy — a human can solve them with prior knowledge):
- [[AddNIST]], [[MultNIST]] (sum/multiply 3 stacked MNIST digits), [[CIFARTile]] (count distinct CIFAR-10 classes in a 2×2 grid), [[GameOfLife]] (count alive cells next gen), [[Sudoku]] (predict missing cell), [[Chesseract]] (predict game outcome from board state).

**Encoded-non-image** (Type-2 — a "channels-first 4-D tensor" but the spatial axes don't carry visual structure; a human cannot solve them by looking):
- [[Language]], [[Gutenberg]] (character-position grids), [[Isabella]] (audio spectrograms), [[Voxel]] (3-D shape in voxel grid, with spatial-z cosplaying as channels).

**Encoded-non-image, 2025** ([[Cryptic]], [[Windspeed]]) — *now confirmed*: Cryptic = 6 cryptic-crossword words → 6×768 BERT embeddings; Windspeed = 6 h ERA5 temp+pressure (12×32×32) → ~21 ordered Beaufort bins (a discretised regression, with a temporal train/test shift).

## NAS-rewardingness (CVPR-era datasets)

NAS reward = `best_NAS - ResNet-18`. The largest gap is the canonical "search-rewarding" signal:

| Dataset | NAS reward | Comment |
|---|---:|---|
| [[CIFARTile]] | **+46.7 pp** | The poster-child for NAS-vs-CNN on this benchmark. ResNet-18 stuck near random; NAS jumps to 92%. |
| [[GeoClassing]] | +15.7 pp | Real-image dataset — gain comes from architecture *and* depth tuning. |
| [[AddNIST]] | +5.8 pp | Modest. Off-the-shelf already at 92%. |
| [[MultNIST]] | +6.6 pp | Same shape, similar story to AddNIST. |
| [[Chesseract]] | +11.0 pp | Random search beats every named NAS method here — see [[Chesseract]]. |
| [[Language]] | +3.1 pp | NAS gives little; ResNeXt-50 actually beats every NAS method (93.97% > 90.12%). |
| [[Gutenberg]] | +8.1 pp | But absolute ceiling ~50%. Probably noise floor, not NAS limit. |
| [[Isabella]] | +4.3 pp | Small, no NAS method clearly wins. |

[[CIFARTile]] is the clear NAS-search-justifying dataset. Several others reward better-engineered training more than search.

## Score-relevance ranking (April 2026)

For Phase 2/3, the 2025 leaderboard mix was [[GameOfLife]] + [[Windspeed]] + [[Cryptic]]. See [[2025 - EVAI]] for the breakdown. Working hypothesis: 2026 will rotate in some 2024-era datasets ([[Sudoku]], [[Voxel]], [[Myofibre]]) and may keep one or two 2025 datasets.

In any single run, the [[Scoring]] formula rewards 1% accuracy gains *much* more on a high-benchmark dataset (slope = `10 / (100 - benchmark)`):
- benchmark 47% ([[GameOfLife]]) → +0.19 / 1%
- benchmark 71% ([[Voxel]]) → +0.34 / 1%
- benchmark 90% ([[AddNIST]] / [[MultNIST]]) → +1.00 / 1%

Implication for [[Score-Efficient Targeting]]: **don't over-invest in low-benchmark datasets** — even a perfect score on a 50%-benchmark dataset is worth less than a +5% gain on a 90%-benchmark one. Conversely, the biggest cliff is the −10 fail; **never miss a benchmark**.

## Additional datasets (ours)

Custom datasets we build ourselves, spec-frozen before the pipeline sees them, used for the RQ2 generalisation test. They live in `additional/`. Two are home-built (Quincy), four come from a teammate's generator tool vendored at `dev/datagen/` (The Count: mathvision, connect4, polygon_collision, domino), triaged 2026-06-13 with corr1 + logreg.

| Dataset | Task | Shape | corr1 / route | Status |
|---|---|---|---|---|
| [[Quincy]] | decode the class token from a version-1 QR code, 10% ECC-recoverable corruption | 1x21x21 | 0.05 / encoded | built, benchmark owed |
| [[Magnus]] | predict the degree (1-5) of a complex polynomial phase portrait | 3x64x64 | 0.65 / image-like | GOOD, spec to freeze |
| [[Cornelius]] | predict the immediate winning column of a Connect-4 board (or none) | 3x64x64 | 0.75 / image-like | GOOD, balanced 2026-06-13 (none-class 50%->9%), spec to freeze |
| [[Candidates - Pavel + Dominik\|Pavel]] | does the bouncing object hit a wall next step | 4x64x64 (was 3x64x256) | image-like | frames-as-channels DONE, CNN check owed |
| [[Candidates - Pavel + Dominik\|Dominik]] | is the domino chain valid or one-mismatch | 3x64x64 | 0.70 / image-like | NEEDS WORK: snake layout + tiny pips (fix in note) |

Triage takeaway: Magnus and Cornelius are genuine NAS-rewarding tasks (logreg far below a real net's reach) and exercise the IMAGE-LIKE branch, complementing Quincy's encoded branch. Pavel and Dominik need generator fixes before use (ideas in [[Candidates - Pavel + Dominik]]). Still planned: tic-tac-toe encoded, minesweeper (see [[Pipeline v10 - Design + Idea Ledger]]).

## Open dataset questions

- ~~[[Cryptic]] and [[Windspeed]] sources — DOIs wrong~~ **RESOLVED 2026-05-19**: DOIs were fine, datasets just published late (2025-12-05) to Edinburgh DataShare, CC-BY-4.0. Specs in those notes.
- [[Isabella]] generation script said "coming soon" by the organisers — not yet visible at github.com/Towers-D/NAS-Unseen-Datasets.
- Whether the Phase 2/3 dataset rotation in 2026 keeps any 2025 set.
