# Useful docs

Reference material for the NAS Unseen Data Challenge 2026: papers, past-winner approaches, dataset notes.

## Papers (in `papers/`)

### 1st place 2025 — Team EVAI (University of Cyprus / Thakaa)

Two versions of the same paper. The journal version is the canonical citation.

- **`1st-2025_EGNAS_Siddiqui_SN-CompSci-journal.pdf`**
  Siddiqui, Kyrkou, Theocharides — *Efficient Global Neural Architecture Search* — SN Computer Science (2025) 6:282.
  doi:10.1007/s42979-025-03790-z. Received Mar 2024, accepted Feb 2025.

- **`1st-2025_EGNAS_Siddiqui_arxiv-preprint.pdf`**
  Same authors, arXiv:2502.03553v1 (Feb 5 2025). Free preprint.

### Organisers' analysis

- **`Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf`**
  Geada, Towers, Forshaw, Atapour-Abarghouei, McGough — *Insights from the Use of Previously Unseen Neural Architecture Search Datasets* — CVPR 2024.
  Authoritative reference on the 8 CVPR-era datasets (AddNIST, Language, MultNIST, CIFARTile, Gutenberg, Isabella, GeoClassing, Chesseract). Read this for what the organisers themselves consider "easy / hard / search-rewarding."

## 2025 leaderboard (top 3)

| Rank | Team | Affiliation | Score | GameOfLife | Wind | Cryptic |
|------|------|-------------|------:|-----------:|-----:|--------:|
| 1 | EVAI | U. Cyprus + Thakaa | **12.192** | 9.998 (99.99%) | 0.449 (17.4%) | 1.745 (76.1%) |
| 2 | Jiawen Deng | Sichuan University | 10.885 | 9.942 (99.66%) | 0.206 (15.3%) | 0.737 (73.2%) |
| 3 | Aojun Lu, Yanan Sun | Sichuan University | 10.430 | 9.995 (99.97%) | 0.033 (13.8%) | 0.402 (72.3%) |

Key reads: GameOfLife was solved by everyone (~10/10 each); Wind was effectively impossible (top score 17%); the entire ranking was decided by **a 4% accuracy gap on Cryptic**.

## What worked — by team

### 1st: EVAI / EGNAS (Siddiqui et al., U. Cyprus)

Their approach **is** the EGNAS paper. Method summary, from the journal version + the public code at `github.com/siddikui/Efficient-Macro-Micro-NAS`:

- **Search space (macro-micro):**
  - Macro = depth (number of layers) and width (channel count).
  - Micro = per-layer choice between regular `Conv` vs `SepConv`, with kernel size in {3, 5, 7}.
  - 2x channel doubling at the 1/3 and 2/3 depth points (standard ResNet-style downsampling).
- **Search algorithm (sequential, not weight-sharing):**
  1. Depth search: start min-depth max-width, train short, grow depth until accuracy plateaus.
  2. Width search: shrink channels while accuracy stays within tolerance.
  3. Operations + kernel search: per-layer hill-climb over {Conv, SepConv} × {3, 5, 7}.
- **Trick that cuts cost:** train each candidate for fewer epochs early in the search (e.g. 10), more epochs as the architecture grows. Architecture-aware training protocol → fairer ranking under tight budgets.
- **Public training defaults** (final retrain): SGD, lr=0.025, momentum=0.9, weight_decay=3e-4, cosine schedule, cutout=16, gradient clip=5, 50 epochs.
- **Reported claim:** 2× to 4× faster than the fastest prior global NAS, SOTA on EMNIST and KMNIST, competitive on CIFAR-10/100/FashionMNIST.

Code: https://github.com/siddikui/Efficient-Macro-Micro-NAS

### 2nd: Jiawen Deng (Sichuan University, Yanan Sun's group)

No paper is publicly tagged as "this is my NAS Unseen 2025 submission". What's known from Yanan Sun's group page:

- Jiawen Deng has work accepted at **PRICAI 2025** on a **performance predictor** for NAS.
- The Sun group's general direction in 2025 includes:
  - **DAP: Domain Adaptive Performance Predictor for Efficient NAS** (IEEE Trans. Computers, 2026, Song et al.).
  - **Online Performance Predictor for Evolutionary NAS** (IEEE TSMC, 2025, Song et al.).
  - **Listwise Ranking Predictor for Evolutionary NAS** (Swarm and Evolutionary Computation, 2025, Li et al.).
  - **Accurate and Robust NAS via a Flexible Supernet** (IEEE TNNLS, 2025, Feng et al.).

Inferred approach: **predictor-guided evolutionary NAS**. Train a surrogate model that predicts architecture accuracy from architecture encoding, search the space cheaply by querying the predictor, only fully train the top candidates. This explains why their score on Cryptic (the differentiator) was below EVAI's: predictor quality is the bottleneck, and predictor calibration on novel datasets like Cryptic is hard.

### 3rd: Aojun Lu + Yanan Sun (Sichuan University)

Yanan Sun's PhD student page states Aojun Lu "won 1st place at NAS global competition + 2 papers" — possibly referring to a different competition (likely 2024). For 2025 they placed 3rd.

Known publications by Aojun Lu:
- **Revisiting Neural Networks for Continual Learning: An Architectural Perspective** (arXiv:2404.14829, 2024). Continual learning, not NAS Unseen directly — but the "ArchCraft" idea (architecture-aware design that recrafts AlexNet/ResNet) shows the team's broader direction.
- Cited as co-author on "bilevel surrogate model for NAS" with Sun's group.

Inferred approach: similar to 2nd place — predictor-assisted evolutionary search from the same group's machinery. The very small score gap between 2nd and 3rd (10.885 vs 10.430, decided almost entirely by Cryptic) suggests two variants of the same toolkit.

## Lessons distilled

1. **The 1st-place winner used a sequential, hill-climbing macro-micro search — not weight-sharing, not predictor-based.** This is contrary to the "predictor / DARTS / ZC-proxy is required" intuition. Simpler than expected.
2. **Sichuan's predictor-guided approach took 2nd and 3rd.** Predictors transfer poorly to genuinely unseen data; that's where points were lost.
3. **The competition is decided on the *hardest* dataset, not the easiest.** GameOfLife was free for all top-3. Cryptic is where the leaderboard was settled.
4. **Reliability still matters more than peak performance.** All three top teams scored on every dataset; nobody got a -10 penalty.

## Reference repos and resources

| Resource | URL |
|----------|-----|
| Competition site | https://www.nascompetition.com/ |
| Past leaderboards | https://www.nascompetition.com/results (or similar; check the site) |
| Starter kit | https://github.com/Towers-D/NAS-Comp-Starter-Kit |
| Public datasets repo | https://github.com/Towers-D/NAS-Unseen-Datasets |
| EGNAS code (1st place 2025) | https://github.com/siddikui/Efficient-Macro-Micro-NAS |
| Yanan Sun's group page (2nd & 3rd 2025) | https://yn-sun.github.io/ |
| AutoML 2025 conference comp page | https://2025.automl.cc/nas-unseen-data-competition-2025/ |

## Public datasets (for local validation)

13 of 14 known datasets are downloadable; **Isabella is held back** (license issue). Total ~22 GB.

| Dataset | Size | Era | Notes |
|---------|-----:|-----|-------|
| Sudoku | 0.02 GB | 2024 | Tiny |
| GameOfLife | 0.03 GB | 2025 | Tiny — easy points for all top-3 |
| Gutenberg | 0.13 GB | CVPR | Near noise floor; ~50% acc is the ceiling |
| Language | 0.16 GB | CVPR | ResNet-18 at 97% — barely benefits from NAS |
| Chesseract | 0.22 GB | CVPR-2023 | Chess board state; small NAS gains |
| AddNIST | 0.66 GB | CVPR | Three MNIST → sum-1 |
| MultNIST | 0.66 GB | CVPR | Three MNIST → product mod 10 |
| Voxel | 2.24 GB | 2024 | 3D-flavoured |
| GeoClassing | 2.65 GB | CVPR | BigEarthNet satellite imagery |
| CIFARTile | 3.44 GB | CVPR | **Best NAS-vs-CNN delta (45→92%)** — ideal NAS test |
| Myofibre | 9.71 GB | 2024 | Largest; binary classification |
| Cryptic | ? | 2025 | Starter-kit URL is broken; ask organisers |
| Windspeed | ? | 2025 | Same — broken URL |
| Isabella | n/a | CVPR-2023 | Music spectrograms; no public download |

Use CIFARTile, Chesseract, GameOfLife, and (when available) Cryptic / Windspeed as your primary local benchmarks — they're the closest to "what the eval will look like."
