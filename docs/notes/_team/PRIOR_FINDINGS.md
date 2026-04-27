# Prior Findings — authoritative starting context

> Synthesised from extensive prior research. Treat as authoritative — don't re-research these facts unless you have specific reason to doubt them. Use WebSearch / WebFetch for gaps and for claims marked `(verify)`.

## 1. Competition basics

- **5th edition** of the NAS Unseen-Data Competition, hosted at AutoML 2026.
- Site: https://www.nascompetition.com/
- Organisers: D. Towers, R. Geada, A. Atapour-Abarghouei, M. Forshaw, A. S. McGough — Newcastle University.
- Contact: nas-competition-contact@newcastle.ac.uk

### Phases

| Phase | When | What | Where |
|-------|------|------|-------|
| Phase 1 | Open now (April 2026) | Development. Public datasets available. Teams build and test locally. Unlimited iterations. | Your machines |
| Phase 2 | July 1 – July 31 | "Validation" / smoke test. ≤7 submissions. ~1h per dataset. | Organisers' servers, hidden datasets |
| Phase 3 | August 1+ | Final scored run. **Same submission as last working Phase 2 entry — no separate Phase 3 push.** Unknown wall-clock budget. | Organisers' servers, same hidden datasets |

### Scoring

- Per dataset: `adj_score = (raw_acc - benchmark) * 10 / (100 - benchmark)`, capped at +10, floored at −10.
  - 100% accuracy → +10
  - accuracy == benchmark → 0
  - failure (timeout / OOM / crash) → −10
- Total = sum of `adj_score` across datasets.
- Slope by benchmark: high-benchmark datasets reward more points per 1% raw gain; low-benchmark datasets the opposite.
  - benchmark 50% → +0.2 pts per 1%
  - benchmark 70% → +0.33 pts per 1%
  - benchmark 90% → +1.0 pts per 1%
  - benchmark 95% → +2.0 pts per 1%

### Rules (from competition site)

- 7-submission cap in Phase 2.
- Last working Phase 2 submission is auto-promoted to Phase 3.
- Code must respect a `run_clock` passed through. Time overruns score −10.
- Code must respect available RAM and VRAM. Crashes score −10.
- Don't attempt to download the final-stage datasets. Disqualifying.
- Teams allowed (single submission per team). Multiple teams from same institution allowed only if member-disjoint and approaches distinct.
- Finalists present (in person or virtually) at AutoML 2026.

## 2. Pipeline contract

- Submission = 4 files: `data_processor.py`, `nas.py`, `trainer.py`, `helpers.py`.
- Classes: `DataProcessor`, `NAS`, `Trainer`. Filenames and class names hardcoded in `evaluation/main.py`.
- **Method signatures** (from starter kit):
  - `DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, clock)`, `.process() → (train_loader, valid_loader, test_loader)`
  - `NAS(train_loader, valid_loader, metadata, clock)`, `.search() → PyTorch model`
  - `Trainer(model, device, train_loader, valid_loader, metadata, clock)`, `.train() → trained model`, `.predict(test_loader) → list[int]`
- A `clock` object is passed to every stage. `clock.check()` returns seconds remaining for the current dataset.
- `metadata` dict has: `num_classes`, `codename`, `input_shape`, `benchmark`, `time_limit` (hours, default 0.5), `time_remaining` (set per stage).
- Datasets are 4D image classification: `(N, C, H, W)`.
- Test loader **must** have `shuffle=False` and `drop_last=False`. Asserted in `evaluation/main.py`; violation crashes immediately.
- Don't include `main.py` or `score.py` in submission — they're overwritten by the grader.
- Failure-mode model: any exception in `run_submission` is caught by the grader, logged, and the dataset is marked failed (-10). So `try/except` inside your code is defensive padding, not strictly required.
- Per-dataset clock resets between datasets (each dataset gets its own `time_limit`).

## 3. 2025 leaderboard (top 3)

Datasets used in 2025: **GameOfLife, Windspeed, Cryptic**. Total possible: 30. Winning total was 12.192 — even winners were far from cap.

| Rank | Team | Affiliation | Total | GameOfLife (acc / adj) | Windspeed (acc / adj) | Cryptic (acc / adj) |
|------|------|-------------|------:|------:|------:|------:|
| 1 | EVAI | U. Cyprus + Thakaa | 12.192 | 99.99% / +9.998 | 17.375% / +0.449 | 76.145% / +1.745 |
| 2 | Jiawen Deng | Sichuan University | 10.885 | 99.66% / +9.942 | 15.278% / +0.206 | 73.235% / +0.737 |
| 3 | Aojun Lu, Yanan Sun | Sichuan University | 10.430 | 99.97% / +9.995 | 13.776% / +0.033 | 72.265% / +0.402 |

**Observations**:
- GameOfLife was solved by all top-3 (~10/10 each). Free points for any competent pipeline.
- Windspeed was effectively impossible (top accuracy 17.4%). Don't burn budget chasing it.
- Cryptic decided the entire ranking (1st = 76% vs 3rd = 72%, ~3% absolute).
- All top-3 avoided the −10 cliff on every dataset. Reliability is required.

## 4. EGNAS — the 2025 winner method

Authors: **Shahid Siddiqui, Christos Kyrkou, Theocharis Theocharides** — University of Cyprus + KIOS.

Papers (both in `docs/papers/`):
- `1st-2025_EGNAS_Siddiqui_SN-CompSci-journal.pdf` — SN Computer Science (2025) 6:282, doi:10.1007/s42979-025-03790-z. Canonical citation. Received Mar 2024, accepted Feb 2025.
- `1st-2025_EGNAS_Siddiqui_arxiv-preprint.pdf` — arXiv:2502.03553v1, Feb 2025. Same content, free.

Public code: https://github.com/siddikui/Efficient-Macro-Micro-NAS

### Search space (macro-micro)

- **Macro**: depth (number of layers) and width (max channel count).
- **Micro**: per-layer choice between `Conv` (regular) and `SepConv` (depthwise separable). Per-layer kernel size in {3, 5, 7}.
- ResNet-style: 2× channel doubling at the 1/3 and 2/3 depth points (downsampling).

### Search algorithm (sequential, **not** weight-sharing, **not** predictor-based, **not** DARTS)

1. **Depth search**: start with min-depth + max-width baseline. Train short. Grow depth 1 layer at a time until accuracy plateaus.
2. **Width search**: shrink channel count gradually while accuracy stays within tolerance.
3. **Operations + kernel search**: per-layer hill-climb over {Conv, SepConv} × {3, 5, 7}.

Key trick: **architecture-aware training schedule** — more epochs as architecture grows (cheaper search early). Also called "variable training schemes".

### Defaults (from public training script)

- SGD, lr=0.025, momentum=0.9, weight_decay=3e-4
- Cosine schedule
- Cutout=16
- Gradient clip=5
- Final retrain: 50 epochs
- Search inner training: starts at 10 epochs, increases as depth grows

### Reported results (from paper)

- 2× to 4× faster than the fastest prior global NAS.
- SOTA on EMNIST and KMNIST.
- Competitive on CIFAR-10/100 and FashionMNIST.
- Demonstrated transfer to face recognition (1:1 verification, top-n identification).

## 5. Sichuan teams (2nd, 3rd 2025)

Both from **Yanan Sun's NAS group**, Sichuan University. https://yn-sun.github.io/

- **2nd: Jiawen Deng** — has work accepted at PRICAI 2025 on a **performance predictor** for NAS (per Sun's group page).
- **3rd: Aojun Lu, Yanan Sun**.

### Group's published direction (2025–2026)

| Paper | Venue |
|-------|-------|
| DAP: Domain Adaptive Performance Predictor for Efficient NAS — Song et al. | IEEE Trans. Computers (2026) |
| Online Performance Predictor for Evolutionary NAS — Song et al. | IEEE TSMC (2025) |
| Listwise Ranking Predictor for Evolutionary NAS — Li et al. | Swarm and Evolutionary Computation (2025) |
| Accurate and Robust NAS via a Flexible Supernet — Feng et al. | IEEE TNNLS (2025) |

### Inferred submission approach

**Predictor-guided evolutionary NAS**: train a surrogate accuracy predictor on architecture encodings, search the space cheaply via predictor queries, only fully train the top-k candidates. Aligns with the group's broader research direction.

The Cryptic gap to EVAI (76% vs 73%) hints that predictor calibration on novel datasets is the bottleneck — the predictor learned priors don't transfer to "weird" data.

### Other Aojun Lu work

- "Revisiting Neural Networks for Continual Learning: An Architectural Perspective" — arXiv:2404.14829 (2024). Continual learning, not NAS Unseen, but shows the team's broader architectural-design instincts.

Group supervisory page: Aojun Lu credited with "NAS global competition first place + 2 papers" — likely refers to a prior 2024 competition iteration, not 2025.

## 6. Datasets

### Public 13 (Isabella unavailable)

| Dataset | Size | Era | Notes |
|---------|-----:|-----|-------|
| Sudoku | 0.02 GB | 2024 | Tiny. Sudoku grid → digit class |
| GameOfLife | 0.03 GB | 2025 | Tiny. Conway's GoL prediction |
| Gutenberg | 0.13 GB | CVPR | Three encoded words from public-domain texts. Near-noise floor (~50% ceiling) |
| Language | 0.16 GB | CVPR | Encoded multi-language strings, 24×24, 10 classes |
| Chesseract | 0.22 GB | CVPR-2023 | Chess-board state as 3D one-hot tensor |
| AddNIST | 0.66 GB | CVPR | Three MNIST images summed minus one, 20 classes |
| MultNIST | 0.66 GB | CVPR | Three MNIST images multiplied mod 10, 10 classes |
| Voxel | 2.24 GB | 2024 | 3D-flavoured |
| GeoClassing | 2.65 GB | CVPR | BigEarthNet satellite, country classification |
| CIFARTile | 3.44 GB | CVPR | Four tiled CIFAR-10 → number of unique sub-classes |
| Myofibre | 9.71 GB | 2024 | Largest. Binary medical imagery |
| Cryptic | unknown | 2025 | Starter-kit DOI `10.7488/ds/8054` resolves to wrong content (Tower Blocks UK photos) — broken. Ask organisers |
| Windspeed | unknown | 2025 | Starter-kit DOI `10.7488/ds/8053` similarly broken |

Total ~22 GB. Isabella (CVPR-2023, music spectrograms) held back due to license issues.

### Public benchmark tables (Towers-D readme + Geada CVPR 2024)

Off-the-shelf CNNs (averaged across 7 architectures: ResNet-18, AlexNet, VGG16, ConvNext, MNASNet, DenseNet, ResNeXt):

| Dataset | ResNet-18 | Best CNN | PC-DARTS (NAS) |
|---------|-----------|----------|----------------|
| AddNIST | 92.1% | 94.9% | 97.9% |
| Language | 97.0% | 97.0% | 90.1% |
| MultNIST | 91.6% | 94.0% | 98.1% |
| CIFARTile | 45.6% | 51.3% | 92.3% |
| Gutenberg | 50.0% | 50.0% | 49.1% |
| Isabella | 62.0% | 63.3% | 65.8% |
| GeoClassing | 80.3% | 94.2% | 96.0% |
| Chesseract | 57.8% | 59.6% | 60.8% |

**Best competition results** (the 2023 ceiling, per Towers-D readme):

| Dataset | Best |
|---------|------|
| AddNIST | 95.06% |
| Language | 89.71% |
| MultNIST | 95.45% |
| CIFARTile | 73.08% |
| Gutenberg | 50.85% |
| Isabella | 61.42% |
| GeoClassing | 96.08% |
| Chesseract | 62.98% |

**Key insight**: CIFARTile has the largest NAS-vs-CNN delta (45→92%), making it the canonical "search-rewarding" dataset. Gutenberg appears to have a hard ceiling around 50%.

### File layout per dataset (from figshare)

Each downloaded dataset has:
- `train_x.npy`, `train_y.npy`
- `valid_x.npy`, `valid_y.npy`
- `test_x.npy`, `test_y.npy` (held back during real evaluation)
- `metadata` (JSON with `num_classes`, `codename`, `input_shape`, `benchmark`)
- README and example image

This format matches what `evaluation/main.py` expects — no conversion needed.

## 7. Geada et al. CVPR 2024 — organisers' analysis paper

`docs/papers/Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf`

Authors: Geada, Towers, Forshaw, Atapour-Abarghouei, McGough.

Authoritative reference on the 8 CVPR-era datasets. Contains the benchmark tables above and analysis of what works on each. The 2024 paper is from the organisers' own perspective — read it for what they consider important.

## 8. Repos

| Repo | Purpose |
|------|---------|
| https://github.com/Towers-D/NAS-Comp-Starter-Kit | Official starter kit. Contains `evaluation/main.py`, `evaluation/score.py`, template + example submission, Makefile |
| https://github.com/Towers-D/NAS-Unseen-Datasets | Organisers' canonical dataset descriptions. CVPR-era datasets only |
| https://github.com/siddikui/Efficient-Macro-Micro-NAS | EGNAS public code (1st place 2025) |
| https://yn-sun.github.io/ | Yanan Sun's group page (2nd & 3rd place 2025) |

## 9. Eval-environment unknowns

- **GPU**: explicitly withheld by organisers. Quote: *"we have decided to withhold the GPU that we will use to evaluate final submission, though each submission will be tested on machines of equivalent specs."*
- **RAM/VRAM**: not specified. Code must "be adaptable to the amount of memory available."
- **PyTorch / CUDA versions**: starter kit's `requirements.txt` pins `torch==1.10.1+cu113` plus the deprecated `sklearn` package name — clearly stale (2021-era) and probably not the actual eval env.
- **Per-dataset clock**: default 0.5h, overridable via `metadata['time_limit']`.
- **Phase 2 budget**: ~1 hour per dataset (smoke test).
- **Phase 3 budget**: explicitly unspecified ("an unknown amount of runtime").

The intent is design intent: submissions must adapt at runtime, not target a fixed environment.

## 10. Internal repo facts

- Repo: `Lys-fy/HerriML-NAS-Unseen-Data-Challenge-2026` (private GitHub).
- Team members: Luis Melzer, Sebastian Seifert (GitHub: `Neterich04`), Marc Leander von Gottberg.
- `submission/` = the 4 contest files (currently template stubs, not yet implemented).
- `dev/mock_harness.py` = local mirror of `evaluation/main.py`. Same call sequence, same Clock, same assertions, plus `--score` flag for scoring against local `test_y.npy`.
- `scripts/download_datasets.sh` = idempotent figshare downloader for the 11 available datasets (Cryptic + Windspeed are not in the script — broken DOIs).
- Datasets stored at `~/local/nas-comp-datasets/`, symlinked into the repo as `datasets/` (gitignored). Out-of-tree to avoid Syncthing replication.
- `docs/papers/` has the EGNAS papers (preprint + journal) and the Geada CVPR 2024 paper.
- Notes vault is `docs/notes/` (Obsidian-style). 47 stub notes already exist across 8 numbered sections.

## 11. Mock harness behaviour

`dev/mock_harness.py`:
- Adds `submission/` to `sys.path`, then `from nas import NAS`, etc. — same import shape as the real grader.
- Uses identical `Clock` class.
- Asserts `not isinstance(test_loader.sampler, RandomSampler)` and `not test_loader.drop_last`.
- Wraps `run_submission` in `try/except`, writes a fail stats pkl on error — same behaviour as grader.
- Saves predictions to `dev/predictions/<codename>.npy`.
- Optional `--score` flag computes `adj_score` per dataset using local `test_y.npy` and the formula above.
- CLI: `python dev/mock_harness.py [DATASET ...] [--time-limit HOURS] [--score]`.

## 12. Stub structure already created

47 notes in `docs/notes/`:

```
01 - Competition/   Overview, Phases and Timeline, Scoring, Rules
02 - Framework/     Submission Contract, Pipeline Flow, Clock and Time Budget,
                    Failure Modes, Mock Harness
03 - Datasets/      Overview + AddNIST, Language, MultNIST, CIFARTile, Gutenberg,
                    Isabella, GeoClassing, Chesseract, Sudoku, Voxel, Myofibre,
                    GameOfLife, Cryptic, Windspeed
04 - Approaches/    Overview, EGNAS, Predictor-Guided Evolutionary, DARTS Family,
                    Zero-Cost Proxies, Multi-Fidelity, Weight-Sharing One-Shot
05 - Past Winners/  2025 - EVAI, 2025 - Sichuan teams, 2023 Finalists
06 - Strategy/      Reliability vs Peak, Budget Allocation, Score-Efficient Targeting,
                    Search vs Train Split, Threats to Validity
07 - Workflow/      Dev Loop, Submission Packaging, Testing Strategy,
                    Phase 2 Submission Strategy
08 - References/    Papers, Repositories, Official Pages, Researchers and Groups
```

Each stub has the standard frontmatter blockquote (`> Status / Topic / Related`) with wikilinks pre-filled where obvious. Replace the body, keep the pattern.
