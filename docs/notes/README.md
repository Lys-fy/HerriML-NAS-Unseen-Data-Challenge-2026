# Map of Content

> **Status:** current | **Topic:** Central index for the HerriML NAS competition notebase.
> **Related:** [[Competition Overview]] (start here)

Open `docs/notes/` as the Obsidian vault root. Wikilinks resolve across subfolders.

## Sections

### 01 - Competition

- [[Competition Overview]] — What the NAS Unseen Data Challenge is and why it exists.
- [[Phases and Timeline]] — Phase 1, 2, 3 — what happens in each, key dates.
- [[Scoring]] — Per-dataset scoring formula, the +10 / -10 caps, slope by benchmark.
- [[Rules]] — Submission limits, prohibited actions, contact, mailing list.

### 02 - Framework

- [[Submission Contract]] — Fixed file names, class signatures, asserted constraints.
- [[Pipeline Flow]] — DataProcessor → NAS → Trainer.train → Trainer.predict.
- [[Clock and Time Budget]] — How clock.check() works, per-dataset budgets, grace window.
- [[Failure Modes]] — What causes a -10 (timeout, OOM, crash, wrong shape).
- [[Mock Harness]] — What dev/mock_harness.py does and how to use it.

### 03 - Datasets

- [[Datasets Overview]] — All 13/14 known datasets in one table; era split (CVPR vs AutoML).
- [[AddNIST]] — Three MNIST images summed minus one, 20 classes.
- [[Language]] — Encoded multi-language word strings as 24x24 images, 10 classes.
- [[MultNIST]] — Three MNIST images multiplied mod 10, 10 classes.
- [[CIFARTile]] — Four tiled CIFAR-10 images; label = number of unique classes.
- [[Gutenberg]] — Three encoded words from public-domain texts.
- [[Isabella]] — Music spectrograms, classification by era. Not publicly downloadable.
- [[GeoClassing]] — BigEarthNet satellite imagery, country classification.
- [[Chesseract]] — 3D-encoded chess board state.
- [[Sudoku]] — Sudoku grids; small dataset.
- [[Voxel]] — 3D-flavoured dataset.
- [[Myofibre]] — Largest dataset (~10 GB), binary medical-imagery classification.
- [[GameOfLife]] — Conway's Game of Life prediction; tiny dataset.
- [[Cryptic]] — 2025 dataset; starter-kit DOI is broken.
- [[Windspeed]] — 2025 dataset; starter-kit DOI is broken.

### 04 - Approaches

- [[Approaches Overview]] — Decision tree: which method to use under which constraint.
- [[EGNAS]] — Efficient Global NAS — sequential macro-micro hill-climbing search.
- [[Predictor-Guided Evolutionary]] — Surrogate accuracy predictor + evolutionary search; Sun's group approach.
- [[DARTS Family]] — DARTS, PC-DARTS, DrNAS — gradient-based weight-sharing search.
- [[Zero-Cost Proxies]] — Synflow, Jacov, Zen-NAS — no-train architecture ranking.
- [[Multi-Fidelity]] — Hyperband, ASHA, BOHB — budget-allocation across many candidates.
- [[Weight-Sharing One-Shot]] — Supernet methods: OFA, ProxylessNAS, SPOS.

### 05 - Past Winners

- [[2025 - EVAI]] — 1st place 2025: Siddiqui et al. (U. Cyprus + Thakaa). Method = EGNAS.
- [[2025 - Sichuan teams]] — 2nd (Jiawen Deng) and 3rd (Aojun Lu, Yanan Sun). Sichuan University.
- [[2023 Finalists]] — Top 5 from 2023 (Isabella, GeoClassing, Chesseract).

### 06 - Strategy

- [[Reliability vs Peak]] — Why avoiding -10 dominates pursuing +10. Expected score math.
- [[Budget Allocation]] — How to split per-dataset time between NAS and Trainer.
- [[Score-Efficient Targeting]] — Where points actually live. Slope formula by benchmark.
- [[Search vs Train Split]] — Per-dataset adjustment based on benchmark and headroom.
- [[Threats to Validity]] — Dev/eval drift, overfit-to-public-13, randomness, transfer assumptions.

### 07 - Workflow

- [[Dev Loop]] — Edit submission/, run mock_harness, score, iterate.
- [[Submission Packaging]] — How to zip submission/, what to exclude, size limits.
- [[Testing Strategy]] — Integration tests against public datasets, stress tests.
- [[Phase 2 Submission Strategy]] — How to spend the 7 attempts.

### 08 - References

- [[Papers]] — Annotated bibliography of papers in docs/papers/ and beyond.
- [[Repositories]] — Starter kit, EGNAS code, datasets repo, mirrors.
- [[Official Pages]] — Rules, info, contact, archive, mailing list.
- [[Researchers and Groups]] — Yanan Sun's group, U. Cyprus team, organisers (Towers, Geada, McGough).

## Conventions

Each note opens with a frontmatter blockquote:

```markdown
> **Status:** [drafting / current / stale] | **Topic:** [one-line]
> **Related:** [[Note A]], [[Note B]]
```

Wikilinks (`[[Note Name]]`) work across folders — Obsidian resolves by basename.
