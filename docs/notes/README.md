# Map of Content

> **Status:** project complete | **Topic:** Central index for the HerriML NAS competition notebase.
> **Related:** [[Competition Overview]] (context) · [[Pipeline v13 - Final Spec]] (the result)

Open `docs/notes/` as the Obsidian vault root. Wikilinks resolve across subfolders.

## Final state — start here

The project is finished. The shipped pipeline is **v13** (`submissions/v2/`), which passed the
Phase-2 grader and is our Phase-3 entry. To understand the outcome, read in this order:

1. **[[Pipeline v13 - Final Spec]]** — the complete shipped pipeline, stage by stage, with the
   pipeline diagram and every constant. *(The single best entry point to what we built.)*
2. **[[v12.1 - Validation Results]]** — the evidence: v13 validation, the 16-dataset tier, the
   finals dress rehearsal (ahead of the 2025 winner), and the cleaned-code equivalence proof.
3. **[[Pipeline v12 - Screening Report]]** — the ablation history: what was tried and rejected
   (BDH, independent-init, wider models, weight soups, difficulty routing) with evidence.

The sections below index the full notebase (competition rules, datasets, the research history).

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

### 09 - Implementation

Split into **plan** (what we intend to build) and **implementation** (what we built and measured, with embedded graphs). Folder layout:
`09 - Implementation/plan/` and `09 - Implementation/implementation/`.

**Plan** — staged build: shipping path (T0–T4) + experimental branches (B1–B4):

- [[Probe-Driven Self-Configuration]] — **The central thesis.** Pipeline probes the unseen dataset at runtime and self-configures; prior methods become the menu it selects from. Read this first.
- [[Implementation Plan Overview]] — Roadmap, axes (shipping vs experimental), recommended order, reframed around the thesis.
- [[Tier 0 - Hardcoded CNN]] — Fixed CNN, no search. Validates harness.
- [[Tier 1 - Augmentation + Op Search]] — Augmentation + regularization + a first random op-type search layer.
- [[Tier 2 - EGNAS-Light]] — Sequential macro-micro hill-climb (the [[2025 - EVAI|2025-winning method]] scaled). Shipping candidate.
- [[Tier 3 - EGNAS + Reliability]] — T2 + per-regime budget classifier + OOM-safe batch + in-search fallback + ensemble + TTA.
- [[Tier 4 - Phase-2 Polish]] — Data-driven refinement once Phase-2 leaderboard data arrives.
- [[Branch B1 - Zero-Cost Proxies]] — Synflow + NASWOT + params 3-way vote. Pre-filter for larger search spaces.
- [[Branch B2 - Predictor-Guided EA]] — Sun group's approach. Comparator; expected to lose on novel data.
- [[Branch B3 - DARTS Comparator]] — PC-DARTS / DrNAS. Organisers' published reference benchmark.
- [[Branch B4 - Weight-Sharing Supernet]] — SPOS-lite offline pretrain. High risk, possibly only as warm-start.

**Implementation** — measured results + findings + graphs:

- [[Implementation Results Overview]] — Scoreboard, headline findings, graph index.
- [[Fixed CNN]] — Hardcoded-CNN baseline: total adj −0.355. Established overfitting (not search) is the binding constraint.
- [[Aug + Op Search]] — Augmentation + reg + op-search: total adj +24.644. Includes the single-feature ablation quantifying search/EMA/aug.

## Conventions

Each note opens with a frontmatter blockquote:

```markdown
> **Status:** [drafting / current / stale] | **Topic:** [one-line]
> **Related:** [[Note A]], [[Note B]]
```

Wikilinks (`[[Note Name]]`) work across folders — Obsidian resolves by basename.
