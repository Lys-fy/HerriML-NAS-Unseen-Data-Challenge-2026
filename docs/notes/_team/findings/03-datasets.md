# Section 03 — Datasets — dossier

Owner: `datasets-researcher`. Filled all 15 notes; consolidated facts, flagged unknowns, called out cross-section dependencies.

## Established beyond PRIOR_FINDINGS.md

- The 14-dataset table in [[Overview]] now includes **codename**, **shape**, **classes**, **benchmark**, **best NAS** in one place. Codenames not previously consolidated: Adaline (AddNIST), Mateo (MultNIST), LaMelo (Language), Sokoto (Sudoku), conway (GameOfLife), Volga (Voxel), Sadie (GeoClassing), Chester (Chesseract), Gutenberg (same name).
- Benchmarks read from local `metadata` JSON for all 11 downloaded datasets are sometimes lower than the Geada Tab. 1 numbers (e.g. AddNIST: 89.85 metadata vs 92.08 Geada). Trust the local metadata at submission time — it's what the grader will use in the score formula.
- [[Voxel]] has shape `[N, 20, 20, 20]` with **no channel dimension** in metadata. Unique among the 14 datasets. Pipelines must not assume `input_shape[1]` is a small channel count.
- [[Sudoku]] ships a benchmark of `0.0` in its metadata — almost certainly a placeholder. Score formula behaviour with benchmark=0 is `adj_score = raw_acc / 10` — verify with a Phase-2 test submission before relying on this.
- [[Chesseract]]'s 12-channel input is the canonical pipeline trap; combined with 8×8 spatial extent it breaks default ResNet-style downsampling. Documented in note as a [[Failure Modes]] hook.
- [[GameOfLife]] solvable analytically via a single 3×3 conv; the 2025 top-3 confirms ≥99.6% is achievable for any working pipeline.
- [[Windspeed]] benchmark inferred from 2025 leaderboard math at **~13–14%** — the lowest of any dataset. Score-relevance analysis: even 100% accuracy nets at most +10, and the teams that won were sitting at +0.03 to +0.45.
- [[Cryptic]] inferred to have benchmark in the **70%+ range** based on adj_score ≈ +0.40 to +1.75 at 72–76% accuracy. Was the 2025 leaderboard-deciding dataset.
- [[Myofibre]] is **2024 era**, not "newer" — the README explicitly says "one of the three hidden datasets used by the 2024 NAS Unseen-Data Challenge". Same for [[Sudoku]] and [[Voxel]]. So the "2024-era three" is Sudoku + Voxel + Myofibre.

## Dataset-specific pathologies worth flagging

- [[CIFARTile]]: 2×2 tile boundary at H/2,W/2=32 — cropping/padding through the boundary scrambles the label.
- [[Language]] / [[Gutenberg]]: standard image augmentations (flip, crop, colour) **break the encoding**. Augmentation must be label-aware.
- [[Gutenberg]]: test split is 6,000 (not 10,000); easy to miss.
- [[Voxel]]: missing channel-axis in metadata interacts badly with NAS pipelines that hardcode 4-D input.
- [[Isabella]]: 1×64×128 spectrogram shape is unique; 4:8 aspect ratio fails square-only assumptions.
- [[Chesseract]]: 12-channel + 8×8 — ResNet-18-style stride-2 stages collapse spatial dim to <1.

## Conflicts with PRIOR_FINDINGS.md

- PRIOR_FINDINGS.md table reports Language ResNet-18 as **97.0%**. Geada Tab. 1 says **87.0%**. Local metadata says **85.20%**. The 97% is almost certainly a typo (off-by-10 from the actual 87%). Used 87% / 85.2% as appropriate in the note.
- PRIOR_FINDINGS lists Cryptic / Windspeed sizes as "unknown". Note now adds inferred benchmarks from leaderboard math (Windspeed ~13–14%, Cryptic ~70%+).
- Geada paper §4.7 says GeoClassing is 3×60×60; the local `metadata` and README both say 3×64×64. Trust metadata (current).
- Geada paper says GeoClassing splits 43,821 / 8,758 / 8,751; local metadata + README say 50,000 / 10,000 / 10,000. The figshare version was re-released with rounder splits — note flags both.

## Open questions / out-of-scope items

- [[Cryptic]] and [[Windspeed]] sources still broken — recommend the team email `nas-competition-contact@newcastle.ac.uk` for correct links.
- [[Isabella]] generation script status — Geada §7 says "coming soon", not yet visible in the github.com/Towers-D/NAS-Unseen-Datasets repo as of April 2026.
- 2026 Phase 2/3 dataset rotation is not announced. Speculation: 3 picks from {2024 era, 2025 era, fresh-2026}.
- [[Myofibre]] benchmark is unconfirmed (we haven't downloaded the 9.7 GB dataset yet — `scripts/download_datasets.sh` includes it but it would be the largest single download).

## Cross-section wikilinks used (for consolidator to verify)

- [[Scoring]], [[Phases and Timeline]] — section 01 (`competition-framework`).
- [[Failure Modes]], [[DataProcessor]], [[NAS]], [[Clock and Time Budget]] — section 02 (`competition-framework`).
- [[DARTS Family]], [[Predictor-Guided Evolutionary]] — section 04 (`approaches-researcher`).
- [[2025 - EVAI]], [[2025 - Sichuan teams]], [[Official Pages]] — sections 05/08 (`winners-references`).
- [[Reliability vs Peak]], [[Score-Efficient Targeting]], [[Search vs Train Split]], [[Budget Allocation]], [[Threats to Validity]] — section 06 (`strategy-workflow`).

Note: I link to `[[DataProcessor]]` and `[[NAS]]` but those are class names within `[[Submission Contract]]`, not separate notes. Consolidator: convert these to `[[Submission Contract]]` if needed.

## Coordination notes

- No SendMessage threads opened — facts already in PRIOR_FINDINGS.md were sufficient for the table-style notes. If `winners-references` adds detail to [[2025 - EVAI]] regarding the Cryptic gap, consider revising [[Cryptic]] downstream.
- The "image-y vs encoded-non-image" grouping in [[Overview]] may belong duplicated in [[Approaches Overview]] — left to `approaches-researcher` to decide.
- For the strategy team: my notes assume the [[Scoring]] formula slope analysis stays canonical. If [[Score-Efficient Targeting]] derives a different per-dataset prioritisation, my "implications for our pipeline" sections in each note may need a re-pass.
