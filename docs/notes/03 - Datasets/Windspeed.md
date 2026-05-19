# Windspeed

> **Status:** ✅ source RESOLVED 2026-05-19 (DOI was not broken — dataset just hadn't been published yet; live since 2025-12-05). Specs confirmed; our earlier working hypothesis was correct.
> **Related:** [[Datasets Overview]], [[Cryptic]], [[2025 - EVAI]], [[Score-Efficient Targeting]]

![[Windspeed.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


One of the three hidden datasets used in the **2025 NAS Unseen-Data Challenge**. **Now public** (Edinburgh DataShare, CC-BY-4.0).

## Source — RESOLVED

DOI `10.7488/ds/8053` now resolves correctly → **https://datashare.ed.ac.uk/handle/10283/9131**. It was never "wrong content" — the dataset was simply **published 2025-12-05**, after our April-2026 checks (so the handle was empty/unindexed then; no organiser email needed). `windspeed.tar.gz` (1.42 GB), CC-BY-4.0. Authors: Ericsson, Crowley, McGough, Towers, Moeller, Atapour-Abarghouei (Univ. Edinburgh / Newcastle).

| Field | Value |
|---|---|
| Codename | unknown (assigned per-run by the harness) |
| Era | 2025 |
| Shape | **`[12, 32, 32]`** — 6 hours × 2 vars (2 m temperature, surface pressure) × 32×32 ERA5 grid (west Kazakhstan, lat/long 46–54) |
| Train / val / test | **temporal split — train 2020–2022, val 2023, test 2024** (built-in year-over-year distribution shift) |
| Classes | **20** (verified from shipped `metadata`) — discretised Beaufort bins; an **ordered/ordinal** target |
| Benchmark | **13.492** (verified — shipped `metadata`; matches the ~13–14 % leaderboard inference exactly) |
| Verified data | downloaded 2026-05-19: 34 822 samples (train 20 979 / val 6 787 / test 7 056), `x` float32 `[·,12,32,32]`, `y` int64 0–19. Standard 6-npy layout. |
| License | CC-BY-4.0 |
| DOI | `10.7488/ds/8053` → datashare.ed.ac.uk/handle/10283/9131 ✓ |

## What the 2025 leaderboard tells us

| Team | Windspeed accuracy | adj_score |
|---|---:|---:|
| EVAI (1st) | 17.375% | +0.449 |
| Jiawen Deng (2nd) | 15.278% | +0.206 |
| Aojun Lu / Y. Sun (3rd) | 13.776% | +0.033 |

The third-place adj_score of +0.033 implies the team's 13.776% was **just barely above the benchmark**. Working backwards through `adj_score = (raw − benchmark) × 10 / (100 − benchmark)`:

- If `adj_score ≈ 0.033` and `raw = 13.776`, then `benchmark ≈ 13.5%`.

So the benchmark is **roughly 13–14%** — the lowest of any known dataset. If the "natural" random-baseline rate at the dataset's class count is similar (e.g. 1/8 ≈ 12.5%, or 1/10 = 10%), the dataset is essentially **at the noise floor**. None of the top 2025 teams cracked even 20%.

## Implications

This dataset is the textbook example of a **−10 / +10 cliff trap**:

- Floor risk (failure → −10) is enormous.
- Upside is near-zero. Even doubling the top-3 to 35% accuracy would only buy ~+2.5 pts.
- Per the 2025 outcome, **don't burn budget chasing it**. The right strategy is "make sure it doesn't crash" and accept whatever just-above-benchmark score the model produces.

This is captured in the strategy section under [[Reliability vs Peak]] — Windspeed is the canonical "ignore for peak, just don't fail" dataset.

## Content — CONFIRMED (the working hypothesis was right)

Confirmed exactly as hypothesised: **numerical regression cosplaying as classification.** Predict the average wind speed in the central 50 % of the region from 6 h of whole-region temperature + pressure (ERA5), discretised into **20 ordered Beaufort bins** (verified from the data, not estimated).

Direct pipeline implications (now actionable, not speculation):
- **Classes are ordered by magnitude** → ordinal structure. Plain cross-entropy ignores ordering; label-smoothing (already in our stack) helps a little; an ordinal/soft-label head or distance-aware loss is the principled lever — adjacent-bin confusions are "almost right".
- **Temporal distribution shift** (test = 2024, train = 2020–22) — the hardest part. Validation (2023) is itself shifted from train, so val-acc tracking is *more* honest here than usual; a model that overfits 2020–22 climate patterns will degrade on 2024.
- **12-channel non-visual input** — like [[Voxel]]/[[Chesseract]], conv's spatial-locality bias is only partly right (it *is* a 32×32 spatial grid, so some locality holds; the 12 = 6h×2vars "channel" axis is sequence-like, not RGB).
- A model outputting near-uniformly over the ~21 bins scores ≈ random — consistent with the 2025 leaderboard (nobody cleared 20 %).

**It is now public (CC-BY-4.0)** → unlike the truly-hidden eval sets, we *can* legitimately download it to **stress-test the pipeline** on a real 12-channel, ~21-ordered-class, temporally-shifted input before Phase 2 (this is exactly the "test pipeline with arbitrary low-benchmark high-class-count inputs" action below — now doable for real).

## Open actions

- [x] ~~Email organisers for the correct Windspeed DOI~~ — **not needed**; DOI resolves, dataset was just published late (2025-12-05). Same for [[Cryptic]] (`10.7488/ds/8054` → datashare.ed.ac.uk/handle/10283/9132).
- [ ] (Optional, now possible) Download `windspeed.tar.gz` and use it as a real pipeline stress-test fixture: 12-ch input, ~21 ordered classes, temporal shift. Confirms our stack doesn't crash on this shape and shows where it lands vs the ~13–14 % benchmark.
- [ ] Consider an ordinal-aware loss/head as an optional probe-gated component (only if a runtime signal says "ordered classes").

## Implications for our pipeline

- **Cannot tune for it locally**.
- Must not crash — see [[Failure Modes]]. Test pipeline with arbitrary low-benchmark, high-class-count inputs before Phase 2.
- Don't allocate disproportionate compute to it. Even if the metadata at runtime looks "interesting", the 2025 evidence is that returns to effort are minimal.
- **Strategy decision**: spending 90% of the dataset's clock on Windspeed is almost certainly a bad bet. Better to deliberately under-invest and reallocate budget to high-leverage datasets.
