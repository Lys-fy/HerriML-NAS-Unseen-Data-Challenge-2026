# Windspeed

> **Status:** drafting | **Topic:** 2025 competition dataset. Source DOI broken — content unknown. Was effectively impossible (top score ~17%) in the 2025 leaderboard.
> **Related:** [[Datasets Overview]], [[Cryptic]], [[2025 - EVAI]], [[Score-Efficient Targeting]]

One of the three hidden datasets used in the **2025 NAS Unseen-Data Challenge at AutoML**. As of April 2026, **not publicly available** through any verified source.

## Source — broken DOI

The starter-kit references DOI `10.7488/ds/8053` for Windspeed, but the handle resolves to wrong content. `scripts/download_datasets.sh` deliberately omits Windspeed for this reason. WebSearch (April 2026) on "Windspeed NAS Unseen Data Competition" surfaces unrelated wind-power forecasting datasets and Newcastle figshare entries for *other* NAS datasets ([[Voxel]], [[Sudoku]], [[AddNIST]], [[Gutenberg]]) but **no Windspeed**. The dataset is not yet listed under the Towers-D figshare profile we can verify.

Action: contact `nas-competition-contact@newcastle.ac.uk` for the correct link. See [[Official Pages]].

| Field | Value |
|---|---|
| Codename | unknown |
| Era | 2025 |
| Shape | unknown |
| Train / val / test | unknown |
| Classes | unknown (the name suggests this could be **regression-flavoured**, framed as multi-class) |
| Benchmark | unknown — but inferred to be in the 10–15% range (see leaderboard math below) |
| License | unknown |
| DOI | broken (`10.7488/ds/8053`) — placeholder until corrected |

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

## Working hypothesis on content

The name suggests numerical-regression-cosplaying-as-classification: predicting wind speed (or wind-power output) from satellite / weather / sensor input, bucketed into discrete classes. Newcastle has academics in environmental modelling, and BigEarthNet-style data is in their pool ([[GeoClassing]] uses BigEarthNet). Plausible-but-unverified.

If true, this would be a **regression-shaped task**:

- Classes ordered by magnitude → label-smoothing / ordinal regression head might help.
- Standard cross-entropy may waste capacity on the unordered-classes assumption.
- A model that outputs near-uniformly across all classes scores at random. Most teams might be doing exactly this.

But this is speculation. Until we obtain the dataset we can't confirm.

## Open actions

- [ ] Email organisers for the correct Windspeed DOI.
- [ ] Watch `github.com/Towers-D/NAS-Unseen-Datasets` for additions.
- [ ] On obtaining Windspeed: rerun the metadata-signature analysis and update this note.

## Implications for our pipeline

- **Cannot tune for it locally**.
- Must not crash — see [[Failure Modes]]. Test pipeline with arbitrary low-benchmark, high-class-count inputs before Phase 2.
- Don't allocate disproportionate compute to it. Even if the metadata at runtime looks "interesting", the 2025 evidence is that returns to effort are minimal.
- **Strategy decision**: spending 90% of the dataset's clock on Windspeed is almost certainly a bad bet. Better to deliberately under-invest and reallocate budget to high-leverage datasets.
