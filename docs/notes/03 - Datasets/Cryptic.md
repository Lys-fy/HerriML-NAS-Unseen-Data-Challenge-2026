# Cryptic

> **Status:** ✅ source RESOLVED 2026-05-19 — DOI was fine, dataset just published late (2025-12-05). The **decisive dataset** in the 2025 leaderboard.
> **Related:** [[Datasets Overview]], [[Windspeed]], [[2025 - EVAI]], [[2025 - Sichuan teams]]

![[Cryptic.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


One of the three hidden datasets used in the **2025 NAS Unseen-Data Challenge**. **Now public:** DOI `10.7488/ds/8054` → **https://datashare.ed.ac.uk/handle/10283/9132** (Edinburgh DataShare, CC-BY-4.0, pub. 2025-12-05; authors Ericsson/Crowley/McGough/Towers/Moeller/Atapour-Abarghouei). **Content (confirmed):** built from George Ho's *Cryptic Crossword Clues* (NYT, Guardian, Times, Hindu, …); clues of exactly 6 words, each word → BERT embedding (dim 768) ⇒ **input ≈ 6×768**. The broken-DOI sections below are historical; the resolved facts above supersede them.

## Source — RESOLVED 2026-05-19 (was never broken)

The DOI `10.7488/ds/8054` resolves **correctly** → **https://datashare.ed.ac.uk/handle/10283/9132**. The earlier "resolves to unrelated content" finding was a *timing artefact*: the dataset was only **published 2025-12-05**, after the April-2026 checks (the handle was empty/unindexed then, and search engines misrouted "Cryptic" to the tuberculosis-genomics project / Tower Blocks photos). No organiser email needed. Downloaded + verified locally + on Spark via the now-DataShare-aware `scripts/download_datasets.sh`.

| Field | Value (verified from the data, 2026-05-19) |
|---|---|
| Codename | `Cryptic` (shipped); assigned per-run by the harness at eval |
| Era | 2025 |
| Shape | **`[1, 6, 768]`** — a 6-word clue, each word a 768-dim BERT embedding |
| Train / val / test | **59 718 / 10 441 / 17 833** (= 87 992 total), `x` float32, `y` int64 |
| Classes | **2 — binary classification** (labels {0, 1}) |
| Benchmark | **placeholder** — shipped `metadata` had `"benchmark": ?` (invalid JSON); we normalise to `0.0`, same convention as [[Sudoku]]. Real ResNet-18 benchmark still unpublished. |
| License | CC-BY-4.0 (Edinburgh DataShare, authors Ericsson/Crowley/McGough/Towers/Moeller/Atapour-Abarghouei) |
| DOI | `10.7488/ds/8054` → datashare.ed.ac.uk/handle/10283/9132 ✓ |

## What the 2025 leaderboard tells us

The 2025 competition used three datasets — [[GameOfLife]], [[Windspeed]], **Cryptic**. From [[PRIOR_FINDINGS]] §3:

| Team | Cryptic accuracy | adj_score |
|---|---:|---:|
| EVAI (1st) | 76.145% | +1.745 |
| Jiawen Deng (2nd) | 73.235% | +0.737 |
| Aojun Lu / Y. Sun (3rd) | 72.265% | +0.402 |

A few observations:

- **Cryptic decided the entire 2025 leaderboard.** GameOfLife was max-ed by all three teams (+9.99); Windspeed was near-impossible (+0.03 to +0.45). The 4-percentage-point gap between 1st and 3rd on Cryptic translates to a 1.3-point swing — large enough to flip the ranking.
- The accuracy range (72–76%) sits in a regime where NAS / training tuning has real differential value. Not at the noise floor (so models *can* learn it), not saturated (so optimisation matters).
- The benchmark must be in the 70%+ range to give those modest adj_scores — implying Cryptic is a **medium-benchmark dataset** where the [[Scoring]] slope is ~0.33–0.50 per 1%.

## Content — CONFIRMED

Built from George Ho's *A Dataset of Cryptic Crossword Clues* (cryptics.georgeho.org; NYT/Guardian/Times/Hindu). Each sample = a clue of **exactly 6 words**, each word encoded by **BERT to a 768-dim vector** → input `[1, 6, 768]`. The target is **binary** (2 classes) — a yes/no property of the clue (exact semantics per the dataset README; not a 9-class etc. task). So it is an **encoded-non-image, sequence-flavoured** dataset (Type-2, like [[Language]]/[[Gutenberg]] in spirit) — conv's spatial-locality bias is questionable on the 6×768 layout; the 6 = word axis is sequence-like.

## Implications for our pipeline

- **Now available locally** (CC-BY public) — usable as a dev/stress-test fixture; but it's a *public* dataset, the actual Phase-2/3 eval sets remain hidden, so the pipeline must still be generic, not Cryptic-tuned.
- Likely included in 2026 Phase 2/3 (organisers tend to reuse hidden datasets across years where licenses allow).
- Reading the metadata at runtime is critical — our pipeline must adapt to whatever shape arrives.
- If we're competitive on the rest of the 2025 set, **Cryptic is where we win or lose**. Worth pre-budgeting a "long-tail dataset" branch in the [[Submission Contract]] strategy that does heavier search when the metadata signature is unfamiliar.

## Open actions

- [x] ~~Email organisers for the correct Cryptic DOI~~ — **not needed**; resolved (published late, now on Edinburgh DataShare CC-BY-4.0; downloaded + verified).
- [ ] Real ResNet-18 benchmark still unpublished (shipped placeholder) — if pursuing honest tier-vs-tier scoring on Cryptic, add an internal override in `dev/internal_benchmarks.py` (like Sudoku/Sokoto), or treat its adj as non-comparable.
- [ ] (Optional) Use it as a pipeline stress-test fixture: a `[1,6,768]` binary, sequence-flavoured input — confirms our conv stack doesn't crash on this shape.
