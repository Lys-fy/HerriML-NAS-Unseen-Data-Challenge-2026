# GameOfLife

> **Status:** drafting | **Topic:** Conway's Game of Life next-step prediction. Predict the count of alive cells in the next generation. 25 classes.
> **Related:** [[Datasets Overview]], [[Sudoku]], [[2025 - EVAI]]

![[GameOfLife.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


A 10×10 binary grid showing a randomly-initialised Conway's Game of Life starting state, with random initial alive-cell density. The label is the **number of alive cells in the next generation** (i.e. after one Conway step). Limited to 25 classes (0–24) because higher counts become exponentially rare.

| Field | Value |
|---|---|
| Codename | conway |
| Era | 2025 (one of three hidden datasets in the 2025 competition; **publicly downloadable** via `scripts/download_datasets.sh`) |
| Shape | 1×10×10 (channels-first) |
| Train / val / test | 50,000 / 10,000 / 10,000 |
| Classes | 25 (labels 0–24 = number of alive cells next generation) |
| Class balance | 2,800 train / 2,000 val / 400+400 test per class — evenly distributed |
| Benchmark (`metadata`) | 47.530% (ResNet-18) |
| 2025 competition top score | 99.99% (EVAI), 99.97% (Sichuan team) — see [[2025 - EVAI]] |
| License | inferred CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/30000835 (figshare ID 30000835, used by `scripts/download_datasets.sh`) |

## Why every 2025 top-3 team near-perfected this

Conway's rule for cell `(i,j)` depends on **only the 3×3 neighbourhood centred on `(i,j)`**. So a single 3×3 convolution with the right weights can compute the next-generation state exactly; summing the result yields the count. Geada notes the dataset was deliberately constructed to test that competitors could handle "simple tasks along with the more complicated ones" without overengineering.

The 2025 leaderboard confirms the design intent: every top-3 team scored ≥99.66%. **This is essentially free points for any competent pipeline**.

## Type and difficulty

**Type-1** at machine difficulty (trivially solvable by a hand-coded 3×3 conv); slightly tedious for a human but doable. The dataset is essentially a sanity check — it distinguishes "submission that works at all" from "submission that crashes on simple data".

## Pathologies — surprisingly few

- 25 classes is high relative to typical small-dataset classification, but the class structure is well-balanced.
- Score slope at benchmark 47.53% is ~0.19 / 1% — the lowest of any dataset with a non-trivial benchmark. So even reaching 100% gives only +10. Nothing on the upside, but **failure here is unconscionable** (−10 cliff).
- The 1×10×10 shape is small. Pipelines that aggressively downsample early will collapse to <1×1 and fail.

## Implications for our pipeline

- **Reliability target #1** (alongside [[Sudoku]] in the 2024-era pool). Every submission must score ≥+9 here, ideally +10. Failure here = guaranteed not to win.
- Train budget can be tiny — the dataset is ~30 MB. A very small CNN (2–3 conv layers) reaches >99% in minutes. Don't burn search budget here.
- Consider hand-coding a "Conway baseline": one 3×3 conv with weights `[[1,1,1],[1,10,1],[1,1,1]]` followed by a lookup. Useful as a sanity-check during dev.
- This is the dataset where the [[Reliability vs Peak]] discussion bites: a clever pipeline that scores +9.5 here and innovates elsewhere wins; a clever pipeline that crashes here loses by 10+ points.
