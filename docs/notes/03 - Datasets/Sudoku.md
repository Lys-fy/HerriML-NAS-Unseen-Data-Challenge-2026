# Sudoku

> **Status:** drafting | **Topic:** 9×9 partially-masked sudoku grids; predict the value of one target cell (label = value − 1).
> **Related:** [[Datasets Overview]], [[GameOfLife]]

![[Sudoku.png]]
> *Example samples + the classification target — raw `.npy` arrays rendered for viewing. Regenerate: `python3 dev/plot_dataset_examples.py`.*


Algorithmically generated sudoku grids stored as 9×9 single-channel "images". Real sudoku digits 1–9 are stored as 0.1–0.9; missing cells are 0; **the target cell is encoded as 1**. The task is to classify what digit (1–9) the target cell should hold. Different masking levels (number of missing cells) per grid.

| Field | Value |
|---|---|
| Codename | Sokoto |
| Era | 2024 (one of three hidden datasets in the 2024 competition) |
| Shape | 1×9×9 (channels-first) |
| Train / val / test | 50,000 / 10,000 / 10,000 |
| Classes | 9 (labels 0–8, where label `n` means cell value `n+1`) |
| Benchmark (`metadata`) | **0.0%** — placeholder, almost certainly not a real ResNet-18 result |
| License | inferred CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/Sudoku_Dataset/26976121 |

## Notable construction details

- **70,000 grids total** with class-balanced sampling: 7,777 per class evenly distributed across the three sets, with a small top-up to round numbers up to 7,778 max per class. Reasonably balanced.
- The label is `cell_value − 1` (zero-indexing). Off-by-one bugs are easy here.
- **Variable masking** within the dataset: some grids have many filled cells (easier), some have few (harder). The benchmark probably averages over the masking distribution.
- The "image" is not a visual encoding — it's a structured tensor in image clothing. Pixel-space augmentations (rotation, flipping) **can** be label-preserving with care (sudoku has rotational/reflective symmetry that maintains validity), but **scaling, blurring, colour jitter** are nonsense.

## Type and difficulty

**Type-1**: a human can solve sudoku and so identify the target cell's digit. Difficulty depends on masking level.

A constraint-aware model could in principle solve this perfectly. A pure CNN has to learn the constraint propagation implicitly, which is exactly the kind of inference Geada's Type-1 framing was designed to test.

## Implications for our pipeline

- **Tiny dataset** (~20 MB total). Search budget is cheap here; many architectures can be evaluated quickly.
- **Benchmark of 0.0% is a metadata defect** — score formula `(raw - 0)/(100 - 0) * 10 = raw/10` would give a trivial mapping (50% acc → +5 pts, 100% → +10). Verify what the actual evaluation server uses; if it really ships 0.0%, the dataset is essentially scored as `acc * 10 / 100`, with no benchmark cliff. (unconfirmed: needs a test submission.)
- Score-relevance: **probably high for Phase 2** — 2024 dataset, organisers may rotate it in. Tiny size + low compute requirement = cheap to nail down to >90%.
- Consider a sudoku-specific architecture (graph- or constraint-aware) as a side experiment in [[Search vs Train Split]]. But baseline first — a small ResNet should already crush this.
