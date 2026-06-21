# Sedol (Go outcome)

> **Status:** built 2026-06-17, spec frozen | **Topic:** predict the eventual winner of a dan-level Go game from one mid-to-late board position. The hard ENCODED-board sibling of [[Chesseract]], built blind to the pipeline for the RQ2 generalisation test.
> **Related:** [[Datasets Overview]], [[Dataset Ideas - Backlog]], [[Chesseract]], [[Quincy]], [[Magnus]], [[Sloan]], [[Cauchy]], [[Pipeline v10 - Design + Idea Ledger]]

![[Sedol.png]]
> *Four sampled positions rendered as gobans (a human-readable view; the network sees two binary 19x19 planes, not this render). The label is who eventually WON the whole game, not who is ahead in the shown position. Regenerate: `python3 dev/plot_sedol_examples.py`.*

The network sees a 2x19x19 stack of binary board planes (black stones, white stones) and must predict the eventual game result from a single position taken partway through. Like Chesseract this is a value-prediction task: the outcome depends on reading the position, not on counting a finished board. It is our second hard encoded board game, the direct test of whether the v11 macro/attention lever that helps chess (+5.94 on Chesseract) generalises to Go.

| Field                  | Value                                                                       |
| ---------------------- | --------------------------------------------------------------------------- |
| Codename               | Sedol (Lee Sedol, the AlphaGo match)                                        |
| Shape                  | 2x19x19 (black / white stone planes, binary float32)                        |
| corr1 / route          | **+0.15 / encoded** (sparse stone symbols; near the 0.2 boundary)           |
| Train / val / test     | 50,000 / 10,000 / 10,000 (balanced, one position per game)                  |
| Classes                | 2 (0 = Black wins, 1 = White wins); chance 50%                              |
| Source                 | KGS dan-level SGF, u-go.net (7d+ or both 6d+); 228,849 games scanned        |
| Benchmark (`metadata`) | **53.92%** (ResNet-18 zero-point, 2026-06-18)                                |
| Generator              | `dev/make_go_dataset.py` (seed 13, sgfmill replay)                          |
| Location               | `dev/datasets/Sedol` (gitignored) + spark `~/local/nas-comp-datasets/Sedol` |

## How it was generated

1. **Corpus.** All KGS dan archives from u-go.net (`dev/fetch_go_sgf.sh`), 228,849 SGF games. Games are strong amateur to professional; we redistribute only the derived board tensors, never the SGF.
2. **Parse + replay.** `sgfmill` reads the result `RE[...]` and replays the main line with full Go capture rules. Games are kept only if 19x19, with a decisive winner, and at least 60 moves (skips blitz / forfeits).
3. **One position per game.** A single move index is sampled at a uniform fraction in [0.50, 0.75] of the game length (mid-to-late). One position per game means no game spans the train/test split, so there is no position-level leakage.
4. **Encoding.** Two binary 19x19 planes: black stones, white stones. This minimal board-state encoding is deliberate (see measured properties): a redundant empty plane and a constant side-to-move plane were dropped because they inflated the fingerprint to image-like. Balanced to 25,000 Black-win and 25,000 White-win positions for train, 5,000 + 5,000 each for val/test.
5. **Packaging.** float32 `train/valid/test_x.npy`, int64 `*_y.npy`, JSON `metadata`, the official format (`dev/mock_harness.py` loads it unchanged).

## Results (2026-06-18)

![[newdata_curves_grid_4x5.png]]
> *Custom-dataset grid (bottom three rows are the new sets): v10 best-val (green, 3 seeds + aggregate) over the ResNet-18 zero-point (blue). Regenerate: `python3 dev/plot_newdata_grid.py`.*

ResNet-18 benchmark **53.92%**. v10 3-seed test accuracy **56.55%** (s42 57.00, s1337 55.17, s2024 57.48), so **adjusted +0.57**. Sedol lands almost exactly on Chesseract (57.66%): a hard encoded board-outcome task, ~56-58%, where v10 has a small but real edge over ResNet. Strong sibling validation, and the prime candidate for the macro/attention lever in the in-depth search.

## Measured properties (before any pipeline run)

- **Routes encoded (corr1 = +0.15), and the encoding choice is the whole story.** A first 4-plane version (black, white, empty, side-to-move) routed image-like at +0.42, but that was an artifact: the constant side-to-move plane contributes +1.0 to corr1 (every pixel equals its neighbour) and the empty plane is a redundant blobby mask (+0.39). The actual stone planes are +0.15 each. The minimal 2-plane encoding therefore routes ENCODED, just inside the 0.2 boundary, which is the honest nature of Go-board-as-symbols and what makes Sedol a true hard-encoded sibling of Chesseract. Nice secondary finding: Go sits right at the encoded/image-like boundary, between chess (~0) and natural images.
- **Hard by construction.** Predicting the winner from a mid-game position is a value task; a from-scratch net in 2h is not AlphaGo, so the ResNet benchmark should land modest (the difficulty dial is `SAMPLE_LO/HI`: later = easier).
- **Balanced**, so accuracy is meaningful and chance is exactly 50%.

## Open items

- ResNet-18 benchmark and v10 3-seed run: DONE (see Results). Macro/attention lever to be tested by the in-depth per-stage macro search.
- First v10 run: Go routes encoded (settled), so the live question is which family the backbone gate picks within the encoded route, and whether the macro/attention candidate that wins Chesseract also wins here. That comparison is the whole point of the set.
- Difficulty is tunable without re-downloading: change `SAMPLE_LO/HI` and rerun the generator.
