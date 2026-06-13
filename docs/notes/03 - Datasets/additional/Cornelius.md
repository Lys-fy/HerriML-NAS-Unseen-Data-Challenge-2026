# Cornelius (connect4)

> **Status:** rebuilt 2026-06-13, GOOD, v10 3-seed sweep queued | **Topic:** rendered Connect-Four boards; predict the immediate winning column. A teammate-built additional dataset (The Count), vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Magnus]], [[Pavel]], [[Dominik]], [[Chesseract]], [[Sudoku]], [[Pipeline v10 - Design + Idea Ledger]]

![[Cornelius.png]]
> *Six train examples: a no-winning-move board plus boards whose winning column is 0, 2, 4, 6, 8. Red and yellow are the two players; the active player is the one to move.*

The network sees a 10x10 Connect-Four position (default `board_width=board_height=10`, `connect_k=4`) and must output which column gives the active player an immediate win, or a "no winning move" class. With a width-10 board that is 11 classes (columns 0..9 plus "none"). Boards are sampled from legal self-play games, then a state is kept and analytically checked for winning columns; when several columns win, the label is the one chosen to keep the classes balanced (see below).

| Field | Value |
|---|---|
| Codename | Cornelius |
| Generator | `dev/datagen/` (teammate tool), `--dataset connect4` |
| Shape | 3x64x64 (RGB board render) |
| Classes | 11 (columns 0..9 + "none"; scales with `--board_width`) |
| Split | 70/15/15 |
| corr1 | +0.75 -> routes IMAGE-LIKE in v10 (2-D TCN direct) |
| Learnability | logreg 63.7% vs 9.1% chance: strongly learnable, but see caveat |

## Why it is interesting

- **Board logic, the regime where the random arm and attention have won.** The two project datasets where the v10 random arm beat the gated default are board games (Sudoku s42, [[Chesseract]] s1337, the latter via attention-after-stem). Cornelius is another spatial board-reasoning task and a clean test of whether the encoded-style ops surface on the image-like route too.
- **Spatially structured multi-class output.** Predicting *which* column wins forces the model to scan every column for a connect-k threat, a real lookahead-by-pattern task rather than a global property.
- Built blind to the pipeline (valid RQ2 test); exercises the IMAGE-LIKE branch.

## Class balance: FIXED 2026-06-13

The original generator balanced only win vs no-win, which left ~50% of samples in the single "no winning move" class while the other ~50% scattered across 10 column labels (a model predicting "none" scored ~50% learning nothing). The vendored generator (`dev/datagen/connect4/generator.py`) now buckets boards by their actual label across all 11 classes and fills each bucket to the same target by rejection (a board whose label-bucket is already full is dropped). The "none" class drops from ~50% to ~1/11. Raw accuracy is now meaningful. (Combined with the unique-winner rule below, each board contributes its single winning column to that column's bucket; high columns are filled by continuing to sample until their bucket reaches target.)

Difficulty / framing dials still available:
- **Binary reframe** ("does an immediate winning move exist?") -> a clean 2-class task if we want an easier second dataset.
- **Smaller board** (`--board_width 7`, classic Connect-4) -> fewer classes, larger pieces at 64px.
- `--noise_level`, `--label_noise` as usual.

## Multiple winning moves: FIXED 2026-06-13 (unique-winner only)

A position can have several immediate winning columns (an open three wins at either end; stacked threats win in two places). Measured on 10x10 boards: of all win-boards, **23.5% have 2+ winning columns** (7.3% have 2, 1.1% have 3, 0.3% have 4). A single-label classifier can only be credited for one of them, so labelling such a board with one column means a model that names a *different valid* winning column is marked wrong - a ~23% noise ceiling on the win classes.

Fix: the generator now keeps **only positions with exactly one winning move** (`unique_winner_only=True`, the default; `--c4_allow_multi_winner` opts out). Every win label is then unambiguous: there is one correct column. The "none" class is unchanged. This discards the multi-winner boards during collection (a minority, but it does make generation slower, so the shipped dataset is 20k samples / 14k train rather than 50k). The task is now cleanly "name the unique winning move, or say none".

## Open items

- v10 3-seed 2h sweep queued (2026-06-13); spec frozen at default 10x10 board, balanced 11-way. Build/convert: `dev/datagen/generate.py --dataset connect4` then `dev/datagen_to_npy.py --dataset connect4` (3x64x64, 11 classes).
- ResNet-18 benchmark to replace the 0.0 placeholder.
