# Cornelius (connect4)

> **Status:** assessed 2026-06-13, GOOD with a metric caveat, spec not yet frozen | **Topic:** rendered Connect-Four boards; predict the immediate winning column. A teammate-built additional dataset (The Count), vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Magnus]], [[Chesseract]], [[Sudoku]], [[Pipeline v10 - Design + Idea Ledger]]

![[Cornelius.png]]
> *Six train examples: a no-winning-move board plus boards whose winning column is 0, 2, 4, 6, 8. Red and yellow are the two players; the active player is the one to move.*

The network sees a 10x10 Connect-Four position (default `board_width=board_height=10`, `connect_k=4`) and must output which column gives the active player an immediate win, or a "no winning move" class. With a width-10 board that is 11 classes (columns 0..9 plus "none"). Boards are sampled from legal self-play games, then a state is kept and analytically checked for winning columns (the label is the smallest winning column).

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

The original generator balanced only win vs no-win, which left ~50% of samples in the single "no winning move" class while the other ~50% scattered across 10 column labels (a model predicting "none" scored ~50% learning nothing). The vendored generator (`dev/datagen/connect4/generator.py`) now buckets boards by their actual label across all 11 classes and fills each bucket to the same target; when a board has several winning columns it goes to the emptiest bucket so the rare high columns fill up. Measured on a 1100-sample run (770 train): per-class counts 60-83, the "none" class down from ~50% to **9.4% (= 1/11)**. Raw accuracy is now meaningful.

Difficulty / framing dials still available:
- **Binary reframe** ("does an immediate winning move exist?") -> a clean 2-class task if we want an easier second dataset.
- **Smaller board** (`--board_width 7`, classic Connect-4) -> fewer classes, larger pieces at 64px.
- `--noise_level`, `--label_noise` as usual.

Note: with the balanced rule the label is *a* winning column (the one chosen to balance), not necessarily the smallest; any winning column is a correct answer, so a prediction of a different valid winning column counts as wrong. That single-label limitation is inherent to framing this as classification.

## Open items

- Freeze the spec (board size, sample counts, noise) before any pipeline run.
- Convert with `dev/datagen_to_npy.py --dataset connect4` (PNG/CSV -> npy, 3x64x64, 11 classes). ResNet-18 benchmark; first v10 run behind the sweep.
