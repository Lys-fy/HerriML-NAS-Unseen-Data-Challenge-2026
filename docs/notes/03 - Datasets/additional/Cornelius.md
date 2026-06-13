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

## Caveat: class imbalance makes raw accuracy misleading

The generator balances the dataset 50% "a winning move exists" vs 50% "none". But the 50% with a winning move are split across up to 10 column labels, while the single "none" class holds the whole other 50%. Measured on 1260 train samples: the "none" class had ~617 members, individual columns 26-100. So:

- A model that always predicts "none" scores ~50% accuracy while learning nothing about columns. logreg's 63.7% is only modestly above that floor.
- **Recommendation:** report per-class / macro-F1, not raw accuracy, OR reframe (see below). For the internal adjusted score we use, the imbalance is not fatal but the headline number must not be read as "63% of winning moves found".

## Difficulty / framing dials (decide before freezing)

- **Binary reframe** ("does an immediate winning move exist?") -> a clean balanced 2-class task, easier and metric-honest. Good if we want a reliable signal.
- **Keep 11-way but balance columns** -> condition generation to draw winning columns evenly and cap the "none" class; harder and more interesting, needs a small generator tweak.
- **Smaller board** (`--board_width 7`, the classic Connect-4 width) -> fewer classes, pieces larger at 64px, easier to read.
- `--noise_level`, `--label_noise` as usual.

## Open items

- Decide the framing (recommend: balanced 11-way with even columns, or the binary version as a second easier dataset), then freeze the spec before any pipeline run.
- Shared PNG/CSV -> npy converter; ResNet-18 benchmark; first v10 run behind the sweep.
