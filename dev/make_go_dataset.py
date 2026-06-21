#!/usr/bin/env python3
"""Sedol: predict the eventual winner of a Go game from a single mid-to-late
board position. The Chesseract sibling -- a hard ENCODED board task that tests
whether the macro/attention lever that helps chess generalises to Go.

Source: free dan-level KGS SGF (u-go.net, 7d+ or both 6d+), parsed with sgfmill
(full Go capture rules). One position per game (no game spans the train/test
split). The position is sampled mid-to-late so the label genuinely depends on
reading the position, not on a near-finished board (too easy) or an opening
(near chance). Difficulty dial: SAMPLE_LO/HI (fraction of game length).

Encoding: (2, 19, 19) float32 planes -- black stones, white stones. This minimal
board-state encoding is deliberate: it is the honest representation (no redundant
empty plane, no constant side-to-move plane), and it routes ENCODED (corr1 ~0.15,
sparse symbols) so Sedol is a true hard-encoded sibling of Chesseract rather than
an image-like task inflated by a constant plane. Balanced 2-way (Black / White).

Runs on spark CPU (sgfmill is pure Python). Reads ~/local/go-sgf/sgf, writes
~/local/nas-comp-datasets/Sedol. Regenerable from the seed (file order sorted).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sgfmill import sgf, boards

SGF_ROOT = Path.home() / "local/go-sgf/sgf"
OUT = Path.home() / "local/nas-comp-datasets/Sedol"
SEED = 13
SIZE = 19
MIN_MOVES = 60                      # skip blitz / forfeits / tiny games
SAMPLE_LO, SAMPLE_HI = 0.50, 0.75   # sample fraction of game length (mid-late)
PER_CLASS = 35_000                  # balanced pool per winner
SPLIT = {"train": 25_000, "valid": 5_000, "test": 5_000}   # per class

rng = np.random.default_rng(SEED)


def game_moves(game):
    """Return (winner 'b'/'w', setup (b,w) sets, [ (colour,(r,c)) ... ] stones)."""
    winner = game.get_winner()
    if winner not in ("b", "w") or game.get_size() != SIZE:
        return None
    root = game.get_root()
    sb, sw, _ = root.get_setup_stones()
    moves = []
    for node in game.get_main_sequence():
        colour, move = node.get_move()
        if colour is not None and move is not None:   # skip passes
            moves.append((colour, move))
    if len(moves) < MIN_MOVES:
        return None
    return winner, (sb, sw), moves


def board_at(setup, moves, k):
    """Board after the first k moves (full capture rules via sgfmill)."""
    b = boards.Board(SIZE)
    for (r, c) in setup[0]:
        b.play(r, c, "b")
    for (r, c) in setup[1]:
        b.play(r, c, "w")
    for colour, (r, c) in moves[:k]:
        b.play(r, c, colour)
    return b


def planes(board):
    blk = np.zeros((SIZE, SIZE), np.float32)
    wht = np.zeros((SIZE, SIZE), np.float32)
    for colour, (r, c) in board.list_occupied_points():
        (blk if colour == "b" else wht)[r, c] = 1.0
    return np.stack([blk, wht], axis=0)   # (2,19,19) black, white


def main():
    files = sorted(SGF_ROOT.rglob("*.sgf"))
    order = rng.permutation(len(files))
    print(f"{len(files)} sgf files; collecting {PER_CLASS} positions per winner ...")
    pools = {"b": [], "w": []}
    seen = bad = 0
    for idx in order:
        if len(pools["b"]) >= PER_CLASS and len(pools["w"]) >= PER_CLASS:
            break
        fp = files[idx]
        try:
            g = sgf.Sgf_game.from_bytes(fp.read_bytes())
            parsed = game_moves(g)
        except Exception:
            bad += 1
            continue
        if parsed is None:
            continue
        winner, setup, moves = parsed
        if len(pools[winner]) >= PER_CLASS:
            continue
        k = int(round(len(moves) * rng.uniform(SAMPLE_LO, SAMPLE_HI)))
        k = max(1, min(k, len(moves) - 1))
        try:
            b = board_at(setup, moves, k)
        except ValueError:
            bad += 1
            continue
        pools[winner].append(planes(b))
        seen += 1
        if seen % 10_000 == 0:
            print(f"  {seen} kept (b={len(pools['b'])} w={len(pools['w'])}), "
                  f"{bad} bad, scanned {np.where(order==idx)[0][0]}")

    n = min(len(pools["b"]), len(pools["w"]))
    print(f"pool: b={len(pools['b'])} w={len(pools['w'])} -> using {n} per class")
    if n < sum(SPLIT.values()):
        sys.exit(f"not enough games: have {n}/class, need {sum(SPLIT.values())}")

    OUT.mkdir(parents=True, exist_ok=True)
    pb = rng.permutation(len(pools["b"]))[:n]
    pw = rng.permutation(len(pools["w"]))[:n]
    off = 0
    for name, per in SPLIT.items():
        ib, iw = pb[off:off + per], pw[off:off + per]
        off += per
        xb = np.stack([pools["b"][i] for i in ib])
        xw = np.stack([pools["w"][i] for i in iw])
        x = np.concatenate([xb, xw]).astype(np.float32)
        y = np.array([0] * per + [1] * per, np.int64)   # 0=black wins, 1=white wins
        sh = rng.permutation(len(y))
        np.save(OUT / f"{name}_x.npy", x[sh])
        np.save(OUT / f"{name}_y.npy", y[sh])
        print(f"  {name}: x{x.shape} balance {np.bincount(y).tolist()}")

    meta = {"num_classes": 2, "input_shape": [2 * SPLIT['train'], 2, SIZE, SIZE],
            "codename": "Sedol", "benchmark": 0.0}
    (OUT / "metadata").write_text(json.dumps(meta))
    print(f"wrote {OUT}  meta={meta}")


if __name__ == "__main__":
    main()
