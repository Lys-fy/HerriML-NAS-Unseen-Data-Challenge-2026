# Dominik (domino)

> **Status:** rebuilt 2026-06-13, GOOD, v10 3-seed sweep queued | **Topic:** a left-to-right domino chain; predict which junction breaks the matching rule (or that the chain is valid). Teammate-built (The Count), vendored at `dev/datagen/`; rebuilt from a broken binary task into a localisation task.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Magnus]], [[Cornelius]], [[Pavel]], [[Pipeline v10 - Design + Idea Ledger]]

![[Dominik.png]]
> *A five-domino chain read left to right. The chain rule is that the touching halves across each gap match; the label is which junction (gap) breaks it, or "valid". Regenerate examples: `dev/datagen/generate.py --dataset domino` then the converter below.*

A chain of dominoes is laid left to right. Adjacent dominoes should match where they touch (domino i's right half equals domino i+1's left half). Exactly one junction may be wrong; the model must read every junction's pip counts and point to the broken one, or say the chain is valid.

| Field | Value |
|---|---|
| Codename | Dominik |
| Generator | `dev/datagen/` `--dataset domino` (localisation task, default) |
| Shape | 3x44x374 (a wide chain strip; cf. Cryptic 6x768) |
| Classes | 5 (class 0 = valid; class k = mismatch at junction k-1), set by `--domino_count` |
| Split | 70/15/15 (tool default -> train/valid/test) |
| Class balance | even across all classes (bucketed during generation) |
| corr1 | +0.84 -> routes IMAGE-LIKE in v10 (2-D TCN direct) |
| Learnability | logreg 35.2% vs 20.0% chance: clearly learnable, far from solved |

## Why it is interesting

- **Fine-detail symbol reading + localisation.** The model must count pips at every half-tile and compare across each gap, then output a position. That is a denser supervision signal than a single valid/invalid bit and forces attention to all junctions.
- **Wide aspect ratio.** A 1-D chain is honestly a wide image (3x44x374), so like Cryptic it probes the pipeline on a non-square input where 1-D-style processing could matter.
- **Built blind to the pipeline** (valid RQ2 generalisation test); exercises the image-like branch.

## Rebuilt 2026-06-13 (was broken)

The original was a binary valid/invalid task drawn as a self-avoiding walk snaking on a 2-D grid, scaled to 64x64. With up to 9 tiles each cell collapsed to a few pixels and the pips merged into blobs, so the label was not recoverable from the image (logreg 52.2% vs 50%). Four fixes, all in the vendored generator:

1. **Linear layout** (`render_dominoes` rewritten): the chain is a single left-to-right row, so order is unambiguous and the touching halves flank the gap they are checked across. The 2-D walk is gone.
2. **Readable pips:** half-cell size set by `--domino_tile` (default 34px), independent of `image_size`; larger pip radius. Pips are countable.
3. **Localisation reframe** (`--domino_task localisation`, the default): fixed `--domino_count` (default 5) chain; label = the mismatch junction index, or a "valid" class; balanced across all classes. The original binary task remains via `--domino_task validity`.
4. **Re-measured:** balanced 5 classes; logreg 35.2% vs 20% chance, confirming the signal is readable.

## Build + convert

```
python dev/datagen/generate.py --dataset domino --num_samples 70000 --seed S --output_dir <png>
python dev/datagen_to_npy.py --dataset domino --in_dir <png> --out_dir <npy>   # 3x44x374, 5 classes
```

## Open items

- v10 3-seed 2h sweep queued (see [[Run State + Handoff]]).
- ResNet-18 benchmark to replace the 0.0 placeholder.
- Difficulty dials: raise `--domino_count` (more junctions and classes), `--noise_level`, `--label_noise`.
