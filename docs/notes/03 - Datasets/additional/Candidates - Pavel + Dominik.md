# Candidates - Pavel (polygon_collision) + Dominik (domino)

> **Status:** assessed then FIXED 2026-06-13 | **Topic:** the two teammate datasets that were not usable as-is. Both are now addressed: Pavel channel-stacked, Dominik rebuilt as a localisation task. Generators vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Magnus]], [[Cornelius]], [[Quincy]]

Both probe a genuinely different regime from [[Magnus]] / [[Cornelius]] (temporal for Pavel, fine-detail symbol reading for Dominik). Neither was usable in its original form; the fixes below are implemented in `dev/datagen/`.

## Pavel (polygon_collision) - frames-as-channels DONE, pending a CNN check

Original: 3x64x256 (a horizontal strip of 4 frames), binary, does the moving object collide with a wall next step. **logreg 48.5% vs 50% chance** because the informative pixels are a small cyan dot whose position changes frame to frame, and on the wide strip the four positions sit 64px apart so a conv cannot relate them without a huge receptive field.

**Fixed (2026-06-13): the converter `dev/datagen_to_npy.py --dataset polygon_collision --frames_as_channels` splits the strip into its T frames, greyscales each, and stacks them as a (T, 64, 64) channel tensor.** Verified shape `(N, 4, 64, 64)`. This is the competition's own idiom (Windspeed 12x32x32, GameOfLife multi-step): now a single 3x3 conv at each location sees all four timesteps at once, so "did the object move toward a wall" is a local cross-channel pattern instead of a long-range one. (Note: logreg is permutation-invariant to feature order, so the strip and the channel stack give it the same ~chance score; the gain is for a conv, not for logreg, so the real test is a small CNN.)

Still recommended before a full build:
- **CNN learnability smoke** (a few epochs on spark) to confirm the channel stack is learnable. If a small CNN cannot beat ~55-60%, apply the next two ideas.
- **Make the object bigger / brighter** (radius 2-3px is current; bump it) so the signal survives augmentation.
- **Motion trail** option: draw the last k positions as a fading streak so heading is readable within one frame.
- **Richer label** once visible: which wall/edge it bounces off (multi-class) instead of binary.

## Dominik (domino) - REBUILT 2026-06-13, now a localisation task

![[Dominik.png]]
> *Five-domino chain read left to right; the label is which junction (the gap between two dominoes) breaks the matching rule, or "valid". Pips are now countable.*

Original: 3x64x64 binary (valid vs one mismatch), **logreg 52.2% vs 50%: barely above chance**, teammate "not quite working". Root cause (from `dev/datagen/domino/render.py`): the chain was a **self-avoiding walk snaking on a 2-D grid** scaled to fit 64x64, so with up to 9 tiles each cell collapsed to a few pixels and the pips merged into blobs. The label was not recoverable from the image, and the task stacked three sub-problems (recover chain order from the snake, read every pip, check adjacency).

All four fixes are now implemented in the vendored generator:
1. **Linear layout.** `render_dominoes` was rewritten to draw the chain as a single left-to-right row; chain order is just left to right and the touching halves the rule checks flank the gap between consecutive dominoes. The 2-D walk is gone.
2. **Bigger tiles + larger pips.** Half-cell size is set by `--domino_tile` (default 34 px) independent of `image_size`, and the pip radius was bumped (`min(hw,hh)//4`). Pips are countable (see figure).
3. **Localisation reframe (the richer task).** New `--domino_task localisation` (now the default) uses a fixed `--domino_count` (default 5) chain. Label = the junction index of the single mismatch, or a dedicated "valid" class: **class 0 = valid, class k = mismatch at junction k-1**, so `domino_count` classes total, balanced across all of them. The model must read every junction and point to the wrong one. The original binary task stays available via `--domino_task validity`.
4. **Re-measured.** Balanced 5-class run (770 train): per-class counts 151-160; **logreg 35.2% vs 20% chance** (was 52 vs 50), confirming the signal is now readable. Shape 3x44x374 (a wide chain strip; a 1-D chain is honestly a wide image, cf. Cryptic 6x768).

Build + convert:
```
python dev/datagen/generate.py --dataset domino --num_samples 70000 --seed S --output_dir <png>
python dev/datagen_to_npy.py --dataset domino --in_dir <png> --out_dir <npy>   # 3x44x374, 5 classes
```
Open items: freeze the spec (domino_count, tile, noise), ResNet-18 benchmark, first v10 run behind the sweep. To make it harder: raise `--domino_count` (more junctions and classes) or add `--noise_level` / `--label_noise`.

## Shared next step

Convert any of these to competition npy with `dev/datagen_to_npy.py` (handles the label remap and, for Pavel, `--frames_as_channels`). A small-CNN learnability smoke on spark is the remaining gate for Pavel; Dominik already clears logreg. Keep specs frozen before the pipeline sees them (RQ2 no-design-leakage).
