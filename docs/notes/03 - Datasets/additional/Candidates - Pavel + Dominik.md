# Candidates - Pavel (polygon_collision) + Dominik (domino)

> **Status:** assessed 2026-06-13, NEED WORK before use | **Topic:** the two teammate datasets that are not usable as-is, with concrete ideas to make them interesting. Generators vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Magnus]], [[Cornelius]], [[Quincy]]

![[_candidates_polygon_domino.png]]
> *Top: Pavel, a 4-frame film strip of a tiny object (cyan dot) bouncing inside a polygon; predict whether it hits a wall next step. Bottom: Dominik, a domino chain; predict valid vs one-mismatch.*

Both probe a genuinely different regime from [[Magnus]] / [[Cornelius]] (temporal for Pavel, fine-detail symbol reading for Dominik), so they are worth fixing rather than dropping, but neither is usable in its current form.

## Pavel (polygon_collision) - frames-as-channels DONE, pending a CNN check

Original: 3x64x256 (a horizontal strip of 4 frames), binary, does the moving object collide with a wall next step. **logreg 48.5% vs 50% chance** because the informative pixels are a small cyan dot whose position changes frame to frame, and on the wide strip the four positions sit 64px apart so a conv cannot relate them without a huge receptive field.

**Fixed (2026-06-13): the converter `dev/datagen_to_npy.py --dataset polygon_collision --frames_as_channels` splits the strip into its T frames, greyscales each, and stacks them as a (T, 64, 64) channel tensor.** Verified shape `(N, 4, 64, 64)`. This is the competition's own idiom (Windspeed 12x32x32, GameOfLife multi-step): now a single 3x3 conv at each location sees all four timesteps at once, so "did the object move toward a wall" is a local cross-channel pattern instead of a long-range one. (Note: logreg is permutation-invariant to feature order, so the strip and the channel stack give it the same ~chance score; the gain is for a conv, not for logreg, so the real test is a small CNN.)

Still recommended before a full build:
- **CNN learnability smoke** (a few epochs on spark) to confirm the channel stack is learnable. If a small CNN cannot beat ~55-60%, apply the next two ideas.
- **Make the object bigger / brighter** (radius 2-3px is current; bump it) so the signal survives augmentation.
- **Motion trail** option: draw the last k positions as a fading streak so heading is readable within one frame.
- **Richer label** once visible: which wall/edge it bounces off (multi-class) instead of binary.

## Dominik (domino) - weak; root cause is layout + resolution, not the task

3x64x64, binary: is the chain valid (all adjacent halves match) or does exactly one pair mismatch. **logreg 52.2% vs 50%: barely above chance**, and the teammate flags it as "not quite working".

Root cause (from reading `dev/datagen/domino/render.py`): the chain is laid out as a **self-avoiding walk on a 2-D grid** that snakes around, and the whole walk is scaled to fit 64x64. With up to 9 tiles the grid can be ~9 cells wide, so `cell_size` collapses to a handful of pixels and the pips (`pip_r = cell//9`, ~2-3 px) merge into the blobs we see. So the label is not recoverable from the image: the dataset is closer to broken than hard. The task also stacks three sub-problems (recover chain order from the 2-D snake, read every pip count, check adjacency), which is too much at this resolution.

How I would fix it (in priority order):
1. **Drop the 2-D snake for a linear layout.** Lay the chain left-to-right in a single row (wrap to a second row only if needed). This removes the "find the chain order" sub-problem (order is just left-to-right) and frees pixels for larger tiles.
2. **Fewer, bigger tiles + larger canvas.** Cap at 4-5 dominoes (`--max_dominoes 5`) and render at 96-128 px so each half-tile gets ~16 px and the standard pip layout is countable. Bump `pip_r` to ~`cell//6`.
3. **Re-measure** with the same corr1 + logreg + small-CNN triage; only freeze a spec once a human can read the pips and a small CNN clears ~60%.
4. **Optional richer reframe** once readable: predict the *position* of the mismatching junction (multi-class localisation) rather than a single valid/invalid bit; forces the model to read every junction and gives a denser signal.
5. Lowest priority of the four. If the linear-layout + resolution fix does not lift it clearly above chance, drop it in favour of [[Magnus]] / [[Cornelius]] / [[Quincy]].

(Steps 1-2 are generator changes I have not made yet, since the task framing should be a team call; Cornelius's balancing and Pavel's channel stack were unambiguous fixes so those are done.)

## Shared next step

Convert any of these to competition npy with `dev/datagen_to_npy.py` (handles the label remap and, for Pavel, `--frames_as_channels`). Re-run the corr1 + logreg + small-CNN triage before freezing a spec; keep specs frozen before the pipeline sees them (RQ2 no-design-leakage).
