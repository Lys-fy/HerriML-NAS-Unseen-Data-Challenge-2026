# Candidates - Pavel (polygon_collision) + Dominik (domino)

> **Status:** assessed 2026-06-13, NEED WORK before use | **Topic:** the two teammate datasets that are not usable as-is, with concrete ideas to make them interesting. Generators vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Magnus]], [[Cornelius]], [[Quincy]]

![[_candidates_polygon_domino.png]]
> *Top: Pavel, a 4-frame film strip of a tiny object (cyan dot) bouncing inside a polygon; predict whether it hits a wall next step. Bottom: Dominik, a domino chain; predict valid vs one-mismatch.*

Both probe a genuinely different regime from [[Magnus]] / [[Cornelius]] (temporal for Pavel, fine-detail symbol reading for Dominik), so they are worth fixing rather than dropping, but neither is usable in its current form.

## Pavel (polygon_collision) - promising but the signal is invisible

3x64x256 (a horizontal strip of 4 frames), binary: does the moving object collide with a wall at the next timestep. **logreg 48.5% vs 50% chance: a linear model sees literally nothing**, because the only informative pixels are a ~1-pixel cyan dot whose position changes frame to frame. That could mean "hard but learnable by the right net" or "too hard to be useful"; we cannot tell without a CNN smoke, and the dot is small enough that the second is a real risk.

Ideas to make it work (cheap generator changes):
1. **Stack frames as channels, not a wide strip.** 4 frames -> a 4-or-12-channel 64x64 tensor instead of 3x64x256. This is the competition's own idiom (Windspeed is 12x32x32 = 6h x 2 vars; GameOfLife is 10x10 multi-step) and gives a 2-D conv a clean per-pixel temporal stack instead of having to learn frame boundaries. Also broadens our branch coverage: a multi-channel input is more encoded-like (lower corr1) than the current image-like strip.
2. **Make the object visible.** Radius 2-3 px, high-contrast colour. A 1-px dot at 64px is below what augmentation-heavy training can track.
3. **Render a motion trail.** Draw the last k positions as a fading streak so velocity and heading are readable, turning "needs cross-frame memory" into a single-glance geometry task (or keep frames AND add the trail).
4. **Fewer, larger frames** (2-3 frames, image_size 96-128) if we keep the strip.
5. **Richer label** once the object is visible: predict which wall/edge it bounces off (multi-class) instead of binary.

Recommended: do (1)+(2), then a 5-minute CPU CNN learnability smoke before committing to a full-size build. If a small CNN cannot beat 55-60%, park it.

## Dominik (domino) - weak, the pips are unreadable

3x64x64, binary: is the chain valid (all adjacent halves match) or does exactly one pair mismatch. **logreg 52.2% vs 50%: barely above chance.** The teammate already flags it as "not quite working, not so exciting", and the render confirms why: the pip markings look like small hooks/arrows rather than countable dots, so even a human struggles to read the values at 64px, and the task (compare adjacent pip counts) is unreadable.

Ideas:
1. **Fix the pip rendering first** - real countable dot pips, white on a dark tile, large enough to resolve at 64px. Without this the label is not recoverable from the image and the dataset is broken, not just hard.
2. **Fewer / bigger tiles** (lower `--max_dominoes`) or larger `--image_size`, so each tile has enough pixels for its pips.
3. **Reframe to localisation** - "which tile boundary is the mismatch" (multi-class position) is richer than binary valid/invalid and forces the model to read every junction.
4. Lowest priority of the four; fix the render and re-measure, or drop in favour of Magnus/Cornelius/[[Quincy]].

## Shared next step

Once Pavel is channel-stacked and Dominik's pips are legible, run the same corr1 + logreg + small-CNN triage we used here (`dev/datagen/` generate -> measure) before freezing any spec. Keep specs frozen before the pipeline sees them (RQ2 no-design-leakage).
