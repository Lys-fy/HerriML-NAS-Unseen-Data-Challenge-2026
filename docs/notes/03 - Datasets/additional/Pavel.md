# Pavel (polygon_collision)

> **Status:** rebuilt 2026-06-13, v10 3-seed sweep queued | **Topic:** a short film of an object bouncing inside a polygon; predict whether it hits a wall at the next step. Teammate-built (The Count), vendored at `dev/datagen/`; restructured to a 4-channel temporal stack.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Magnus]], [[Cornelius]], [[Dominik]], [[Pipeline v10 - Design + Idea Ledger]]

![[Pavel.png]]
> *The 4 timesteps stacked as 4 channels (N, 4, 64, 64). Top row (collision next): the object reaches the wall by the last channel. Bottom row (no collision): it stays interior. A conv sees all four positions at one location. Regenerate examples: `dev/datagen/generate.py --dataset polygon_collision` then the converter below.*

An object moves inside a polygon, bouncing off the walls. The model sees four consecutive frames and must predict whether the object collides with a wall on the next step. This is the only temporal task in the suite: the answer depends on the object's position AND velocity, which are only recoverable by comparing frames.

| Field | Value |
|---|---|
| Codename | Pavel |
| Generator | `dev/datagen/` `--dataset polygon_collision`, then `dev/datagen_to_npy.py --frames_as_channels` |
| Shape | 4x64x64 (4 timesteps as channels; was a 3x64x256 strip) |
| Classes | 2 (collision next step: yes / no), balanced 50/50 |
| Split | 70/15/15 (tool default -> train/valid/test) |
| corr1 | +0.75 -> routes IMAGE-LIKE in v10 (2-D TCN direct) |
| Learnability | logreg ~chance (expected for a moving-dot temporal task); CNN check is the gate |

## Why it is interesting

- **The only temporal / spatiotemporal task.** The label is a function of motion, not a static image, so it tests whether the pipeline can use the channel axis as time. This is the competition's own multi-frame idiom (Windspeed 12x32x32 is 6 hours x 2 variables; GameOfLife is multi-step), so it is a fair unseen-data probe of that regime.
- **Built blind to the pipeline** (valid RQ2 generalisation test); exercises the image-like branch with a non-trivial channel structure.

## Restructured to channels (2026-06-13)

The original render was a horizontal strip of 4 frames (3x64x256), where the four object positions sit 64px apart, so a conv cannot relate them without a huge receptive field, and logistic regression scored at chance (48.5% vs 50%). The converter `dev/datagen_to_npy.py --dataset polygon_collision --frames_as_channels` splits the strip into its 4 frames, greyscales each, and stacks them as a (4, 64, 64) tensor. Now a single 3x3 conv at each location sees all four timesteps at once, so "did the object move toward a wall" is a local cross-channel pattern. (logreg is permutation-invariant to feature order, so it is unchanged by the restack; the gain is for a conv, which is why the CNN check below is the real test.)

## Open items

- **CNN learnability check** is the one remaining gate: a few epochs on spark should clear chance comfortably; if not, the fallbacks are a bigger/brighter object (radius is ~2px now) or a motion trail.
- The v10 3-seed 2h sweep is queued (see [[Run State + Handoff]] and the lineage in [[Pipeline v10 - Design + Idea Ledger]]); the first run doubles as a robustness test of the 4-channel input.
- ResNet-18 benchmark to replace the 0.0 placeholder.
- Difficulty dials: bigger object, `--num_timesteps`, `--num_sides`, `--noise_level`, `--label_noise`, or a multi-class "which wall" reframe.
