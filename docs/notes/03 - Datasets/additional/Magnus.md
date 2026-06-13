# Magnus (mathvision)

> **Status:** assessed 2026-06-13, GOOD, v10 3-seed sweep queued | **Topic:** domain-coloured complex polynomial phase portraits; predict the polynomial degree. A teammate-built additional dataset (The Count), vendored at `dev/datagen/`.
> **Related:** [[Datasets Overview]], [[Quincy]], [[Cornelius]], [[Pavel]], [[Dominik]], [[Pipeline v10 - Design + Idea Ledger]]

![[Magnus.png]]
> *Six train examples spanning degrees 1-5. The degree equals the number of phase singularities (the points where all colours meet, one per root). Regenerate examples: see the command below.*

The network sees a full-frame domain-coloured plot of a complex polynomial p(z) (hue = phase of p(z), brightness banding = magnitude) and must output the polynomial degree, 1 to 5. The degree shows up as the number of colour-wheel centres (roots), so the task is essentially "count the phase singularities / total winding" under heavy nuisance variation in root placement, zoom and translation.

| Field | Value |
|---|---|
| Codename | Magnus |
| Generator | `dev/datagen/` (teammate tool), `--dataset mathvision` |
| Shape | 3x64x64 (RGB, full-frame colour field) |
| Classes | 5 (degree 1..5; `--max_degree` is the dial) |
| Split | 70/15/15 (the tool's default; maps to train/valid/test) |
| Class balance | even (degree sampled uniformly) |
| corr1 | +0.65 -> routes IMAGE-LIKE in v10 (2-D TCN direct, no gate) |
| Learnability | logreg 23.0% vs 20.0% chance: NOT linearly separable, needs real vision |

## Why it is interesting

- **Genuine visual reasoning, not a texture shortcut.** Logistic regression on raw pixels sits barely above chance (23 vs 20), so the degree is not encoded in any global colour statistic; the model has to localise and count singularities. This is the same "hard for linear, easy in principle for the right architecture" profile that makes a dataset NAS-rewarding.
- **Rotational / cyclic structure.** Each root is a 2-pi phase winding. That is exactly the kind of local-periodic structure where the random arm's `SpatialSelfAttn2D` op or specific oriented filters could matter, so Magnus is a clean image-like probe of whether the pipeline finds non-default ops (compare the board-logic wins on [[Cornelius]] and Chesseract).
- **Built blind to the pipeline**, so it is a valid RQ2 generalisation test, and it exercises the IMAGE-LIKE branch of v10 (unlike [[Quincy]], which is encoded).

## Difficulty notes (read before freezing the spec)

- **Higher degrees get ambiguous at 64x64.** When roots cluster or fall off-frame, degree 4 vs 5 is genuinely hard to read (visible in the degree-5 example, which looks almost smooth). This is fine as difficulty, but it caps the ceiling below 100%; if we want a cleaner task, constrain roots to the visible disc in the generator or cap `--max_degree` at 4.
- **Difficulty dials:** `--max_degree` (more classes, harder), `--noise_level` (pixel noise), `--label_noise` (label corruption), `--image_size` (more pixels eases the count).

## Open items

- v10 3-seed 2h sweep queued (2026-06-13); spec frozen at default max_degree 5, image_size 64. Build/convert: `dev/datagen/generate.py --dataset mathvision` then `dev/datagen_to_npy.py --dataset mathvision` (remaps degrees 1-5 -> classes 0-4).
- ResNet-18 benchmark to replace the 0.0 placeholder.
