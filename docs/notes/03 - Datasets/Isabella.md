# Isabella

> **Status:** drafting | **Topic:** 5-second music-recording spectrograms classified by composition era. **Not publicly downloadable.**
> **Related:** [[Datasets Overview]]

5-second snippets of music recordings from the Isabella Stewart Gardner Museum's public concert library, converted to 64-band spectrograms. Label = composition era (Baroque, Classical, Romantic, 20th Century).

| Field | Value |
|---|---|
| Codename | (none in published metadata) |
| Era | CVPR-2023 |
| Shape | 1×64×128 (channels-first) |
| Train / val / test | 50,000 / 10,000 / 10,000 (Geada §4.6) |
| Classes | 4 |
| Benchmark (ResNet-18) | 62.02% (Geada Tab. 1) |
| Best off-the-shelf CNN (Geada) | DenseNet-161 63.27% |
| Best NAS method (Geada Tab. 2) | random-DARTS 66.35% (PC-DARTS 65.77%) |
| Best 2023-comp submission | 61.42% (slightly below ResNet-18 — likely time-constraint limited) |
| License | **withheld** — Isabella Stewart Gardner Museum prohibits redistribution of modified recordings |

## Availability

**The Isabella dataset is not on figshare.** Geada §7 ("Rights and Reproduction"): the museum permits use of the underlying audio but not redistribution of derived spectrograms. The organisers provide a **Python script** to convert music files (downloaded by the user from the museum's site at `gardnermuseum.org/experience/music`) into the competition format. As of April 2026 that script is listed as "coming soon" at github.com/Towers-D/NAS-Unseen-Datasets — **status unconfirmed**.

This means:

- Isabella will not be in our local development pool.
- It *may* be one of the hidden Phase 2 / Phase 3 datasets, since the organisers control it directly.
- We cannot tune for it. Any submission must rely on the metadata signature (1-channel, 64×128, 4 classes, ~62% benchmark) at runtime.

## Type and difficulty

**Type-2** (Geada): the task is era classification from a spectrogram, requiring a model to learn era-defining acoustic-spectral features (instrumentation, tonality, recording quality). No human can do this from a single 5-second spectrogram with confidence.

**Recording-level data leakage** is prevented: no recording in train appears in val or test.

## What's notable

- The 64×128 spectrogram shape is **unique among the 14 datasets** — much wider than tall, and 1-channel. Very different from anything else, so a NAS that hardcodes 3-channel input or square spatial shape will silently fail.
- Random search wins here (66.35%), as does Bonsai random (58.53%). Among Geada-tested methods, **architecture choice barely matters** — the dataset rewards reasonable training more than search.
- Memory-efficient feature reuse seems to help: DenseNet-161 is the best off-the-shelf CNN, in line with its showing on the other "memory-intensive" datasets ([[GeoClassing]], [[Chesseract]], [[CIFARTile]]).

## Implications for our pipeline

- Treat as **possibly-included, untunable**. The pipeline must:
  - Accept arbitrary `(N, 1, H, W)` inputs without assuming square or 3-channel.
  - Have a sane default architecture for spectrogram-shaped data (avoid square-only padding logic).
- If Isabella shows up in Phase 2 with our pipeline failing on shape, we eat a −10. See [[Failure Modes]] in framework section.
- Score slope is moderate (benchmark ~62% → ~0.26 / 1%). Don't optimise hard for it.
