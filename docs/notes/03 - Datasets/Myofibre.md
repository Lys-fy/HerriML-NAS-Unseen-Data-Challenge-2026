# Myofibre

> **Status:** drafting | **Topic:** Binary medical classification of segmented muscle-fibre images (analysable vs not). Largest dataset in the suite.
> **Related:** [[Datasets Overview]]

A binary medical-imaging classification dataset derived from the **NCL_SM dataset** (Newcastle University's skeletal muscle imaging collection). Each sample is a manually-segmented muscle fibre. Label: AM ("Analysable Myofibre") vs NAM ("Not Analysable Myofibre") — a quality / readiness assessment for downstream muscle pathology analysis.

| Field | Value |
|---|---|
| Codename | (no codename in published metadata) |
| Era | 2024 (one of three hidden datasets in the 2024 competition) |
| Shape | 3×128×128 (channels-first) |
| Train / val / test | 39,497 / 4,937 / 4,937 (asymmetric splits — note the small test set) |
| Classes | 2 (binary: AM, NAM) |
| Class balance | ~30,000 AM, ~20,000 NAM (≈3:2) — randomly distributed across splits |
| Benchmark | unconfirmed — not exposed by figshare without download (verify by reading `metadata` after `scripts/download_datasets.sh` Myofibre) |
| License | inferred CC BY 4.0 |
| DOI / URL | https://data.ncl.ac.uk/articles/dataset/Myofibre_-_Binary_Classification_Dataset/26969998 |
| Authors | David Towers, Linus Ericsson, Andrew McGough, Amir Atapour-Abarghouei, Elliot J. Crowley |

## Why it stands out

- **Largest dataset in the suite by disk size (~9.7 GB)**: 3 × 128 × 128 × 4-byte float × 50,000 samples ≈ 9.4 GB raw. RAM budget will matter — see [[Failure Modes]].
- **First "real medical" image** in the challenge — derived from human tissue scans, not synthetic / encoded data. Standard image augmentation (rotation, flip, mild colour) is plausibly label-preserving (a flipped fibre is still the same quality).
- **Binary classification** is unique in the suite. A 50/50 random baseline scores 50%, but the 3:2 class skew lets a "predict AM always" baseline score 60% — non-trivial floor.
- The "analysable / not analysable" judgement is a **dataset-construction quality signal**, not a biological feature. A model essentially learns "what the human annotators thought was good for analysis" — possible label noise.

## Type and difficulty

Effectively **Type-1**: a domain-trained pathologist can do this readily. The benchmark question is whether NAS finds an architecture suited to high-resolution medical imagery without medical priors.

## Implications for our pipeline

- **High memory pressure**: with 3×128×128 inputs, batch size needs care. The [[Submission Contract]] should not load all of train into memory; use lazy loading from `.npy`.
- Score slope and budget allocation: depends on the benchmark we observe. If ResNet-18 already lands ~85–90%, treat like [[AddNIST]] (small headroom, high slope). If it lands ~60% (just-better-than-majority), treat like a high-leverage NAS target.
- **Augmentation usable**: standard medical-imaging augmentations (random rotation, h/v-flip, small affine) are likely safe. Verify that fibre orientation isn't part of the AM/NAM signal before flipping.
- Likely candidate for Phase 2/3 inclusion in 2026.

## Open questions

- Real benchmark % (only visible after download).
- Whether the 3-channel input is RGB-from-microscopy or some channel-stacked staining — affects useful augmentation.
