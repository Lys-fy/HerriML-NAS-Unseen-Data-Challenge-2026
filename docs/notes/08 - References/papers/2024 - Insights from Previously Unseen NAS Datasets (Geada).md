# 2024 - Insights from the Use of Previously Unseen NAS Datasets (Geada)

> **Status:** reference (PRIMARY - this is our benchmark's origin paper) | **Topic:** the CVPR-2024 paper that introduces the eight Unseen-NAS Challenge datasets and motivates the "generalise to unseen data" framing | **Found by:** research scout July 2026
> **Source:** https://arxiv.org/abs/2404.02189 (CVPR 2024) | **PDF:** `docs/papers/Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf` | **Authors:** Rob Geada, David Towers, Matthew Forshaw, Amir Atapour-Abarghouei, A. Stephen McGough (Newcastle / Durham / Turing)
> **Related:** [[Papers]], [[2025 - Transferrable Surrogates in Expressive NAS Spaces]], [[2025 - ONNX-Net Universal Performance Prediction]], [[Past Winners]], [[Key Learnings]], [[Pipeline v10 - Design + Idea Ledger]]

## TL;DR
The organisers' own paper for the predecessor challenge. Introduces eight bespoke datasets designed so that memorised architecture priors do not transfer: **AddNIST, Language, MultNIST, CIFARTile, Gutenberg, Isabella, GeoClassing, Chesseract**. Split into Type-1 (human-solvable with insight, e.g. AddNIST = sum RGB channels) and Type-2 (human-hard, e.g. GeoClassing = which country from satellite topology). The thesis: standard NAS benchmarks (CIFAR-10/NAS-Bench) are unrepresentative, and a method that only wins there has not shown generalisation. Our 2026 competition is the successor (now 13-16 datasets).

## Dataset facts worth keeping
- **Chesseract** = 12x8x8 one-hot piece/colour board -> {White win, Draw, Black win}. Explicitly noted: the 12-channel shape "caused errors for hardcoded algorithms that only accepted 1 or 3 channel dimensions" -> a robustness trap our pipeline must (and does) handle.
- **GeoClassing** (aka Sadie) = 3x60x60 Sentinel satellite -> 10 European countries.
- **Language** = words from 10 Latin-script languages encoded as letter-frequency images; **Gutenberg** = predict-the-author/era from text-derived images.
- **Isabella** = music-era classification from spectrograms (Type-2, human-hard).
- AddNIST/MultNIST = 3-channel MNIST-derived where the label is an arithmetic function of the three digits (sum / product mod 10).

## Table 1 - stock CNN baselines on the 8 datasets (test acc %)
Their headline reference: strong hand-designed CNNs, NO NAS, no per-dataset tuning. Directly comparable in spirit to our informed-defaults floor.

| Model | AddNIST | Language | MultNIST | CIFARTile | Gutenberg | Isabella | GeoClassing | Chesseract |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 92.08 | 87.00 | 91.55 | 45.56 | 40.98 | 62.02 | 80.33 | 57.83 |
| AlexNet | 94.87 | 85.71 | 94.01 | 48.88 | 45.53 | 61.37 | 92.49 | 57.45 |
| VGG16 | 92.06 | 84.54 | 90.43 | 24.43 | 44.00 | 58.13 | 83.67 | 55.69 |
| **ConvNeXt** | **38.06** | 83.40 | 64.20 | 31.06 | 31.93 | 57.18 | 72.76 | 52.74 |
| MNASNet | 90.51 | 84.63 | 87.70 | 48.49 | 38.00 | 60.69 | 86.00 | 56.26 |
| DenseNet | 93.52 | 84.57 | 92.81 | 51.28 | 43.28 | 63.27 | 94.21 | 59.60 |
| ResNeXt | 91.42 | 93.97 | 90.57 | 46.23 | 40.30 | 60.46 | 89.99 | 55.15 |
| Random | 5 | 10 | 10 | 25 | 16.7 | 25 | 10 | 33.3 |

## What this tells us (relevance is high)
- **Modern heavy nets are NOT safe defaults here.** ConvNeXt collapses on AddNIST (38%, barely above VGG's own weak 24% CIFARTile) and is the worst or near-worst on most columns. The "obvious SOTA backbone" is a trap on small/encoded data - direct empirical support for our informed-simple-defaults thesis over blindly importing ImageNet SOTA.
- **The best baseline varies per dataset** (ResNeXt wins Language 93.97; DenseNet wins CIFARTile/Isabella/GeoClassing/Chesseract; AlexNet wins AddNIST/MultNIST). No single fixed backbone dominates -> justifies a heterogeneous / per-dataset-adaptive pipeline rather than one net.
- **Our v10 numbers sit right in this band** (e.g. our Chesseract ~57.7 vs their 52.7-59.6; Language ~90 vs 84.5-94) - a good external sanity check that our floor is competitive with tuned stock CNNs.
- CIFARTile and Gutenberg are hard for everyone (best ~51 and ~45) - low ceilings to keep in mind when judging our own scores.

## Concrete uses
1. **Writeup:** cite Table 1 as the organisers' own evidence that (a) generic SOTA backbones fail on these datasets and (b) no fixed net generalises - the motivation for informed-defaults + verified probes. Pair with [[2025 - Transferrable Surrogates in Expressive NAS Spaces]] + [[2025 - ONNX-Net Universal Performance Prediction]] (predictor transfer collapses on the misaligned datasets) for a complete "prediction/transfer is unreliable here" argument.
2. **Sanity target:** use these per-dataset baseline numbers as a floor our pipeline should beat or match on the overlapping datasets.
