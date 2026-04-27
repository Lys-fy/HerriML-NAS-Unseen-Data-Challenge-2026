# GeoClassing

> **Status:** drafting | **Topic:** Sentinel-2 satellite RGB images classified by European country. 10 classes.
> **Related:** [[Datasets Overview]]

Satellite photography from BigEarthNet, relabelled by country (rather than the original ground-cover labels). The country was determined by cross-referencing the sentinel patch coordinates against a map of Europe.

| Field | Value |
|---|---|
| Codename | Sadie |
| Era | CVPR (2021–23) |
| Shape | 3×64×64 (channels-first) — note: the README says 64×64, Geada §4.7 says 60×60. The local `metadata` reports 64×64; trust the metadata. |
| Train / val / test | 50,000 / 10,000 / 10,000 (local README); Geada paper says 43,821 / 8,758 / 8,751 (older split) |
| Classes | 10 (Ireland, Lithuania, Austria, Finland, Serbia, Belgium, Luxembourg, Kosovo, Switzerland, Portugal) |
| Benchmark (ResNet-18) | 80.33% |
| Best off-the-shelf CNN (Geada) | DenseNet-161 94.21% |
| Best NAS method (Geada Tab. 2) | DrNAS 96.03% (PC-DARTS 94.61%, Bonsai 95.66%, random-DARTS 95.54%) |
| Best 2023-comp submission | **96.08%** — competition-grade NAS edged out offline NAS |
| License | CC BY 4.0 (inherited from BigEarthNet) |
| DOI | https://data.ncl.ac.uk/articles/dataset/GeoClassing_Dataset/24050256 |

## Type and difficulty

**Type-2** (Geada): humans without geographic / vegetation expertise can't reliably tell which country a satellite tile is in. The task requires the model to learn topology, vegetation, urban-density signatures.

This is the **most "real" image dataset** in the suite — it's actual photography, with all the noise and compositional structure of real satellite data, not a synthetic encoding. Standard image augmentations (random crops, flips) are mostly safe, since the labels are translation-invariant within a country.

## Why it rewards NAS modestly

- Off-the-shelf DenseNet already at 94.21%; NAS pushes it +1.8 pp to 96.03%. Real, but small.
- ResNet-18's 80% lag (vs 94% for DenseNet) shows ResNet-18 lacks the depth/density to capture the fine vegetation signatures. Any NAS that finds a deeper, denser-connected network wins here.
- Score slope at 80.33% is **0.51 / 1%**. A move from baseline 80% → 95% is +7.6 points. Strongly worth optimising.

## Pathologies

- **Country-label imbalance possible**: BigEarthNet patches per country are not equal in number — some European countries (Switzerland, Belgium) have small areas. The README claims 7,000 per class evenly distributed, so post-hoc rebalanced.
- The 64×64 RGB shape collides with [[CIFARTile]]'s shape — they look identical to a generic shape sniffer. The codename and metadata distinguish them.
- Possible domain-shift hazard: BigEarthNet patches are season-specific. If Phase 2/3 holdout shifts seasons, the model may underperform. Not flagged in any organiser doc — speculative.

## Implications for our pipeline

- High-leverage dataset: real benchmark, real headroom (16 pp), and the [[Scoring]] slope rewards it.
- Standard image-classification augmentation works.
- Suggested approach: use a deep DenseNet-style backbone or let NAS find one. ResNet-18 is **specifically known to underperform here** — don't default to it as a "safe baseline".
- Memory: 3×64×64 floats × 50,000 train samples = ~2.4 GB raw — biggest of the CVPR-era datasets after CIFARTile. Watch RAM in the [[Submission Contract]].
