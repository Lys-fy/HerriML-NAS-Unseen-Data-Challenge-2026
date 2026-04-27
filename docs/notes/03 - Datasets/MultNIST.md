# MultNIST

> **Status:** drafting | **Topic:** Three MNIST images stacked as RGB; label = (r × g × b) mod 10, 10 classes.
> **Related:** [[Datasets Overview]], [[AddNIST]]

Sister dataset to [[AddNIST]]: same channel-stacked MNIST construction, different arithmetic. Label is `l = (r * g * b) mod 10`.

| Field | Value |
|---|---|
| Codename | Mateo |
| Era | CVPR (2021–23) |
| Shape | 3×28×28 (channels-first) |
| Train / val / test | 50,000 / 10,000 / 10,000 |
| Classes | 10 (labels 0–9) |
| Benchmark (ResNet-18) | 90.87% (`metadata`) — Geada Tab. 1 reports 91.55% |
| Best off-the-shelf CNN (Geada) | AlexNet 94.01% |
| Best NAS method (Geada Tab. 2) | DrNAS 98.10% |
| Best 2023-comp submission | 95.45% |
| License | CC BY 4.0 |
| DOI | https://data.ncl.ac.uk/articles/dataset/MultNIST_Dataset/24574678 |

## Type and difficulty

**Type-1** (Geada): same human-solvable structure as [[AddNIST]] — recognise three digits, multiply, mod 10. The mod-10 step removes the magnitude bias that AddNIST has (in AddNIST, larger digits inflate the label; in MultNIST, the mod folds them).

NAS reward over ResNet-18 is +6.6 pp (DrNAS), the highest gain among the three CVPR Type-1 NIST-flavoured datasets — multiplication likely demands slightly more representational capacity than addition.

## Class distribution quirk

The mod-10 multiplication produces a **massively skewed** label distribution: any digit-zero in any channel zeros the product, so label 0 is over-represented relative to the other classes. Even labels (2, 4, 6, 8) and 5 are produced in many ways; odd primes (especially 1, 3, 7, 9) are rarer. The README claims "7,000 examples of each, distributed evenly between the three subsets" — this suggests the dataset was **subsampled to balance** classes rather than raw-generated. Worth verifying empirically before training.

## Pathologies

- ConvNext underperforms badly (64.20% — better than AddNIST's 38% but still poor).
- Random search in DARTS space hits 96.55%, very close to PC-DARTS — most of the gain is from the search-space, not the strategy. See [[DARTS Family]].

## Implications for our pipeline

- Very similar to [[AddNIST]]: high benchmark (~90%), high scoring slope (+1.0 / 1%), modest absolute headroom.
- Same recipe as [[AddNIST]]: short, well-trained CNN. Consider the same data pipeline — they share `(N, 3, 28, 28)` shape and class-balance flavour.
- Don't waste search budget here unless the pipeline already handles AddNIST well.
