# 2023 Finalists

> **Status:** current | **Topic:** Top 5 from the CVPR-NAS 2023 Unseen Data Track. Datasets: Isabella, GeoClassing, Chesseract.
> **Related:** [[Isabella]], [[GeoClassing]], [[Chesseract]], [[Researchers and Groups]], [[2025 - EVAI]], [[2025 - Sichuan teams]]

## Context

The 2023 edition ran at the CVPR Neural Architecture Search workshop. Hidden datasets: [[Isabella]] (music spectrograms), [[GeoClassing]] (BigEarthNet country classification), [[Chesseract]] (chess-board state tensors).

This was the third CVPR-NAS Unseen Data Track (2021, 2022, 2023). The series moved to AutoML in 2024.

## Top 5 leaderboard (raw accuracies)

From the [Towers-D/NAS-Unseen-Datasets readme](https://github.com/Towers-D/NAS-Unseen-Datasets):

| Rank | GeoClassing | Isabella | Chesseract |
|-----:|------------:|---------:|-----------:|
| 1 | 92.37% | 59.97% | 58.20% |
| 2 | 89.62% | 60.14% | 58.10% |
| 3 | 93.34% | 46.87% | 59.31% |
| 4 | 90.69% | 44.93% | 58.74% |
| 5 | 87.81% | 43.92% | 63.05% |

The published table omits team names. Aojun Lu's entry on the [Sun group page](https://yn-sun.github.io/) credits him with **1st place at the CVPR 2023 NAS competition** — the most likely identification for Rank 1.

Other 2023 finalists are not publicly named (unconfirmed: contact organisers for the original press release).

## Observations

- **No dataset is saturated.** Top score on each: GeoClassing 93%, Isabella 60%, Chesseract 63%. Compare to the 2024 paper's PC-DARTS baseline: 96%, 65.8%, 60.8%. Competition methods underperformed PC-DARTS on Isabella and GeoClassing — likely a budget vs ceiling effect.
- **Ranking is not consistent across datasets.** Rank 3 was best on GeoClassing; Rank 5 was best on Chesseract; Rank 2 was best on Isabella. The total-score formula determined the final order.
- **The Towers-D readme calls these "Best competition results"**: GeoClassing 96.08%, Isabella 61.42%, Chesseract 62.98%. Slightly higher than the table top — these may include later non-finalist scores or different scoring runs.

## What 2023 tells us

- **2023 had three datasets, like 2025.** Same evaluation shape — three hidden, scored individually, summed.
- **No dataset was a free win.** Unlike [[GameOfLife]] in 2025, none of the 2023 datasets was solved to ~10/10. The benchmark structure — and the lower ceilings on Isabella and Chesseract — meant adj_score totals stayed modest.
- **Pre-trained ImageNet/CIFAR weights are useful for [[GeoClassing]].** Best CNN: 94.2%, PC-DARTS: 96%, finalists: 87–93%. Off-the-shelf transfer beat custom search there.
- **[[Chesseract]] needs domain-specific structure.** Tensor-shaped (3D one-hot) input. Off-the-shelf CNNs cap at ~60%; finalists got 58–63%. Marginal NAS gain.
- **[[Isabella]] held back from 2026.** License issues. Don't plan around it for [[Phase 2 Submission Strategy]].

## Identification gaps

The competition's archives don't publish the 2023 finalist roster by name. Known facts:

- **Aojun Lu** (Sichuan U., Sun group): 1st place at CVPR 2023 — likely Rank 1 above.
- Other 2021/2022/2023 finalists not findable via public web. CVPR-NAS Twitter (`@CVPR_NAS`) and the codalab pages no longer host detailed leaderboards.

If we want past-edition methodology details, the route is direct contact: [nas-competition-contact@newcastle.ac.uk](mailto:nas-competition-contact@newcastle.ac.uk).

## Implications for HerriML

- The 2023 edition shows that on hard, unfamiliar datasets, a ceiling exists that NAS can approach but rarely break. [[Score-Efficient Targeting]] should plan around realistic accuracy targets per dataset, not 100%.
- The Sichuan group has won this competition family at least twice (CVPR 2023, AutoML 2024); we should treat their methodology as the strong-baseline competitor alongside [[2025 - EVAI|EGNAS]].
- The lack of public finalist info means we cannot study most 2021–2023 submissions directly. Methodology must be inferred from the Geada CVPR 2024 insights paper and the group/team papers we *can* access (see [[Papers]]).
