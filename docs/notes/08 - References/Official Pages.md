# Official Pages

> **Status:** current | **Topic:** Competition site, rules, contact, mailing list, and conference pages.
> **Related:** [[Rules]], [[Phases and Timeline]], [[Competition Overview]], [[Researchers and Groups]], [[Repositories]]

## Competition

| Resource | URL | Purpose |
|----------|-----|---------|
| Main site | https://www.nascompetition.com/ | Landing page. Rules, phase descriptions, organiser contact. Authoritative for [[Rules]] and [[Phases and Timeline]]. |
| Mailing list | https://lists.ncl.ac.uk/wws/subscribe/nas-workshops | `nas-workshops` Sympa list. Subscribe for announcements and Phase 2/3 notifications. |
| Contact email | [nas-competition-contact@newcastle.ac.uk](mailto:nas-competition-contact@newcastle.ac.uk) | Primary contact for organiser questions, broken DOIs ([[Cryptic]], [[Windspeed]]), and historical results not on the website. |

## AutoML 2026 conference

The competition is hosted at AutoML 2026, held in Ljubljana, Slovenia. Final results announced at the conference, **September 28 – October 1, 2026**.

| Resource | URL | Purpose |
|----------|-----|---------|
| AutoML 2026 conference | https://2026.automl.cc/ (unconfirmed: verify when site goes live) | Programme, registration, finalist presentation slot info. |
| AutoML conference series | https://www.automl.org/ | Parent body. Houses AutoML.org community pages and earlier proceedings. |

Finalists must present in person or virtually at the conference (see [[Rules]]).

## Past edition archives

The series moved venues over time. Past pages:

| Year | Venue | Pointer |
|------|-------|---------|
| 2021 | CVPR-NAS workshop | [Codalab competition #29853](https://competitions.codalab.org/competitions/29853) — leaderboard for 2021 Unseen Data Track. 77 participants. |
| 2022 | CVPR-NAS workshop | (unconfirmed: codalab page exists but not linked from current site) |
| 2023 | CVPR-NAS workshop | Twitter handle [@CVPR_NAS](https://twitter.com/CVPR_NAS); detailed leaderboard not archived publicly. See [[2023 Finalists]]. |
| 2024 | AutoML 2024 (Paris) | First AutoML-conference edition. [Mailing list announcement](https://groups.google.com/g/ml-news/c/qhDrHluegF4). Winners: Aojun Lu et al., Sichuan U. ([[Researchers and Groups]]). |
| 2025 | AutoML 2025 | Datasets: [[GameOfLife]], [[Cryptic]], [[Windspeed]]. Winner [[2025 - EVAI]]. |

The current `nascompetition.com` site replaced earlier per-year sites. There is no consolidated archive of leaderboards across years — historical results are scattered across Codalab, Twitter, group pages, and the Towers-D readme.

## Datasets (canonical)

CVPR-era datasets are hosted on Newcastle University's figshare:

| Dataset | DOI / page |
|---------|------------|
| [[AddNIST]] | https://data.ncl.ac.uk/articles/dataset/AddNIST_Dataset/24574354 |
| [[Language]] | https://data.ncl.ac.uk/articles/dataset/Language_Dataset/24574729 |
| [[MultNIST]] | https://data.ncl.ac.uk/articles/dataset/MultNIST_Dataset/24574678 |
| [[Gutenberg]] | https://data.ncl.ac.uk/articles/dataset/Gutenberg_Dataset/24574753 |
| [[Chesseract]] | https://data.ncl.ac.uk/articles/dataset/Chesseract_Dataset/24118743 |
| [[Voxel]] | https://data.ncl.ac.uk/articles/dataset/Voxel_Dataset/26970223 |

[[CIFARTile]], [[GeoClassing]], [[Sudoku]], [[GameOfLife]], [[Myofibre]] also on figshare — see `scripts/download_datasets.sh` for the resolved DOIs.

[[Cryptic]] and [[Windspeed]] DOIs in the starter kit resolve incorrectly. Use the contact email above to request the correct ones.

## Related conferences / venues

These are where adjacent NAS work is published — useful for tracking what the Sun group and the U. Cyprus team will publish next:

- **AutoML conference** — [automl.cc](https://2026.automl.cc/)
- **CVPR** — paper venue for the Geada 2024 insights paper
- **ICLR / ICML / NeurIPS** — standard NAS publication venues
- **IEEE Transactions** (TEVC, TSMC, TNNLS, Trans. Computers) — Yanan Sun's group's preferred outlets
