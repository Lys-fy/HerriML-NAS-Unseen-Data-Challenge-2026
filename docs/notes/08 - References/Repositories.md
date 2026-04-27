# Repositories

> **Status:** current | **Topic:** GitHub repos for the competition, prior winners, and related tooling.
> **Related:** [[Papers]], [[Official Pages]], [[2025 - EVAI]], [[EGNAS]], [[Submission Contract]], [[Mock Harness]]

## Competition-official

| Repo | Purpose |
|------|---------|
| [Towers-D/NAS-Comp-Starter-Kit](https://github.com/Towers-D/NAS-Comp-Starter-Kit) | Official starter kit. Contains `evaluation/main.py`, `evaluation/score.py`, the 4-file submission template, an example submission, and the Makefile. Pin: `torch==1.10.1+cu113` (stale — see [[Mock Harness]]). |
| [Towers-D/NAS-Unseen-Datasets](https://github.com/Towers-D/NAS-Unseen-Datasets) | Organisers' canonical descriptions of the **8 CVPR-era datasets**: [[AddNIST]], [[Language]], [[MultNIST]], [[CIFARTile]], [[Gutenberg]], [[Isabella]], [[GeoClassing]], [[Chesseract]]. Includes the [[2023 Finalists]] table. |
| [NewcastleRSE/CVPR-NAS-API](https://github.com/NewcastleRSE/CVPR-NAS-API) | Internal API used to run prior CVPR-NAS competitions. Useful as a reference for the original evaluation harness, but **not** what the 2026 grader runs — the AutoML 2026 grader is the starter-kit `main.py`. |

## Past winners' code

| Repo | Purpose |
|------|---------|
| [siddikui/Efficient-Macro-Micro-NAS](https://github.com/siddikui/Efficient-Macro-Micro-NAS) | Official [[EGNAS]] implementation — the [[2025 - EVAI]] winning method. Includes the depth/width/op search loops and the architecture-aware training schedule. Direct dependency for our own EGNAS baseline. |
| [yn-sun.github.io](https://yn-sun.github.io/) (group page, not a repo) | Yanan Sun's group landing page — see [[Researchers and Groups]]. The group does not publicly release a single "competition-winning" repo for AutoML 2025; predictor implementations live across individual papers' artefacts. |

## Predecessor / mirror repos

These are older codalab / CVPR-era utilities that may help us understand prior submissions:

| Repo | Purpose |
|------|---------|
| [SangbumChoi/CVPR-NAS-2021-Unseen-Data-Track](https://github.com/SangbumChoi/CVPR-NAS-2021-Unseen-Data-Track) | A 2021 participant's repo. Contains a walkthrough notebook and an example submission for the original codalab evaluation. Useful for archaeology, not direct reuse. |
| [RobGeada/cvpr-nas-datasets](https://github.com/RobGeada/cvpr-nas-datasets) | Rob Geada's earlier dataset distribution repo — pre-Towers-D. Likely superseded by `NAS-Unseen-Datasets`. |

## Internal repo (this team)

| Repo | Purpose |
|------|---------|
| `Lys-fy/HerriML-NAS-Unseen-Data-Challenge-2026` (private) | This repo. Contains [[Submission Contract|the 4-file submission scaffold]] under `submission/`, [[Mock Harness|local grader mirror]] under `dev/`, dataset downloader at `scripts/download_datasets.sh`, and the Obsidian notes vault under `docs/notes/`. Members: Luis Melzer, Sebastian Seifert, Marc Leander von Gottberg. |

## NAS framework reference (not directly used)

For comparison or borrowed components:

| Repo | Purpose |
|------|---------|
| [quark0/darts](https://github.com/quark0/darts) | Original DARTS implementation — reference for [[DARTS Family]]. |
| [yuhuixu1993/PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS) | PC-DARTS — the baseline used in the Geada 2024 paper. |
| [SamsungLabs/zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas) | Zero-cost proxy implementations — see [[Zero-Cost Proxies]]. |

## Stale or broken

- The starter kit `requirements.txt` pins `torch==1.10.1+cu113` and the deprecated `sklearn` package name (last valid 2021). Treat as a hint, not a target — see [[Mock Harness]] for the actual versions we run locally.
- Cryptic and Windspeed DOIs in the starter kit (`10.7488/ds/8054`, `10.7488/ds/8053`) resolve to wrong content (Tower Blocks UK photos). Tracked: ask organisers.
