# Papers

> **Status:** current | **Topic:** Annotated bibliography. Local PDFs in `docs/papers/` plus key external references.
> **Related:** [[EGNAS]], [[Predictor-Guided Evolutionary]], [[DARTS Family]], [[Zero-Cost Proxies]], [[2025 - EVAI]], [[2025 - Sichuan teams]], [[Researchers and Groups]]

## Local PDFs (`docs/papers/`)

### `1st-2025_EGNAS_Siddiqui_SN-CompSci-journal.pdf`

Siddiqui, S., Kyrkou, C., Theocharides, T. *Efficient Global Neural Architecture Search.* SN Computer Science 6:282 (2025). [doi:10.1007/s42979-025-03790-z](https://doi.org/10.1007/s42979-025-03790-z). Received Mar 2024, accepted Feb 2025.

Canonical [[EGNAS]] paper — the method that won AutoML 2025 ([[2025 - EVAI]]). Proposes a sequential global NAS over a macro-micro search space (depth, width, per-layer op + kernel) trained without weight sharing or a predictor. Key trick: an architecture-aware ("variable") training schedule that scales epoch count with depth, making early hill-climbs cheap. Reports 2–4× speedup over prior global NAS, SOTA on EMNIST and KMNIST, competitive on CIFAR-10/100 and FashionMNIST, plus transfer to face recognition. Public code at [github.com/siddikui/Efficient-Macro-Micro-NAS](https://github.com/siddikui/Efficient-Macro-Micro-NAS).

### `1st-2025_EGNAS_Siddiqui_arxiv-preprint.pdf`

Same authors, same content. [arXiv:2502.03553v1](https://arxiv.org/abs/2502.03553), Feb 2025. Open-access mirror of the SN Computer Science paper. Cite the journal version when possible; use this for retrieval and figures.

### `Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf`

Geada, R., Towers, D., Forshaw, M., Atapour-Abarghouei, A., McGough, A. S. *Insights from the Use of Previously Unseen Neural Architecture Search Datasets.* CVPR 2024. [arXiv:2404.02189](https://arxiv.org/abs/2404.02189).

Authoritative reference on the **8 CVPR-era datasets**: [[AddNIST]], [[Language]], [[MultNIST]], [[CIFARTile]], [[Gutenberg]], [[Isabella]], [[GeoClassing]], [[Chesseract]]. Authored by the competition organisers ([[Researchers and Groups]]) — read it as their own perspective on what "good NAS" looks like on this problem. Contains the off-the-shelf CNN benchmark tables (ResNet-18, AlexNet, VGG16, ConvNext, MNASNet, DenseNet, ResNeXt) and PC-DARTS baselines used for adj_score formulation. Highlights [[CIFARTile]] as the canonical "search-rewarding" dataset (45% CNN → 92% PC-DARTS) and [[Gutenberg]] as effectively noise-bound (~50% ceiling). The 2024 paper is the closest thing we have to a published review of the competition family.

## Key external papers (no local copy)

### Sun group — predictor papers

These describe the methodological direction inferred for [[2025 - Sichuan teams]]'s submissions ([[Predictor-Guided Evolutionary]]).

- **DAP: Domain Adaptive Performance Predictor for Efficient NAS** — Song, X., et al. *IEEE Transactions on Computers*, 2026. Cross-dataset predictor adaptation. Likely a direct response to the Cryptic-gap signal at AutoML 2025.
- **Online Performance Predictor for Evolutionary NAS** — Song, X., et al. *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 2025. Surrogate trained during search, not pre-trained.
- **Listwise Ranking Predictor for Evolutionary NAS** — Li, N., et al. *Swarm and Evolutionary Computation*, 2025. Rank-based loss — predicts pairwise ordering, not absolute accuracy.
- **Accurate and Robust NAS via a Flexible Supernet** — Feng, Y., et al. *IEEE TNNLS*, 2025. One-shot supernet with robustness focus.

### Other Sun group / competition-adjacent

- **Revisiting Neural Networks for Continual Learning: An Architectural Perspective** — Lu, A., et al. [arXiv:2404.14829](https://arxiv.org/abs/2404.14829), 2024. Aojun Lu's continual-learning paper; not NAS Unseen, but signals the team's architectural-design instincts.

### NAS foundations

These are not local but are the reference implementations behind [[DARTS Family]] and the PC-DARTS baseline numbers in the Geada paper:

- **DARTS: Differentiable Architecture Search** — Liu, H., Simonyan, K., Yang, Y. ICLR 2019. [arXiv:1806.09055](https://arxiv.org/abs/1806.09055). Original gradient-based one-shot NAS.
- **PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search** — Xu, Y., et al. ICLR 2020. [arXiv:1907.05737](https://arxiv.org/abs/1907.05737). The PC-DARTS baseline used in the Geada paper.
- **DrNAS: Dirichlet Neural Architecture Search** — Chen, X., et al. ICLR 2021. [arXiv:2006.10355](https://arxiv.org/abs/2006.10355). Cited in Geada 2024 as one of the methods evaluated on the unseen datasets.
- **Bonsai-Net** — Geada, R., McGough, A. S. *Bonsai-Net: One-Shot Neural Architecture Search via Differentiable Pruning.* The organisers' own NAS method, used as a baseline in the Geada 2024 paper.

### Zero-cost proxies (for [[Zero-Cost Proxies]])

- **Zero-Cost Proxies for Lightweight NAS** — Abdelfattah, M. S., et al. ICLR 2021. [arXiv:2101.08134](https://arxiv.org/abs/2101.08134).
- **Neural Architecture Search without Training** — Mellor, J., Turner, J., Storkey, A., Crowley, E. J. ICML 2021. [arXiv:2006.04647](https://arxiv.org/abs/2006.04647). Crowley is also a [[Voxel]] co-author — same Newcastle/Edinburgh axis as the organisers.

## Notes on citation hygiene

- When citing EGNAS in our own writeup, prefer the SN Computer Science DOI; fall back to arXiv if a free reference is needed.
- When citing the competition methodology / dataset definitions, the Geada CVPR 2024 paper is the canonical reference.
- For the Sichuan group's approach, no single paper covers their 2025 submission. Cite the predictor-line collectively or ask the group directly for a 2026 writeup ([[Researchers and Groups]] has the contact route).
