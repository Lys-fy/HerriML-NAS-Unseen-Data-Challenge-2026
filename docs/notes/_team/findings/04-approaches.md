# 04 - Approaches — Section dossier

Owner: `approaches-researcher`. All 7 notes in `04 - Approaches/` filled.

## Key facts established beyond `PRIOR_FINDINGS.md`

- **EGNAS public-code defaults differ slightly from paper**: code uses `W_max=256` and `K ∈ {3, 5}`, paper uses `W_max=128` and `K ∈ {3, 5, 7}`. Paper's "tiny / mobile" uses `W_max=64 / 128`. Resolved in EGNAS.md by listing both.
- **EGNAS strikes / tolerance**: code uses 3 consecutive failures and 0.10/0.05 thresholds for grow/prune (paper says 3 strikes, 0.10/0.05). Search inner training starts at `E_min=10` (macro) / `2` (micro).
- **EGNAS Section 4 ablation** quantifies the architecture-aware-schedule benefit: 1-epoch ranking → 0.65 Spearman vs 0.91 Spearman with the dynamic schedule. Documented in EGNAS.md as the load-bearing trick.
- **EGNAS init choice** (Table 2 of paper): `Sep + 3×3` with width 64 dominates `Conv + 3×3 / 7×7` and `Sep + 7×7` — it's not a tunable, it's the validated baseline.
- **Search cost on EGNAS paper machine** (single Quadro RTX 8000): 0.11–0.94 GPU-d depending on dataset / setting. Tiny CIFAR-10 was 0.24 GPU-d (~5.8 h); FashionMNIST tiny was 0.11 GPU-d (~2.6 h). Phase 2 budget is ~1 h per dataset — so paper defaults need trimming to fit.
- **Predictor calibration is the inferred Sichuan bottleneck**: 2.9% gap to EVAI on Cryptic (the novel dataset) vs 0.3% on GameOfLife (the conventional one) — this is exactly the failure pattern of distribution-shifted predictors. Documented in `Predictor-Guided Evolutionary.md`.
- **DARTS vs PC-DARTS vs DrNAS**: PC-DARTS = DARTS + channel sampling (memory fix). DrNAS = DARTS with Dirichlet relaxation + progressive shrinking (skip-collapse fix). PC-DARTS appears in the organisers' benchmark table → [[Language]] drops to 90.1% vs ResNet-18 97.0%, evidence DARTS's cell-only macro is wrong for some Unseen datasets.
- **Zero-cost proxy correlation drops on top-10%**: Synflow ρ ≈ 0.74 globally on NAS-Bench-201, but ρ ≈ 0.42 on the top decile. So they are filters, not final rankers. Important caveat for any pipeline that uses them.
- **Synflow has a documented param-count bias** — it scales with weight magnitude, biases toward large nets. NASWOT and Zen-NAS are data-aware alternatives.
- **Hyperband / ASHA / BOHB framing**: ASHA is the right primitive for our setting because it's parallelisable and the sync brackets of Hyperband add overhead at small candidate count.
- **EGNAS already does multi-fidelity** internally via its architecture-aware schedule — adding ASHA on top is redundant for arch ranking but useful for outer LR/WD sweeps.
- **OFA's progressive-shrinking schedule cannot run inside the per-dataset clock**, only as offline pretraining. Documented in `Weight-Sharing One-Shot.md`.

## Open questions for the team

- Per-dataset EGNAS time budget on the unknown eval GPU. If slower than the paper's RTX 8000, default `D_max=100` and 50 retrain epochs both need cutting. Strategy / Workflow team should propose concrete budgets — see [[Budget Allocation]].
- Whether channel-doubling at 1/3 and 2/3 of depth is safe for tiny inputs ([[Sudoku]] 9×9, [[GameOfLife]] 16×16). Spatial dims will collapse — needs a clamp. Datasets team should flag affected datasets.
- Whether to cache discovered architectures across datasets (skip depth-grow on later datasets). Strategy decision, not in section 04 scope.
- Whether the team has bandwidth to attempt a [[Predictor-Guided Evolutionary]] approach as a Phase 2 backup — it's the second-most-validated path (2nd and 3rd 2025) but more engineering.

## Conflicts / inconsistencies noticed

- None vs `PRIOR_FINDINGS.md`. The PF section 4 EGNAS summary is consistent with the paper. PF section 6 reports PC-DARTS scores from the Geada CVPR table — kept those numbers verbatim and only used DARTS-family theory for context.
- Paper vs public code for EGNAS hyperparameters (noted above) — flagged in EGNAS.md, paper considered authoritative, code defaults listed for reproducibility.

## Wikilinks used to other sections (consolidator: please verify these resolve)

- `[[2025 - EVAI]]` — `05 - Past Winners/2025 - EVAI.md` (used in EGNAS.md)
- `[[2025 - Sichuan teams]]` — `05 - Past Winners/2025 - Sichuan teams.md` (used in `Predictor-Guided Evolutionary.md`)
- `[[CIFARTile]]` — `03 - Datasets/CIFARTile.md`
- `[[AddNIST]]`, `[[MultNIST]]`, `[[Language]]`, `[[Sudoku]]`, `[[GameOfLife]]`, `[[Voxel]]`, `[[Myofibre]]` — `03 - Datasets/*.md`
- `[[Gutenberg]]` — `03 - Datasets/Gutenberg.md`
- `[[Phases and Timeline]]` — `01 - Competition/Phases and Timeline.md`
- `[[Scoring]]` — `01 - Competition/Scoring.md`
- `[[Clock and Time Budget]]` — `02 - Framework/Clock and Time Budget.md`
- `[[Failure Modes]]` — `02 - Framework/Failure Modes.md`
- `[[Reliability vs Peak]]` — `06 - Strategy/Reliability vs Peak.md`
- `[[Budget Allocation]]` — `06 - Strategy/Budget Allocation.md`
- `[[Score-Efficient Targeting]]` — `06 - Strategy/Score-Efficient Targeting.md`
- `[[Search vs Train Split]]` — `06 - Strategy/Search vs Train Split.md`
- `[[Threats to Validity]]` — `06 - Strategy/Threats to Validity.md`

Internal section-04 wikilinks (`[[Overview]]`, `[[EGNAS]]`, `[[DARTS Family]]`, `[[Zero-Cost Proxies]]`, `[[Multi-Fidelity]]`, `[[Weight-Sharing One-Shot]]`, `[[Predictor-Guided Evolutionary]]`) all resolve to existing files in section 04. Verified.

## Wikilink hygiene notes for the consolidator

- I avoided the ambiguous bare `[[Overview]]` for cross-section references (it would clash with section 04's own `Overview.md`). Section-03 references go via specific dataset notes (`[[CIFARTile]]`, `[[Language]]`, etc.), not via the section's overview.
- No top-level `[[Past Winners]]` index note exists in section 05 — used `[[2025 - EVAI]]` and `[[2025 - Sichuan teams]]` directly where the leaderboard is referenced.
- The `[[ASHA]]` token in early drafts was de-wikilinked since no `ASHA.md` exists; ASHA is described inside `[[Multi-Fidelity]]`.
- Internal section-04 wikilinks (`[[EGNAS]]`, `[[DARTS Family]]`, `[[Zero-Cost Proxies]]`, `[[Multi-Fidelity]]`, `[[Weight-Sharing One-Shot]]`, `[[Predictor-Guided Evolutionary]]`, plus the section's own `[[Overview]]`) all resolve to existing files.
