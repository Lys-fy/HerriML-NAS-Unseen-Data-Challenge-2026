# SOTA Scan - Novel Directions (June 2026)

> **Status:** ✅ scan complete (2026-06-09, four parallel research agents, 19 paper notes written to `08 - References/papers/`) | **Topic:** What the current literature says about our findings, which of our ideas are genuinely novel, and what to steal.
> **Related:** [[Papers]] (index of the per-paper notes), [[Key Learnings]], [[TCN Backbone Excursion]], [[Next-Run Ideas - lecture-informed]], [[Probe-Driven Self-Configuration]]

## Why this scan
After the TCN excursion we had three candidate "novel logic" ideas and needed to know: published already, or open? The scan checked each against 2023 to 2026 work, plus the competition lineage's own future-work sections.

## The three novelty gaps (all three are open)

**Gap 1: search only when a runtime steepness probe says searching pays.**
Not found in the literature. Near-neighbors to cite and differentiate: [[2023 - PASHA]] adapts how much budget the search consumes from a measured ranking-stability signal, but always searches and never gives the savings to training. [[2019 - Tunability]] measures "gain from tuning over defaults" per dataset (often near zero, matching our dead-micro finding), but only offline over a benchmark collection, for HPO. [[2019 - HyperSched]] gates by pure time arithmetic, blind to landscape shape. The runtime, prior-free, search-vs-train reallocation version is unclaimed.

**Gap 2: measure the data, pick the backbone (content-based fingerprints).**
Not published as a runtime selection procedure. Nearest: [[2025 - L2M Mutual Information Scaling Law]] proves the principle (the data's mutual-information scaling dictates the architecture's state capacity) but only for text, run as analysis, needing a huge reference model; extension to vision is their declared open problem. [[2022 - DASH Efficient Architecture Search for Diverse Tasks]] agrees receptive field (kernel + dilation) is the dominant per-task axis but selects it by differentiable search, not measurement. And [[2025 - EG-ENAS]] (our exact competition) uses only num_classes and num_channels as meta-features: the content-measurement slot in this lineage is empty.

**Gap 3: adapt dilation / receptive field per dataset at runtime.**
Unclaimed between two poles: [[2022 - RF-Next]] searches per-layer dilations per task but offline with several trainings; [[2024 - Frequency-Adaptive Dilated Convolution]] adapts dilation per pixel from frequency content, segmentation only. The per-dataset, 1-hour-budget, blind version does not exist. [[2024 - UniRepLKNet]] independently legitimises our finding that one fixed wide-receptive-field ConvNet travels across modalities.

## Competition lineage: what their future-work sections say (proposal gold)
- **EGNAS (Siddiqui 2025, the 2025 winner):** future work is "joint macro architecture and training protocol search". Our v8 (aug-probe + train-to-budget around the search) is exactly that, already built and measured.
- **EG-ENAS (AutoML 2025, same challenge):** declares reliable augmentation selection for unseen data an open problem (our two-arm probe answers it), and their ablation echoes us: "search space size and augmentation selection method had the greatest impact", strategy components disappointed. Their surrogate's rank correlation collapses on Chesseract (0.2 vs ~0.85 elsewhere), independently confirming our random-search finding there.
- **Geada 2024 (organisers):** a call for generalisation machinery over benchmark wins; they repeat that random search often matched PC-DARTS/DrNAS.
- **einspace (NeurIPS 2024, organiser-adjacent):** the field's direction is expressive spaces plus cheap transferable predictors ([[2025 - Transferrable Surrogates in Expressive NAS Spaces]], which evaluates on our datasets and maxes out at Kendall tau ~0.46 cross-dataset: coarse filter, not a decision-maker).

## Ranked steal list (cheap, budget-legal, evidence-fit)
1. **TSE / SoTL-E ranking signal** ([[2021 - Training Speed Estimation (TSE)]]): rank candidates by the sum of recent training losses instead of early val-acc. Spearman ~0.85 at 5% fidelity where early val-acc gives ~0.45. Zero overhead, one-line change, directly attacks the proxy wall. Validate offline on our 13-dataset curves first.
2. **Steepness-probe budget gate (Gap 1, ours):** train 2 or 3 deliberately different archs briefly; if they tie within noise, skip search, train the default to budget. Mechanics from PASHA's rank-stability test; the reallocation is our contribution.
3. **MI-decay fingerprint (Gap 2, ours):** quantized mutual information between pixels at distance tau on raw tensors, fit the decay; power-law tail votes dilated or sequence backbone. Seconds of CPU, no training, sidesteps the proxy wall entirely.
4. **Significance-gated acceptance** ([[2024 - AdaptSH Hypothesis-Test Successive Halving]] + racing lineage): eliminate or accept only when significantly separated, otherwise keep the incumbent default. Formalizes our verification.
5. **Proxy pre-filter for the aug probe** ([[2025 - EG-ENAS]]): fisher + jacob_cov on one minibatch to shortlist transforms, then our trusted two-arm probe on the survivor.
6. **LC extrapolation as verify pre-filter** ([[2023 - LC-PFN Bayesian Learning Curve Extrapolation]], [[2024 - ifBO In-Context Freeze-Thaw BO]]): caveat, published early-stopping criteria fail on curves with LR-schedule inflection points (our shape); must validate offline before trusting.

## Offline validation results (2026-06-09, `dev/validate_offline_signals.py`)
Ran against the existing curves (3 backbones x 13 datasets, seed 42). Task: pick the right backbone from a 3/5/10-minute curve prefix. References: always-tcn2d +59.1, oracle +63.2.

| budget | earlyval | TSE | powlaw | **gated (abstain to default)** |
|---|---|---|---|---|
| 3 min | +56.8 | +59.4 | +53.3 | **+60.2** |
| 5 min | +61.3 | +59.1 | +55.9 | **+60.2** |
| 10 min | +58.7 | +55.6 | +60.7 | **+60.8** |

Three findings:
1. **The gate is the only signal that beats the fixed default at every budget** (+1.0 to +1.7), and it is the only one that never falls below it. Early val-acc swings from -2.3 to +2.2 depending on budget; the gate is stable. This validates the abstention principle: trust the leader only when the separation exceeds the measured noise, otherwise keep the default. Exactly the racing/[[2024 - AdaptSH Hypothesis-Test Successive Halving]] mechanism, and it is the load-bearing piece of the proposal story.
2. **TSE does NOT transfer to cross-family ranking** (honest negative). It degenerates to voting for whichever arch drops its training loss fastest (tcn2d almost everywhere), adding nothing over the fixed default and failing at 10 min. The published TSE results rank similar architectures within one space; comparing raw loss trajectories across different families is not sound. Keep TSE in mind for within-family decisions only (e.g., our depth search), not the backbone choice.
3. **Power-law extrapolation fails at early budgets** (-5.9 at 3 min), exactly as the LC-PFN literature warns for curves with LR-schedule inflection points (our cosine schedule). It only becomes useful at 10 min (+1.6, 8/13 top-1). Extrapolation is not a shortcut around the proxy wall in our regime.

Caveats: single seed, and the prefix curves and the final outcomes come from the same runs (noise is correlated), so live gains would be somewhat smaller. The structural result (gate never loses to the default, rankers swing) is the durable part. Top-1 accuracy is the wrong metric here; regret is what matters (GameOfLife's "wrong" picks cost 0.002).

## The proposal story this supports
A NAS pipeline that measures the dataset and its own search landscape, then spends its 1-hour budget only on decisions the evidence can support: fingerprint probes (Gap 2) pick the architecture family, a steepness probe (Gap 1) decides whether searching is worth its budget slice, and significance-gated verification protects every acceptance. Each component is individually validated against the proxy wall, and three components sit in confirmed publication gaps.
