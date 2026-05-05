# DeepCAVE: NAS & HPO Diagnostic Framework

- **Authors**: Sass et al. (ICML 2022 / JMLR 2025)
- **Topic**: Interactive Analysis, Visualization, HPI, NAS Diagnostics
- **Framework Compatibility**: SMAC3, Optuna, Ray Tune, AMLTK, BOHB.

## Core Utility for NAS
DeepCAVE is the "diagnostic layer" for the search process. It treats architectural decisions as hyperparameters, enabling deep analysis of NAS runs.

### Key Features
- **Hyperparameter Importance (HPI)**: Identify which architectural choices (e.g., cell structure vs. operations) drive performance.
- **Pareto Front Visualization**: Essential for **Multi-Objective NAS** to analyze accuracy vs. hardware cost trade-offs.
- **Budget Correlation**: Evaluate if low-fidelity search (proxies) accurately represents full-training performance.
- **Configuration Footprint**: Visualize the search space coverage and identify unexplored regions.

## Integration
- Can be used to debug search space design in the **HerriML Challenge**.
- Works post-hoc on optimization logs or in real-time.
