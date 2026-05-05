# What to Expect of Hardware Metric Predictors in NAS

- **Authors**: Laube et al. (2022)
- **Topic**: Benchmarking Hardware Predictors, MLP vs GBT

## Core Findings
- **Predictor Benchmarking**: Evaluated 18 predictor types across multiple devices.
- **MLP Dominance**: Simple Multi-Layer Perceptrons (MLPs) are generally the most robust and promising predictors for latency.
- **Inaccuracy Impact**: Quantified how predictor errors reduce the quality of selected architectures (Pareto hypervolume loss).
- **Verification Strategy**: Recommends verifying top candidates with real measurements to mitigate predictor error.

## Key Takeaway
- For limited budgets, fast predictors are often superior to slow, exact measurements, provided a verification step is included.
