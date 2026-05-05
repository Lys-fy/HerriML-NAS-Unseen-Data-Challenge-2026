# How Powerful are Performance Predictors in NAS?

- **Authors**: White et al. (NeurIPS 2021)
- **Topic**: Performance Predictors, Benchmarking, Ensemble Methods (OMNI)

## Core Problem
- Lack of systematic comparison between the many performance prediction techniques in NAS (Model-based, Zero-cost, Learning curves, etc.).

## Key Insights
- **31 Techniques Benchmarked**: Evaluated across Model-based (GCN, Boosted Trees), Learning curve extrapolation, Zero-cost proxies, and Weight-sharing.
- **Complementary Strengths**:
    - **Zero-cost proxies**: Best for start of search (no training budget).
    - **Model-based**: Superior once a small training pool (~50-100) exists.
    - **OMNI Predictor**: A proposed ensemble that combines multiple families to outperform any single method by up to 30%.
- **Generalization**: Model-based predictors using latent features (GNNs) generalize better across complex search spaces.

## Relevance to HW-NAS
- Performance predictors are not just for accuracy; they form the basis for many hardware cost predictors (as seen in BRP-NAS or Laube et al.).
- The **OMNI** approach can be adapted to combine hardware measurements with surrogate models for better multi-objective optimization.

## Data/Library
- Integrated into **NASLib**.
