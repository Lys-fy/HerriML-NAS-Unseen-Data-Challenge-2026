# BRP-NAS: Prediction-based NAS using GCNs

- **Authors**: Dudziak et al. (NeurIPS 2020)
- **Topic**: Graph Convolutional Networks (GCN), Binary Relation, Ranking

## Core Problem
- Layer-wise lookup tables miss complex hardware interactions (kernel fusion, caching).
- Absolute value prediction is often unstable.

## Key Innovations
- **GCN Predictor**: Processes the architecture graph directly to capture inter-layer interactions.
- **Binary Relation Predictor (BRP)**: Instead of predicting exact latency, it predicts the *relative* performance (ranking) between two architectures.
- **LatBench**: Release of a comprehensive latency dataset for reproducibility.

## Outcome
- State-of-the-art performance on NAS-Bench-101 and 201.
- Demonstrated that ranking is more effective than regression for hardware metrics.
