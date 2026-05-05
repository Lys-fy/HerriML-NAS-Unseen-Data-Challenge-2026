# Hardware-aware NAS (HW-NAS)

- **Topic**: Designing neural architectures optimized for specific hardware constraints (latency, energy, memory).
- **Project**: [[HerriML NAS Unseen Data Challenge 2026]]

## Key Literature
- **"A Comprehensive Survey on Hardware-aware Neural Architecture Search"** (2021) by *Hadjer Benmeziane et al.*
    - [[Benmeziane_Hardware_Aware_NAS_Survey|Survey Summary]]
    - Taxonomy: Search Space, Search Strategy, Hardware Cost Estimation, and Acceleration.

## Core Concepts
### Constrained NAS
- Searching for architectures that satisfy strict hard constraints (e.g., Latency < 50ms).
- Approaches often involve penalty functions in the reward or loss.

### Multi-Objective NAS (MO-NAS)
- Treating architecture search as a Pareto optimization problem.
- Result: A set of Pareto-optimal architectures for different deployment scenarios.

## Hardware Cost Estimation
- **Lookup Tables (LUT)**: Fast but ignores layer interaction.
- **Surrogate Predictors**: ML models trained to predict hardware metrics (e.g., BRP-NAS).
- **On-device Measurement**: Most accurate, high overhead.


### Hardware Approximation
- [[Hardware Approximation]] (Predicting Cost vs. Measuring)
- [[Cai_2019_ProxylessNAS|ProxylessNAS]] (Direct search on target hardware)
- [[Laube_2022_Hardware_Metric_Predictors|Hardware Metric Predictors in NAS]] (Predictor benchmarking)
- [[Dudziak_2020_BRP_NAS|BRP-NAS]] (GCN-based ranking predictors)

## Competition Relevance
- Optimization for resource-constrained environments (Unseen Hardware/Data).
