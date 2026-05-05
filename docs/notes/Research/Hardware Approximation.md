# Hardware Approximation & Cost Prediction

- **Topic**: Estimating and modeling hardware metrics (latency, energy, memory) to guide architecture search without expensive physical measurements.
- **Project**: [[HerriML NAS Unseen Data Challenge 2026]]

## Key Strategies
### 1. Lookup Tables (LUT)
- Pre-measure latency of individual operations.
- Sum them up for an end-to-end estimate.
- *Problem*: Ignores layer interactions (fusion, caching).

### 2. Surrogate Predictors
- ML models (MLPs, GCNs, GBTs) trained on measured data.
- **BRP-NAS**: Uses GCNs and binary relations (ranking).
- **Laube et al.**: Found MLPs to be the most robust baseline.

### 3. Direct/Proxyless Search
- Searching on the target task/hardware directly.
- **ProxylessNAS**: Path-level binarization for memory-efficient direct search.

## Literature
- [[Cai_2019_ProxylessNAS|ProxylessNAS: Direct NAS on Target Task and Hardware]] (Cai et al., 2019)
- [[Laube_2022_Hardware_Metric_Predictors|What to Expect of Hardware Metric Predictors in NAS]] (Laube et al., 2022)
- [[Dudziak_2020_BRP_NAS|BRP-NAS: Prediction-based NAS using GCNs]] (Dudziak et al., 2020)

## Data Sources
- **LatBench**: Comprehensive latency dataset for various devices.
