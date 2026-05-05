# ProxylessNAS: Direct NAS on Target Task and Hardware

- **Authors**: Cai et al. (ICLR 2019)
- **Topic**: Direct NAS, Memory Efficiency, Hardware-aware Loss

## Core Problem
- Traditional NAS uses **proxy tasks** (e.g., small datasets or cells) which don't generalize to the final hardware/task.
- Differentiable NAS (DARTS) has memory explosion on large search spaces.

## Key Innovations
- **Path-level Binarization**: Only one path is active in memory during training, allowing search directly on ImageNet.
- **Hardware-aware Loss**: Latency is added as a differentiable objective using a lookup table.
- **Search on Target**: Optimized specifically for Mobile, CPU, or GPU without proxies.

## Outcome
- Outperformed MobileNetV2 with better accuracy/latency trade-offs.
- Proved direct search on large-scale tasks is feasible.
