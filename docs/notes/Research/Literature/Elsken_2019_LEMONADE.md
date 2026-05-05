# LEMONADE: Efficient Multi-Objective NAS

- **Paper**: *Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution* (ICLR 2019)
- **Authors**: Elsken, Metzen, Hutter, Lindauer
- **Topic**: Multi-Objective NAS, Evolutionary Algorithms, Function Preserving Morphisms

## Core Problem
- NAS is typically slow and often optimizes only for a single objective (accuracy), ignoring hardware constraints.

## Key Innovation
- **Lamarckian Evolution**: Uses network morphisms to generate child architectures that inherit weights from parents.
- **Warm-starting**: Children don't need to be trained from scratch, resulting in a **20-40x speedup**.
- **Cheap vs. Expensive Objectives**: Uses "cheap" metrics (FLOPs, Param count) to filter architectures before evaluating "expensive" metrics (Validation Accuracy).
- **Pareto Front**: Maintains a set of non-dominated architectures for different hardware/performance trade-offs.

## Relevance to HW-NAS
- Directly optimizes for multiple hardware constraints using an efficient evolutionary approach.
- The use of "cheap" proxies for hardware cost is a foundational pattern in HW-NAS.
