> **Status:** current | **Topic:** Comprehensive survey and taxonomy of NAS.
> **Related:** [[Papers]], [[Approaches Overview]]

# Neural Architecture Search: Insights from 1000 Papers
**Authors:** Colin White et al. (2023)

## Overview
This paper provides a massive survey of over 1000 NAS papers published up to 2022. It establishes NAS as the inevitable next step in AutoML, demonstrating that automatically searched architectures have consistently outperformed human-designed ones on benchmarks like ImageNet.

## Key Insights
- **Taxonomy:** The paper organizes the field into three core pillars: Search Spaces, Search Algorithms, and Acceleration Techniques.
- **Efficiency Shift:** It highlights the transition from early, computationally heavy "black-box" methods to efficient "one-shot" and weight-sharing techniques, making NAS accessible without massive compute.
- **Benchmarking:** Emphasizes the critical need for standardized benchmarks (e.g., NAS-Bench suites) to prevent unfair comparisons and ensure reproducibility.

## Relevance to Challenge
It provides a definitive "map" of the available search algorithms and acceleration proxies, which is useful when deciding which approach to implement or tune in `nas.py`.