# AZ-NAS (Assembling Zero-Cost Proxies)

> **Status:** Research | **Topic:** Ensemble Proxies, Robust Ranking, Single-Batch Evaluation.
> **Related:** [[Zero-Cost Proxies]], [[Predictor-Guided Evolutionary]], [[EGNAS]]

AZ-NAS addresses the instability of individual zero-cost proxies by assembling them into a robust ensemble score. It is particularly designed for **Unseen Data Challenge** scenarios where a single proxy might fail on a novel dataset distribution.

## Main Advantage
**Robustness through Consensus:** By combining scores for *Expressivity*, *Progressivity*, and *Trainability*, AZ-NAS significantly reduces the variance in architecture ranking, making it less likely to pick "fake winners" that only score well on one specific metric (like Synflow's bias towards larger models).

## Core Idea
Instead of relying on a single signal, AZ-NAS calculates an ensemble score across four pillars:
1.  **Linear Regions (NAS-WOT):** Captures the expressivity and input-separation capability.
2.  **Synaptic Flow (Synflow):** Ensures the topology supports gradient flow without layer collapse.
3.  **Fisher Information:** Estimates the importance of parameters for training.
4.  **Gradient Norm:** Measures the signal strength during a backward pass.

## Workflow in Pipeline
1.  **Generate** $10^4$ candidates from the Macro-Search Space.
2.  **Calculate** the AZ-Score (weighted sum or rank-average of individual ZCPs).
3.  **Prune** the search space to the Top-100 candidates.
4.  **Hand-off** to a [[Predictor-Guided Evolutionary]] search for fine-grained selection.

## Implementation Notes
- **Normalization is key:** Since different ZCPs have different scales, rank-averaging is often more stable than raw weighted sums.
- **Hardware-Aware:** Can be combined with a latency lookup table to ensure the ensemble winner fits the competition time budget.

## See also
- [AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search (2024)]
