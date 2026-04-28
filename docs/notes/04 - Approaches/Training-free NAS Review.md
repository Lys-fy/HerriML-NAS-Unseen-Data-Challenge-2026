# Training-Free NAS: A Review

> **Status:** Reference | **Topic:** Taxonomy, Theoretical Foundation, ZCP Overview.
> **Related:** [[Zero-Cost Proxies]], [[AZ-NAS]], [[Zen-NAS]]

This is a comprehensive review paper that categorizes the "Zoo" of training-free NAS methods.

## Main Advantage
**Decision Support:** It provides a clear taxonomy that helps us decide *which* proxy to use for which dataset. For example, it explains why data-independent proxies (Synflow) are better for high-noise data, while data-aware proxies (NAS-WOT) are better for feature-rich images.

## Core Idea
The paper divides the field into:
1.  **Initialization-based:** Evaluating the network's structural properties before any training.
2.  **Gradient-based:** Using the gradient flow to estimate trainability.
3.  **Perturbation-based:** (like Zen-NAS) using input noise to measure model capacity.

## Strategic Value
If our [[Predictor-Guided Evolutionary]] search fails, this review provides the "menu" of alternative ranking strategies we can implement in `nas.py`.
