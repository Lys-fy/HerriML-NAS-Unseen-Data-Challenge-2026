# Architecture-Aware Minimization (A2M)

> **Status:** Research | **Topic:** Flat Minima, Sharpness-Aware, Robustness.
> **Related:** [[GeNAS]], [[Trainer.py]]

A2M introduces the concept of **Sharpness-Aware** search. It argues that architectures with "flat" loss minima generalize much better than those with "sharp" minima.

## Main Advantage
**Robustness to Weight Noise:** An architecture found via A2M is much less sensitive to the specific weights found during training. This makes the [[Trainer.py]] stage much more reliable, as the model will perform well even if the training budget is too short to find the absolute global minimum.

## Core Idea
- **Flatness as a Score:** It evaluates the sensitivity of the output to small perturbations in the weights.
- **Co-Optimization:** It optimizes the architecture and the sharpness-aware training objective simultaneously.

## Application in Challenge
- This is a "Train-time" insight. We can incorporate A2M principles into our `Trainer.train()` function to make our final models more robust to the "Unseen" evaluation sets.

## Reference
- [Architecture-Aware Minimization (A2M).pdf]
