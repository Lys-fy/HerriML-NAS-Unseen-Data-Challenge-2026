# GeNAS (Generalization-Aware NAS)

> **Status:** Research | **Topic:** Generalization Gap, Overfitting, Robustness.
> **Related:** [[EGNAS]], [[Predictor-Guided Evolutionary]]

GeNAS focuses on finding architectures that have a smaller **Generalization Gap**—the difference between training accuracy and performance on unseen test data.

## Main Advantage
**Overfitting Prevention:** Standard NAS often finds "over-parameterized" models that excel on the validation set but fail on truly unseen data. GeNAS uses specific architectural constraints to ensure the found network actually generalizes.

## Core Idea
- **Generalization Estimation:** It estimates the generalization capability based on the **curvature** of the loss landscape. 
- **Topology Bias:** It favors architectures with smoother transition boundaries, which are theoretically less prone to overfitting.

## Why it matters for "Unseen Data"
The entire competition is about generalization. An architecture that "wins" on the public datasets (Sudoku) might fail on the private evaluation datasets if it is too specialized. GeNAS principles help us pick "safer" architectures.

## Reference
- [GeNAS: Neural Architecture Search with Better Generalization]
