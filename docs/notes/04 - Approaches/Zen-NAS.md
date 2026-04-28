# Zen-NAS

> **Status:** Research | **Topic:** Zen-Score, Gaussian Complexity, Data-Free NAS.
> **Related:** [[Zero-Cost Proxies]], [[AZ-NAS]], [[EGNAS]]

Zen-NAS introduces the **Zen-Score**, a zero-shot proxy based on the Gaussian Complexity of the network. It is unique because it measures the "representational capacity" of an architecture without requiring any training data.

## Main Advantage
**Dataset Independence:** Since the Zen-Score is calculated using Gaussian noise, it is completely immune to dataset noise or bias. It finds the "strongest" architecture purely based on its mathematical structure, making it a powerful baseline for **Completely Unseen Data**.

## How it works (The Zen-Score)
```python
1. Sample two Gaussian inputs: x and x + Δ
2. Feed both through the randomly initialized network (f).
3. Score = E[ ||f(x + Δ) - f(x)|| / ||Δ|| ] * sqrt(layer_wise_stds)
```
- It essentially measures how much the network "stretches" or "amplifies" input perturbations.
- High Score = High Model Capacity = Better potential accuracy.

## Performance in Competitions
- Reported to match or exceed **EfficientNet-B5** performance when searching on ImageNet.
- Extremely robust for **Voxel** and **3D data** because it evaluates the 3D-Convolution topology's capacity directly.

## Caveats for the Challenge
- **Memory Consumption:** Calculating Zen-Scores for very large kernels or deep models can trigger OOMs during the search phase.
- **Latency Bias:** Often selects very deep models; must be used with a strict FLOPs/Latency constraint to stay within the competition's `clock` limits.

## See also
- [Zen-NAS: A Zero-Shot NAS for High-Performance Image Recognition]
