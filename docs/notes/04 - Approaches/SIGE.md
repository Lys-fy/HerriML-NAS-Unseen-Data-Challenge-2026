# SIGE (Sub-One-Shot NAS via Information Theory)

> **Status:** Research | **Topic:** Information Theory, Entropy-driven Pruning, Efficiency.
> **Related:** [[Weight-Sharing One-Shot]], [[Zero-Cost Proxies]]

SIGE proposes a "Sub-One-Shot" approach that uses Information Theory to efficiently prune sub-networks within a large Supernet.

## Main Advantage
**Information-Preserving Search:** Unlike standard NAS which looks purely at accuracy, SIGE evaluates how much "information" (entropy) each layer or operation contributes to the final output. This makes it extremely effective at finding **highly compressed** models that don't lose the "signal" of the unseen data.

## Core Idea
- **Mutual Information:** It calculates the mutual information between the input features and the output of a specific layer.
- **Pruning:** Operations that contribute low mutual information (redundant operations) are pruned early in the search.
- **Efficiency:** This reduces the "Supernet Bias" where the large model masks the failures of smaller sub-models.

## Application in Challenge
- Use this if we implement a [[Weight-Sharing One-Shot]] supernet. It helps in selecting sub-networks that are robust to the **Gutenberg** or **Language** datasets where the signal-to-noise ratio is low.

## Reference
- SIGE: SUB-ONE-SHOT NAS VIA INFORMATION THEORY AND...
