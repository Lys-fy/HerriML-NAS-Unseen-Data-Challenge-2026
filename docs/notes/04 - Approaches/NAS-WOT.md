# NAS Without Training (NAS-WOT)

> **Status:** Reference | **Topic:** Jacobian Covariance, ReLU Activation Patterns, Zero-Shot.
> **Related:** [[Zero-Cost Proxies]], [[AZ-NAS]], [[Zen-NAS]]

NAS-WOT is one of the foundational papers for training-free NAS. It proposes a way to score architectures using a single minibatch of data and a single forward pass through a randomly initialized network.

## Main Advantage
**Simplicity & Speed:** It requires **zero labels** and **zero training**. It is the fastest known way to capture the "expressivity" of a network. In the Unseen Data Challenge, this is the perfect "first-pass" filter to eliminate thousands of useless architectures in seconds.

## Core Idea
The authors observe that the performance of a network is highly correlated with how well it can distinguish between different inputs at initialization.
- **Jacobian Covariance:** For a batch of $N$ images, it computes an $N \times N$ matrix of how similar the ReLU activation patterns are between inputs.
- **Score:** $Score = \log(\det(K))$, where $K$ is the kernel matrix.
- **Intuition:** A higher score means the network maps different inputs to more distinct activation patterns, indicating higher representational potential.

## Role in the Challenge
- Use it to rank architectures before any [[Multi-Fidelity]] training.
- Highly effective for **AddNIST** or **MultNIST** where distinguishing between image features is key.

## Reference
- [Mellor et al., ICML 2021](https://arxiv.org/abs/2006.04647)
