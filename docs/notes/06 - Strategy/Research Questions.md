> **Status:** current | **Topic:** Open questions for empirical investigation and strategy refinement.
> **Related:** [[Approaches Overview]], [[Budget Allocation]], [[Threats to Validity]]

# Research Questions

This note tracks specific technical and methodological questions that impact our implementation strategy in `nas.py` and our overall competition performance.

## Encoding and Efficiency
- **What Neural Architecture Encoding is the fastest?**
    - *Context:* We need to balance descriptive power (to capture complex macros like those in [[CIFARTile]]) with the speed of mutation/sampling during search.
    - *Goal:* Identify an encoding that minimizes overhead during the NAS phase to preserve more time for the [[Trainer]].
- **Which Neural Architecture Encoding consumes the least memory?**
    - *Context:* High-memory encodings (like full adjacency matrices in DARTS) can trigger OOM on large datasets like [[Myofibre]].
    - *Goal:* Find a memory-efficient encoding that allows for larger batch sizes or deeper architectures during the NAS phase.

## Search Strategy Baselines
- **Is random search actually worth trying when implementing weight sharing, zero cost proxies, or learning curve extrapolation?**
    - *Context:* Many papers (e.g., [[Insights from 1000 Papers]], Li & Talwalkar) suggest that Random Search is a surprisingly strong baseline.
    - *Goal:* Determine if the added complexity of Reinforcement Learning or Evolutionary Algorithms provides a statistically significant "lift" over random sampling when combined with efficient proxies.
