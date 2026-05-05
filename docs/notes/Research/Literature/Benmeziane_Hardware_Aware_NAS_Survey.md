# Benmeziane Hardware-aware NAS Survey

- **Title**: A Comprehensive Survey on Hardware-aware Neural Architecture Search
- **Year**: 2021
- **Authors**: Hadjer Benmeziane, et al.

## Taxonomy
The survey categorizes HW-NAS into four dimensions:
1. **Search Space**: Macro vs. Micro (cell-based), and hardware-specific extensions (quantization/pruning).
2. **Search Strategy**: RL, Evolutionary (EA), or Differentiable (Gradient-based).
3. **Hardware Cost Estimation**:
    - Lookup Tables (LUT)
    - Predictive Models (Surrogates)
    - Direct Measurement
4. **Acceleration**: Weight Sharing, Proxy Tasks.

## Key Challenges
- **Multi-Objective Trade-offs**: Accuracy vs. Latency/Energy.
- **Computer Vision Dominance**: Lack of research in NLP/Time-series for HW-NAS.
- **Estimation Accuracy**: Predictive models are outperforming LUTs by capturing layer interactions.

## Future Directions
- **HW-SW Co-design**: Simultaneously searching for architectures and hardware configurations.
- **Beyond Accuracy**: Reliability and security metrics.
- **Generalization**: Methods that work across different hardware without retraining.
