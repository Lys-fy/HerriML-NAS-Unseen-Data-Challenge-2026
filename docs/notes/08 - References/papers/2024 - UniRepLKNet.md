# 2024 - UniRepLKNet

> **Status:** reference | **Topic:** Design guidelines for large-kernel (large receptive field) ConvNets, one fixed architecture across modalities | **Found by:** SOTA scan June 2026 (search-space angle)
> **Source:** [arXiv:2311.15599](https://arxiv.org/abs/2311.15599) | **Authors:** Ding et al. (CVPR 2024)
> **Related:** [[Papers]], [[EGNAS]]

## TL;DR
The RepLKNet authors' follow-up: four architectural guidelines for large-kernel ConvNets, built around the observation that large kernels "see wide without going deep". The same fixed architecture, with only modality-specific preprocessing, gets strong results on image (88.0% ImageNet, 55.6 mIoU ADE20K, 56.4 box AP COCO), audio, video, point cloud, and time-series tasks. No per-task architecture search at all.

## Key idea
Large kernels decouple receptive field from depth: a few large-kernel layers provide the RF, and depth is then spent on capacity (small-kernel blocks, more channels), not on accumulating RF. Their Dilated Reparam Block trains a large kernel together with parallel small dilated kernels and merges them into one kernel at inference, so the large-RF branch costs nothing at deploy time. The cross-modality result is the headline: a single well-designed wide-RF backbone generalises to unseen modalities better than per-modality designs.

## Results that matter for us
- A fixed wide-RF conv backbone is competitive or SOTA across five modalities with zero architecture adaptation: the strongest published version of our own "fixed dilated backbone beats searched conv pipeline" finding, and very on-message for a competition about unseen data.
- Their guidelines say where the wide RF should live (a few dedicated layers) and where it should not (do not make every layer large-kernel; it wastes compute and hurts local detail). Our uniform exponential-dilation stack may be over-spending on RF in early layers.
- Dilated Reparam: dilated convs are algebraically equivalent to sparse large kernels, so dilation branches can be reparameterised away after training.

## Relevance to our pipeline
Medium-high. It does not address search or budget at all (these are big ImageNet-scale models), so nothing transfers directly at our 1-hour scale. But it is the best citation for the design philosophy our empirical finding points to, and the guidelines are testable on our backbone: concentrate large dilation in a few mid/late blocks, keep other blocks local, and add a parallel undilated 3x3 branch (reparam-merged or just summed) so local detail is not sacrificed, which is plausibly why our dilated net still loses a little on purely local tasks.

## Future work the authors flag
Scaling large-kernel ConvNets further and broader multimodal use; they position large-kernel design as an underexplored alternative to transformers. Nothing on search or per-dataset adaptation.

## Concrete idea to steal
The Dilated Reparam Block, simplified: in each residual block, sum a dilated 3x3 and an undilated 3x3 branch during training (cheap, a few percent extra FLOPs at our scale), optionally merge at inference. Gives every block both local and long-range paths without choosing, which hedges the blind-runtime decision of how much RF a hidden dataset needs.
