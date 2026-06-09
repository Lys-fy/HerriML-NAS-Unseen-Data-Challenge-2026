# 2022 - RF-Next

> **Status:** reference | **Topic:** Cheap search over per-layer dilation rates to match receptive field to the task | **Found by:** SOTA scan June 2026 (search-space angle)
> **Source:** [arXiv:2206.06637](https://arxiv.org/abs/2206.06637) | **Authors:** Gao et al. (TPAMI 2022, extends CVPR 2021 "Global2Local")
> **Related:** [[Papers]], [[EGNAS]]

## TL;DR
The published version of exactly our question: what receptive field does this task need, and can we find it cheaply? RF-Next replaces convs with "RFConv" and searches per-layer dilation rates via a global-to-local scheme: coarse sparse sampling of dilation combinations first, then an expectation-guided iterative local search to refine. Found rates beat human-designed exponential patterns on temporal action segmentation (TCN-style models, our backbone family), object detection, instance segmentation, and speech synthesis. Code: [github.com/ShangHua-Gao/RFNext](https://github.com/ShangHua-Gao/RFNext).

## Key idea
Receptive field requirements are task-dependent: large RF for long-range relations, small RF for local detail, and the right answer is a per-layer combination, not a global knob. The search space of dilation combinations is huge, so they (a) sample it sparsely at the global level (uniform, gradually sparse, gradually dense schedules), then (b) run a local search where a layer's dilation is treated as an expectation over a small set of nearby rates, iteratively collapsing to a single rate. The whole thing is a plug-in: wrap existing convs, search, retrain.

## Results that matter for us
- Consistent gains over hand-designed dilation patterns on MS-TCN-style temporal action segmentation, which is literally the dilated residual no-downsampling family our fixed backbone comes from. Their searched models beat the standard exponential 1,2,4,8,... schedule.
- Works across four very different domains with the same recipe, supporting the claim that RF matching, not architecture novelty, is what transfers.
- Cost: the global stage trains sampled candidates and the local stage is roughly a few model trainings, cheap for NAS but still likely several times one training run. In our 1-hour regime we could only afford a heavily truncated version.

## Relevance to our pipeline
High, with a budget asterisk. Our loudest finding (fixed exponential-dilation backbone wins by +11 points, CIFARTile 48% to 72%) is a special case of their thesis: the conv pipeline lost because its receptive field was wrong, not because search was bad. RF-Next says the exponential schedule we hard-coded is itself probably suboptimal per dataset, and gives a principled way to adapt it. The full method does not fit 1 hour, but the local-search stage alone (start from our exponential schedule, locally perturb dilations with expectation over neighbouring rates) might.

## Future work the authors flag
The paper frames receptive field search as a missing dimension in NAS and notes existing NAS spaces rarely include dilation; they point to integrating RF search into general NAS spaces and more tasks. No dataset-level runtime adaptation is proposed: search is offline per task, then frozen.

## Concrete idea to steal
A one-shot dilation probe: for each residual block, parameterise dilation as a softmax-weighted expectation over {previous, half, double} rates, train the probe net briefly, read off the argmax per layer, then train the final backbone with the chosen schedule. This is their local search collapsed into a single cheap pass, and it directly addresses datasets like [[CIFARTile]] (needs huge RF) vs [[MultNIST]] (local digits).
