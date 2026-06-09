# 2025 - Transferrable Surrogates in Expressive NAS Spaces
> **Status:** reference | **Topic:** surrogate models (graph features, fine-tuned LMs) that predict architecture performance across datasets, evaluated on the Unseen NAS Challenge datasets | **Found by:** SOTA scan June 2026 (dataset-fingerprint angle)
> **Source:** https://arxiv.org/abs/2504.12971 | **Authors:** Shiwen Qin et al. (AutoML Conference 2025, einspace group)
> **Related:** [[Papers]], [[Probe-Driven Self-Configuration]], [[Past Winners]]

## TL;DR
In the einspace grammar-based search space, the authors train surrogates that predict architecture accuracy: GRAF-style neural graph features plus zero-cost proxies, or an off-the-shelf LM (ModernBERT-large) fine-tuned on string representations of architectures. Surrogates trained on CIFAR10 plus the eight Unseen NAS datasets (AddNIST, Language, MultNIST, CIFARTile, Gutenberg, Isabella, GeoClassing, Chesseract) transfer to held-out datasets and speed up evolutionary search by filtering bad candidates.

## Key idea
Instead of ranking candidates by training them (the proxy wall) or by raw zero-cost proxies, learn a reusable mapping from architecture description to performance using all available offline runs, then carry it to new datasets. Meta-corpus: about 9,000 architecture-accuracy pairs on CIFAR10 plus about 2,000 per unseen dataset. At search time the surrogate either filters candidates before real evaluation or serves directly as the search objective.

## Results that matter for us
- Cross-dataset transfer is real but modest: ModernBERT trained on CIFAR10 gets average Kendall tau 0.466 on unseen tasks; leave-one-out reaches 0.658 on Gutenberg and 0.625 on MultNIST but only 0.186 on Isabella. So transfer fails exactly on the weirdest dataset, mirroring our finding that odd datasets (Chesseract-like) break general rules.
- The surrogate is architecture-conditioned but barely dataset-conditioned: it learns "what is a good architecture on average for this dataset family" from the identity of the training dataset, with no content-based dataset fingerprint. The dataset-measurement input slot is empty, which is our angle.
- They confirm surrogate filtering beats raw zero-cost proxies in expressive spaces, consistent with our observation that zero-cost proxies degenerate to parameter counting.

## Relevance to our pipeline
Medium-high. Same datasets, same lineage as the challenge organizers, and the strongest published evidence on what surrogate transfer can and cannot do with a meta-corpus of roughly our size (9 datasets, ours is 13). The Kendall tau around 0.4-0.5 ceiling says: usable as a filter for bad candidates, not reliable for picking between similar good ones. That matches our proxy-wall finding and argues for using any meta-learned ranking only as a coarse pre-filter ahead of a measured probe.

## Future work the authors flag
Limitations: surrogate-guided search does not outperform baseline search on all unseen data (named a challenge for future research); only image classification accuracy, no multi-objective or hardware-aware surrogates. Future work: better performance predictors combined with LLM advancements, custom evolutionary operators.

## Concrete idea to steal
Precompute offline a frozen ranking of our backbone-family templates per public dataset (from our existing runs), train a tiny pairwise ranker mapping (dataset meta-features, family id) to win-probability, and at runtime use it only to order which families our probes test first. Inference is milliseconds; even tau around 0.45 is enough to order three or four probes well. Their numbers also justify hard-capping trust: never let the ranker override a measured probe result.
