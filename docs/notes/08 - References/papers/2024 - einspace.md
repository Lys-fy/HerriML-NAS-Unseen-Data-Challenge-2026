# 2024 - einspace

> **Status:** reference | **Topic:** Expressive PCFG-based NAS search space, evaluated on the same 8 unseen datasets we compete on | **Found by:** SOTA scan June 2026 (search-space angle)
> **Source:** [arXiv:2405.20838](https://arxiv.org/abs/2405.20838) | **Authors:** Ericsson et al. (NeurIPS 2024)
> **Related:** [[Papers]], [[EGNAS]], [[Researchers and Groups]]

## TL;DR
Edinburgh group (Crowley, Storkey, Ericsson) builds a maximally expressive search space from a parameterised probabilistic context-free grammar over four fundamental operation types (branching, aggregation, routing, computation). The space contains ResNets, ViTs, and MLP-Mixers as points. They evaluate exclusively on the 8 Unseen NAS Challenge datasets ([[AddNIST]], [[Language]], [[MultNIST]], [[CIFARTile]], [[Gutenberg]], [[Isabella]], [[GeoClassing]], [[Chesseract]]). Note: Ericsson and Crowley are now co-authors on the 2024 competition datasets (Sudoku, Voxel), so this is effectively the organiser axis describing where they think the field should go.

## Key idea
Stop searching inside narrow cell spaces. Define a grammar whose derivations are whole architectures, then run regularised evolution over derivation trees. Crucially, search does not have to start from scratch: an existing strong architecture (ResNet18) can be expressed in the grammar and used as the initial individual, with evolution mutating subtrees around it.

## Results that matter for us
- Regularised evolution from scratch: 74.72% average over the 8 datasets. With ResNet18 initialisation: 78.17%, improving on ResNet18 on every single dataset. Mixed-baseline initialisation: 77.56%, best on non-vision tasks (Language 97.92% vs 92.16% for ResNet18).
- In their expressive space, random search clearly loses to evolution. Their framing: restrictive spaces are why random search looks competitive in the literature; once the space is expressive, strategy matters again.
- Budget caveat: 1000 evolution iterations (population 100), far beyond our 1-hour-per-dataset regime. Their own limitations section admits the search is slow.

## Relevance to our pipeline
High. Two reasons. First, it is the strongest published support for our "search space and block family matter more than strategy" finding, on exactly our datasets, from the organisers' own circle. Second, their best recipe (start from a strong known backbone, mutate locally) is structurally identical to what we found works: a fixed dilated backbone beats searching from scratch. The difference is they still searched; we froze. Their per-dataset wins from initialised evolution suggest a middle path: our backbone plus a few cheap local mutations.

## Future work the authors flag
"Finding more efficient search strategies for expressive spaces like ours is an important direction for future work." They explicitly call for intelligent (cheaper than evolution) search inside expressive spaces. The follow-up is already out: Transferrable Surrogates in Expressive NAS Spaces (Qin et al., AutoML 2025, [arXiv:2504.12971](https://arxiv.org/abs/2504.12971)) trains GRAF-feature and LM surrogates that transfer across datasets to filter weak candidates in einspace.

## Concrete idea to steal
Treat our dilated backbone as the einspace-style "initial individual": spend leftover budget on 3 to 5 cheap local mutations (swap an op, change a kernel or dilation, add a branch), early-reject with a few-epoch probe, keep the winner. This is initialised local search, not open search, so it fits the 1-hour budget.
