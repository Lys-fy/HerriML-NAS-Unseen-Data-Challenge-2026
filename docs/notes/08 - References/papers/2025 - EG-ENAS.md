# 2025 - EG-ENAS

> **Status:** reference | **Topic:** Evolutionary NAS pipeline built for the NAS Unseen Data Challenge setting, RegNetY design space plus zero-cost-proxy runtime decisions | **Found by:** SOTA scan June 2026 (search-space angle)
> **Source:** [OpenReview 3YWElIrU8a](https://openreview.net/forum?id=3YWElIrU8a), [PMLR 293](https://proceedings.mlr.press/v293/pava25a.html) | **Authors:** Avila Pava et al. (AutoML 2025 Methods Track), FAU Erlangen-Nuernberg
> **Related:** [[Papers]], [[EGNAS]], [[Zero-Cost Proxies]], [[Predictor-Guided Evolutionary]], [[Probe-Driven Self-Configuration]]

## TL;DR
A direct competitor blueprint we had not catalogued: an evolutionary NAS pipeline evaluated on 11 datasets including all the 2023 and 2024 Unseen Challenge datasets, explicitly structured around the challenge's data/search/trainer module format. Instead of an exotic search space, they constrain the RegNetY design space (Radosavovic's "designing network design spaces" lineage) to 4 parameters, then spend their cleverness on runtime decisions: dataset-aware augmentation selection via zero-cost proxies, surrogate-ranked population initialisation, and stage-level weight transfer. Code: [github.com/ankilab/EG-ENAS](https://github.com/ankilab/EG-ENAS).

## Key idea
Pick a design space with a high density of good architectures (RegNetY, 18,564 archs standard, 58,344 extended), so even cheap or random sampling lands near good models, then make per-dataset decisions cheaply at runtime:
- Augmentation selection: rank 22 candidate transforms by the normalised sum of Fisher and Jacob_cov zero-cost proxies over 20 random models, under 8 minutes per dataset.
- Population init: a RandomForest ranking regressor (trained offline on 1,528 models, pairwise comparisons) filters 100 sampled models down to a top 20.
- Fitness evaluation: stem inheritance from parents plus a pretrained stage-weights pool, saving 5 to 10 epochs per candidate.

## Results that matter for us
- Their low-cost T0 mode (surrogate picks best of 100 sampled, no evolution at all) runs about 1 hour per dataset and is "comparable to the best competition scores in the NAS Unseen Challenge 2023", with SOTA on Sadie. T0 relative score 25.76 vs best-competition 29.02.
- Random search (T1) was the worst mode; full pipeline (T6/T7) best, but T1-T7 need 3 to 5 hours per dataset.
- Their own ablation conclusion: "search space size and augmentation selection method were the components that had the greatest impact on the total score". Strategy components (regressor init, weight transfer) helped "but not as significantly as expected".
- Surrogate ranking correlation collapses exactly on [[Chesseract]] (Spearman 0.2 vs 0.82 to 0.91 elsewhere): more evidence that Chesseract breaks learned priors, consistent with our random-search finding there.
- Their ranking regressor (RandomForest, pairwise comparisons, trained offline on 240 models per public dataset; inputs are the RegNet encoding plus dataset metadata) beat 5-epoch and even 20-epoch training rankings in Spearman correlation on most datasets. Even crude meta-features can out-rank short-training bake-offs, which is our proxy-wall finding seen from the other side.
- Their dataset meta-features are shallow (num_classes, num_channels only). Nobody in this competition lineage has used content-based data measurements (correlation structure, spectra) yet: that selection slot is empty.

## Relevance to our pipeline
High. This is the closest published thing to our regime: same datasets, similar budget, modular pipeline, and the same headline conclusion (space and training policy dominate, search strategy is second order). Their T0 mode is essentially our philosophy with a sampled RegNet instead of our fixed dilated backbone. Caveat: RegNetY is a downsampling ImageNet-style space, exactly the family our dilated backbone beat on CIFARTile-like tasks, so we should not copy the space, only the decision machinery.

## Future work the authors flag
Quoted from the conclusion: "Future work could explore other search spaces, such as MobileNetV3, for better comparison and new proxies tailored to specific architectures or search spaces. Further improvements include exploring more robust surrogate models and alternative genetic operators for RegNet models. Lastly, improving weight transfer methods between models, independent of their architectures, could improve NAS efficiency and applicability across diverse datasets and tasks." They also flag that no single augmentation works for all datasets and that reliable augmentation selection for unseen data is open.

## Concrete idea to steal
The 8-minute augmentation selector: score a shortlist of transforms with normalised Fisher + Jacob_cov on one minibatch through a handful of random tiny models (seconds each), then run our trusted two-arm training probe only on the top survivor vs no-aug. Cuts probe cost while keeping the measured binary decision we trust. Also worth pulling: their per-dataset augmentation rankings table for the public datasets, as a sanity check of our probe's outputs.
