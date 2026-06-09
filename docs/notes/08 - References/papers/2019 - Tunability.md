# 2019 - Tunability

> **Status:** reference | **Topic:** Quantifies, per algorithm and per dataset, how much performance tuning gains over good defaults, the offline ancestor of "is searching worth it" | **Found by:** SOTA scan June 2026 (budget-aware NAS angle)
> **Source:** https://jmlr.org/papers/v20/18-444.html | **Authors:** Philipp Probst et al. (Probst, Boulesteix, Bischl; JMLR 20, 2019)
> **Related:** [[Papers]], [[Multi-Fidelity]], [[Probe-Driven Self-Configuration]]

## TL;DR
Defines "tunability" as the performance gain achievable by tuning a hyperparameter (or a whole algorithm) versus running it at a good data-independent default, and measures it across 6 classic ML algorithms and 38 OpenML datasets via surrogate performance models over about 2.5 million runs. Headline finding: tunability varies enormously, glmnet and xgboost gain a lot from tuning, random forest barely gains anything over defaults. This is the principled, published version of our empirical finding that search often just returns the default.

## Key idea
Make "should we tune at all" a measurable quantity: (1) compute optimal data-independent defaults over a dataset collection, (2) define tunability of an algorithm on a dataset as best-tuned performance minus default performance, (3) aggregate over datasets to decide which algorithms and which hyperparameters justify tuning effort. The whole framework is about allocating scarce tuning budget where the expected gain is large, which is the meta-decision we want, except they answer it offline over a benchmark collection while we need it online on one hidden dataset.

## Results that matter for us
- Tunability is a property of the algorithm-dataset pair, not a universal constant: some algorithms are well served by defaults on nearly all datasets. Substitute "architecture search" for "tuning" and "default backbone" for "defaults" and this is our finding (3) with 38 datasets of evidence behind it.
- A small number of hyperparameters carry most of the tunability; the rest can be frozen. Analogy for us: most of our macro-search moves never change the outcome, so the searchable space should shrink to the moves that have ever paid off.
- Their defaults-first framing presaged later work like "Mining Robust Default Configurations for Resource-constrained AutoML" (Cai et al. 2022) and Auto-Sklearn 2.0's meta-learned policy selection: the field's answer to tight budgets is consistently "invest in strong defaults, tune only what is provably tunable".

## Relevance to our pipeline
**Medium.** Nothing here runs under a 1-hour deadline, it is a giant offline benchmark study of classic ML (binary classification, no neural nets, no NAS), so no machinery transfers directly. Its value to us is conceptual and rhetorical: it is the citation that legitimizes treating "expected gain from search over a strong default" as a first-class measurable quantity, and it confirms that this quantity is often near zero. Crucially, their measurement is retrospective across a dataset collection; nobody in this line of work estimates tunability at runtime from a cheap probe on a single unseen dataset, which is the gap our landscape-steepness probe targets.

## Future work the authors flag
From the conclusion and discussion: (a) only binary classification was studied, extension to multiclass, regression, survival analysis is open; (b) uniform random sampling of configurations does not scale to high-dimensional spaces, smarter sequential sampling is needed; (c) they learn static defaults, and defaults that depend on dataset characteristics (number of features, statistical measures) might do considerably better but need a more complex approach (pointing to van Rijn et al. 2018); (d) the method needs initial hyperparameter ranges as input. Point (c) is exactly the direction our probe-driven self-configuration pushes, taken online.

## Concrete idea to steal
Define and log a runtime "searchability" score per dataset: short-train the default backbone plus 2-3 cheap perturbations of it (the probes we already pay for), and report the spread of their validation scores relative to the noise floor (estimable PASHA-style from rank swaps). Across our dev datasets this gives us an empirical tunability table for NAS, both as the gating signal for search and as evidence for the writeup that the signal correlates with realized search gains.
