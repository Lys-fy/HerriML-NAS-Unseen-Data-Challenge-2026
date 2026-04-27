# Multi-Fidelity

> **Status:** current | **Topic:** Hyperband, ASHA, BOHB — bandit-style budget allocation across many candidates.
> **Related:** [[Approaches Overview]], [[Budget Allocation]], [[Zero-Cost Proxies]], [[Predictor-Guided Evolutionary]]

Spend cheap budget on many candidates; only spend expensive budget on the survivors. The competition's per-dataset clock is fixed, so this maps directly to "how many epochs do I give each architecture before deciding". Not strictly NAS — these are HPO algorithms — but they slot into any NAS pipeline that has more candidates than budget.

## Successive Halving (SHA)

The base primitive. Given budget `B`, configurations `n`, halving rate `η`:

```
1. Train all n candidates for B/n resource.
2. Keep top 1/η.
3. Train survivors for η · B/n resource.
4. Repeat until 1 survivor remains.
```

Fixed-`n` is sensitive to the choice of `n`: too small and you may discard good late-bloomers; too large and the per-candidate budget is too small to discriminate.

## Hyperband (Li et al. 2017)

Run SHA at multiple `n` values ("brackets"), picking the best winner across brackets. Hedges against the choice of `n`. Budget split is fixed by η (typical 3); bracket count is `log_η(R/r)` for max/min resources `R, r`.

Reported in HPO benchmarks: ~3× faster to a target accuracy than random search, ~1.5× faster than vanilla Bayesian opt. See [Sage benchmark](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-provides-up-to-three-times-faster-hyperparameter-tuning-with-hyperband/).

## ASHA — Asynchronous SHA (Li et al. 2018, [arXiv:1810.05934](https://arxiv.org/abs/1810.05934))

Vanilla SHA / Hyperband is synchronous: a rung waits for all candidates to finish before halving. ASHA promotes asynchronously — as soon as a candidate finishes a rung, if it's currently in the top 1/η of finishers, promote. Trivially parallelisable; near-linear scaling with workers.

This is the right multi-fidelity primitive for our environment: per-dataset clock is fixed, parallelism is unknown but possibly available, and we don't know the right `n` ahead of time. ASHA self-tunes.

## BOHB — Bayesian Opt + Hyperband (Falkner et al. 2018)

Same bracketed structure as Hyperband, but the seed configurations for each rung come from a Bayesian optimiser (TPE) fit on prior rung observations, not random sampling. Best-of-both: Hyperband's anytime efficiency + BO's final-quality exploitation.

## Mapping to this competition

The thing being multi-fidelity-tuned is **training epochs of an architecture**, not a hyperparameter sweep:

```
Resource = epochs trained.
Configuration = candidate architecture (and possibly LR, WD).
Cheap rung = 1–2 epochs.
Final rung = full retrain (e.g. 50 epochs as in [[EGNAS]] defaults).
```

Useful when paired with [[Predictor-Guided Evolutionary]] (use ASHA to score the predictor's top-k cheaply) or [[Zero-Cost Proxies]] (proxy filters 10⁴ → 10², ASHA filters 10² → 10).

## Why [[EGNAS]] doesn't need this

EGNAS already implements its own custom multi-fidelity via the architecture-aware training schedule (see [[EGNAS]]) — bigger nets get more epochs, smaller nets fewer. It's a domain-specific successive-halving that exploits the param-count → epochs-needed correlation directly. Generic Hyperband/ASHA on top of EGNAS would be redundant for arch ranking, but **could** still be used for an outer LR / WD sweep.

## When multi-fidelity actually helps us

- **Dataset where short-budget val acc tracks final acc.** AddNIST, MultNIST, GameOfLife — early signal is reliable, halving is safe.
- **Many candidates from a cheap source** (predictor or proxy). 50+ candidates is the regime where Hyperband / ASHA pay off.
- **Multiple parallel GPUs.** ASHA's main strength is parallelism. If we get a 4-GPU eval, ASHA scales nearly linearly.

## When it hurts

- **CIFARTile-class datasets where short and long acc disagree.** [[EGNAS]]'s own ablation (Sect. 4 of paper) shows 1-epoch acc has Spearman 0.65 with final — meaning ~35% of "winners" at the cheap rung are wrong. Halving aggressively will drop genuinely strong late-bloomer architectures.
- **Single-GPU sequential setup.** No parallelism advantage; ASHA degrades to SHA, and the bracket overhead is dead weight.
- **Tiny candidate set (≤ 10).** Just train them all to convergence. Successive halving on a small set is just early stopping with extra ceremony.

## Practical defaults

If we add ASHA on top of an EA / predictor:
- **η = 3** (the standard).
- **Min resource = 1 epoch** for image-class datasets, 2 epochs for noisy ones (Cryptic, Voxel).
- **Max resource = match [[EGNAS]] retrain epochs (50)** or shorter if the per-dataset budget is tight.
- **Grace period**: never promote/halve before epoch 2 — early loss is dominated by warmup noise.
- **Top-k tracking**: keep the best 3 candidates from each rung, not just 1. Cheap insurance against premature halving.

Library options: [Optuna](https://optuna.org/) (`SuccessiveHalvingPruner`, `HyperbandPruner`), [Ray Tune](https://docs.ray.io/en/latest/tune/) (`ASHAScheduler`), [SyneTune](https://github.com/awslabs/syne-tune). Standard, don't roll our own.

## Open questions

- Does the per-dataset epoch budget on Phase 2 (~1 h) leave enough room for a meaningful Hyperband bracket? With 1 GPU and 50-epoch full training, you fit maybe ~20 candidate-epochs total. ASHA-style early-rung filtering on tiny `n` is barely better than random.
- Whether the [[EGNAS]] schedule + ASHA on outer LR/WD is worth implementing vs just fixing LR/WD to paper defaults.

## See also

- [[Approaches Overview]] cross-family decision tree.
- [[Budget Allocation]] for the strategy-level dollar/epoch math.
- [[Zero-Cost Proxies]] — natural upstream filter.
- [[EGNAS]]'s built-in multi-fidelity scheme.
