# Probe-Driven Self-Configuration

> **Status:** current — the submission's central thesis (adopted 2026-05-15 after the [[Tier 1 - Results|Tier 1 ablation]]) | **Topic:** Instead of copying a prior winner's NAS method, the pipeline runs cheap runtime probes to characterize the unseen dataset, then configures itself. Prior methods (EGNAS, predictor-EA, DARTS) become the *menu it selects from*, not the submission.
> **Related:** [[Implementation Plan Overview]], [[Tier 1 - Results]], [[Threats to Validity]], [[EGNAS]], [[Predictor-Guided Evolutionary]], [[Multi-Fidelity]], [[Reliability vs Peak]], [[CIFARTile]], [[Score-Efficient Targeting]]

## The problem this competition actually poses

[Geada et al.'s framing](../../08%20-%20References/Papers.md) is: you receive a dataset with **zero domain knowledge** — only `(C, H, W)`, `num_classes`, `benchmark`, `time_remaining`. The defining difficulty is *prior-free*: you cannot use what you know about the data because you know nothing about it.

Every prior approach handles this badly in a *characteristic* way:

- **Predictor-guided EA** ([[Predictor-Guided Evolutionary]], the Sichuan group) carries an accuracy predictor trained on a corpus of known datasets. On novel data its learned `arch → acc` mapping does not extrapolate — the 2025 Cryptic gap (76.1 % EVAI vs 73.2 % Sichuan) is exactly this distribution-shift failure.
- **EGNAS** ([[EGNAS]], the 2025 winner) rebuilds from scratch per dataset, which is *robust* but *blind* — it ignores every signal the dataset could give it about its own structure and just hill-climbs.
- **Metadata heuristics** (what we tried in Tier 1's `aug_policy_for`) guess dataset properties from `(C, H, W)`. We have **hard proof this fails**: see below.

So a faithful reimplementation of any of these caps our ceiling at "match the 2025 winner" — who will have improved for 2026, and whose method we'd be running strictly worse than they do.

## The thesis

> The pipeline does not commit to a NAS method in advance. It spends a small slice of the per-dataset budget running **cheap probes** that measure the unseen dataset's actual properties, then **configures itself** — choosing augmentation, regularization strength, and search strategy from what the data reveals about itself.

EGNAS, predictor-EA, DARTS, zero-cost proxies stop being "the method." They become **components a probe-driven meta-controller selects between**, per dataset, at runtime, with no transferred priors.

This is the rare direction that is simultaneously:

1. **The most novel** — systematic runtime self-characterization for prior-free NAS is under-explored; prior winners do not do it.
2. **The highest-EV competition play** — it is literally what fixes our biggest Tier 1 regression (see numbers below).
3. **The best grade story** — a thesis + the rigorous ablation methodology we already run, motivated by *our own measured failures* rather than by what someone else published.

### What this is *not* — relation to meta-learning for NAS

The nearest concept in the literature ([[Papers|White et al. 2023 survey]], Sec. 5.3 "meta-learning for NAS": Lee et al. 2021; Shala et al. 2022) is the **opposite** of this thesis. Those methods meta-train a *(architecture, dataset)-conditioned surrogate* over a corpus of known datasets and adapt it at test time. That is still a **transferred prior** — exactly the mechanism that fails on truly novel data (the 2025 Cryptic gap; the survey's own "hyperparameters do not transfer well across datasets and search spaces", Mehta et al. 2022, Sec. 9.1).

Our contribution is **prior-free**: the probes carry *no* meta-trained model and *no* corpus of prior datasets. They measure the dataset in front of them from scratch. State the novelty precisely as **"prior-free runtime self-configuration / method dispatch"** and explicitly distinguish it from meta-learned dataset-aware surrogates — that pre-empts the obvious reviewer objection ("isn't this just meta-learning for NAS?"). It is not: meta-learning transfers; we probe.

**Scoping the transfer claim.** The survey notes a genuine counter-point (Sec. 2.4): cell *topology* does transfer CIFAR→ImageNet when the distributions are similar. So our claim is **not** "all transfer fails" — it is "transferred priors fail under *distribution shift / unknown modality*", which is precisely the regime this competition targets (the Language encoded-text failure, Cryptic). Scope it that way; do not overclaim.

## Why this is grounded, not hand-waving — the Tier 1 evidence

The thesis is motivated directly by three measured failures in [[Tier 1 - Results]], not by speculation:

### 1. Metadata cannot decide augmentation — proven

Single-feature ablation, per-dataset value of augmentation (Tier 1 full minus aug-off):

| Dataset | aug value | What it means |
|---|---:|---|
| AddNIST | **+11.86** | aug is *the* thing that lifts it off the −10 floor |
| CIFARTile | +1.17 | aug helps modestly |
| Language | **−5.65** | aug is *catastrophic* — it breaks encoded-text structure |

AddNIST and Language are **indistinguishable from metadata** (both are small-ish low-channel image-shaped grids). A static rule cannot get both right. An oracle picking aug-on/off per dataset scores −0.371 on the 3-dataset subset vs Tier 1 full's −5.278 → **+4.9 adj on 3 datasets**, projected **~+30.3 total**. A 2-epoch probe *is* that oracle.

### 2. Over-regularization hurts memorization tasks — proven

GameOfLife regressed −0.78 in Tier 1 purely from universal DropPath + label smoothing on a deterministic-function task. The signal that it's a memorization task (train loss collapses instantly) is **already sitting in our logs, unused**.

### 3. Architectural ceilings are real and detectable — proven

CIFARTile stays ≤ −0.498 under *every* Tier 1 configuration. The signal that a dataset is macro-sensitive (a 2-epoch +depth/+width probe moves val_acc) is cheap to measure and tells the pipeline *when to spend search budget vs trainer budget*.

All three are the same lesson: **the unseen dataset will tell you how to configure for it if you ask it cheaply, and guessing from metadata is provably worse.**

## The probe taxonomy

Four probes, run at the start of `NAS.search()`, each ≈ 1–2 epochs on a tiny model (total cost target ≤ 3–5 % of the per-dataset clock):

| Probe | Question | Mechanism | Configures |
|---|---|---|---|
| **P1 — Aug safety** | Does augmentation help or break this task? | train 2 tiny models (aug-on / aug-off) ~2 ep, compare val_acc | augmentation policy (the proven one) |
| **P2 — Invariance** | Which symmetries hold? (h-flip, channel-perm, small translation) | apply transform to a held-out batch on a briefly-trained model; does prediction survive? | which augmentations are safe + architecture hint (translation-invariant → conv-friendly; position-critical → position-aware) |
| **P3 — Task structure** | Memorization-friendly (deterministic) or noisy/hard? | train-loss trajectory of a tiny model over ~3 ep — collapses instantly vs never | regularization strength (dial DropPath/smoothing down for memorization tasks → fixes GameOfLife) |
| **P4 — Macro sensitivity** | Does the task reward bigger architecture? | 2-ep val_acc of {small, +depth, +width} candidates | search budget allocation + whether to invoke EGNAS macro-search at all |

P1 is **Probe v1** — already justified by hard ablation data, lowest risk, ships first. P2–P4 extend the framework.

## How prior methods become components

The meta-controller's menu, selected per dataset by the probes:

```
        ┌─────────────────── probes (P1–P4) ───────────────────┐
        │  characterize the unseen dataset in ≤5% of the clock  │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
   P4 says macro-sensitive?  ──► EGNAS depth/width macro-search   ([[EGNAS]])
   P4 says macro-flat?       ──► skip search, all clock to Trainer
   large candidate pool?     ──► zero-cost proxy pre-filter        ([[Branch B1 - Zero-Cost Proxies]])
   P1 says aug-safe?         ──► full augmentation, else none
   P3 says memorization?     ──► light regularization, else heavy
   P2 finds symmetry?        ──► matched augmentation + arch bias
```

EGNAS is no longer "our method" — it is the macro-search the macro-sensitivity probe *triggers when warranted*. Predictor-EA, DARTS, SPOS become alternative search backends the controller can dispatch to. The contribution is the **controller**, not any backend.

### Additional backends surfaced by the [[Papers|White et al. 2023 survey]]

Three literature-anchored options that fit a blind 1 h budget better than DARTS and are worth slotting into the menu / a branch:

- **Local search + network morphisms** (survey Sec. 3.1, 5.4) — iteratively try 1-op / 1-edge neighbours, function-preserving morphism warm-starts (near-zero retrain). The cheapest credible macro-search and more robust than DARTS for a no-tuning budget. Strong candidate for the backend P4 triggers, alongside or instead of EGNAS.
- **Neural Ensemble Search** (Zaidi et al. 2021; multi-head variant Narayanan 2021) — explicitly improves accuracy *and robustness under dataset shift*, directly hedging the −10 clip. A cheap diverse-seed / diverse-arch ensemble of the final model. High-value, low-risk → folded into [[Tier 3 - EGNAS + Reliability|Tier 3]].
- **DASH / XD** (survey Sec. 2.3) — fix a standard backbone, search only kernel sizes / dilations via a Fourier-diagonalised mixed-op; explicitly designed as a minimal-bias single pipeline for arbitrary image tasks. The closest published algorithm to "one pipeline, unknown image task" — a comparator branch and a citation, not the thesis (it's still a single fixed method, not self-configuring).

## Revised Tier 2+ roadmap (reframed around the thesis)

| Tier | Was | Now (probe-framed) |
|---|---|---|
| **Tier 2a** | aug-probe + per-regime reg | **Probe v1**: P1 (aug-safety) + P3-lite (task-structure → regularization strength). Both proven-needed by Tier 1 data. Low risk, ships first. Target ≈ +30. |
| **Tier 2b** | EGNAS-Light | **Probe-gated EGNAS**: P4 (macro-sensitivity) decides whether to run depth-grow/width-prune at all and how much budget. EGNAS becomes a *triggered component*, not the default. |
| **Tier 3** | reliability layer | **Full probe suite + reliability**: add P2 (invariance), unify the controller, plus OOM-safe / ensemble / fallback. The complete self-configuring pipeline. |
| **Branches B1–B4** | comparators | the **menu** the controller dispatches to + honest baselines to prove the controller beats any single fixed method |

The shipping path is unchanged in *risk ordering* (cheap proven things first), but the *framing* changes: every tier is now "extend the self-configuration," not "bolt on another technique."

## Honest caveats

- **"Novel" means novel synthesis for the prior-free unseen-data setting**, not "nobody ever built a probe." Probing is multi-fidelity ([[Multi-Fidelity]]); invariance detection exists in equivariance literature. The contribution is aiming this machinery squarely at the competition's actual problem, where it is genuinely under-explored.
- **Probe signal is noisy.** The [[EGNAS]] paper measures 1-epoch val-acc at only Spearman ~0.65 with final accuracy. P1's binary aug-on/off decision is robust to this (it's a coarse comparison, and the Language/AddNIST gaps are huge — −5.65 vs +11.86, far above noise). P4's finer macro-ranking is more fragile — needs ≥2 ep and averaging, or it degrades to "bigger = better."
- **Our headline justification is currently single-seed — confirm before leaning on it.** The [[Papers|White et al. 2023 survey]] (Sec. 9.2) is emphatic that NAS has high variance and single-seed deltas are untrustworthy. The numbers this thesis rests on — Language −5.65, the per-feature ablation values, the +4.9 oracle-aug gain — are each from one seed ([[Tier 1 - Results]]). The *magnitude* of the aug effect (−5.65 vs +11.86) is far above plausible seed noise so the qualitative conclusion is safe, but the **precise deltas must be ≥3-seed confirmed** before they go in any write-up or drive a probe threshold. Prerequisite, not optional polish.
- **Probe thresholds must be frozen on held-out data, never tuned on eval data** (survey Sec. 9.1; also literally the [[Phase 2 Submission Strategy|challenge protocol]]). The aug-on/off margin, macro-sensitivity cutoff, task-structure threshold — decided once on a subset of the public 11, then frozen for every dataset including the hidden ones. Tuning a probe threshold per-dataset on eval data is the same overfitting-to-priors mistake dressed up as adaptivity.
- **Probe cost is real budget.** Target ≤ 5 % of the clock. On a 1 h Phase-2 dataset that's ≤ 3 min for all probes combined — tight but feasible with tiny models at low resolution.
- **Reliability still dominates** ([[Reliability vs Peak]]). The probe must *fail safe*: if a probe is inconclusive or errors, default to the conservative known-good config (no aug, medium reg, no macro search). The probe adds upside; it must never add a failure mode.
- **Grade vs competition.** For the grade, this thesis + our ablation rigor is a strong submission regardless of placement. For the competition, EVAI won 2025 on reliability + clean engineering with *no* novelty — so novelty is not required to place, but it is the only way to *beat* rather than match. The thesis happens to also be the highest-EV reliability play, so the two goals do not conflict.

## Open questions

- Probe budget split: fixed 5 % vs adaptive (cheaper on tiny datasets where the whole run is short anyway)?
- Can P1–P4 share a single tiny probe model trained once, reading different signals off it, instead of training one model per probe? Big budget saving if so.
- Does P2 (invariance) generalize beyond flip/permute/translate to the symmetries that actually matter on encoded datasets (e.g. Language)? Unknown — needs a Tier-3 experiment.
- Meta-controller as hand-written rules (transparent, debuggable, what we'll start with) vs a learned method-family classifier ([[Predictor-Guided Evolutionary]]-style but coarser — sidesteps the Cryptic accuracy-prediction failure). Start rules-based; revisit.

## Next action

Implement **Probe v1** (P1 aug-safety + P3-lite regularization-strength) as Tier 2a — it is cheap, the ablation already proved the P1 mechanism, and it converts the single largest Tier 1 regression (Language −5.65) into a gain. See [[Tier 2 - EGNAS-Light]] (to be reframed as "Probe-gated EGNAS") for the subsequent step.
