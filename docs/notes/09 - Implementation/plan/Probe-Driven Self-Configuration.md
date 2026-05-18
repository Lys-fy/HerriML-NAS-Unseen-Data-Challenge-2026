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

Probes run at the start of `NAS.search()`. They share **one tiny model trained twice in lockstep** (aug-on arm / aug-off arm), with a **dynamic racing stop** (not fixed epochs — see below). Total cost target ≤ 5 % of the per-dataset clock.

| Probe | Question | Signal read | Configures |
|---|---|---|---|
| **P1 — Aug safety** | Is augmentation *breaking* the task (not just slowing it)? | shape of aug-arm val_acc trajectory vs no-aug arm | augmentation on/off (the proven one) |
| **P2 — Invariance** | Which symmetries hold? (h-flip, channel-perm, small translation) | does prediction survive the transform on the probe model? | which augmentations are safe + architecture hint (translation-invariant → conv-friendly; position-critical → position-aware) |
| **P3 — Task structure** | Memorization-friendly (deterministic) or noisy/hard? | slope of no-aug-arm train-loss collapse | regularization strength (dial DropPath/smoothing down for memorization tasks → fixes GameOfLife) |
| **P4 — Macro sensitivity** | Does the task reward bigger architecture? | val_acc delta of {small, +depth, +width} on the probe budget | search budget allocation + whether to invoke EGNAS macro-search |

P1 + P3-lite are **Probe v1** — ship first. P2, P4 extend the framework.

### Why not fixed epochs — and the trap it hides

Fixed "~2 epochs" is wrong twice over:

1. **Wall-cost varies ~30×**: 2 epochs ≈ 2 s on Sudoku, ≈ 60 s on Myofibre. Fixed epochs blows the ≤ 5 % budget on slow datasets, wastes it on fast ones.
2. **The deeper trap — short probes are structurally biased *against* augmentation.** Augmentation *always* lags early: it removes the memorization shortcut, so the aug arm has *lower* val_acc at any short horizon even when aug is worth +11.86 long-term (directly visible in the Tier 1 curves — aug models climb for 100+ epochs). A naive "pick the higher short-horizon val_acc" probe — fixed-epoch *or* fixed-time — would flip **AddNIST to aug-off → −10**. Dynamic stopping alone does **not** fix this; it just reaches the wrong answer faster.

### P1's actual mechanism — race on trajectory *shape*, asymmetric, fail-safe

The decision is **not** "which arm is ahead?" It is **"is augmentation breaking the task or merely slowing it?"**

- **Language (aug breaks it):** aug arm flat / near-chance / capped far below — *not improving*. Encoding destroyed.
- **AddNIST (aug just lags):** aug arm rising, behind in absolute terms but clearly climbing and tracking the no-aug shape.

Both arms train in lockstep on a train subset; val-subset accuracy logged every K steps. After a short warmup, each check applies an **asymmetric** rule:

```
switch AUG OFF  only if  aug-arm slope ≈ 0 (not improving)  AND  far below no-aug arm
keep   AUG ON   if       aug-arm is rising (even if behind)        ← the default
ambiguous / probe deadline reached  →  keep AUG ON  (Tier 1 known-good)
```

Killing aug requires **positive evidence it is broken** (collapsed trajectory), never merely "it's behind." This makes AddNIST safe (rising → keep) and catches Language (flat near-chance → off). The dynamic racing stop is the *cost* layer on top of this *signal*: huge-gap datasets resolve in a fraction of an epoch; ambiguous ones run to the hard wall-time cap, then fail safe to keep-aug.

P3-lite uses the same probe model: classify memorization-friendliness from the **slope of the no-aug train-loss collapse**, not a fixed-epoch loss value — and apply the same asymmetric fail-safe (only reduce regularization on a *strong* memorization signal, because wrongly dropping reg on AddNIST re-floors it).

### Other probes worth testing (exploration menu)

Beyond P1–P4, the unseen-data setting opens more cheap runtime measurements. Ranked by (impact × cheapness × fit to our measured failures):

| Candidate probe | What it measures | Why it could matter | Cost |
|---|---|---|---|
| **LR-range probe** | short LR sweep, find divergence point (Smith range-test) | we *hardcode* lr=0.025; optimal LR is dataset-dependent and unknown on unseen data — likely the single biggest untested assumption | very low (~tens of steps) |
| **Resolution-tolerance probe** | full-res vs half-res tiny train, compare | if a dataset tolerates ½ res → 4× faster epochs → far more training in budget. Huge lever on slow datasets (Myofibre 98 % budget-bound) | low |
| **Budget-allocation probe** | extrapolate epochs-to-converge from probe learning-curve slope | a *measured* replacement for the *guessed* [[Search vs Train Split]] benchmark-regime classifier — decide NAS-vs-Trainer split from data, not from `metadata['benchmark']` | ~free (reuses probe curve) |
| **Class-imbalance probe** | label histogram of `train_y` | detect skew → weighted/focal loss; cheap insurance, no training at all | free |
| **Channel-semantics probe** | 2D-conv-on-N-channels vs 3D-conv quick train | directly attacks the [[Voxel]] 20-channel-as-spatial pathology; the hidden set may have more 3D-ish data | medium |
| **Aug-strength probe** | if aug helps, sweep light/medium/heavy | granular, not just binary on/off; squeezes more from datasets where aug helps | medium |
| **Architecture-family probe** | conv vs MLP-ish on a quick train | boldest — for truly encoded data (Language/Gutenberg) convs may be the wrong inductive bias entirely; could structurally fix the encoded-data failures | high, research-y |

**Free wins to fold in early:** class-imbalance (no training), LR-range (tiny), budget-allocation (reuses the P1 probe curve). The last one is conceptually important — it replaces a *guessed* heuristic from the strategy notes with a *measured* one, which is the whole thesis applied to budget allocation.

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
- **Probe signal is noisy.** The [[EGNAS]] paper measures 1-epoch val-acc at only Spearman ~0.65 with final accuracy. P1's binary aug-on/off decision is robust to this (it's a coarse comparison, and the aug gaps are huge *and seed-stable* — Language −6.82 ± 1.88 vs AddNIST/MultNIST aug-off pinned at the −10 floor every seed; [[Multi-Seed - Results]]). P4's finer macro-ranking is more fragile — needs ≥2 ep and averaging, or it degrades to "bigger = better."
- **Seed variance is now measured ([[Multi-Seed - Results]], 3 seeds) — and it is large.** The [[Papers|White et al. 2023 survey]] (Sec. 9.2) warned single-seed NAS deltas are untrustworthy; confirmed empirically — the Tier 1 recipe's total is **+8.4 ± 8.4 internal** (std = the mean), and the headline +24.644 was a favourable tail draw above all three re-runs. **What survived:** the aug double-edge *direction* is rock-stable (Language aug-cost −6.82 ± 1.88, negative every seed; AddNIST/MultNIST aug-off a dead −10.00 ± 0.00 floor) — P1's signal is real. **What did not:** AddNIST/MultNIST's *sign* is seed-noise (−0.4 ± 5.4 — never "fixed"), and all tier-vs-tier point totals. Design consequence, now baked in: **probe targets must be seed-stable to be probeable**; every write-up/threshold number is ≥3-seed mean ± std; the +4.9 oracle-aug and ~+30 projections are re-expressed with this spread.
- **Probe thresholds must be frozen on held-out data, never tuned on eval data** (survey Sec. 9.1; also literally the [[Phase 2 Submission Strategy|challenge protocol]]). The aug-on/off margin, macro-sensitivity cutoff, task-structure threshold — decided once on a subset of the public 11, then frozen for every dataset including the hidden ones. Tuning a probe threshold per-dataset on eval data is the same overfitting-to-priors mistake dressed up as adaptivity.
- **Probe cost is real budget.** Target ≤ 5 % of the clock. On a 1 h Phase-2 dataset that's ≤ 3 min for all probes combined — tight but feasible with tiny models at low resolution.
- **Reliability still dominates** ([[Reliability vs Peak]]). The probe must *fail safe*: if a probe is inconclusive or errors, default to the conservative known-good config (no aug, medium reg, no macro search). The probe adds upside; it must never add a failure mode.
- **Grade vs competition.** For the grade, this thesis + our ablation rigor is a strong submission regardless of placement. For the competition, EVAI won 2025 on reliability + clean engineering with *no* novelty — so novelty is not required to place, but it is the only way to *beat* rather than match. The thesis happens to also be the highest-EV reliability play, so the two goals do not conflict.

## Extension experiments (beyond probing — to run when bandwidth allows)

These are not probes; they are independent experiments/components that emerged from the variance + budget + ceiling analysis. Each is high-EV and on the exploration backlog.

### E1 — Budget-filling diverse-restart ensemble

**Motivation:** datasets early-stop using a fraction of the 2 h budget (Phase 3's budget is larger and *unspecified*); and runs are non-reproducible even at fixed seed (`cudnn.benchmark=True`), so single runs are luck-exposed. **Mechanism:** while `clock.check()` has budget, keep training additional *diverse* members (vary seed / arch / aug / LR), then **ensemble all of them** at predict (average softmax). Crucially **ensemble, do not select-best-by-val** — we have no test labels, so max-over-K by validation is winner's-curse selection bias; the test gain is far below the val spread and can backfire. Diversity is the lever: the search *already trains K=8 candidate archs and discards 7* — ensembling the top-k of those is nearly free and more decorrelated than seed-only. Clock-adaptive (reads the budget, fills it, always reserves predict + K-forward-pass time — the −10 timeout cliff is unforgiving). This is the convergence point of three threads: it fills the unused **clock**, spends the unused **VRAM** (K model copies — the concrete answer to "isn't the memory wasted"), and averages out the **variance/luck**. It is the structural form of [[Tier 3 - EGNAS + Reliability|Neural Ensemble Search]]. **Validated 2026-05-18 ([[E1 - Results]], 3 seeds) — MIXED, NOT a blanket fix:** on the high-variance target AddNIST the mechanism works as designed (σ 5.45→1.00, ×0.18, *and* mean −0.38→+2.75); CIFARTile a clean small win. **But** it does *not* collapse total submission variance (total σ only ×0.75; per-seed totals still −0.04…+12.56), it *regressed* a budget-saturating dataset (GeoClassing −0.34 — when m0 eats the clock the lone extra member is under-trained and dilutes a strong solo model), and its largest apparent gain (Language +4.74) is noise-inflated (σ ×2.78, and Language's real problem is aug damage = P1's job, not variance). ⇒ E1 is a **conditional, probe-gated reliability component for AddNIST-class high-variance datasets** with a member-quality + slack gate (Tier-3 design note) — *not* a default wrapper, *not* "empirically mandatory". The key strategic takeaway: the residual variance is **structural / dataset-level, not per-run noise ensembling can average away** — which *reinforces* this thesis (characterize the dataset and configure; you can't just ensemble harder). Still does **not** fix architectural ceilings (CIFARTile needs E2 / Tier 2).

### E2 — Non-convolutional primitive in the search space

**Motivation:** our entire ceiling is "best CNN findable" — conv inductive bias is *actively wrong* on relational (CIFARTile: compare 4 tiles) and positional/encoded (Language, Gutenberg) datasets, which is exactly where we're stuck. **Mechanism:** add a lightweight **self-attention block** as an op in the search vocabulary (hybrid conv-early / attention-late, à la CoAtNet/BoTNet — *not* a pure from-scratch ViT, which loses on small data with no pretraining and a tight clock). Folds into the architecture-family probe: conv vs MLP vs attention, decided per-dataset by measurement. This is the strongest single candidate to break the **CIFARTile** ceiling that "more conv" (even EGNAS macro-search) likely won't, because attention is built for relating distant elements. Exploration *branch*, not Tier 2 core: it enlarges the search space and attention is unstable on tiny data, so it must be probe-gated and fail-safe to conv.

### E3 — Architecture × dataset "gold" grid (the thesis validation harness)

Run a grid of **{plain-CNN, conv+Deep-Sets/relational head, conditional Conv3d, conv→attention hybrid, MLP-Mixer}** × the 11 public datasets, multi-seed, at the 2 h budget → the **oracle**: the best architecture per dataset. Then characterise which *cheap signals* (input shape, channel count, benchmark, learning-curve shape, detected symmetry, overfit speed) predict the oracle's choice, validated **leave-one-out** (derive the rule on 10 datasets, test on the held-out 1, rotate).

Critical framing: the deliverable is **not** a shippable "features → architecture" predictor — that *is* the meta-learning approach that cost the Sun group Cryptic ([[Predictor-Guided Evolutionary]]). The grid's job is to be the **oracle the runtime probe is validated against**: "does the probe's cheap measured signal track the oracle's choice?" The probe stays prior-free (measures the actual dataset); the grid only tells us whether the probe's signal is *sufficient*. This experiment is effectively the **validation harness for the entire thesis** — it proves or kills the probe approach before Phase 2, and should be prioritised accordingly.

### Architecture-primitive menu (search-space / head options the arch-family probe selects from)

No rulebook restriction exists ([[Rules]]: no size/FLOPs/family cap — any `nn.Module` from `NAS.search()` is legal). Ranked by fit-to-measured-failures × cheapness × small-data safety:

| Primitive | Targets | Why | Catch |
|---|---|---|---|
| **Deep-Sets / permutation-invariant head** | **CIFARTile** (the ceiling), AddNIST/MultNIST | CIFARTile is *literally* a symmetric set-function (count distinct over unordered tiles) — the *correct* bias, not just different. Cheap (a head). **The standout.** | needs input segmentable into "elements" |
| **conv backbone + relational head (hybrid)** | CIFARTile, Chesseract | practical synthesis: conv (small-data-safe) + small permutation-inv/attention head for the relational step | best single concrete bet |
| **Conv3d (conditional)** | **Voxel** | exact structural match (Voxel is true 20³ volume, we fake it as 20-ch 2D) | only helps truly-3D → needs detection signal |
| **DASH-style dilated/multi-scale search** | unknown Phase-2/3 generalist | minimal-bias, designed for "one pipeline, unknown modality"; survey-anchored | fixed method, not self-configuring → branch |
| **MLP-Mixer / gMLP** | **Language, Gutenberg** | no locality bias → right for encoded-positional data where conv bias is *actively wrong* | data-hungry; small-data instability |
| self-attention op (E2) | CIFARTile, encoded | relates distant elements; probe-gated, fail-safe to conv | search-space size, instability |
| group-equivariant CNN | any dataset P2 finds symmetry on | principled form of "augment for invariance" | heavy/slow, exotic for budget |

All probe-gated, all fail-safe to plain conv (the small-data-safe default). The point is *not* "transformers/GNNs are better" — by default they are not — it is "we can detect *when* conv is the wrong inductive bias, and a CNN-only field structurally cannot."

## Questions we want answered (consolidated)

The experiments above (multi-seed run, E1–E3, probes P1–P4) exist to answer these. Roughly in priority order:

1. **Is the cheap probe signal sufficient?** Does a ≤5 %-budget probe make the aug-on/off (P1) and arch-family decisions the E3 *oracle* would — without false-flagging (AddNIST stays aug-on)? If no, the whole thesis fails and we fall back to EGNAS-copy.
2. **How large is run-to-run variance (even at fixed seed, `cudnn.benchmark=True`)?** → **ANSWERED ([[Multi-Seed - Results]]):** Tier 1 total +8.4 ± 8.4 internal (std = mean). Above the noise floor = the aug *direction* (Language −6.82 ± 1.88; aug-off −10 floor on AddNIST/MultNIST) and the stable wins (GeoClassing +7.68 ± 0.29, Sudoku +5.94 ± 0.34). Within it = AddNIST/MultNIST *sign* (±5.4 / ±3.0) and every tier-vs-tier point total. Partially answers Q1/Q6.
3. **What is the per-dataset gold architecture, and what cheap signal predicts it (leave-one-out)?** → E3.
4. **Does budget-filling diverse-restart *ensembling* (not selection) convert unused clock+VRAM into robust gain?** How much, and does it saturate by ~3–5 members? → E1.
5. **Can a non-conv primitive break the CIFARTile ceiling** where EGNAS macro-search (more conv) cannot? → E2/E3, Deep-Sets head first.
6. **Does P3-lite recover GameOfLife without re-flooring AddNIST?** (the dangerous false-positive — over-reg-detection safety).
7. **What is the optimal model size *per unit wall-clock* per dataset type?** (capacity–budget frontier, run at simulated VRAM caps so it also informs the tiny-GPU degradation ladder).
8. **Is the search itself variance-amplifying?** → **ANSWERED — yes.** Tier-1 op-search (Step 0/Q8): instability universal but not the *downstream* driver. **Tier 2b EGNAS-macro ([[Tier 2b - Results]]): definitively yes** — it picks a different depth/width per seed on a cuDNN-noisy short-train val; on s2024 it chose D=5 for AddNIST → AddNIST collapsed to −2.81. The architecture-aware schedule's paper Spearman-0.91 does **not** hold at our budget. ⇒ arch selection must be stabilised (average the short-train signal over ≥2 restarts / robust pick) before EGNAS can ship — the active next action.

## Open questions

- Probe budget split: fixed 5 % vs adaptive (cheaper on tiny datasets where the whole run is short anyway)?
- Can P1–P4 share a single tiny probe model trained once, reading different signals off it, instead of training one model per probe? Big budget saving if so.
- Does P2 (invariance) generalize beyond flip/permute/translate to the symmetries that actually matter on encoded datasets (e.g. Language)? Unknown — needs a Tier-3 experiment.
- Meta-controller as hand-written rules (transparent, debuggable, what we'll start with) vs a learned method-family classifier ([[Predictor-Guided Evolutionary]]-style but coarser — sidesteps the Cryptic accuracy-prediction failure). Start rules-based; revisit.

## Next action

Implement **Probe v1** (P1 aug-safety + P3-lite regularization-strength) as Tier 2a — it is cheap, the ablation already proved the P1 mechanism, and it converts the single largest Tier 1 regression (Language −5.65) into a gain. See [[Tier 2 - EGNAS-Light]] (to be reframed as "Probe-gated EGNAS") for the subsequent step.
