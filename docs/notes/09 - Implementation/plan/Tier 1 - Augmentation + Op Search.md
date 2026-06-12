# Tier 1 — Augmentation + Random Op-Type Search

> **Status:** planned (re-scoped from "Random Op Search" after Tier 0 evaluation) | **Topic:** Make the per-dataset clock count by adding the regularization Tier 0 deliberately skipped, *plus* a first random-search layer over per-layer op type. Two stacked improvements: training that actually uses the budget, and search-loop infrastructure for later tiers.
> **Related:** [[Implementation Plan Overview]], [[Tier 0 - Hardcoded CNN]], [[Tier 2 - EGNAS-Light]], [[Approaches Overview]], [[Budget Allocation]], [[Multi-Fidelity]]

## Goal

Tier 0's overnight measurement exposed two problems that the original Tier 1 plan ("random op search") doesn't fix:

1. **Budget under-use.** With no augmentation / no label smoothing / no DropPath, the small CNN memorizes the train set in 20–30 epochs. Train loss → ~0.01, val plateaus, patience early-stop fires, the remaining ~95 % of the 2 h budget sits idle.
2. **Architectural ceiling on some datasets.** [[CIFARTile]] caps at ~40 % regardless of training time — a linear conv stack with global pooling can't learn the multi-instance reasoning the task requires. No amount of additional epochs fixes it; the fix is structural.

Tier 1 attacks both:

- **A — Training that uses the budget**: augmentation + regularization slow overfitting so train loss takes 100+ epochs to flatten instead of 20. The 2 h clock now fills with productive epochs.
- **B — A first search layer**: random op-type sampling validates the per-candidate timing + 30/65/5 budget split + best-tracker that [[Tier 2 - EGNAS-Light]] will inherit.

Either half alone isn't enough. A without B is "Tier 0 with regularization." B without A is "search over architectures that all overfit equally fast" — ranking them on memorization speed, not learning quality.

## Core ingredients

These are the things Tier 1 commits to shipping. Picked for low risk × high expected payoff. Ablations and stretch ideas live in [Ideas to explore](#ideas-to-explore) below.

### A. Training improvements (universal)

| Addition | Default | Why |
|---|---|---|
| **Label smoothing** | ε=0.1 | Softens hard labels; universally safe; reduces overconfidence on misclassified train samples |
| **LR warmup** | linear 0 → base over 5 epochs, then cosine | Stabilises the first updates; standard for non-trivial models |
| **DropPath (stochastic depth)** | 0.1 per ConvBlock | Another regularizer; trades convergence speed for generalization |
| **Bump base_width** | 32 → 48 | More capacity → more training needed to overfit → budget gets used. Memory hit: ~2.3× (still well under 1 GB on Spark) |

### A'. Augmentation (dataset-conditional)

A runtime classifier on `metadata` picks the augmentation policy:

```python
def aug_policy(metadata):
    C, H, W = metadata['input_shape'][-3:]
    codename = metadata['codename']
    # Encoded / symbolic — augmentation destroys structure
    if codename in {'LaMelo', 'Gutenberg', 'Chester', 'Sokoto', 'conway'}:
        return AugPolicy(flip=False, crop=False, cutout=False, mixup=False)
    # Image-shaped natural inputs
    if H == W and H >= 24:
        return AugPolicy(flip=True, crop_pad=4, cutout=8, mixup=0.2)
    # Default: conservative — no aug
    return AugPolicy(flip=False, crop=False, cutout=False, mixup=False)
```

Conservative by codename blacklist. Worth refining once we see Tier 1 ablation numbers.

| Dataset | Expected policy | Why |
|---|---|---|
| AddNIST, MultNIST, CIFARTile, GeoClassing, Myofibre, Voxel | flip + crop + cutout + mixup | square natural-image inputs |
| Language, Gutenberg | none | encoded character grids — flip scrambles reading order |
| Chesseract | none | board orientation matters; black/white asymmetry |
| Sudoku, GameOfLife | none | symbolic positional encoding; spatial transforms are meaningless |

### B. Random op-type search

Same shape as the original Tier 1 plan ([[Tier 0 - Hardcoded CNN]] foundation + a search wrapper):

- 6-block linear stack, downsampling positions from `plan_downsamples()` (inherited)
- Per-block op sampled from `{Conv3, Conv5, SepConv3, MaxPool, Identity}` — 5 ops × 6 positions = 15,625 architectures
- K = 8 candidates per dataset
- Short-train each at E = 3 epochs (now meaningful because regularization stops the model from memorizing in 1 epoch)
- Pick best by val_acc → full-train from scratch
- Budget split: 30 % search / 65 % final retrain / 5 % buffer
- `try/except` around each candidate; OOM / NaN → silently drop, continue
- Best-arch tracker across all evaluated candidates

## Acceptance criteria

- [ ] All 11 public datasets complete without `Failed=True`
- [ ] **Wall-clock budget utilization ≥ 50 %** per dataset (vs Tier 0's ~6 %) — proves regularization is biting
- [ ] **Train loss at end-of-training ≥ 0.1** on AddNIST / MultNIST (vs Tier 0's ~0.01) — confirms the model isn't memorizing
- [ ] No more `−10` floors on AddNIST / MultNIST — the LR fix + regularization should resolve both
- [ ] [[CIFARTile]] raw accuracy ≥ +5 pp over Tier 0 — Mixup specifically helps multi-instance tasks
- [ ] Score ≥ Tier 0 on at least 9/11 datasets
- [ ] Search loop: induced NaN-loss arch is dropped without failing the dataset
- [ ] `Trainer.train()` always gets ≥ 60 % of the per-dataset clock (proves the search budget split works)

## Tradeoff this tier doesn't make

Tier 1 deliberately doesn't:

- Search depth or width — that's [[Tier 2 - EGNAS-Light]]'s macro phase.
- Use any predictor / zero-cost proxy — [[Branch B1 - Zero-Cost Proxies]] / [[Branch B2 - Predictor-Guided EA]] live elsewhere.
- Hill-climb or evolve — K=8 random samples is small enough that smarter generators don't matter yet.
- Reuse weights between candidates — each short-train from scratch.
- Add the per-regime budget classifier (Regime A/B/C/D from [[Search vs Train Split]]) — that's [[Tier 3 - EGNAS + Reliability]]'s job. T1 uses the default 30/65/5.

## Ideas to explore

What follows is *not* the Tier 1 shipping list — it's the menu of things worth a small ablation (~1 h each on Sudoku / AddNIST / CIFARTile, the canonical representatives of fast / standard / hard). Categorised. Each independently mock-harness-evaluable.

### Training tricks — low risk, often +0.3 to +1.5 % acc

- **EMA of weights** (decay=0.999): shadow copy of model parameters updated each step; use for prediction. ~5 lines. Helps almost universally on small datasets. Very strong "add this first" candidate.
- **Lookahead optimizer wrapped around SGD** (k=5, α=0.5): outer "slow" steps based on inner SGD trajectory. Documented +0.5 % on CIFAR. ~30 lines.
- **`torch.channels_last` memory format** (`model.to(memory_format=torch.channels_last)` + matching input transform): free 1.3–2× speedup on Ampere+ for conv-heavy models. Probably also lifts Grace+Blackwell throughput. One-line addition; biggest "free lunch" available.
- **CosineAnnealingWarmRestarts** instead of plain cosine: cosine that restarts every T_0 epochs (doubling). Escapes local minima. Adds the T_0 knob.
- **SAM** (Sharpness-Aware Minimization, [Foret et al. 2020](https://arxiv.org/abs/2010.01412)): two-step update that finds flat minima. Costs ~2× compute per step. Often +1–2 % on high-benchmark datasets like AddNIST. Worth it when slope is steep — see [[Score-Efficient Targeting]].
- **NoiseSGD / SAM-like noise injection**: add small Gaussian noise to gradients. Cheap regularizer.

### Model tweaks (1–2× helpers.py size)

- **Residual skip connections** inside each ConvBlock: `out = relu(bn(conv(x)) + identity_or_proj(x))` instead of just `out = relu(bn(conv(x)))`. Helps gradient flow; trivial to add; fundamental architecture upgrade. **High priority.**
- **Squeeze-and-Excitation (SE) blocks** between conv blocks: adds channel attention. ~10 lines per block, often +0.5–1 %. Cheap.
- **Deeper stem**: replace the single Conv3 stem with `[Conv3(in→16) → BN → ReLU → Conv3(16→32) → BN → ReLU]`. Better feature extraction for small inputs.
- **Pre-activation order** (BN → ReLU → Conv instead of Conv → BN → ReLU): sometimes better for deeper nets.
- **Bigger base width still** (48 → 64): more capacity, more training time required. Bounded only by VRAM.

### Augmentation — data-dependent payoff

- **CutMix** instead of (or alongside) Mixup: cut a patch from one image, paste onto another, mix labels by area. Often outperforms Mixup on image classification. Stronger structural prior.
- **RandAugment**: pick N transforms with random magnitude, apply in sequence. No tuning. Standard.
- **TrivialAugment**: simpler than RandAugment — pick one transform per image. Surprisingly competitive, almost no implementation cost.
- **AugMix**: multiple augmentation pipelines mixed back into the original image. Improves OOD generalization specifically — relevant given [[Threats to Validity|hidden datasets are by design out-of-distribution]].
- **Symmetry augmentation for [[Chesseract]]**: flip board state with appropriate label-flip (white-wins ↔ black-wins). Dataset-specific; would need to be opted in.
- **Stronger Mixup α schedule**: start α=0.1, increase to 0.4 across training. Theory: lighter regularization early when the model needs to learn the easy patterns, stronger later.

### Search-loop variations

- **Successive halving inside search**: start with K=16 candidates × 1 epoch, keep top 8, train 2 more epochs, keep top 4, etc. ASHA-style. See [[Multi-Fidelity]]. Multi-fidelity within the search budget.
- **Weight sharing between candidates** (light version): share the stem across all candidates' short-train, only diverge on the inner blocks. Saves stem-warmup cost per candidate.
- **Val-loss instead of val-acc as ranking signal**: finer-grained for tied candidates, less noisy on small validation sets.
- **Top-3 ensemble at predict time**: train the top-3 candidates from search, average softmax at prediction. Cheap upgrade; reduces variance per [[Threats to Validity]] §4.
- **Multiple random initialisations per arch**: try 2 inits per candidate during short-train, pick the better-trained one. Reduces "lucky init" bias on a small K.

### Reliability borrows (small subset of [[Tier 3 - EGNAS + Reliability]])

- **NaN-loss recovery**: catch `torch.isnan(loss).any()` inside `_train_one_epoch`, restore from `best_state`, reduce LR by 5×, continue. ~10 lines. Prevents catastrophic divergence killing the dataset.
- **OOM-safe batch sizing**: try batch 256 first, halve on `torch.cuda.OutOfMemoryError`. Doesn't help on Spark (128 GB unified memory) but lets the same code run on a smaller eval server.
- **Different seed per dataset**: `seed = 42 + hash(codename) % 1000`. Reduces single-seed bias across the whole sweep. Each dataset still reproducible.

### Speculative / research-y (probably defer to Tier 2+)

- **Multi-task warmup**: pre-train the model on multiple public datasets with shared backbone + per-dataset heads, then specialise. Inputs have different shapes — would need shape-projection at the input. Complex. Possibly the closest thing to "use [[Threats to Validity|the public 13 as a prior]] without overfitting to them" we could try.
- **Self-supervised init**: train a small autoencoder on the input data for ~10 epochs, use the encoder weights as initialisation. Cheap; possibly helps on noisy datasets like Cryptic-class.
- **Distribution-aware regularization**: stronger Mixup α when `metadata['benchmark']` is low (suggests noisier task → don't memorize), weaker α when high. Driven by the same heuristic as [[Search vs Train Split]]'s regime classifier.
- **Per-layer LR scaling**: different LR per depth (e.g. 0.5× LR for layer 0, 1.0× for layer 3, 1.5× for layer 5). Recent NAS papers find this matters for deep nets.
- **Knowledge distillation from a teacher**: train one "teacher" (Tier 0 hardcoded arch, regularized) for the full clock, then train candidate architectures during search to match teacher logits + true labels. Probably too complex for T1, well-defined for T2.

## What to actually try first

Out of all the above, the priority ablations I'd run (each ~1 h on Sudoku / AddNIST / CIFARTile) before committing to a Tier 1 shipping version:

1. **EMA of weights** — almost free, almost always helps, no risk.
2. **`channels_last` memory format** — free throughput, more epochs in the same wall time.
3. **Residual skip connections in ConvBlock** — fundamental architecture fix, cheap.
4. **CutMix vs Mixup** — pick whichever wins on image-shaped datasets.
5. **NaN-loss recovery** — defensive, no acc impact, cheap insurance.

Lower priority (more code, smaller expected delta):

- SAM, Lookahead, SE blocks, deeper stem, RandAugment.

Save for [[Tier 2 - EGNAS-Light]] or later:

- Multi-task warmup, self-supervised init, knowledge distillation, per-layer LR — these all become more natural inside the EGNAS architecture-aware schedule.

## Repo layout

```
baselines/aug_op_search/
  data_processor.py      # adds aug_policy(metadata), TensorDataset → Dataset subclass
                         # with __getitem__ applying transforms on-the-fly
  nas.py                 # random op-type sampler + short-train loop + best-tracker
  trainer.py             # Tier 0 + label_smoothing + mixup-aware loss + droppath
                         # + warmup-then-cosine LR + (optional) EMA
  helpers.py             # Tier 0 helpers + op vocab + arch builder + EMA wrapper
                         # + augmentation transforms
```

Same 4-file submission shape. Runnable through `dev/mock_harness.py --submission-dir baselines/aug_op_search`.

## Open questions

- **Where does Mixup's loss live?** Inside `_train_one_epoch` or as a wrapper around the criterion? Cleaner as the latter, but Tier 0's structure assumes plain cross-entropy on `(logits, hard_labels)`. Mixup needs `(logits, soft_labels)` or `(logits, λ * y_a + (1-λ) * y_b)` style. Small refactor.
- **Which datasets does Mixup actively hurt?** Probably symbolic ones (Sudoku, Chesseract, GameOfLife) — the label arithmetic is fragile when inputs are blended. The codename blacklist handles this conservatively, but a principled metadata heuristic would be better.
- **How aggressive should DropPath be?** 0.1 is the standard starting value. With our 6-block model, even 0.1 means each block has a 10 % chance of being dropped — non-trivial. Could be 0.05 to start.
- **Augmentation policy by metadata vs codename blacklist?** Codename is reliable for the public 13 but won't transfer to hidden Phase-2 datasets. The principled answer is metadata-driven (`H == W and H >= 24` + a benchmark threshold + classes count), but harder to get right. Worth a small ablation: "policy correctly applied" vs "policy disabled" per public dataset.
- **Whether to keep K=8 or push to K=16** for the random search. Lower K is faster per dataset; higher K explores more. K=8 is conservative.

## What this tier feeds into [[Tier 2 - EGNAS-Light]]

The reusable pieces Tier 2 will import:

- The augmentation policy and dataset-conditional gate (same heuristic still applies).
- The op vocabulary and `ConvBlock` variants (Conv3 / Conv5 / SepConv3 — extended by EGNAS with kernel-5/7 variants).
- The clock-aware short-train + best-tracking loop (depth-grow and width-prune are structurally similar).
- All the training improvements (label smoothing, DropPath, EMA, etc.).
- The NaN-loss recovery if we ship it.

EGNAS's macro/micro is the structurally smarter generator — it inherits everything else.
