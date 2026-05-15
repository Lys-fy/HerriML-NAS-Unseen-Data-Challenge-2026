# Tier 3 — EGNAS + Reliability Layer

> **Status:** planned | **Topic:** [[Tier 2 - EGNAS-Light]] wrapped in defensive code: per-regime budget classifier, OOM-safe batch sizing, in-search fallback, top-k ensemble, TTA. The version most likely to actually be the Phase-2 entry.
> **Related:** [[Implementation Plan Overview]], [[Tier 2 - EGNAS-Light]], [[Tier 4 - Phase-2 Polish]], [[Failure Modes]], [[Reliability vs Peak]], [[Search vs Train Split]], [[Threats to Validity]]

## Goal

Tier 2 makes EGNAS work; Tier 3 makes it **survive Phase-2's unknown conditions**. Wraps T2's search algorithm in everything the strategy notes call for under [[Failure Modes]] "Defensive patterns" and [[Reliability vs Peak]]. **No new search algorithm** — the changes are all around the search.

Splitting T2 and T3 is deliberate: a bug introduced by the reliability layer must not be confused with a search-quality regression. Each piece below is independently mock-harness-testable.

## What changes

### 1. Per-regime budget classifier

From [[Search vs Train Split]] / [[Budget Allocation]]. Drives NAS-vs-Trainer split from `metadata['benchmark']`:

```python
b = metadata['benchmark']
if b >= 85:
    nas_share, train_share = 0.15, 0.80   # Regime A: easy / near-ceiling
elif b >= 70:
    nas_share, train_share = 0.30, 0.65   # Regime D: default / unknown
elif b >= 40:
    nas_share, train_share = 0.50, 0.45   # Regime B: NAS-rewarding (CIFARTile)
else:
    nas_share, train_share = 0.10, 0.85   # Regime C: noise floor (Gutenberg, Windspeed)
buffer_share = 1 - nas_share - train_share
```

The 5 % buffer is shared overhead — `DataProcessor`, `Trainer.predict()`, CUDA warmup, grader I/O. Used inside both `NAS.search()` (caps EGNAS macro+micro time) and `Trainer.train()` (sets the SAFETY_MARGIN).

### 2. OOM-safe batch sizing

A real risk per [[Threats to Validity]] §1 — the eval-server VRAM is unknown. Implementation pattern:

```python
def run_with_oom_retry(fn, batch_size_attr, min_batch=8):
    while True:
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            current = getattr(model, batch_size_attr)
            if current <= min_batch:
                raise        # genuinely too big — let outer fallback take over
            setattr(model, batch_size_attr, current // 2)
            print(f"  OOM: halved batch to {current // 2}")
```

Applied around: per-candidate short-train (during NAS), final retrain (Trainer), and prediction (Trainer.predict — important for [[Myofibre]] at 128×128).

### 3. In-search fallback to Tier 0

Inside `NAS.search()`:

```python
try:
    arch = run_egnas(train_loader, valid_loader, metadata, clock)
    return build(arch)
except Exception as ex:
    print(f"NAS failed ({type(ex).__name__}: {ex}). Falling back to Tier-0 CNN.")
    return build_fixed_cnn(in_channels, num_classes, H, W, base_width=32, n_blocks=6)
```

Vendoring `build_fixed_cnn` into Tier-3's `helpers.py` (not importing from the Tier-0 baselines dir — the submission zip won't include `baselines/`). The fallback's only job is to be *trainable* — Trainer will still get its 65–85 % of the clock.

Same `try/except` applies around `Trainer.train()` (NaN loss / divergence → return best-checkpointed state regardless), and around `Trainer.predict()` is *not* wrapped because a failed predict can't be recovered from — but we ensure the test loader is robust to varying batch sizes.

### 4. Top-k candidate ensemble at predict time

EGNAS's macro+micro phases produce many trained candidates as a byproduct. The top-3 by val-acc become an ensemble:

```python
# inside NAS.search() — record state dicts of top-3 candidates seen
self.topk = []  # list of (val_acc, state_dict, arch_spec)
def record_candidate(val_acc, model, arch):
    if len(self.topk) < 3 or val_acc > min(x[0] for x in self.topk):
        self.topk.append((val_acc, copy.deepcopy(model.state_dict()), arch))
        self.topk.sort(reverse=True)
        self.topk = self.topk[:3]
```

At `Trainer.predict()`: build the 3 architectures, load their state dicts, average their softmax probabilities, argmax. Cost: 3× predict-time forward pass (still seconds). Reduces variance on high-leverage datasets (see [[Threats to Validity]] §4).

Optionally controlled by `metadata['benchmark']` — only ensemble when `slope = 10 / (100 - b) > 0.5`, i.e. high-leverage. On low-slope datasets (b<70), ensembling is wasted compute.

### 5. Test-time augmentation (TTA)

Horizontal-flip TTA at `Trainer.predict()`:

```python
preds_a = predict_once(model, test_loader)             # untransformed
preds_b = predict_once(model, test_loader.flipped())   # h-flipped
final = avg_softmax(preds_a, preds_b).argmax(1)
```

**Conditional**: only on datasets where flipping is label-preserving. Skip on:
- [[Language]], [[Gutenberg]] — encoded text grids, flip destroys reading order.
- [[Chesseract]] — board orientation matters (black vs white pieces).
- [[Myofibre]] — verify fibre orientation is not part of the AM/NAM label before flipping.
- Anything detected as Type-2 (`metadata['input_shape']` H == W with H ≤ 32 and ambiguous semantics).

Heuristic at runtime: enable TTA iff `H == W` and `H >= 16` and `codename` not in a known-asymmetric blacklist. Conservative by default.

### 5b. Neural Ensemble Search (robustness under dataset shift)

From the [[Papers|White et al. 2023 survey]] (Sec. 6.3): Zaidi et al. (2021) show that *searching* for a **diverse ensemble** of architectures — rather than ensembling top-k of one search — improves both accuracy **and robustness under dataset shift / OOD**, which is exactly the Phase-2/3 regime. The multi-head variant (Narayanan 2021) makes it cheap via shared-trunk heads.

Distinction from §4 (top-k ensemble): §4 averages the *k best* candidates of a single search (correlated, same inductive bias). NES deliberately selects for *diversity* (decorrelated errors), which is what actually buys robustness under shift and hedges the −10 clip.

Cheapest viable form for our budget: **diverse-seed ensemble** of the chosen architecture (3 seeds, different inits/augmentation draws), or a multi-head trunk if budget allows. Gated like §4 — only on high-leverage datasets (`slope > 0.5`) where the extra forward passes are worth it. High-value, low-risk; the survey explicitly names ensembling as the one technique it ties to shift-robustness.

> Note: §5 TTA still uses a `codename` blacklist — that's the metadata-vs-codename mistake we corrected in [[Tier 1 - Results]]. When Tier 3 is built, replace it with the metadata-only / probe-driven gate (see [[Probe-Driven Self-Configuration]]).

### 6. Deterministic seeding

```python
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Per [[Threats to Validity]] §4 — one bad seed shouldn't tank an entire submission. Costs ~5 % training throughput (cudnn benchmark off) but eliminates a class of irreproducibility bugs.

## What stays the same as Tier 2

- The macro+micro EGNAS algorithm itself.
- The architecture-aware training schedule.
- The op vocabulary `{Conv3, Conv5, SepConv3, SepConv5}`.
- The clock-aware Trainer with cosine LR.

## Acceptance criteria

- [ ] All 11 public datasets complete without `Failed=True` at `--time-limit 0.05` AND `--time-limit 1.0`.
- [ ] Per-regime budget split observable in logs (Regime A datasets like AddNIST should show ≤ 15 % time in NAS).
- [ ] Forced OOM (cap VRAM to 25 %) recovers via batch halving on at least 3 datasets.
- [ ] Forced NaN loss in NAS phase triggers the Tier-0 fallback without failing the dataset.
- [ ] Ensemble of top-3 vs single-best: variance across 3 seeds smaller for ensemble.
- [ ] TTA-enabled vs disabled: ≥ +0 adj on CIFARTile/AddNIST, ≤ 0 on Gutenberg (i.e. doesn't hurt).
- [ ] Full [[Testing Strategy]] matrix (Tiers 1–5) passes.

## Open questions

- Whether the OOM-retry should also reduce *model* size, not just batch size. A Myofibre run that OOMs at batch=8 might still OOM at batch=4 due to model VRAM. Possibly cap `base_width` based on input dims × num_classes.
- Whether to drop ensembling for [[Tier 4 - Phase-2 Polish]] if the variance reduction is not statistically significant. T3 adds it; T4 evaluates whether it's worth keeping.

## Why this is the most likely Phase-2 entry

[[Phase 2 Submission Strategy]] argues "the last working Phase-2 submission" is what auto-promotes to Phase-3. Tier 3 is engineered to maximise that "working" probability. If Tier 4's polish breaks something, Tier 3 is the rollback.

[[Reliability vs Peak]] math: `E[score] = p * s + (1-p) * (-10)`. T3's defensive layer is exactly the `p`-raising machinery. The expected delta from T2 → T3 is not +adj on any single dataset — it's reducing failure probability across the suite, which the expected-value math rewards heavily.

## Repo layout

```
baselines/tier3_egnas_reliable/
  data_processor.py     # + deterministic seeds + dataloader robustness
  nas.py                # T2 EGNAS wrapped in try/except + topk recorder + regime classifier
  trainer.py            # + OOM-retry + TTA hooks + ensemble predict
  helpers.py            # T2 helpers + build_fixed_cnn (vendored from T0) + TTA utils
```

Same 4-file submission shape. Direct `zip` of this folder → Phase-2 submission candidate.
