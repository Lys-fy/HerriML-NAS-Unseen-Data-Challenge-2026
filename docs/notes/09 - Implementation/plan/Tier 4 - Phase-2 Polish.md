# Tier 4 — Phase-2 Polish

> **Status:** speculative (cannot be written in advance) | **Topic:** Data-driven refinement of [[Tier 3 - EGNAS + Reliability]] using Phase-2 leaderboard feedback. Per-codename overrides, mixed precision, SAM/SWA, anything else suggested by what surprises us.
> **Related:** [[Implementation Plan Overview]], [[Tier 3 - EGNAS + Reliability]], [[Phase 2 Submission Strategy]], [[Threats to Validity]], [[Score-Efficient Targeting]]

## Goal

Tier 4 is what happens **after Phase-2 attempt 1 or 2** when we know what the hidden datasets look like. Until then it's a placeholder — anything we plan now risks the [[Threats to Validity]] §3 trap of overfitting to the public 13.

The structural rule: Tier 4 is the *last 2 of the 7 Phase-2 attempts*. Per [[Phase 2 Submission Strategy]], attempts 6–7 are progressively *less* aggressive — Tier 4 is the iteration window where we are allowed to push, conditional on Tier 3 working.

## Candidate refinements (worth considering when Phase-2 data arrives)

### Per-codename overrides

If a hidden dataset's `codename` matches a known one (the 2025 datasets [[GameOfLife]], [[Cryptic]], [[Windspeed]] might reappear), apply pre-tuned hyperparameters:

```python
overrides = {
    "conway":      dict(D_init=4, W_init=32, max_epochs=20),
    # add more as we see leaderboard results
}
spec = overrides.get(metadata['codename'], default_spec)
```

Risk: brittle if the organisers rotate the dataset pool. Add only with strict fallback to default_spec on KeyError (already the pattern above).

### Mixed precision (`torch.amp`)

`torch.cuda.amp.autocast()` + GradScaler. Easy 1.5–2× throughput on Ampere+ GPUs.

**Blocker**: the eval-server PyTorch version is unknown. `torch.cuda.amp` is stable in PyTorch ≥ 1.10 but `torch.amp` (device-agnostic) only since 2.0. If the eval pin is old, this breaks silently.

Don't add until [[Threats to Validity|env is confirmed]] (open action in `docs/environment.md`).

### Sharpness-Aware Minimization (SAM)

[Foret et al. 2020](https://arxiv.org/abs/2010.01412). Replaces SGD with a two-step update that finds flat minima → better generalization. Costs 2× compute per step.

When it might help: **high-leverage datasets** ([[Score-Efficient Targeting]] slope > 1.0 — AddNIST, MultNIST, GeoClassing). Each percentage point of improved generalization is worth >+1 adj.

When it hurts: noise-floor datasets ([[Gutenberg]], [[Windspeed]]) where the gain is below noise.

Conditional on regime classifier: SAM only when `metadata['benchmark'] >= 85`.

### Stochastic Weight Averaging (SWA)

[Izmailov et al. 2018](https://arxiv.org/abs/1803.05407). Average model weights from the last few epochs of training. Cheap (one extra parameter copy in RAM), often +0.2–0.5 % accuracy.

Trivial to add to `Trainer.train()` — start averaging from epoch `n_total * 0.75`. Should be safe across PyTorch versions.

### Better data augmentation

Conditional on dataset class:

| Class signal | Augmentation |
|---|---|
| Standard 3-channel image, `H == W >= 32` | RandAugment / Cutout / Mixup |
| 1-channel symbolic ([[Sudoku]], [[Language]]) | None — augmentation is destructive |
| Medical ([[Myofibre]]) | RandomRotation, ColorJitter (verify orientation-safe first) |
| 3D-flavored ([[Voxel]]) | Random axis-flip if the dataset is rotation-invariant; tricky |

Implementation: a registry of augmentation policies keyed by heuristics on `(C, H, W, num_classes, benchmark)`. Default = no augmentation.

### Phase-2 attempt-N-aware reduction in NAS budget

If attempts 1–3 show NAS isn't helping on certain dataset shapes (i.e. Trainer alone matches NAS+Trainer), shift those into Regime A (15/80) regardless of benchmark. Trades NAS search for Trainer epochs.

This is the kind of refinement that can *only* be validated against real leaderboard scores — pure speculation pre-Phase-2.

## What absolutely doesn't go into Tier 4

Per [[Rules]]:
- No internet calls.
- No download attempts of the hidden datasets.
- No model >200 MB after zipping (assume informal cap).

Per [[Threats to Validity]]:
- No hyperparameter tuning *on public 13 leaderboards we made up*. The Phase-2 leaderboard is the only signal that counts.
- No per-input-shape special cases without confirming the same shape appears in the hidden set.

## Acceptance criteria

Tier 4 doesn't have fixed acceptance criteria because its content is data-dependent. The structural criterion is:

- **Tier 4 must score ≥ Tier 3 on every dataset Tier 3 already scored ≥ 0 on.**

Any Tier 4 refinement that regresses an existing dataset is reverted. The 7-attempt cap means we can't recover from a Tier-4 regression in week 4 of Phase 2.

## How Tier 4 fits the attempt cadence

From [[Phase 2 Submission Strategy]]:

| Attempt | Tier | Rationale |
|--------:|------|-----------|
| 1 | Tier 0 or Tier 3 | Smoke. Confirm package imports and runs. |
| 2 | Tier 3 | First real attempt. Most defensive working version. |
| 3 | Tier 3 + minor fix | Address attempt-2 surprises (per-dataset budget, OOM observations). |
| 4 | Tier 3 baseline | Re-validate the fixes didn't regress. |
| 5 | **Tier 4** | First ambitious push — add 1–2 refinements (e.g. SWA + targeted augmentation). |
| 6 | Tier 4 or rollback | Only ship if attempt 5 improved. Otherwise re-submit attempt-4 unchanged. |
| 7 | Conservative final | Re-submit best-confirmed-working attempt. Anchors Phase 3. |

Tier 4 is **two attempts at most**. It exists to grab leaderboard points the defensive Tier 3 leaves on the table — and is permitted to fail-soft as long as Tier 3 is the fallback.

## Repo layout

Tier 4 isn't a single baseline — it's a *set of feature flags* on top of Tier 3:

```
submission/                      # ← the live Phase-2 candidate
  (Tier 3 code with selected Tier 4 features turned on per metadata)
```

Or alternatively:

```
baselines/tier4_phase2_polish/   # checked-in variant frozen for a specific Phase-2 attempt
  ...
```

We pick a layout when we actually have data to drive Tier 4 — until then, this is a placeholder.

## Open questions

- Whether Tier 4 should live as branches/PRs against Tier 3 instead of a separate baseline folder. Probably yes — easier to bisect.
- Whether to record per-attempt git tags (`phase2-attempt-N`) so we can confidently roll back. [[Phase 2 Submission Strategy]] §"git-tag step" says yes.
