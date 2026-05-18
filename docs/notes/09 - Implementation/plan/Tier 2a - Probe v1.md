# Tier 2a — Probe v1 (P1 aug-safety)

> **Status:** planned — design locked 2026-05-18 | **Topic:** Tier 1 + a runtime P1 aug-safety probe that decides aug-on/off per dataset, prior-free, ≤5 % clock, asymmetric keep-aug fail-safe. The first actual probe of the [[Probe-Driven Self-Configuration]] thesis.
> **Related:** [[Probe-Driven Self-Configuration]] (§P1), [[Multi-Seed - Results]] (the aug-oracle), [[E1 - Results]], [[Tier 1 - Augmentation + Op Search]], [[Reliability vs Peak]], [[Threats to Validity]], [[Run State + Handoff]]

## Goal

Recover the Language-class augmentation loss (~+6.8 adj) **without false-positiving** on the datasets where augmentation is the difference between working and a −10 floor — by *measuring* at runtime whether aug helps this specific (unseen) dataset, rather than guessing from metadata.

## The data-driven spec (from the multi-seed aug-oracle, [[Multi-Seed - Results]] §C)

P1 is a **binary, high-precision detector of one case**. The oracle (full vs aug-off, 3 seeds):

| keep aug (do NOT flip) | turn aug off |
|---|---|
| AddNIST, MultNIST (aug-off = dead −10 floor), GeoClassing +4.6, Myofibre +3.3, CIFARTile +0.7, Gutenberg ≈0 | **Language only** (aug-cost −6.82 ± 1.88, seed-stable) |

(Sudoku/GameOfLife/Chesseract/Voxel: `aug_policy_for` already returns `none` → probe is a no-op there.) The bar is **precision, not recall**: a false "turn off" on AddNIST costs ≈ −9.6; missing a Language-type costs ≈ −6.8. ⇒ default keep-aug, flip only on strong evidence.

## Mechanism — two-arm trajectory-shape race

*Why not "train 2 epochs aug-on vs aug-off, pick the higher val":* short probes are **structurally biased against aug** — aug lags early on *every* dataset — so absolute-early-val would wrongly flip almost all of them.

- Two **tiny** models (small width, downsampled input, time-boxed few epochs): arm A = Tier-1 aug-on policy, arm B = aug-off. Train on the train split, eval on valid.
- **Decision rule (asymmetric, frozen): default = KEEP aug.** Flip to aug-off **only** on strong evidence aug *breaks learning*: arm A fails to learn (val ≤ chance + δ, or non-positive val slope) **while** arm B clearly learns (val ≥ chance + Δ, with Δ ≫ δ). That is the Language signature — crop/cutout corrupt the character-position encoding so the aug model cannot fit; "aug merely lags" is **not** sufficient to flip.
- Any exception / timeout / ambiguous → keep aug (≡ Tier-1). Pure upside, never a new failure mode ([[Reliability vs Peak]]).

## Locked design decisions (2026-05-18)

1. **Built on pure Tier 1 + P1 only** — one isolated new variable (clean attribution, tiered methodology). E1 stays a separate [[Tier 3 - EGNAS + Reliability]]-gated component per its mixed verdict ([[E1 - Results]]). **P3-lite dropped** (multi-seed showed GameOfLife stable +0.91±0.50; its single-seed regression was noise).
2. **Binary all-aug vs none** — one race, full Tier-1 policy vs no aug (matches Language, where *both* crop and cutout destroy the encoding). Per-component (crop vs cutout) deferred to a v1.1.

## Contract placement — inside `DataProcessor`

Cleanest fit: `DataProcessor` already holds train/valid data + `clock` and builds the loaders; the aug decision *is* a data-processing decision. The probe slots in exactly where `policy = aug_policy_for(self.metadata)` is called (`baselines/tier1_aug_op_search/data_processor.py:61`): pick the metadata policy → if non-empty, run the race → keep or empty the policy → build the train loader. **`nas.py` and `trainer.py` stay byte-identical to Tier 1.**

Reuse: `helpers.build_cnn` (tiny config: base_width ≈16, 3–4 blocks), `AugmentedTensorDataset`/`aug_policy_for`, a minimal SGD loop modelled on `nas.py:_short_train`; downsample via `F.interpolate` to hold the probe ≤5 % clock.

## Thresholds & methodology

δ, Δ, probe epochs/resolution frozen against the multi-seed aug-oracle, **validated leave-one-out** (derive on 10 public datasets, test the held-out 1, rotate) — never tuned on eval data ([[Threats to Validity]] §3 / survey 9.1). Documented in-code as oracle-derived with the LOO check.

## Validation plan (cheap diagnostic FIRST — standing discipline)

1. **Probe-only diagnostic** (no full train, ~minutes/dataset): run just P1 across 11 datasets × 3 seeds, log decision + trajectory metrics. **Pass = flips only Language, all 3 seeds, with margin.** ~1–2 h Spark; de-risks the rule before ~30 h.
2. Only if (1) passes → full Tier 2a multi-seed sweep, paired vs the `full_s*` baseline → total ≥ Tier 1, Language recovered ~+6.8, no regression on the aug-helped 6.

## Files, hooks, success, risk

- `baselines/tier2a_probe_v1/` = copy of Tier 1; `data_processor.py` += probe; `helpers.py` += probe model/loop/decision-rule + frozen thresholds; `nas.py`/`trainer.py` unchanged.
- Env hooks: `P1_OFF=1` (≡ Tier 1), `P1_FORCE=on|off` (diagnostics). Driver `dev/run_p1_diagnostic.sh` + `dev/run_tier2a.sh`.
- **Success:** diagnostic flips only Language ×3 seeds; full sweep ≥ Tier 1 with Language recovered & no regression; ≤5 % clock; zero new failure modes.
- **Primary risk:** separating "aug breaks learning" from "aug just lags" on a tiny/short probe. Mitigation: the asymmetric high bar + the cheap diagnostic gate before any large Spark spend.
