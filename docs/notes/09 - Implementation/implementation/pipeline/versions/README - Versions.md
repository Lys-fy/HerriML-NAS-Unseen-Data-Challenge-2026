# README - Versions

> **Status:** index | **Topic:** A one-note-per-version layer over the pipeline history. Each note synthesizes what was done / why / what was tested / results / verdict for a single version, and links back to the detailed deep-dive sources.
> **Related:** [[Pipeline v1-v7 - Aug Probe]] (v1-v9 deep-dive), [[Pipeline v10 - Design + Idea Ledger]] (v10 + v11 deep-dive), [[README - Map and Lineage]], [[Key Learnings]], [[Run State + Handoff]]

The deep-dive sources stay authoritative; these notes are the navigable summary + index. v1-v7 are the aug-probe era (the probe counter, built inside what was Tier 2a); from v8 the counter refers to the whole pipeline. The TCN excursion and Language deep dive are side studies, not version notes (see [[TCN Backbone Excursion]], [[Language Deep Dive]]).

![[v10_vs_egnas_aggregate_curve.png]]
> *Where the version line landed: v10 (green) vs EGNAS-full (blue, the 2025-winner reimplementation = our method baseline), total adjusted score over training, 3-seed mean +/- std. Final test: v10 +62.2 vs EGNAS +35.5 (+26.7). Each version note carries its own result curves.*

## The version line, one hook each

- **[[v1]]** - aug-probe raw-val rule; failed smoke (aug-off over-fits faster, false-OFF crater).
- **[[v2]]** - gap-penalised symmetric threshold; net -4.14 (coin-flips in the noise band).
- **[[v3]]** - asymmetric -6 pp threshold; net -10.11 (the K=4 noise floor is ~+/-8 pp, wider than any threshold).
- **[[v4]]** - 3-probe shadow; net -4.32 but best single seed +41.29. Shadow data proves **K=8 is the lever, not multi-trial**.
- **[[v5]]** - K=8 driver + Mixup + new ops; ABORTED (noisy ranker false-picks ResNeXt, GameOfLife -9.4).
- **[[v6]]** - rich 9-op vocab + verify + zero-shot; net -12.10 (rich vocab POISONS plain-conv data). Only Language probe (+4.23) survives.
- **[[v7]]** - SIMPLIFY: EGNAS-full + aug-probe + macro-prune-verify; **+41.94, first to beat EGNAS-full**.
- **[[v8]]** - clean aggregate + train-to-budget + adaptive stem; **+48.2** (train-to-budget is the big cheap lever).
- **[[v9]]** - searched dilated backbone; +52.4, but the search is net-negative vs the fixed TCN-2d +59.2 (motivates the default-incumbent rule).
- **[[v10]]** - measure-gate-train (fingerprint routing + family-matched probes + train-to-budget); **+62.23 +/- 0.19, all-time best**.
- **[[v11]]** - extensions: sweep DONE (only Chesseract has headroom; forced-macro +5.94 there); in-depth per-stage macro search running. Tests, graphs and the v11 design are all in the [[v11]] note.

## Lineage table

| Version | One-line | Code dir | Total adj | Deep-dive |
|---|---|---|---|---|
| [[v1]] | aug-probe raw-val rule, failed smoke | (none kept) | n/a | [[Pipeline v1-v7 - Aug Probe]] |
| [[v2]] | gap-penalised symmetric threshold | `baselines/v2` | net -4.14 vs EGNAS-full | [[Pipeline v1-v7 - Aug Probe]] |
| [[v3]] | asymmetric -6 pp threshold | `baselines/v3` | net -10.11 vs EGNAS-full | [[Pipeline v1-v7 - Aug Probe]] |
| [[v4]] | 3-probe shadow (K=8 is the lever) | `baselines/v4` | net -4.32 (s42 +41.29) | [[Pipeline v1-v7 - Aug Probe]] |
| [[v5]] | K=8 driver + Mixup + new ops (ABORTED) | `baselines/v5` | net negative (partial) | [[Pipeline v1-v7 - Aug Probe]] |
| [[v6]] | rich vocab + verify + zero-shot | `baselines/v6` | +23.41, -12.10 vs EGNAS-full | [[Pipeline v1-v7 - Aug Probe]] |
| [[v7]] | SIMPLIFY, beats EGNAS-full | `baselines/v7` | +41.94 (+6.43 vs EGNAS-full) | [[Pipeline v1-v7 - Aug Probe]] |
| [[v8]] | aggregate + train-to-budget + stem | `baselines/v8` | +48.2 (s42) | [[Pipeline v1-v7 - Aug Probe]] |
| [[v9]] | searched dilated backbone | `baselines/v9` | +52.4 (s42) | [[Pipeline v1-v7 - Aug Probe]] |
| [[v10]] | measure-gate-train (all-time best) | `baselines/v10` | +62.23 +/- 0.19 | [[Pipeline v10 - Design + Idea Ledger]] |
| [[v11]] | extensions (running / queued) | `baselines/v10` + probes | TBD | [[Pipeline v10 - Design + Idea Ledger]] |
