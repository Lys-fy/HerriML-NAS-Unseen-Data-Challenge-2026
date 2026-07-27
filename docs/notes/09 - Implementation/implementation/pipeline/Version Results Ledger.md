# Version Results Ledger (whole project history)

> One place to see, per pipeline version, what was tested and what it scored. **Newest on
> top, oldest at the bottom.** Full per-dataset tables exist for the mature pipeline
> (v12–v13); the probe era (v2–v11) is summarised by its documented total + verdict +
> named datasets, because those versions ran on dataset *subsets* / different metrics (no
> full 16-row tables were recorded — nothing here is interpolated). Sources per section.

**Scoring conventions.** `raw` = raw test accuracy %. `adj` = competition adjusted score vs
the ResNet-18 benchmark, `10·(raw−benchmark)/(100−benchmark)`, floored at −10. `internal adj`
substitutes internal benchmarks for the placeholder-benchmark datasets (Sudoku, Cryptic).
Default hardware: DGX Spark (NVIDIA GB10) unless noted.

## Lineage at a glance

| Version | What changed | Headline total | Verdict |
|---|---|---|---|
| **v13** (shipped) | v12.1 + tie-break OFF + saturation stop | +72.38 adj (16@2h, GB10) | **final; ahead of 2025 winner on finals trio** |
| v12.3 | token family integrated | +73.08 adj (16@2h, 2g-MIG) | adopted (token = Cryptic win) |
| v12.2 | overnight A/B battery | — (ablations) | indep-init/BDH/W128 rejected; half-split adopted |
| v12.1 | hardened probes, alt/top-up seeds, capacity arm | +74.52 adj (16@2h, A100-4g) | adopted; all 8h records |
| v12 | throughput ladder + TrivialAug + K-seed + val-gated select | V100 +69.5 < A100 +73.5 < H100 +76.4 | beats v10 12/16 |
| v11.3 | evolutionary hetero search | −25.9 pp vs v10 | rejected |
| v11.2 | hetero micro-macro hill-climb | 0/5 hold-out wins vs v10 | rejected (search mirage) |
| v11.1 | informed-defaults floor + 3-way bake-off | −0.63 pp vs v10 (best of 3) | best challenger, still ≈v10 |
| v11 | extensions sweep on v10 | only Chesseract improves | "v10 has no arch headroom left" |
| v10 | measure-gate-train | +62.23 adj (13@2h, 3-seed) | the v-line peak; basis for v12 |
| v9 | searched dilated backbone | +52.4 adj (13@2h, s42) | search net-negative vs fixed TCN-2d |
| v8 | clean aggregate + train-to-budget | +48.2 adj (13@2h, s42) | train-to-budget adopted |
| v7 | SIMPLIFY (EGNAS + aug-probe) | +41.94 internal (13@2h, 3-seed) | first to beat EGNAS-full |
| v6 | rich vocab + zero-shot proxies | +23.41 internal | rejected (vocab poison) |
| v5 | K=8 driver + Mixup/ResNeXt | aborted (s42 only) | pivot to v6 |
| v4 | 3-probe shadow (K=4 vs K=8) | +31.19 internal (net −4.32) | "K=8 is the lever" |
| v3 | asymmetric −6pp threshold | net −10.11 vs EGNAS | worst probe iteration |
| v2 | gap-penalised probe, ±0.5pp | net −4.14 vs EGNAS | threshold coin-flips in noise |
| baselines | ResNet-18 · Fixed CNN · EGNAS-full | Fixed CNN −0.36 · EGNAS +35.5 | reference points |

---

## v13 — FINAL shipped submission
*Source: [[v12.1 - Validation Results]] §v13; spec [[Pipeline v13 - Final Spec]]. Shipped 2026-07-16 as `submissions/v2`.*

**Tier-16 @2h (raw %)** — v13 (GB10) next to the two comparable full-16 @2h tiers. The v13 vs
v12.1 gap is entirely the two compute-sensitive scalers (GB10 vs A100 hardware, marked ⚙️):

| Dataset | ResNet | v12.1 (A100-4g) | v12.3 (2g-MIG) | **v13 (GB10)** |
|---|---|---|---|---|
| AddNIST | 89.85 | 97.19 | 97.16 | 96.65 |
| CIFARTile | 47.01 | 83.46 | 80.46 | 79.56 ⚙️ |
| Cauchy | 92.21 | 94.86 | 94.69 | 94.76 |
| Chesseract | 57.83 | 64.07 | 65.00 | 63.74 |
| Cryptic | — | 73.68 | 76.68 | 75.44 |
| GameOfLife | 47.53 | 99.99 | 99.99 | 99.99 |
| GeoClassing | 80.33 | 95.83 | 95.29 | 95.25 |
| Gutenberg | 40.98 | 49.75 | 49.18 | 50.03 |
| Language | 85.20 | 90.48 | 90.61 | 91.59 |
| MultNIST | 90.87 | 95.47 | 95.42 | 95.23 |
| Myofibre | 87.93 | 93.09 | 92.06 | 91.45 ⚙️ |
| Sedol | 53.92 | 59.31 | 60.20 | 60.68 |
| Sloan | 84.17 | 84.65 | 84.71 | 84.49 |
| Sudoku | — | 96.71 | 96.18 | 96.05 |
| Voxel | 71.35 | 83.31 | 83.14 | 83.50 |
| Windspeed | 13.49 | 18.71 | 18.04 | 19.37 |
| **Total adj** | — | **+74.52** | **+73.08** | **+72.38** |

**@8h validation (raw %, GB10)** — the saturation-stop thesis: recovers Chesseract, does not
over-stop the long-trainers, Cryptic record.

| Dataset | v13 @8h | note |
|---|---|---|
| Chesseract | **67.41** (SAT on) / 64.59 (SAT off) | the v12.4 inversion, recovered (+2.8) |
| CIFARTile | 87.08 | scaler not over-stopped |
| Sudoku | 96.72 | small board not over-stopped (seeds released → 8-seed roster) |
| **Cryptic** | **78.60** | new all-time best; +2.4 over 2025 winner (76.15) |

**Finals dress rehearsal @8h (Phase-3 shape, shipped `submissions/v2`):** GameOfLife 99.99 ·
Windspeed 19.30 · **Cryptic 78.24** → ahead of the 2025 winner on all three, projects to **1st**.

---

## v12.3 — token family integrated
*Source: [[v12.1 - Validation Results]]; per-dataset tier in `dev/logs/`.* Full 16-row @2h table is the v12.3 column above. **Total +73.08 adj (A100 2g-MIG @2h).** Headline: the shape-gated token family fires only on Cryptic (76.68, best-of-tier), zero collateral elsewhere.

## v12.2 — overnight A/B battery (ablations, not a tier)
*Source: [[v12.1 - Validation Results]] §3 (2× H100 @2h).*
- indep-init: 61.61 / 95.97 / 99.99 vs shared-init 64.07 / 96.71 → **REJECT**.
- split-threshold: CIFARTile K=2 **85.68** vs K=1 84.95 → adopt half-split.
- BDH mult-128: Sudoku 96.38 (vs 96.73 / 96.46) → **BDH closed, flat across 15× params**.
- capacity W96 (85.38/64.89/96.79) ≈ W128 (85.56/63.29/96.11) → W96 sufficient, W128 rejected.

## v12.1 — hardened probes + alt/top-up seeds + capacity arm
*Source: [[v12.1 - Validation Results]] §§1–2.* **@2h all-16, A100-4g: +74.52 adj** (full row above). **@8h — all records (raw %):**

| Dataset | v12.1 @8h | v10 | prev best |
|---|---|---|---|
| Sudoku | **98.36** (H200, 7-seed ens) | 97.70 | 97.70 |
| Language | **92.91** (H200, K=2 ens+tta) | 90.03 | 91.49 |
| CIFARTile | **89.18** (H200, K=3) / 86.68 (A100) | 73.65 | 84.95 |
| Chesseract | **67.01** (A100, 8-seed top-up) | 57.66 | 66.86 |
| Cryptic | 72.46 (A100) / 72.84 (GB10) | 72.08 | 73.82 |
| GameOfLife | 99.99 (GB10, ~1.9h used) | 100.0 | — |
| Windspeed | 18.93 (GB10) | 19.36 | — |

## v12 — throughput-keyed budget-adaptive ladder
*Source: [[Pipeline v12 - Screening Report]].* **@2h all-16 × 4 GPUs (raw %); adj totals scale with throughput: V100 +69.5 < A100 +73.5 < H100 +76.4; GB10 grid +49.7.** Beats v10 on 12/16 (mean +2.3 at matched budget).

| Dataset | ResNet | v10 | GB10 | V100 | A100 | H100 |
|---|---|---|---|---|---|---|
| Chesseract | 57.83 | 57.66 | 66.69 | 62.33 | **66.86** | 61.96 |
| Sudoku | — | 97.70 | 97.22 | 96.54 | 96.68 | 97.14 |
| Language | 85.20 | 90.03 | 89.64 | 90.21 | 88.95 | **91.49** |
| Cryptic | ~71.1 | 72.08 | **73.82** | 72.77 | 73.13 | 72.04 |
| GeoClassing | 80.33 | 95.36 | 95.40 | 95.72 | 95.47 | **96.22** |
| CIFARTile | 47.01 | 73.65 | 80.01 | 78.71 | 82.79 | **84.95** |
| AddNIST | 89.85 | 96.33 | 96.54 | 96.37 | 96.77 | **97.63** |
| MultNIST | 90.87 | 95.52 | 95.90 | 95.36 | 95.73 | **96.47** |
| Gutenberg | 40.98 | 47.28 | 46.85 | 47.90 | **50.13** | 49.08 |
| GameOfLife | 47.53 | 100.0 | 99.98 | 99.96 | 99.99 | 99.99 |
| Myofibre | 87.93 | 91.09 | 90.52 | 90.60 | 92.81 | **93.23** |
| Windspeed | 13.49 | 19.36 | 17.35 | 17.43 | 19.09 | 18.96 |
| Sedol | 53.92 | 56.55 | 58.16 | **59.58** | 58.77 | 58.90 |
| Sloan | 84.17 | 84.60 | 84.55 | **84.63** | 84.59 | 84.59 |
| Cauchy | 92.21 | 93.91 | **95.14** | 94.91 | 95.01 | 94.71 |
| Voxel | 71.35 | 81.56 | 82.86 | 83.27 | 83.22 | **83.55** |

---

## Probe / search era (v2–v11) — totals + verdicts

Per-dataset tables were not recorded for these (subset runs, different metrics). Numbers below
are exactly as documented; see `pipeline/versions/vN.md` and [[Pipeline v1-v7 - Aug Probe]].

- **v11.3** (evolutionary hetero search, 16 ds, s42): **net −25.9 pp vs v10** (mean −1.6, wins 5/16). Rediscovers attention on Chesseract (+4.3) but loses CIFARTile −13, Language −11. *Rejected.*
- **v11.2** (hetero hill-climb): Chesseract validated **+4.84**; blind hold-out **0/5** wins vs real v10 (Sudoku search 87.78 vs 97.70 = −9.92, the mirage). *Rejected.*
- **v11.1** (informed-defaults floor + 3-way bake-off, 16 ds, s42): **−0.63 pp vs v10** (best of the three challengers; evolution −1.62, EGNAS-faithful −4.37). Chesseract **62.48 raw (+4.82)** via the attention probe; CIFARTile 70.8; losses attributed to single-seed variance. Best challenger, still ≈v10.
- **v11** (extensions sweep on v10, s42 @2h, deltas vs v10): arch gauntlet wins **only** Chesseract (+0.4…+1.6), craters elsewhere; macro (ungated) Chesseract +5.94 but Sudoku −33.6. Headline: **"v10 has almost no architectural headroom left."**
- **v10** (measure-gate-train): **+62.23 ± 0.19 adj** (13 ds × 3 seeds @2h, GB10); beats EGNAS-full by +26.7 and fixed TCN-2d (+59.2). The v-line peak; its per-dataset raw numbers survive as the "v10" column above. Phase-A routing: fingerprint+gate +61.2 vs fixed tcn2d +59.2 vs oracle +63.2.
- **v9** (searched dilated backbone, s42): **+52.4 adj** vs fixed TCN-2d +59.2 → search **net-negative** on encoded data (Sudoku searched D=8 75.8 vs fixed 88.1).
- **v8** (clean aggregate + train-to-budget + adaptive stem, s42): **+48.2 adj** (vs v7 +41.94); train-to-budget = 5 wins / 2 ties / 0 regressions → adopted.
- **v7** (SIMPLIFY: EGNAS-full + aug-probe, 13 ds × 3 seeds): **+41.94 ± 3.45 internal** vs EGNAS-full +35.51 (Δ +6.43). Language +7.34 (aug-probe OFF), one regression MultNIST −1.15. **First version to beat the EGNAS-full baseline; the aug-probe is the engine.**
- **v6** (rich 9-op vocab + zero-shot proxies): **+23.41 ± 9.17 internal** (Δ −12.10 vs EGNAS, worse mean AND variance). Sudoku −7.94 (vocab poison), GameOfLife −4.85 (depth lottery); only win Language +4.23. *Rejected.*
- **v5** (K=8 + Mixup-off + ResNeXt): **aborted** during s1337; GameOfLife best_val 90.5% (ResNeXt false positive). Pivot to v6.
- **v4** (3-probe shadow, 13 ds × 3 seeds): **+31.19 ± 7.62 internal (net −4.32)**; per-seed s42 **+41.29** (best single-seed of the era), s1337 +35.49, s2024 +32.62. Finding: **"K=8 is the lever, not multi-trial."**
- **v3** (asymmetric −6pp threshold): **net −10.11 vs EGNAS** — worst probe iteration; K=4 noise floor ~±8pp, no threshold fixes it.
- **v2** (gap-penalised probe, ±0.5pp): **net −4.14 vs EGNAS**; Language fixed (+5) but MultNIST false-OFF craters. Symmetric threshold coin-flips in the noise band.
- *(v1: failed AddNIST smoke, never swept.)*

---

## Reference baselines
*Sources: `baselines/*.md`, `reliability/*.md`.*

- **ResNet-18** — the competition benchmark (no standalone run); its per-dataset numbers are the "ResNet" column above (Chesseract 57.83, Language 85.20, CIFARTile 47.01, Cauchy 92.21, Voxel 71.35, …).
- **Fixed CNN ("Tier 0")** — 11 ds @2h, GB10: **Total −0.355 adj**. GameOfLife 96.53 (+9.34), Sudoku 70.95 (+7.10), GeoClassing 90.23 (+5.03); AddNIST & MultNIST at the −10 floor (the failure that motivated the search).
- **Aug + Op Search ("Tier 1")** — 11 ds @2h, single seed: **+24.6 adj** (flagged as a favourable tail draw); 3-seed reliability probe: **+8.39 ± 8.37 internal / +11.27 ± 8.21 shipped** (huge variance → motivated EGNAS).
- **EGNAS-full** (2025-winner reimplementation, the paired baseline for v2–v7) — 11 ds × 3 seeds @2h: **+27.99 ± 3.28 internal** (13-ds figure +35.51 ± 3.07). Per-dataset (internal adj, mean): GameOfLife +8.15, Sudoku +7.07, GeoClassing +7.65, AddNIST +5.41, MultNIST +3.97, Voxel +2.41, Gutenberg +1.01, Myofibre +0.49, CIFARTile +0.03, Chesseract +0.13, **Language −8.32** (the weakness v7's aug-probe fixed). Faithful re-test (v11.1, s42, 16 ds): net −69.9 vs v10.
- **TCN Backbone Excursion** — fixed archs, 13 ds @2h s42: conv +48.2, TCN-1d +44.7, **TCN-2d +59.2** (adopted as the image-like default), oracle +63.2.
- **LSTM Language probe** — row-BiLSTM 85.76 ± 0.41 raw (adj +0.38), first architecture reliably above the Language benchmark.
