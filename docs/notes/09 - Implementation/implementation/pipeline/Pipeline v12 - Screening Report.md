# Pipeline v12 - Screening Report (all tests, 2026-07-01 to 07-06)

> **Status:** COMPLETE - 93 runs on the GB10, zero failures | **Purpose:** the comprehensive evidence base for designing v12. Every technique screened, with BOTH anchors: delta vs its own controlled baseline AND vs v10-real (the deployed 3-seed pipeline). | **Raw data:** `dev/logs/harvest_screen/*.json` (synced from spark `~/local/harvest/`), per-epoch curves on spark in `~/local/harvest/{tag}/{codename}.csv` | **Related:** [[v11.1]], [[Pipeline v10 - Design + Idea Ledger]], [[Key Learnings]], [[2025 - Dragon Hatchling (BDH)]]

## Design of the screen (read this before the numbers)

- **Testbed:** fixed all-Conv3 6L W48 backbone + full v10 recipe (SGD cosine train-to-budget, EMA, aug policy), 1.5h, seed 42, on a 9-dataset archetype-spanning set: Chesseract (board), Sudoku (reasoning), Language (sequence), GeoClassing (satellite), CIFARTile (hard natural), AddNIST (arithmetic), GameOfLife (CA), Myofibre, Windspeed (scientific). BDH-harvest ops (C1/C2/C4) additionally on Cryptic; soups/ensembles as K=3 x 0.5h shared-init seeds.
- **The controlled anchor is the `default` arm** (same testbed, no technique). Deltas vs default isolate each technique's marginal value.
- **CAVEAT (the Language lesson):** the testbed is far below v10-real on 7/9 datasets because v10 ROUTES to per-dataset families (tcn2d etc.). Language default arm = 72.0 vs v10-real 90.0. So a delta measured here answers "does X help a conv net", NOT "does X improve the deployed pipeline" - deltas can shrink on strong backbones. Only Chesseract (58.9 vs 57.7) and GeoClassing (96.1 vs 95.4) are deployment-representative contexts.

## Master table - test accuracy (bold = beats v10-real)

| dataset | v10-real | default | C1 | C2 | C4 | RepConv3 | SAM | SelfDist | TrivAug | AdamW |
|---|---|---|---|---|---|---|---|---|---|---|
| Chesseract | 57.66 | **58.92** | **58.29** | **58.28** | **57.72** | **58.93** | **59.50** | 56.38 | **59.39** | **58.41** |
| Sudoku | 97.70 | 91.78 | 92.25 | 90.31 | 93.38 | 91.82 | 92.07 | 86.46 | 94.66 | 87.67 |
| Language | 90.03 | 72.04 | 75.86 | 68.69 | 75.66 | 71.45 | 72.17 | 72.74 | 72.11 | 70.34 |
| Cryptic | 72.08 | 71.17 | 70.75 | 70.81 | 68.68 | - | - | - | - | - |
| GeoClassing | 95.36 | **96.09** | **96.47** | **96.02** | **95.98** | **96.07** | **95.39** | 94.85 | **96.10** | **96.31** |
| CIFARTile | 73.65 | 43.40 | - | - | - | 42.41 | 24.41 | 43.05 | 46.60 | 44.16 |
| AddNIST | 96.33 | 92.03 | - | - | - | 91.80 | 91.10 | 91.64 | 92.77 | 92.14 |
| GameOfLife | 100.0 | 74.62 | - | - | - | 75.54 | 75.54 | 72.08 | 88.28 | 68.66 |
| Myofibre | 91.09 | 88.21 | - | - | - | 88.29 | 87.08 | 87.62 | 89.06 | 89.06 |
| Windspeed | 19.36 | 16.70 | - | - | - | 18.88 | 17.60 | 18.11 | 15.87 | 19.33 |

## Verdicts, ranked (delta vs own default)

| technique | n | mean | wins | min..max | verdict |
|---|---|---|---|---|---|
| **TrivialAug** | 9 | **+2.34** | 6/9 | -0.8..+13.7 | **ADOPT (default-on candidate)** - biggest win of campaign, zero compute cost. GameOfLife +13.7 (default stalls at memorizing, TA forces the translation-invariant rule; val curves confirm). CIFARTile +3.2, Sudoku +2.9 (crosses over ~ep50, higher final train loss = true regularization). One real loss: Windspeed -0.8 (physical channels distorted by brightness/contrast; visible on val -> val-gate catches it). Impl: batch-level, tensor-safe op set (identity/flip_h/translate/brightness/contrast/noise/cutout), `TRAIN_TRIVIAL_AUG=1`. |
| **logit-ensemble K=3** | 9 | **+1.45** | 8/9 >= 0 | -0.1..+4.4 | **ADOPT (surplus-budget policy)** - shared-init seeds, average softmax. Biggest where acc lowest (GameOfLife +4.4, Language +3.1, Sudoku +2.0). Chesseract ens+TTA 62.0 = **+4.3 over v10-real in a representative context** (campaign's flagship above-v10 result). |
| C1 LinearAttn2D | 5 | +0.72 | 3/5 | -0.6..+3.8 | **PROBE-CANDIDATE, unproven** - the +3.8 (Language) and +0.5 (Sudoku) are weak-baseline contexts; representative contexts show -0.6..+0.4. Needs the in-pipeline v11.1 probe test before adoption. O(HW), odd-grid-safe - would also REPLACE the crashy SpatialSelfAttn2D if adopted. |
| C4 shared-depth | 5 | +0.28 | 2/5 | -2.5..+3.6 | same status as C1 (reasoning/sequence specialist; same caveat). |
| RepConv3 | 9 | +0.16 | 2/9 | -1.0..+2.2 | **WASH** - reparam buys nothing at 1.5h here. Drop. |
| AdamW | 9 | -0.86 | 4/9 | -6.0..+2.6 | **REJECT as default** - faster early, worse plateau (Sudoku val 0.883 vs 0.923 at same epochs); losses (Sudoku -4.1, GameOfLife -6.0) dwarf wins; Windspeed +2.6 is test-noise (val gap only 0.26). SGD stays. |
| SelfDistill | 9 | -1.21 | 2/9 | -5.3..+1.4 | **REJECT** - EMA-teacher KL fights the existing LS+EMA recipe. |
| C2 SparseGated | 5 | -1.18 | 0/5 | -3.4..-0.1 | **REJECT** - 0 wins anywhere. |
| SAM | 9 | -2.10 | 4/9 | -19.0..+0.9 | **REJECT as default** - CIFARTile collapse (24.4 vs 43.4) disqualifies; small wins elsewhere. At most a val-gated arm at big budgets. |
| soup_greedy | 9 | -0.06 | 0 | -0.5..0 | no-op (degenerates to single-best). Drop. |
| soup_uniform | 9 | -34.9 | 0 | -82.8..-0.5 | **DANGEROUS** - collapses (Sudoku 12%) even with shared init; long training diverges basins. Dead. |
| flip-TTA | 9 | -1.49 (single) / +0.04 (on ens) | 3-4/9 | -12..+6.9 | **GATE-ONLY** - helps flip-invariant grids (Chesseract +1.2..+2.3, Sudoku +1.0..+1.6, GameOfLife ens +6.9), wrecks text (Language -12). Deploy only behind a val with-vs-without check (2 val passes, ~free). |

## Structural results

- **CIFARTile depth ladder:** conv 6/12/20L = 43.4/51.6/53.2; tcn2d 6/12/20L = 73.2/66.9/64.2. Depth helps conv (+9.8) but stays 20pp under tcn2d-d6; MORE depth actively hurts tcn2d. **The 19pp PC-DARTS gap is NOT a depth artifact** - it is family/recipe (cell structure + heavy aug + long schedules). tcn2d-d6 73.2 ~ v10-real 73.65 confirms v10's winner. CLOSED as "not depth"; the remaining gap needs a different backbone idea (or heavier aug: TA gave +3.2 on conv).
- **BDH-bidir (B1 O(T) attention + B2 all-layer readout):** Sudoku 96.73 (vs faithful-BDH 96.21, v10 97.7 - B1/B2 real but still short); Chesseract 57.5 at only 9 epochs (nearly v10 = the long-budget board candidate); everything else MARGINAL by the viability gate (1-12 proj. epochs; linear-T removed the wall but the wide-N constant keeps medium-T out at 2h). Keep as viability-gated specialist arm only.
- **R4 budget saturation + the EPOCH-CAP ceiling:** CIFARTile 4h = 43.7 vs 1.5h 43.4 (saturated; val 0.454 vs 0.448, LR fully annealed). Chesseract 4h = 57.3 vs 58.9 at 1.5h. CORRECTION (first read called this a schedule bug - it is not): the trainer already clamps the cosine schedule to `min(MAX_EPOCHS, projected)`, so both runs did an identical, correctly-annealed 600 epochs; the -1.6 delta is the single-run noise band (cudnn.benchmark nondeterminism + probe-RNG drift). The REAL finding: **TRAIN_MAX_EPOCHS=600 makes extra budget a strict no-op on fast datasets** - at Phase-3 budgets every fast dataset saturates ~25min in. Whether MORE epochs (cap 2400) help is being answered by the in-flight Chesseract rerun (`Chesseract_ab_long4h_v2`). Verdict so far: surplus budget -> seeds/ensembles, NOT longer single runs; and note the ±1.6pp single-seed noise band when reading any 1-seed delta in this report.
- **Difficulty probes (C3 battery, CPU):** all 16 signals n.s. vs task difficulty at n=16 (best |rho| 0.43); supervised probes have BACKWARDS sign (raw-pixel separability is anti-correlated with CNN difficulty: GameOfLife/Sudoku trivial for CNN, hopeless for kNN). Cheap-fingerprint routing is dead; gates must MEASURE, not predict. `dev/logs/difficulty_probes.json`.

## What this means for v12 (agreed direction)

v12 = **throughput-keyed budget-adaptive ladder** (rungs from measured effective_trainings = clock / calibrated full-train time, GPU-agnostic, re-projected between stages), everything gated by measurement:

1. Core rung: fingerprint -> family -> train-to-budget (v10/v11.1 base) + **schedule-cap fix** + TrivialAug (default-on, pending flip-ablation).
2. Probe rung: aug-safety + verified attention probes (v11.1) + C1/C4 only if the in-pipeline test clears them.
3. Surplus rung: K-seed training -> **val-gated final selection** (single vs logit-ensemble, each +/- flip-TTA - ship the val winner; catches Windspeed/TA, Language/TTA failures by construction). BDH arm on viability-passing boards.

## v12 integration test results (T1-T4, 2026-07-06 to 07-08 - ALL DONE)

Runner: `dev/v12_integration.py` (exact grader sequence + TrivialAug + K=3 shared-init seeds
splitting the post-search clock + val-gated selection over {singles, logit-ensemble} x {+-flip-TTA}).
Raw: `dev/logs/v12tests/*.json`, logs on spark `~/local/v12tests/`.

### T2 - v12 prototype vs v10-real at MATCHED 2h total (9-set)
Chesseract **+5.76** (63.42, ens+tta) | Language **+1.44** (91.47, ens) | AddNIST **+0.47** (96.80, ens)
| GameOfLife -0.01 | GeoClassing -0.71 | Sudoku -1.15 | Windspeed -1.33 | Myofibre -2.41 | CIFARTile -5.59.
Mean -0.4 - but the losses are EXACTLY the slow datasets where 3-way splitting cuts each seed below one
full training (CIFARTile, Myofibre); wins are where training saturates early (seeds are free). ->
**adaptive-K rule: K=1 unless remaining budget >= K full trainings** would have made v12 >= v10 on
effectively every dataset at 2h.

### T4 - budget tiers (the headline)
- Chesseract v12: **0.5h +4.1** (61.81) / 2h +5.8 (63.42) / **6h +9.6** (67.25, ens+tta). Wins at QUARTER budget.
- CIFARTile @6h: v12 **82.36 (+8.7)** vs plain-6h control (single train, cap 2400, no TA) **76.43 (+2.8)**.
  Decomposition: longer-uncapped single training +2.8 over v10 (v10's 73.65 was UNDERTRAINED, not
  architecture-limited - cap/budget was the bottleneck); TA + ensemble + TTA add a further **+5.9** on test.
  (Caveat: plain val was 81.6 vs test 76.4 - CIFARTile has a ~5pp val-test gap; the v12 stack's test win stands.)
- Chesseract cap-2400 @4h = 58.98 ~ the 1.5h number -> fast sets saturate in epochs; surplus must go to
  seeds/ensembles there (+9.6), while slow sets want longer single training FIRST. Both cases detected at
  runtime from projected epochs vs cap -> the two-sided surplus rule of the throughput-keyed ladder.

### T3 - C1/C4 in-pipeline: DEAD (ledger as screen mirages)
--plain arms at 2h: C1 (MACRO_OPS=LinearAttn2D): Language 87.99 (-2.0), Sudoku 88.68 (**-9.0**).
C4 (V11_SHARED_REPEAT=3): Language 89.39 (-0.6), Sudoku 93.36 (-4.3). Both far below the T2 v12 numbers
(91.5 / 96.6) on the same datasets. The weak-baseline screen wins did not survive the deployed context -
fourth instance of the proxy-wall mirage pattern. REJECT both; C1 remains only a possible future
replacement for the odd-grid-crashy SpatialSelfAttn2D probe op (shape-robustness, not accuracy).

### T1 - TrivialAug flip ablation: FLIP STAYS
no-flip vs flip: GameOfLife 62.0 vs 88.3 (**flip carries +26** - horizontal flip is a symmetry of the CA
rule), Sudoku 93.6 vs 94.7, Chesseract 56.3 vs 59.4. flip_h remains in the TA op set.

### v12.1 candidates -- DECIDED, waiting for current validation results (do NOT build yet)

1. **More seeds on cap-bound fast datasets (Exception-1 residue spend).** On datasets where
   `remaining/K > CAP_EPOCHS x epoch_time`, the run ends early and the leftover budget is dead
   (under per-dataset limits). Spend it on extra seeds: raise `V12_MAX_SEEDS` (env already exists;
   planner only fires K>3 where the budget affords full trainings, slow sets untouched by
   construction). Expected: diminishing (~log K) ensemble gains, a few tenths -- but free.
2. **Top-m ensemble selection.** At K>=4 the all-K ensemble can be dragged by a weak seed; add
   "ensemble of top-m seeds by val" (m=2..K-1) to the selection configs -- probs already cached,
   ~10 lines, zero extra GPU.
3. **The decisive A/B (run AFTER the current 2h fleet + budget grid land):** `V12_MAX_SEEDS=6`
   on the cap-bound sets (Chesseract, Sudoku, GameOfLife) @2h vs tonight's K=3 numbers, same
   hardware (~4h GPU). Adopt both changes if >= +0.3 mean, else keep the package simple.
4. ~~Open budget-model question~~ **ANSWERED (David Towers, Discord 2026-07):** time limit is
   PER DATASET; Phase 3 = 24h across THREE hidden datasets, ~8h each; Phase 2 = 7 manual email
   submissions on dummy data at ~30min; LAST successful Phase-2 email advances. See the
   organiser-confirmations block in [[Phases and Timeline]]. v12 already complies (per-dataset
   clock fractions). Eval hardware likely Newcastle Comet: H100 96GB or L40S.

### Net v12 build spec (all questions closed)
1. v11.1 base (fingerprint -> family -> probes) + **TrivialAug default-on** (flip in).
2. **Throughput-keyed rung selector**: calibrate step time -> effective_trainings; K=1 single full train
   below ~K full trainings of headroom (slow sets / small budgets), else K=3 shared-init seeds; epoch cap
   budget-aware (raise toward 2400 when clock allows and val hasn't plateaued).
3. **Val-gated final selection** over {singles, ensemble} x {+-TTA} (2(K+1) val passes, ~free) - proven to
   pick correctly on all 9 (chose single where ensembling didn't help, TTA only on flip-tolerant sets).
4. BDH viability-gated board arm unchanged (optional, Phase-3 budgets only).
Expected vs v10-real: >= parity at 1-2h, +5..+10 on multiple datasets at Phase-3 budgets.
