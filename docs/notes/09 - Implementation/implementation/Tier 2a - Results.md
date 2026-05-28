# Tier 2a — P1 Aug-Safety Probe Results

> **Status:** ✅ v4 sweep COMPLETE (2026-05-26 → 2026-05-28, 3 seeds × 13 datasets). v2/v3 frozen as comparators. | **Topic:** Tier-2b full EGNAS + a runtime aug-safety probe (P1). The first **actual probe** of the [[Probe-Driven Self-Configuration]] thesis — short stratified-subset trainings before main training decide aug on/off per dataset.
> **Related:** [[Tier 2a - Probe v1]] (plan), [[Tier 2a - Language Deep Dive]] (excursion + literature + GroupConv proposal — the Language-ceiling follow-up to §D), [[Tier 2b - Results]] (paired EGNAS-full baseline), [[Multi-Seed - Results]] (the aug-oracle that motivated P1), [[Probe-Driven Self-Configuration]], [[Reliability vs Peak]], [[Implementation Results Overview]], [[Run State + Handoff]]

## Headline

**v4 verdict: mechanism validated on Language, net headline negative.**

Across 4 iterations (v1 → v4) we have a working probe but a methodologically honest negative result:

| Metric | Value |
|---|---:|
| Total internal adj (EGNAS-full baseline, 13 ds, 3 seeds) | **+35.51 ± 3.07** |
| Total internal adj (+P1 v4) | +31.19 ± 7.62 |
| Δmean | **−4.32** |
| σ ratio | **×2.48** (P1 *increased* variance) |
| Language Δmean (the target) | **+3.72** ✓ (the only big win) |
| Other regressions (Δmean < −0.25) | GameOfLife (−3.13), Sudoku (−1.31), MultNIST (−0.37) |

**The Language gain is real and probe-attributable.** The other losses are mostly **EGNAS-internal seed variance** on skip-probe datasets (GameOfLife/Sudoku), not the probe's fault — but they sum to more than the probe's gain.

**More important finding for the Language question** (which you flagged): pulling [[Multi-Seed - Results]]'s aug-off arm shows **even Tier-1 + aug-off doesn't beat the Language benchmark** (mean adj −2.56 ± 1.69, best seed −1.17, raw 83.47% vs benchmark 85.20%). **No configuration we've tested has ever crossed Language benchmark on any seed.** The probe is doing its job — it correctly picks aug-off — but aug-off alone is structurally insufficient for Language. See §"Why Language is still below benchmark — and what could actually fix it" below.

## Visual summary

### Paired bars + σ change (EGNAS-full vs +P1 v4, 13 ds × 3 seeds)

![[tier2a_p1_v4.png]]

Panel A reads left-to-right alphabetic — green = EGNAS-full, orange = +P1 v4; Δ annotated above each pair. The Language Δ+3.7 is the headline. Skip-probe datasets (Chesseract/Cryptic/GameOfLife/Sudoku/Voxel/Windspeed) labelled "(skip-probe)" — their P1 numbers are pure EGNAS reruns + probe-time perturbation noise, not P1 effect. Panel B: σ change; red ×ratios indicate datasets where P1 v4 is *noisier* than EGNAS-full (most of them).

### Aggregated learning curves (EGNAS-full mean ± std vs +P1 v4 mean ± std)

![[tier2a_p1_v4_curves.png]]

For each panel: green band = EGNAS-full mean ± std across 3 seeds; orange band = +P1 v4; red dashed line = dataset's ResNet-18 benchmark. Language panel: orange clearly above green (probe-driven aug-off lift) but still below the red benchmark line — the structural ceiling.

### Per-seed score bars

s42 (best seed, total +41.286 — Language correctly OFF, all probes right):

![[tier2a_p1_v4_s42_scores.png]]

s1337 (total +35.493 — Language correct OFF, BUT GameOfLife skip-probe cratered from +8.88 → +2.14 from pure EGNAS variance):

![[tier2a_p1_v4_s1337_scores.png]]

s2024 (total +32.620 — Language probe WRONG: A multi-trial K=4 missed unanimity, picked ON, Language stuck at −7.22):

![[tier2a_p1_v4_s2024_scores.png]]

### Per-seed learning curves

s42:

![[tier2a_p1_v4_s42_curves.png]]

s1337:

![[tier2a_p1_v4_s1337_curves.png]]

s2024:

![[tier2a_p1_v4_s2024_curves.png]]

## Deep analysis

### A. Per-dataset paired comparison (internal adj, 13 ds × 3 seeds)

| Dataset | EG mean ± σ | P1 v4 mean ± σ | Δmean | σ ratio | Probe calls (s42/s1337/s2024) | Story |
|---|---:|---:|---:|---:|---|---|
| AddNIST | +5.41 ± 0.50 | +5.25 ± 0.48 | −0.16 | ×0.96 | ON/ON/ON ✓ | Probe correct; small noise |
| CIFARTile | +0.03 ± 0.34 | −0.11 ± 0.44 | −0.14 | ×1.29 | ON/ON/ON ✓ | Probe correct; small noise |
| Chesseract | +0.13 ± 0.01 | +0.06 ± 0.41 | −0.07 | ×31× | skip-probe | EGNAS variance under P1 path |
| Cryptic | +6.87 ± 0.22 | +7.10 ± 0.08 | +0.23 | ×0.34 | skip-probe | small EGNAS draw difference |
| GameOfLife | +8.14 ± 2.28 | +5.01 ± 3.65 | **−3.13** | ×1.60 | skip-probe | s1337 cratera +8.88→+2.14 (EGNAS-internal) |
| GeoClassing | +7.65 ± 0.35 | +7.53 ± 0.27 | −0.12 | ×0.77 | ON/ON/ON ✓ | Probe correct, tight |
| Gutenberg | +1.01 ± 0.07 | +1.09 ± 0.06 | +0.08 | ×0.85 | ON/ON/ON ✓ | Probe correct, tight |
| **Language** | −8.32 ± 1.56 | **−4.60 ± 2.25** | **+3.72** ✓ | ×1.44 | OFF/OFF/**ON** ✗ | The win; s2024 false-ON cost ~+5 |
| MultNIST | +3.97 ± 0.38 | +3.60 ± 0.91 | −0.37 | ×2.39 | ON/ON/ON ✓ | Probe correct; some perturbation |
| Myofibre | +0.49 ± 0.69 | +0.60 ± 0.10 | +0.11 | ×0.14 | ON/ON/ON ✓ | Probe correct; σ collapsed (good) |
| Sudoku | +9.15 ± 0.39 | +7.84 ± 1.41 | **−1.31** | ×3.62 | skip-probe | EGNAS variance (s1337 raw 66%) |
| Voxel | +2.41 ± 0.05 | +2.34 ± 0.30 | −0.07 | ×6.51 | skip-probe | EGNAS variance |
| Windspeed | +0.66 ± 0.25 | +0.75 ± 0.07 | +0.10 | ×0.27 | skip-probe | tight |

**Attribution split:**
- Probe-correct ON-call datasets (5 ds): mean Δ = **−0.16** — probe is transparent on its no-ops (within ±0.4 of zero on each).
- Probe-correct OFF-call datasets (1 ds, Language s42+s1337 portion): mean Δ = **+~+4.2** on those 2 seeds.
- Probe wrong-call (Language s2024): −7.22 vs −10.00 EG → still recovered ~+2.78 even with aug-ON, but missed the ~+5 it should have gained.
- Skip-probe datasets (6 ds): mean Δ = **−0.72** — pure EGNAS-internal variance the probe doesn't touch.

**Net: probe gain (+3.72 on Language) is roughly cancelled by skip-probe EGNAS noise (sum ≈ −4.4).** The probe isn't *wrong* — it just doesn't move enough of the total to overcome the system's intrinsic variance.

### B. Cross-probe shadow agreement (the v4 design payoff)

Every aug-applicable dataset ran THREE probes per seed (A multi-trial K=4 *drives*; B single-trial K=8 and C multi-trial K=8 *shadow*). 21 probe calls across 7 datasets × 3 seeds. **One disagreement** — and it's the canonical target dataset on the canonical hard seed:

| Dataset/seed | A_multi_K4 | B_single_K8 | C_multi_K8 | Truth | Driver A right? |
|---|---|---|---|---|---|
| Language s2024 | **ON** (Δ −14.70, 2/3 OFF) | OFF (Δ −15.10, 1/1) | OFF (Δ −18.30, 3/3) | OFF | **NO** |
| (all other 20 calls) | ✓ | ✓ | ✓ | ✓ | ✓ |

**The decisive shadow finding: K=8 separated aug-on/off cleanly enough that B and C both got Language s2024 right; K=4 multi-trial failed because 1 of 3 trials randomly fell above the −6 threshold, breaking unanimity.** K=8 (longer K) is the fix, not multi-trial. The shadow experiment paid off — without C+B we wouldn't know.

### C. Four-iteration history — what each verified or broke

| Version | Rule | Outcome | Failure mode |
|---|---|---|---|
| **v1** | OFF iff raw val_off − val_on > 0.5 pp | Smoke FAIL on AddNIST | At K=4, aug-off over-fits faster → raw val high → false-OFF cratera |
| **v2** | Gap-penalised, symmetric ±0.5 pp | Sweep net **−4.14** vs EG | Coin-flip in ±6 pp noise band → AddNIST s2024 + MultNIST s42 cratera |
| **v3** | Asymmetric, OFF iff Δscore < −6 pp | Sweep net **−10.11** vs EG | K=4 noise floor actually ~±8 pp → AddNIST s42 still cratera to −10 |
| **v4** | 3-probe shadow, A drives K=4 unanim −6 pp | Sweep net **−4.32** vs EG | K=4 unanimity too strict on borderline-Δscore seeds (Language s2024 false-ON) |

### D. Why Language is still below benchmark — and what could actually fix it

**Crucial empirical finding** (from [[Multi-Seed - Results]] §C aug-off arm):

| Configuration | Language adj per seed | Mean ± σ | Raw best % | Crosses benchmark? |
|---|---:|---:|---:|---|
| Tier-1 full (aug-on) | −8.14 / −10.00 / −10.00 | −9.38 ± 0.88 | 73.16 % | NO (floored) |
| **Tier-1 + aug-off** | −1.63 / −1.17 / −4.89 | **−2.56 ± 1.69** | **83.47 %** | NO (best is 1.7 pp below) |
| EGNAS-full | −6.93 / −8.03 / −10.00 | −8.32 ± 1.56 | 75.00 % | NO |
| **P1 v4 (probe-decided)** | −3.57 / −3.01 / −7.22 | **−4.60 ± 2.25** | **80.74 %** | NO (s2024 wrong call; best 4.5 pp below) |

**No configuration has ever crossed Language benchmark on any seed.** The probe is doing its job — aug-off is genuinely the right call for Language — but **aug-off alone is structurally insufficient**. Even the cleanest aug-off configuration (Tier-1 + aug-off) plateaus 1.7 pp below benchmark.

**Why** — three plausible causes, all testable:

1. **Language under aug-off early-stops too fast.** Tier-1 aug-off Language runtimes: 256-342s (vs 625-1004s for aug-on). The model converges quickly, hits patience=30 epochs without val improvement, stops. We're leaving training time on the table. Patience tweak / longer warmup could give the model time to escape the plateau.
2. **Wrong architecture for text-like inputs.** Language input is 24×24 single-channel encoded text. Conv stacks may be the wrong inductive bias here (same conv-ceiling story as CIFARTile and Cryptic). E2 (non-conv primitive — attention, deep-sets) is the architectural lever.
3. **Banned augs may actually help Language.** Our `aug_policy_for()` deliberately bans Mixup (label-mixing can break tasks whose labels are arithmetic functions of inputs) and Flip (breaks orientation-sensitive datasets). But Language's labels are *classes*, not arithmetic functions — Mixup might be fine. And character-level perturbation (a domain-specific aug we don't currently have) is the classic NLP augmentation.

**Variations worth trying to actually beat Language benchmark:**

| Idea | Effort | Expected effect | Failure mode |
|---|---|---|---|
| Extend patience to 60 epochs *just for Language* (per-dataset patience based on probe output) | low (1 day) | +1-3 pp raw via not-stopping-too-early | probe overhead negligible |
| Different LR schedule for aug-off: lower base LR, longer warmup | low (1 day) | +1-3 pp via slower convergence to better optimum | might hurt other datasets if applied globally |
| Add Mixup α=0.2 to the aug-off arm (probe-conditional) | medium (2 days) | +1-3 pp via implicit regularization without spatial aug | might break other aug-off-correct datasets |
| Try character-level aug (random char permutation/dropout) | medium (2-3 days) | +2-5 pp on text-like datasets specifically | needs heuristic to detect "text-like" inputs |
| **E2 (non-conv primitive: attention/deep-sets)** | high (3-5 days) | +5-10 pp on Language + CIFARTile + Cryptic | search-space change, may degrade conv-friendly datasets |
| Pre-train an autoencoder on train_x first, fine-tune classifier | high (4-5 days) | unknown for Language; +2-5 pp generally | added complexity, time budget |

**Recommendation:** the cheapest high-leverage experiment is **extending patience for probe-OFF datasets** (idea 1). Hypothesis: Language under aug-off learns slow but steady; we cut it off prematurely. ~1 day to implement + 12h to sweep. Then E2 (idea 5) is the architecture-level move that addresses Language + CIFARTile + Cryptic simultaneously.

> **UPDATE (2026-05-28) — this hypothesis was tested and refuted, and another (Mixup) was validated.** A focused Language-only excursion ran the patience-extension idea (V1) alongside Mixup (V3), gentler LR (V2), larger model (V4), and a combined arm (V5) on the Language dataset, 6 variations × 3 seeds. Verdict in **[[Tier 2a - Language Deep Dive]]**:
> - **Training-side ceiling on Language is ~81.5 % raw** — none of patience-60 / lr-low / larger-model crossed the 85.20 % benchmark, and the combined arm V5 matches V3 (Mixup alone) within noise.
> - **Mixup is the only meaningful training-side lever** (+1.6 adj vs baseline). Patience-extension and capacity changes give only ~+0.4-0.8 adj individually.
> - **Published literature confirms the architectural lever**: vanilla ResNeXt-50 hits 93.97 % on Language (Geada et al. arXiv 2404.02189), PC-DARTS 90.12 %, best published competition submission 89.71 %. **The 85.20 % harness benchmark is a stripped-config ResNet-18 floor, not a ceiling.**
> - **Aug-off + Mixup is novel in the published literature** — no past team documents per-dataset aug policy on Language.
> - **Next experiment: add ResNeXt-style grouped conv to the OP_VOCAB** as a 10-line search-space extension (Option A in the Deep Dive note). Expected to be the architectural lever for Language (and possibly CIFARTile + Cryptic). v5 P1 pipeline can then ship Mixup-on-OFF + the extended ops together.

### E. Comparison vs prior results (consolidated)

| Method | Total adj (13 ds, mean ± σ) | Δmean vs Tier-1 | σ ratio vs Tier-1 | Language adj |
|---|---:|---:|---:|---:|
| Tier 0 (hardcoded CNN) | −0.36 / 11 ds | — | — | (not in scope) |
| Tier 1 multi-seed (full) | +11.27 ± 8.21 / 11 ds | baseline | 1.0 | −8.14 / −10 / −10 |
| Tier 1 multi-seed (aug-off arm) | (partial sweep) | — | — | **−2.56 ± 1.69** (best Language adj seen) |
| E1 (ensemble) | (mixed verdict, AddNIST σ fix) | — | partial | — |
| **EGNAS-full** | **+35.51 ± 3.07 / 13 ds** (internal) | **+24.24** | **×0.37** ← σ collapse | −8.32 ± 1.56 |
| **P1 v4** | +31.19 ± 7.62 / 13 ds | +19.92 | ×0.93 | **−4.60 ± 2.25** |

The σ-collapse story remains EGNAS-full's headline finding ([[Tier 2b - Results]] §Q8). P1 v4's contribution is **recovering Language by +3.72 mean** — real, probe-attributable, but eaten by probe-time perturbation and EGNAS-internal variance on skip-probe datasets.

The submission-relevant view: **best single seed across methods**:
- EGNAS-full s1337: +39.40 (shipped)
- **P1 v4 s42: +41.29 (shipped)** ← current best single-seed total

So *on the seed where everything goes right*, P1 v4 is the best configuration we have. The robustness question (does it ship consistently across unseen-data seeds?) is open.

## Three-iteration history — the probe-design progression

### v1 (failed smoke, never swept) — raw-val rule
| Aspect | Detail |
|---|---|
| Rule | OFF iff `val_off − val_on > 0.5 pp` |
| Failure | AddNIST smoke: aug-off val 0.208 > aug-on val 0.162 at K=4 — picked OFF, knew this would crater (multi-seed shows AddNIST aug-off = −10 floor) |
| Root cause | At K=4, aug-off over-fits faster → raw val misleadingly high |
| Fix path | Penalize the over-fit signature (train-val gap) |

### v2 (swept; **net −4.14 vs EG**) — gap-penalised score, symmetric ±0.5 pp
| Aspect | Detail |
|---|---|
| Rule | Score = val − 0.3·max(0, train − val); OFF iff Δscore < −0.5 pp |
| Outcome | Language fixed +5 (s42/s1337/s2024 all OFF correctly); AddNIST s2024 OFF crater (−0.37 adj); MultNIST s42+s1337 OFF crater (−4.94 / −3.46) |
| Root cause of false-OFFs | ±6 pp noise band; symmetric ±0.5 threshold deep inside it; coin-flip on borderline |
| Fix path | Asymmetric threshold — only OFF on strong signal |

### v3 (swept; **net −10.11 vs EG**) — asymmetric −6 pp threshold
| Aspect | Detail |
|---|---|
| Rule | OFF iff Δscore < −6 pp; otherwise ON |
| Outcome | Language fixed +5; AddNIST s42 cratered to −10 floor (!); GameOfLife s42 also crater (EGNAS-internal) |
| Root cause of failure | K=4 noise floor wider than estimated (~±8 pp not ±6); AddNIST Δscore swung +7.66 → **−7.71** across reruns; −6 threshold not enough |
| Fix path | Reduce noise at source (multi-trial OR longer K), not threshold |

### v4 (swept; **net −4.32 vs EG**) — 3-probe shadow (A drives K=4 unanim, B/C shadow)
| Aspect | Detail |
|---|---|
| Rule | A: 3 trials of K=4, unanimous OFF for OFF; B/C log only |
| Outcome | Language fixed s42/s1337 (+3.4, +5.0); Language s2024 false-ON (only 2/3 unanimity); GameOfLife s1337 EGNAS-internal crater (−6.7 raw — not probe) |
| Root cause of failure | K=4 unanimity too strict on borderline-Δscore Language s2024 |
| Decisive shadow data | B and C BOTH correct on s2024 — **longer K is the fix, not multi-trial** |
| Fix path | v5 = swap driver A → C (multi-trial K=8) or B (single K=8); both got the disagreement right |

## Decision / carry-forward

1. **Don't ship v4 as-is.** s2024 false-ON costs +5 adj on that seed; net mean is still below EGNAS-full.
2. **v5 path is data-backed and minimal**: swap driver from A (multi-trial K=4) to **C (multi-trial K=8)** in `helpers.py:probe_aug_safety()` — one line change. Probe cost goes 36s → 72s per dataset, still <1 % of 2 h budget. Expected outcome: all Language calls correct, Language Δmean → ~+5.5 (no s2024 miss), v5 net likely **≥ EGNAS-full**.
3. **The Language ceiling problem is independent of P1.** Even perfect probe decisions cap Language at ~83% raw. The fix path is **either longer training (cheap, patience tweak) or architectural (E2, expensive)**. See §D for the menu.
4. **EGNAS-full alone remains the best documented baseline** for shipping right now (+35.51 ± 3.07 internal, 13 ds, 3 seeds). The submission-relevant single-seed champion is **P1 v4 s42 = +41.29 shipped** — but that's one lucky seed, not the expected value.

## Run mechanics

- **Driver baseline (v4):** `baselines/tier2a_p1_augsafety_v4/`. Probe lives in `helpers.py:probe_aug_safety()` + helpers `_run_one_trial()` and `_run_probe_config()`.
- **Prior baselines kept for comparison:** `tier2a_p1_augsafety/` (v2), `tier2a_p1_augsafety_v3/` (v3).
- **Sweep launch:** `dev/run_tier2a_p1_v4.sh`. 13 datasets × 3 seeds × `--time-limit 2.0` ≈ 12-16 h.
- **Per-dataset probe JSON:** `dev/logs/tier2a_p1_v4/p1v4_s<seed>/<codename>_p1_probe.json` — carries all 3 configs' per-trial Δscores, votes, decisions.
- **Aggregator:** `dev/aggregate_tier2a_p1_v4.py` → `_tier2a_p1_v4_aggregate.json` + verdict table.
- **Plots:**
  - Paired bars + σ: `python3 dev/plot_tier_results.py --tier2a-p1-v4` → `tier2a_p1_v4.png`
  - Aggregated curves vs EG: `python3 dev/plot_tier_results.py --tier2a-p1-v4-curves` → `tier2a_p1_v4_curves.png`
  - Per-seed: `python3 dev/plot_tier_results.py --tier tier2a_p1_v4_s{42,1337,2024}` → 3 × scores + curves PNGs
