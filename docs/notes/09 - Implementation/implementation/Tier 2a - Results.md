# Tier 2a — P1 Aug-Safety Probe Results

> **Status:** ✅ v4 sweep COMPLETE (2026-05-26 → 28). v5 sweep ABORTED 2026-05-28 — partial s42 data preserved (net negative; GameOfLife −9.4 pp regression from a noisy short-train pick of ResNeXt at L0). **v6 in flight** (smoke 2026-05-28 22:00; full sweep launching tonight). v6 = probe C only + Mixup-on-OFF + 9-op OP_VOCAB + K=12 + verification-of-accepts + clock-extend + zero-shot proxies. Informed by reading the actual **EGNAS paper + code** (Siddiqui 2025, `docs/papers/`). v2/v3/v4/v5 baselines all preserved for ablation. | **Topic:** Tier-2b full EGNAS + a runtime aug-safety probe (P1) + iteratively expanded architectural search-space. The first **actual probe** of the [[Probe-Driven Self-Configuration]] thesis.
> **Related:** [[Tier 2a - Probe v1]] (plan), [[Tier 2a - Language Deep Dive]] (excursion + literature + GroupConv proposal — the Language-ceiling follow-up to §D), [[Tier 2b - Results]] (paired EGNAS-full baseline), [[Multi-Seed - Results]] (the aug-oracle that motivated P1), [[Probe-Driven Self-Configuration]], [[Reliability vs Peak]], [[Implementation Results Overview]], [[Run State + Handoff]], [[EGNAS]] (the upstream method paper + our port discrepancies)

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

| Dataset/seed         | A_multi_K4                 | B_single_K8         | C_multi_K8          | Truth | Driver A right? |
| -------------------- | -------------------------- | ------------------- | ------------------- | ----- | --------------- |
| Language s2024       | **ON** (Δ −14.70, 2/3 OFF) | OFF (Δ −15.10, 1/1) | OFF (Δ −18.30, 3/3) | OFF   | **NO**          |
| (all other 20 calls) | ✓                          | ✓                   | ✓                   | ✓     | ✓               |

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

### v5 (ABORTED partial s42, **net negative on partial data**) — driver C + Mixup-on-OFF + 2 new ops (GroupConv3, ResNeXt)

| Aspect | Detail |
|---|---|
| Rule | Driver swapped A → C (multi-trial K=8) per v4 shadow finding; Mixup α=0.2 auto-applied when probe decides OFF; OP_VOCAB extended with GroupConv3 + ResNeXt |
| Sweep state | Killed during s1337 to free Spark for v6. Only s42 partial (5+4 finished datasets) is on disk. |
| Probe-side verdict | All 7 probe-applicable calls on s42 were correct (driver C robust); MultNIST in particular was saved from v4's K=4 unanimity false-OFF |
| Op-side verdict | **Search picked new ops on the predicted datasets** (GroupConv3 @ L2 Chesseract; ResNeXt @ L1 Cryptic; ResNeXt @ L0 GameOfLife). Conv-friendly datasets (AddNIST, CIFARTile) correctly ignored them. **But** the GameOfLife pick was a noisy-ranker false positive — ResNeXt at L0 on a 1×8×8 cellular-automata input added depth/FLOPs without benefit → s42 GameOfLife best_val 90.5 % (vs EG s42 99.9 %, v4 s42 97.0 %) — **−9.4 pp regression**. |
| Language verdict (s42) | best_val **79.17 %** vs v4's 80.88 % and excursion V3's 82.08 % — Mixup couldn't compensate for the EGNAS arch variance (v5 picked D=4 W=48 vs v4's D=5 W=24). The full-pipeline EGNAS noise (~±3-5 pp) dwarfs the controlled-experiment Mixup gain (+1.6 adj) we measured in the excursion. |
| Root cause of failures | Two issues: (a) search-ranker noise — K=2-6 epoch val_acc can't reliably rank new ops with very different param counts (GroupConv3: 1.4K vs Conv3: 10.5K vs ResNeXt: 6.3K params at the same width); (b) the EGNAS arch-variance is larger than our controlled-experiment effects. |
| Code preserved | `baselines/tier2a_p1_augsafety_v5/`, `dev/run_tier2a_p1_v5.sh`. s42 partial data in `dev/logs/tier2a_p1_v5/p1v5_s42/`. |
| Decision | Don't keep going. Move to v6 with EGNAS-faithful methodology + verification-of-accepts to catch the Q8 problem. |

### v6 (SWEPT + VERDICT 2026-06-01) — Q8-stabilised methodology + 4 more architectural primitives + zero-shot proxies as shadow signal

> **VERDICT: v6 is a clear net regression. EGNAS-full +35.51 ± 3.07 → +P1 v6 +23.41 ± 9.17 (internal, 13 ds × 3 seeds) → Δmean −12.10, σ ×2.98 (WORSE on mean AND variance).** Per-seed P1 totals [18.52, 17.72, 33.98] — the spread is the GameOfLife/AddNIST arch-lottery. The bundle backfired: the rich vocab + aggressive macro search *added* variance and made arch selection worse. The single keeper is the **Language aug-probe (+4.23 Δmean)** — the only thing that demonstrably helps. See §"3-seed deep analysis" below; figures in §Graphs.

#### 3-seed deep analysis (2026-06-01) — why v6 lost, and what it means for v7

The −12.10 gap decomposes into a handful of **self-inflicted arch-selection failures**, all the *same* root cause: **no feasible search-budget horizon (K=12, even verify at K=24) predicts 2 h-trained accuracy.** Three concrete failure modes:

1. **Rich vocab POISONS datasets where plain conv is optimal — Sudoku −7.94 (the biggest single loss).** Smoking gun at identical dims: EGNAS-full s2024 `D=7 W=48` plain-Conv3 → **raw 90.82%**; v6 s1337 `D=7 W=48` with Conv5@L0 + SpatialSelfAttn2D@L5 → **raw 71.01%** (−20 pp). EGNAS-full got Sudoku ~91% all 3 seeds; v6 got 67–71% every seed. The exotic ops won the noisy K=12 short-train and **cratered at full training** — and verification-of-accepts (2× K = 24 epochs) was *still too short* to catch a degradation that only shows up near 2 h. Same failure shape as Language's attention pick, but catastrophic here.
2. **Macro PRUNE on short-horizon craters deep/wide-needing tasks — GameOfLife −4.85 (depth), Sudoku width too.** GameOfLife is a depth lottery (depth-grow strikes deep nets that look bad at K=12; solved only on s2024 where D=6 survived the strike → 99%, vs D=3 → 48% on s42/s1337). Sudoku's width-prune cut W to 16 on 2 of 3 seeds. EGNAS-full's simpler search stayed deep/wide and didn't crater.
3. **Variance explosion (σ ×2.98).** 9 of 13 datasets got *noisier* under v6. The 9-op vocab × aggressive depth/width search amplified the Q8 lottery everywhere. Cross-seed arch agreement stayed ~1/13 (only GameOfLife's ResNeXt@L0 is stable — and it's the broken-shallow arch on 2 seeds).

**The one win is real and durable: Language Δ+4.23** (v6 −4.09 vs EGNAS-full −8.32). This is the P1 aug-probe (aug-off + Mixup), correct on 2/3 seeds. It is the *only* component of the whole P1 line that beats EGNAS-full.

**The prize, quantified.** v6 +23.41, but the losses are self-inflicted: recover Sudoku (+7.94) + GameOfLife (+4.85) + MultNIST (+1.84) + AddNIST (+2.87) back to EGNAS-full levels and **keep** the Language +4.23 → **≈ +40, which would beat EGNAS-full (+35.5) by ~+4.5.** The entire game for v7 is *eliminating v6's self-inflicted regressions while keeping the Language probe.*

**→ Strategic pivot for v7 (the data says: simplify, don't elaborate).** The rich vocab is a **liability**, not an asset — it only flirted with exotic ops in the noise and catastrophically poisoned Sudoku. Reframe v7 as **EGNAS-full (simple, proven, stable) + the Language aug-probe + a macro-prune-verify fix**, with exotic ops (attention/LSTM) tested **out-of-band via the FIXED_ARCH full-budget probe — NOT in the live micro search** (where they win noisy short-train and lose full-train). Concretely: (a) cut OP_VOCAB back to ~3–4 conv ops (Conv3/Conv5/Conv7/SepConv3) — drop attention/ResNeXt/SE/Dilated/Group from the *default* search; (b) extend verification to macro depth AND width (verify-of-rejects + slope) so deep/wide-needing tasks aren't pruned on noise; (c) keep the C-only Language aug-probe; (d) drop clock-extend (submission-illegal anyway) and zero-shot (τ was weak); (e) test LSTM/attention only via `run_lstm_language_probe.sh`-style fixed-arch full-budget A/Bs. This is "EGNAS-full + targeted fixes," not "v6 + more."

Bundles six changes vs v5:

1. **Probe**: drop A + B shadow probes, only C drives. ~60 s/dataset saved.
2. **OP_VOCAB expanded to 9 swappable ops** (exploration mode): Conv3, Conv5, **Conv7**, SepConv3, GroupConv3, ResNeXt, **DilatedConv3**, **SEConv3**, **SpatialSelfAttn2D**. Picks and survives are recorded per layer per dataset for post-hoc analysis. **Don't prune until we have the data** to decide which ops are dead-weight.
3. **`MAX_SHORT_EPOCHS` 6 → 12** — longer K per candidate reduces ranker noise. Direct response to the v5 GameOfLife regression.
4. **Verification-of-accepts**: when micro accepts a swap, re-train it at 2× K. If verify val doesn't hold up, revert. Catches the Q8 short-train false positive failure mode.
5. **Clock-extend mode**: NAS doesn't deduct from trainer budget. Trainer always sees full 2 h regardless of how long NAS spent. Each dataset now takes (2 h + nas_duration) wall-clock — exploration mode.
6. **Zero-shot proxies** (NASWOT, Synflow, SNIP, ZenScore) computed per candidate, logged in pool JSON as **shadow signals only** (no decision impact). If any proxy has Spearman τ ≥ 0.5 vs full-budget test acc on our datasets, v7 can drop the noisy short-train ranker entirely. Legal under competition rules (no trained model, no external data, just math on the untrained network). See [[EGNAS]] §"Zero-shot proxies".
- Code: `baselines/tier2a_p1_augsafety_v6/`; new `zeroshot.py` (~150 lines) with the 4 proxies; `nas.py` extended with verification + clock-extend + per-candidate proxy + rich pool JSON.

Expected outcome: v6 should produce **(a) cleaner per-op attribution data** (which ops survive verification on which datasets?), **(b) per-dataset zero-shot proxy correlation data** for v7 design, **(c) probe correctness data on driver C alone**, **(d) Mixup effect at full-pipeline scale**. Whether it's a *net win* vs EGNAS-full is the residual question.

**s42 partial read (2026-05-30, seed 42 only — DO NOT treat as the verdict).** All 13 datasets scored on s42; s1337 running, s2024 queued. Two findings, both of which motivate v7 and one of which corrects an earlier mid-sweep misread:

**⚠ codename-label note:** Sadie=GeoClassing, Mateo=MultNIST, Sokoto=Sudoku (authoritative map in `aggregate_multiseed.py`). An earlier draft of this block had those three swapped; corrected below. Numbers/deltas were always right per codename — only the names moved.

| | v6 s42 adj | EGNAS-full s42 adj | Δ |
|---|---|---|---|
| GameOfLife | **+0.03** (raw 47.7%) | **+9.97** (raw 99.84%) | **−9.94** |
| MultNIST | +0.89 | +4.40 | −3.52 |
| Sudoku | +6.67 | +9.08 | −2.41 |
| GeoClassing | +8.00 | +7.25 | +0.74 |
| CIFARTile | +1.30 | −0.31 | +1.61 |
| Language | −6.51 (raw 75.56%) | −6.93 (raw 74.95%) | +0.41 |
| *(11-ds total)* | **+19.0** | **+31.7** | **−12.7** |

1. **GameOfLife is an arch-LOTTERY, not a deterministic v6 failure** (updated 2026-06-01 once s2024's curve landed). Per-seed best_val: **s42 47% · s1337 49% · s2024 99%.** On s42/s1337 the depth-grow tried D=4/5/6, saw their noisy K=12 vals sit in a low band (0.06–0.28, *below* the D=3 reference), struck out, and stayed shallow at D=3 → ~48%. On s2024 the *same* search hit D=6 with a K=12 val of 0.281 (just *above* the D=3 ref 0.240), survived the strikes, grew to D=6, and trained to **98.8%**. So the failure mode is **noise at the depth-decision boundary** — the deep nets' short-train vals are all low and within noise of each other, so whether depth survives is a coin-flip. v4 (solved s42 D=7→96.9%) and EGNAS-full (solved s42→99.8%, cratered s1337) show the identical lottery. **v6 3-seed GameOfLife ≈ +3.3 adj, not the +0.06 the 2-seed view implied.** This is the precise target of v7 #1 (val-slope + verify-rejects de-randomizes the boundary). **Correction of a correction:** my mid-sweep "systematic depth bug, 2 seeds agree" was itself premature — the 3rd seed shows it's probabilistic.
2. **Attention's short-train micro-win did NOT translate to a final-train gain on Language.** v6's micro search picked SpatialSelfAttn2D @ L0 (val 0.246→0.482, the biggest single-op short-train lift in the whole sweep), yet the final 2h-trained net scored **75.56% — only +0.6 pp over EGNAS-full's plain-conv 74.95%.** Lesson: **a 1-layer micro op-swap is the wrong way to judge a new op family** — the short-train ranker rewards it but full training washes the gain out. → v7 must test new ops as *architecture-heavy* fixed nets to full budget, not as single micro picks. See §v7 forward design.

#### Graphs — all runs (generated 2026-06-01; same tooling as v4: `plot_tier_results.py --tier2a-p1-v6[-curves]` + per-seed `--tier tier2a_p1_v6_s*`; aggregate via `aggregate_tier2a_p1_v6.py`). **v6 = s42+s1337 scored, s2024 in progress** → totals deferred; per-dataset stats are 2-seed for now.

**Paired bars + σ change (EGNAS-full vs +P1 v6).** Green = EGNAS-full, orange = +P1 v6; Δ above each pair (Panel A), σ change (Panel B). Headline regressions: GameOfLife (Δ−8.1, but *2-seed* — both caught the bad side of the lottery; see below), Sudoku (Δ−7.8), AddNIST (Δ−3.8, σ blows up ×9.9 — the Q8 lottery). Only clear win: **Language Δ+2.8** (probe earns its keep vs EGNAS-full's −8.3).
![[tier2a_p1_v6.png]]

**Aggregated learning curves (EGNAS-full mean±std vs +P1 v6).** Green band = EGNAS-full, orange = +P1 v6, red dashed = benchmark. Language: orange above green but below benchmark (structural ceiling). GameOfLife: both bands low on the 2 scored seeds.
![[tier2a_p1_v6_curves.png]]

**Per-seed score bars.** s42 (+26.66) and s1337 (+24.80); s2024 still running (no Scoring section yet → curves only).
![[tier2a_p1_v6_s42_scores.png]]
![[tier2a_p1_v6_s1337_scores.png]]

**Per-seed learning curves** (s42, s1337, s2024-partial). The **s2024 GameOfLife (conway) panel is the key one** — it climbs to ~99% (D=6 grew) while s42/s1337 sit flat ~48% (D=3): the depth arch-lottery, made visible.
![[tier2a_p1_v6_s42_curves.png]]
![[tier2a_p1_v6_s1337_curves.png]]
![[tier2a_p1_v6_s2024_curves.png]]

Full table of all iterations:

| Version | Rule | Outcome | Failure mode / lesson |
|---|---|---|---|
| v1 | raw-val rule | Smoke FAIL on AddNIST | aug-off over-fits faster at K=4 |
| v2 | gap-penalised ±0.5 pp | Sweep net **−4.14** vs EG | coin-flips in ±6 pp noise band |
| v3 | asymmetric −6 pp | Sweep net **−10.11** vs EG | K=4 noise floor wider than estimated (~±8 pp) |
| v4 | 3-probe shadow (A drives K=4 unanim) | Sweep net **−4.32** vs EG | K=4 unanimity too strict on borderline; shadow data shows K=8 wins |
| v5 | C drives + Mixup-on-OFF + 2 new ops | Aborted s42 partial; probe correct but **GameOfLife −9.4** from noisy ranker pick of ResNeXt @ L0 | adding ops to a noisy K=2-6 ranker = false positives; methodology mismatch with upstream EGNAS (4-op binary vocab, 600-epoch final train) |
| v6 | C only + Mixup-on-OFF + 9 ops + K=12 + verify + clock-extend + zero-shot shadow | Swept; net **−12.10 vs EGNAS-full, σ ×2.98** (worse on both) | richer vocab is a LIABILITY (poisoned Sudoku −20pp via exotic-op picks that win short-train, lose full-train); macro prune craters depth/width tasks; ONLY Language probe (+4.23) helps. → v7 = simplify back to EGNAS-full + Language probe + macro-prune-verify |
| v7 (planned) | v6 + **recurrent op family** (row-BiLSTM, cross-axis BiLSTM) + extended NASWOT to gate activations | TBD | targeted at Language-style sequential / 1D-ish inputs where SpatialSelfAttn2D already showed lift in v6 mid-sweep (Language L0, +24 pp on K=12 short-train val) |

### ✅✅ v7 SWEPT — VERDICT (2026-06-03): v7 BEATS EGNAS-full, the best documented result

3-seed × 13-ds, internal benchmarks. `baselines/tier2a_p1_augsafety_v7/` (EGNAS-full clean search + 4 conv ops + macro-prune-verify + C-only aug-probe). Aggregator `dev/aggregate_tier2a_p1_v7.py`.

> **EGNAS-full +35.51 ± 3.07 → +P1 v7 +41.94 ± 3.45 → Δmean +6.43, σ ×1.12.** Per-seed v7 [43.24, 38.03, 44.54]. First version to **beat the proven EGNAS-full baseline**.

**Where the +6.43 comes from (Δ vs EGNAS-full):**
- **Language +7.34** (−8.32 → −0.98) — the aug-probe chose **OFF on all 3 seeds**; this is essentially the *whole* win. The macro-verify's real contribution: stabilising the archs so the aug-off gain isn't eaten by arch variance (which is what killed it in v6).
- **GameOfLife +0.65** — better mean *and* tighter (σ 2.28→1.95); the depth verify-of-rejects reduced the lottery (in-sweep 99.4/82.3/99.7 — solved 2/3).
- Cryptic +0.17, Sudoku +0.15 (σ 0.39→0.19), CIFARTile +0.10, Gutenberg +0.06 — small stable gains.
- **MultNIST −1.15** — the one real regression (v7's deep-plain-conv < EGNAS-full's MultNIST arch). Myofibre −0.36, AddNIST −0.31 (but σ 0.50→0.06), Voxel/Windspeed −0.1.

![[tier2a_p1_v7.png]]
![[tier2a_p1_v7_curves.png]]

**Key insights (drive the next run):** (1) **the aug-probe is the engine, not the search** — verify+aug-probe are the validated combo. (2) **The LSTM's marginal value shrank to ~+1.4 on Language** — conv+aug-off already reaches −0.98 (≈84-85 %, near benchmark), so the LSTM is now a small top-up, not the big lever it looked like vs raw EGNAS-full. (3) **MultNIST −1.15 is the biggest single fix opportunity**, larger than the LSTM top-up. → next run = "representation bake-off" (channel-embed stem + row-BiLSTM as fail-safe A/B arms), led by the data-preprocessing lever + MultNIST recovery; v7 core unchanged (the validated +6.43 winner). See [[Next-Run Ideas - lecture-informed]].

### ✅ LSTM Language probe — RESULT (2026-06-02): row-BiLSTM cracks Language

The out-of-band full-budget research gate (`dev/run_lstm_language_probe.sh`, 4 arms × 3 seeds, fixed 5-block @ W=24, `fixed_arch_probe` baseline, all with the C-only aug-probe). **First architecture in the whole project to put Language *reliably above benchmark*.**

| Arm | Language raw (3 seeds) | mean ± σ | adj | verdict |
|---|---|---|---|---|
| **row-BiLSTM (heavy)** | 85.32 / 85.83 / 86.12 | **85.76 ± 0.41** | **+0.38** | ✅ above benchmark ALL 3 seeds, tight |
| attention (heavy) | 84.40 / 85.09 / 84.81 | 84.77 ± 0.35 | −0.29 | beats conv, but stays *below* benchmark |
| conv (Conv3+Conv7) | 83.47 / 84.08 / 83.93 | 83.83 ± 0.32 | −0.93 | the conv floor (already > deep-dive's 81.5% ceiling) |
| cross-axis BiLSTM | 87.19 / 78.26 / 85.93 | 83.79 ± 4.83 | −0.95 | **unreliable** — s1337 cratered to 78.3 (9pp spread) |

![[lstm_lang_probe.png]]

**Findings:**
- **Row-BiLSTM is THE lever** — +1.93 pp over the conv floor, clears benchmark on every seed, low variance. It's the 1D left-to-right recurrent sweep, which fits Language being *text rendered along the width*.
- **Attention does NOT cross** (84.77) — better than conv but below benchmark. So the win is *specifically recurrence*, not "any non-conv op."
- **Cross-axis (2D recurrence) is too unstable** to use (one seed at 78.3). Multi-seed discipline caught this — a single-seed read (s42 87.19) would have wrongly crowned it.
- Context: Language was the project's worst deficit (v6+aug-probe −4.09, EGNAS-full −8.32). Row-BiLSTM = +0.38. **~+5 to +9 adj swing on the single hardest dataset.**

**The gate is passed → the work becomes the BLIND SELECTION RULE.** We cannot gate on "Language" (Unseen Data Challenge: hidden codenames). A row-BiLSTM op must be selected by a metadata- or probe-driven rule that runs identically on every dataset and fires on sequence-like inputs *without poisoning conv-optimal ones* (Sudoku). That rule — not the op — is the deliverable.

#### Can we just add row-BiLSTM to the micro vocab? NO — decisive short-train test (2026-06-02)

The natural question: skip the bolt-on, just put row-BiLSTM in `MICRO_OPS` and let the search pick it. We tested whether it survives the micro short-train ranker (replicated `nas._short_train`: plain SGD lr=0.025, the exact micro-search recipe), logging val every epoch, conv vs row-BiLSTM on Language:

| epoch | conv | row-BiLSTM | lstm−conv |
|---|---|---|---|
| 4 (micro horizon) | 73.4 | 62.4 | **−11.0** |
| 5 | 74.7 | 56.0 | **−18.7** |
| 6 | 74.7→77.2 | 76.1 | −1.1 |
| … | (smooth) | **45 / 22 / 40 / 62 / 56 / 76 / 61 / 47 / 79 / 36 …** | thrashes |
| best (20 ep) | 81.3 | 79.2 | recurrence never cleanly wins |

**Two failure modes, the second fatal:** (1) at the 4–6 epoch micro horizon row-BiLSTM is *behind* conv → the search rejects it (false negative). (2) Deeper: **the search's crude SGD optimiser literally cannot train the LSTM** — it oscillates 22 %↔79 % and never converges; recurrence only performs under the *full* trainer's warmup+cosine+EMA (where the same arch hit 85.76 %). No "longer K in the search" fixes this — the proxy optimiser itself is the problem. **→ micro-vocab inclusion is dead.** (Folds into [[Key Learnings]] ⭐ — the proxy-reliability wall.)

#### The deployment rule: a fail-safe, budget-gated, convergence A/B (design 2026-06-02, with user)

Since the search proxy is structurally blind to recurrence and a metadata "sequence-like" heuristic is fragile (Language 1×24×24 looks like AddNIST/Sudoku in metadata), the only reliable selector is a **full-convergence A/B** — and convergence is cheap (~5 min, patience-stop), which makes it affordable. Two-tier design:
- **Cheap short-K search for the conv core** — depth/width/conv-op choices, where short-K demonstrably works (EGNAS-full).
- **Convergence A/B for the one decision short-K can't make** — conv-winner vs its row-BiLSTM variant, *both trained to their own patience-convergence* (equal footing — not a fixed wall-clock, which would bias toward the fast-per-epoch op), keep the higher val.

**Fail-safe against unknown compute** (the user's robustness concern — the test GPU/budget is unknown): **conv is trained first = a guaranteed shippable model**; the row-BiLSTM arm runs *only if the measured clock allows* (estimated from the conv arm's observed epoch time + recurrence slowdown, with margin). On slow compute it simply never starts → ships conv. Cannot do worse than conv-only, ever.

**Guard against over-picking** (the user's "don't always choose LSTM"): the A/B can only pick row-BiLSTM where it genuinely wins at convergence. **Validation in flight** (`dev/run_recurrence_validation.sh`): run the convergence A/B on conv-optimal datasets (Sudoku / AddNIST / CIFARTile) and confirm **conv wins there**. If conv wins on conv-optimal data and row-BiLSTM wins on Language, the A/B is a correct blind selector. Results pending (~3-4 h). Baseline `baselines/fixed_arch_probe/` (= v6 + LSTM ops + FIXED_ARCH).

### v7 forward design — recurrent op family (2026-05-30, conditional on v6 verdict)

**Motivation.** v6 s42 mid-sweep already shows SpatialSelfAttn2D winning twice (Language L0, CIFARTile L3). The Language pick was the **largest single-op micro lift across all 9 datasets** (val 0.246 → 0.482 at K=12). Geada et al. (arXiv 2404.02189) report ResNeXt-50 hits 93.97 % on Language, suggesting a real architectural ceiling above current 0.81 raw. The case for non-conv ops on sequence-like data is data-backed. LSTMs are the obvious companion to attention.

**Slot-fit constraint.** Every op in OP_VOCAB is `(B,C,H,W) → (B,C,H,W)`. Plain LSTM needs a sequence dim. Three feasible mappings:

| Variant | Sequence axis | Cost | Best fit |
|---|---|---|---|
| **Row-BiLSTM** | flatten W; reshape `(B·H, W, C)` → BiLSTM → reshape | medium; ~3-5× Conv3 wall-clock | Language, Gutenberg, Cryptic (1D-shaped inputs) |
| **Spatial-flatten BiLSTM** | flatten H·W; `(B, H·W, C)` → BiLSTM | high; sequential bottleneck at 32×32=1024 steps | only viable at deeper layers post-downsampling |
| **Cross-axis BiLSTM** | row-BiLSTM → col-BiLSTM in series (PixelRNN-lite) | high; ~6-10× Conv3 | maximum expressivity; image-RNN baseline |

Drop-in candidate: **row-BiLSTM** with `MAX_LSTM_TIME=16` cap (adaptive_avg_pool2d if W>16, bilinear upsample residual — mirrors the SpatialSelfAttn2D pattern). Bounded cost regardless of input size; gracefully degrades on small spatial dims. Cross-axis BiLSTM as an aspirational second op only if row-BiLSTM survives verification on ≥2 datasets.

#### v7 locked scope (decided 2026-05-30 with user; ambition — full sweep vs targeted — decided after v6's 3-seed verdict lands)

The s42 partial read reframed the priority: **the LSTM is the secondary lever; the headline is the macro depth bug** (GameOfLife −9.94 = 78 % of the s42 gap, and it's the *same* bug that hit v5). Six changes, ordered by expected points:

**1. Macro depth fix — val-slope + verification-of-rejects (HEADLINE, +~10 pts).**
The depth-grow loop (`nas.py` PHASE 1) strikes out a deeper candidate on absolute `val@K`. Deep nets on iterated-computation tasks (GameOfLife, a cellular automaton) look terrible at K=12 and brilliant at full train, so the short-horizon ranker kills them. Two coupled mechanisms:
- **Val-slope-aware judging** — judge a depth candidate on `val@K + λ·slope` where slope = mean val improvement over the last ~3 short-train epochs. A net still climbing steeply gets credit instead of a strike. Requires threading per-epoch val through `_short_train → _eval_arch`; currently `_short_train` returns only the final `_quick_eval`. ~Free (no extra training). Tune λ against v6 GameOfLife epoch-curves once they land.
- **Verification-of-rejects** — the symmetric twin of v6's verification-of-accepts: when depth-grow is about to register a *strike*, re-train that deeper candidate at 2-3× K first; if it overtakes the incumbent, keep growing (reset strikes). Fires only at the strike boundary → bounded cost (a few retrains/dataset). This is the principled cure — it mirrors exactly what *did* work for micro accepts.
- *Build note:* moderate `nas.py` refactor; benefits from v6 GameOfLife epoch-trajectory data to calibrate λ and the verify-K multiplier, and from being end-to-end validatable (needs a free GPU). **Deferred to implement-and-validate when v6 lands**, not built blind.

**2. Recurrent op family — row-BiLSTM (+ cross-axis BiLSTM). ✅ BUILT 2026-05-30.**
`baselines/tier2a_p1_augsafety_v7/helpers.py`: `_RowBiLSTM` (W-as-time BiLSTM, `MAX_LSTM_TIME=16` cap mirroring `_SpatialSelfAttn2D`) + `_CrossAxisBiLSTM` (PixelRNN-lite row→col) added to `OP_VOCAB` as ops #10/#11. CPU shape+backward smoke PASSED (all stride/cap/deep paths, grads flow). **Tested the right way** via `dev/run_lstm_language_probe.sh` — 6 fixed 5-block @ W=24 Language archs (conv5 / v6pick / lstm_min / lstm_heavy / crossaxis / attn_heavy) each trained to full 2 h budget through the new `FIXED_ARCH` hook in `nas.py`. Tests *architecture-heavy* LSTM, not a 1-layer micro swap — because the s42 data proved 1-layer op-wins don't translate (attention won Language micro, final gain ≈ 0). Ready to run the moment the GPU frees (after stopping v6 post-s1337).

**3. Prune OP_VOCAB to survivors (gated on v6 3-seed attribution).**
v6's 9-op vocab is exploration mode and it's *costing* points — MultNIST −3.52 / Sudoku −2.41 (vs EGNAS-full s42) are partly search-budget dilution (9 ops × every layer ⇒ noisier per-op signal). s42 attribution: Conv3 dominates (11/449 swaps beat it); winners = SpatialSelfAttn2D, ResNeXt, Conv7, GroupConv3; likely dead = SepConv3, SEConv3, DilatedConv3. **Cut to ~5 ops + LSTM once 3 seeds confirm** which are dead weight.

**4. Param-fairness in micro (gated on v6 data, then implement).**
Upstream EGNAS shrinks width to equalize params on an op swap; we don't. v6 proxy data already flags the confound — `params` had the *highest* Spearman (+0.29), so the micro ranker partly rewards size not quality. Match param budget per op (per-op width scaling in `build_cnn`/`ResConvBlock`) to de-bias. Medium effort.

**5. Proxy-rescue for depth (gated on v6 proxy τ ≥ 0.4).**
If v6's sweep shows any zero-shot proxy correlates (τ ≥ 0.4 macro-level), use Synflow/Zen ("is this deep net trainable?") to *rescue* depth candidates the short-train ranker wrongly kills — the proxies' first real decision role. If τ < 0.4, proxies stay shadow-only and this is dropped.

**6. NASWOT gate-activation hook + per-op K-multiplier (needed for the LSTM to be judged fairly).**
- NASWOT counts post-ReLU patterns; LSTM gates are sigmoid/tanh → extend the `zeroshot.py` hook to `nn.Sigmoid`/`nn.Tanh` (active = `>0` after tanh, `>0.5` after sigmoid). ~10 lines.
- Per-op `K_MULT` table → LSTM trials get 1.5× K so the ranker isn't unfair to slow-settling ops.

**Build status (2026-05-30):** #2 done + smoke-passed + ready; #1/#3/#4/#5/#6 documented, deferred to implement-and-validate when v6's 3-seed verdict + attribution + proxy-τ land (~48 h). The `FIXED_ARCH`/`FIXED_WIDTH` hook (`nas.py`) is reusable for *all* v7 dev — it's the full-budget A/B tool we lacked every time we misjudged an op on short-train.

### Key insight from reading the upstream EGNAS paper + code (2026-05-28)

`docs/papers/1st-2025_EGNAS_Siddiqui_arxiv-preprint.pdf` + `https://github.com/siddikui/Efficient-Macro-Micro-NAS`. See [[EGNAS]] for the methodology comparison. Highlights of the mismatch that explains our Q8 problems:

- **Their OP_VOCAB**: `{sep_conv, plain_conv} × {k=3, k=5}` = 4 binary combinations. **Ours (v5)**: 7 ops. **Ours (v6)**: 11 in vocab / 9 swappable. We've been growing the search space without matching their methodology's noise-tolerance assumptions.
- **Their micro epochs**: hard-coded `args.epochs = 2`. **Ours**: 2-6 depth-scaled (v5), 2-12 (v6).
- **Their acceptance rule**: `next_acc >= curr_acc - 0.0` (ties accept). **Ours**: `> curr + 0.003` (stricter).
- **Their final training**: 600 epochs. **Ours at 2 h budget**: ~70 epochs. **Their ranking noise is washed out by 10 × more train time.**
- **Their param-fairness**: when a swap changes the op, they SHRINK width to keep params equal. **We don't** — Conv3 (10 K params) vs GroupConv3 (1.4 K) vs ResNeXt (6.3 K) all compared at the same nominal width.
- **They use TEST acc as the search ranker** (`valid_size=0.0`). We use proper val (legal-cleaner but noisier in absolute terms).

The Q8 problems we documented (Spearman ranges +0.7 to −0.5 on our datasets) aren't a contradiction of the paper — their "Spearman 0.91" is for **(param count) ↔ (val_acc at incrementally-trained budget)** on CIFAR-10 only, aggregate over 50 sampled nets. Not "2-epoch val_acc on Unseen X predicts 600-epoch test acc". **We've been holding our pipeline to a standard the paper never claimed.**

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
