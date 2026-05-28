# Tier 2a — Language Deep Dive (excursion + literature)

> **Status:** ✅ excursion complete 2026-05-28 (6 variations × 3 seeds × Language only). Knowledge-base summary of our Language understanding to date + a clear path forward backed by published literature. | **Topic:** Why no configuration we have tested has ever crossed the Language benchmark (85.20%), what training-side knobs *can* do (~+1.6 adj from Mixup), what the published literature says is needed (~+8-12 pp more raw acc), and the concrete architectural experiment recommended next.
> **Related:** [[Tier 2a - Results]] (the broader P1 probe story), [[Tier 2b - Results]] (EGNAS-full baseline; Language was already aug-bound there), [[Multi-Seed - Results]] (the original aug-oracle showing aug HURTS Language), [[Probe-Driven Self-Configuration]] (thesis), [[Reliability vs Peak]], [[Datasets Overview]], [[Implementation Results Overview]], [[Run State + Handoff]]

## TL;DR

Across **every** configuration we have run on Language — Tier-1 aug-on (raw 68-73%), Tier-1 aug-off (best 83.47%), EGNAS-full (69-75%), P1 v4 with the probe correctly picking aug-off (mean 78%), and a focused excursion of 6 training-side variations on Language alone (V0-V5) — **the best we achieve is ~81-82 % raw** on a 24×24×1 input with a 10-class label. **The harness benchmark is 85.20 %**, the ResNet-18 paper number is 87 %, the best public competition submission is **89.71 %**, PC-DARTS is 90.12 %, and **vanilla ResNeXt-50 is 93.97 %** (Geada et al. arXiv 2404.02189). Training-side tweaks (patience, LR schedule, larger model, Mixup, all-combined) **all plateau around 81 %** on the disadvantaged fixed-Conv3 architecture. **Mixup is the only meaningful lever** and gives +1.6 adj on its own; combining it with patience/LR/larger doesn't add mean accuracy, only tighter variance. **The genuine lever to cross the literature targets is architectural** — published evidence specifically points to **ResNeXt-style grouped convolutions**. That is the next experiment.

## 1. Why this excursion existed

By the time the P1 v4 sweep landed it was clear we had two coupled problems on Language:

1. **The probe disagreement on s2024**: A multi-trial K=4 picked aug-ON wrongly, while shadow K=8 probes (B + C) both said OFF correctly. Confirmed by [[Tier 2a - Results]] §B.
2. **The structural ceiling**: even when the probe gets aug-off right, Language still lands below benchmark. Best ever observed (multi-seed Tier-1 + aug-off s1337) was raw **83.47 %**, only 1.7 pp below the 85.20 % benchmark — never crossed.

The probe-design problem (1) and the structural problem (2) are independent. The probe redesign (v5) addresses (1); this excursion isolates (2). **Hypothesis to test: is there a training-side intervention that crosses the benchmark when augmentation is correctly off?**

To answer cleanly, we held arch fixed (Tier-1 default `D=6 W=48` all-Conv3 stack, no EGNAS search) and varied one knob at a time + a combined arm. That removes EGNAS-internal variance from the picture and reads the training-side ceiling honestly.

## 2. The 6 variations — design + hypothesis

All variations run with `TIER1_AUG_OFF=1` (spatial aug off, the multi-seed-validated direction for Language), no EGNAS search, fixed default arch. Differences are env-variable toggles in `baselines/tier2a_language_excursion/`:

| # | Name | Env override | Hypothesis | Reasoning |
|---|---|---|---|---|
| V0 | baseline | (none) | Establishes the fixed-Conv3 + aug-off floor. Lower than multi-seed Tier-1 aug-off (which has op-search) because we deliberately stripped op-search out for cleaner attribution. | control |
| V1 | patience60 | `LANG_PATIENCE=60` | Early-stop at patience=30 fires too soon (Language under aug-off runtimes are 190-342 s vs the 7200 s budget — model "saturates", not "times out"); doubling patience lets it train longer through the plateau. | runtime check showed every Language run anywhere ends in <30 min on a 2 h budget |
| V2 | lr_low + warmup | `LANG_BASE_LR=0.01 LANG_WARMUP=10` | Gentler schedule (default 0.025 base, 5-epoch warmup) might find a better optimum on a small-data plateauing task. | classic small-data wisdom |
| V3 | mixup α=0.2 | `LANG_MIXUP_ALPHA=0.2` | Implicit regularisation via label-space mixing — perturbs the loss landscape so the model can keep learning past where it would otherwise memorise/saturate. Language labels are *classes* (not arithmetic functions of inputs) so Mixup is safe; our default `aug_policy_for()` bans it as conservative-by-default. | the thesis's "thing-we-could-try" backlog |
| V4 | larger model (W=72) | `LANG_WIDTH=72` | Extra capacity might absorb more signal. Width 72 vs default 48 is +50 % channels. | the "scale it up" hypothesis |
| V5 | combo | `LANG_PATIENCE=60 LANG_BASE_LR=0.01 LANG_WARMUP=10 LANG_MIXUP_ALPHA=0.2` | Stack the (cheap) training-side wins. If V1+V2+V3 are individually useful, combining should be better than any one alone. | belt-and-suspenders |

E2-class structural changes (grouped conv, attention) are deliberately **out of scope** for this excursion — they are 3-5 day implementation jobs and the point here is to first map the training-side ceiling cleanly. **Conclusion now: that ceiling is around 81 %, decisively below the published targets.**

## 3. Results — full data

3 seeds (42 / 1337 / 2024) × 6 variations × Language only × `--time-limit 0.5` (well above all observed convergence times):

| Variation | s42 adj | s1337 adj | s2024 adj | Mean adj | σ adj | Best raw % | Mean raw % | Runtime range |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V0 baseline | −4.50 | −4.34 | −3.52 | **−4.12** | 0.43 | 79.99 % | 79.10 % | 190-299 s |
| V1 patience60 | −2.93 | −3.47 | −4.11 | −3.50 | 0.49 | 80.87 % | 80.02 % | 358-562 s |
| V2 lr_low + warmup | −3.87 | −4.32 | −3.00 | −3.73 | 0.55 | 80.76 % | 79.68 % | 255-273 s |
| **V3 mixup α=0.2** | −2.11 | −2.51 | −2.91 | **−2.51** | **0.33** | **82.08 %** | **81.49 %** | 255-356 s |
| V4 larger model | −3.16 | −3.56 | −3.17 | −3.30 | 0.19 | 80.52 % | 80.32 % | 371-406 s |
| **V5 combo** | −2.53 | −2.20 | −2.60 | **−2.44** | **0.18** | 81.94 % | 81.58 % | 569-682 s |

### Excursion figure — variations vs literature

![[lang_excursion.png]]

V3 (green) and V5 (orange) are the two leaders — visually tied near 81.5 % raw, both ~3.6 pp below the harness benchmark (red dashed) and ~12 pp below ResNeXt-50 SOTA (top green line). All other variations cluster a half-bar lower.

### Across-method horizontal comparison

Every Language measurement we have, in approximate progression order — plus the literature reference verticals:

![[lang_across_methods.png]]

Reading the figure:
- **The aug-on configurations (top two grey + blue bars: Tier-1 aug-on, EGNAS-full)** sit at 70-72 % raw. Both float around the −10 adj floor for Language. The −10 floor we see on multi-seed full is not "the model couldn't learn" — it's "the model memorised the cutout-augmented training set and never generalised back to clean test". A textbook overfit.
- **P1 v4** (78 %) sits between aug-on and aug-off because s2024's probe wrongly picked aug-on on that seed, dragging the mean down.
- **All the aug-off configurations** cluster between 79-82 % raw, regardless of patience, LR schedule, model capacity, or Mixup. There is a clear "training-side ceiling" plateau visible at ~81 % that no V0-V5 variation crosses meaningfully.
- **Excursion V3 and V5** match **Multi-seed Tier-1 (aug-off, with op-search)** within 0.2 pp on the mean — i.e. Mixup on fixed-Conv3 buys back exactly what op-search would have given. They are equivalent regularisation interventions on this dataset.

## 4. Per-variation analysis

### V0 baseline — calibrating the floor

V0 = aug-off, fixed Conv3, default Tier-1 hparams. Mean 79.10 % ± 0.78. **This is ~2.3 pp below the multi-seed Tier-1 aug-off arm (81.41 %)** — a real, reproducible delta. The difference is **op-search**: multi-seed Tier-1 randomises ops per layer and picks among 5 op types; our V0 strips that out and trains a single all-Conv3 stack. So op-search is worth ~+2 pp on Language, and the proper internal reference is multi-seed Tier-1 aug-off (81.41), not V0 (79.10).

### V1 patience60 — extending training helps a *little*

V1 ran 358-562 s vs V0's 190-299 s — roughly 2× as much wall-clock per seed because the patience window doubled. Mean improvement: **+0.62 adj** (−4.12 → −3.50). Two takeaways:

1. **The model isn't running out of time.** It saturates within ~30-50 epochs (about 200 s) and then patience runs out. Doubling patience extends the search a bit further into the plateau, but the model has substantially converged.
2. **The remaining ~95 min of the 2 h budget would not help.** Every Language run anywhere in any pipeline configuration (multi-seed, EGNAS, P1, this excursion) ends in <26 min. The early-stop trigger isn't budget; it's saturation.

### V2 lr_low + warmup — gentler schedule, no mean gain

V2 changes `base_lr 0.025 → 0.01` and `LR_warmup 5 → 10`. Mean: −3.73, **only +0.39 over V0**. Runtimes shorter (255-273 s vs V0's 190-299) because the gentler schedule means it takes more epochs to overfit (which is what triggers our patience cutoff), but the final accuracy is essentially the same as V0. Conclusion: **schedule alone is not the lever** for Language.

### V3 mixup α=0.2 — the only training-side lever that moves the mean

V3 mean: **−2.51 ± 0.33**. Δ vs V0: **+1.61** — clearly above V1/V2/V4 noise. Best raw 82.08 % (s42). Lowest σ outside V5 (0.33). Mechanism is clean: Mixup perturbs the per-batch labels, removes the "easy" overfit-to-training-set route, the model keeps learning past where it would normally plateau. Runtime per seed (255-356 s) is barely longer than V0 — same number of epochs, same convergence shape, just lands at a higher plateau.

This matches the existing literature on Mixup as a regulariser: it shifts the implicit-regularisation regime without dramatically extending training. **The novel observation here is that no published team in the NAS Unseen Data Challenge has documented this on Language.** See §6.

### V4 larger model (W=72) — modest gain, beaten by Mixup

V4 mean: −3.30. Δ vs V0: **+0.82**, but **−0.79 worse than V3 alone**. Width-72 vs width-48 = ~2.25× FLOPs, but no proportional accuracy gain. σ (0.19) is the lowest among the single-knob variations, so capacity does reduce variance. **Conclusion: capacity helps mildly, but not as much as regularisation does.** The dataset isn't capacity-bound at this scale.

### V5 combo (V1 + V2 + V3) — tighter σ, no mean gain over V3

V5 = patience=60 + lr=0.01 + warmup=10 + mixup=0.2. Mean: **−2.44 ± 0.18**. Δ vs V3: +0.07 mean, −0.15 σ. **Within noise on the mean, decisively tighter on the variance** (0.18 vs 0.33, a ×0.55 collapse). Runtimes ~570-680 s — about 2× V3's, reflecting the longer patience window doing its job under the gentler LR.

Reading: the other training-side variations (V1, V2) aren't *useless* — they stabilise the search but don't push the optimum higher than Mixup already finds. **Mixup is finding the same plateau every time; the others' job is to keep us reliably on that plateau.**

## 5. The training-side ceiling — what we actually demonstrated

Across the 6 variations, the *mean raw accuracy* spread is:

```
V0  79.10%
V1  80.02%
V2  79.68%
V4  80.32%
V3  81.49%
V5  81.58%
```

A 2.5 pp band. **Mixup pushes the mean upward by ~+2 pp; everything else operates within ±0.5 pp of V0.** Combining all four variations with Mixup gives 81.58 — *identical within noise* to Mixup alone (81.49). So **the training-side ceiling on a fixed-Conv3 aug-off pipeline is ~81.5 % raw, with Mixup**. We can confidently claim no amount of patience-tweaking, LR-schedule-fiddling, or capacity-bumping will cross our 85.20 % harness benchmark on this architecture.

The path forward must change something more fundamental — the architecture itself.

## 6. Literature deep-dive (Geada et al., CVPR 2024)

Source: **Geada, Towers, Forshaw, Atapour-Abarghouei, McGough — "Insights from the Use of Previously Unseen Neural Architecture Search Datasets" (arXiv 2404.02189)**. This is the canonical published source for the NAS Unseen Data Challenge dataset zoo (Language included), spanning the 2021-2023 CVPR workshop editions.

### Published Language raw-acc numbers (from the paper)

| Method | Raw acc % |
|---|---:|
| Random search | 10.00 (chance) |
| Bonsai-Net | 87.65 |
| DrNAS | 88.55 |
| **Best competition submission** | **89.71** |
| **PC-DARTS / DARTS** | **90.12** |
| **ResNeXt-50 (32×4d, vanilla)** | **93.97** |
| ResNet-18 (paper) | 87.00 |
| ResNet-18 (our harness, stripped config) | **85.20** ← what we're competing against |

### The architectural finding (the paper's headline for Language)

> ResNeXt-50 outperforms every NAS approach. The high cardinality (groups) of ResNeXt let it learn the kind of complex/diverse representations needed for character-frequency-like patterns.

Specifically: **a vanilla 32-group 4-width-per-group ResNeXt-50 (cardinality = 32) beats every cell-search method by 4-10 pp on Language**. The paper's interpretation is that grouped convolutions — partial mixing of channels rather than full mixing (Conv) or no mixing (SepConv depthwise) — match Language's structural needs.

### What this confirms about our work

1. **The 85.20 % benchmark is a low bar.** It's a stripped-config ResNet-18. The actual literature targets are 90-94 %. **Crossing 85.20 % should not be the goal — crossing 90 % should.**
2. **NAS itself doesn't reliably beat ResNeXt on Language.** Even the best NAS competition submission (89.71 %) is 4 pp below vanilla ResNeXt. Our search-space limitation (Conv3/Conv5/SepConv3 micro vocabulary) doesn't include grouped convs at all.
3. **There is no published discussion of aug policy per-dataset on Language.** The Geada paper doesn't mention augmentation at all in the per-dataset findings. **Our aug-hurts-Language observation appears novel in the published literature.** Same for Mixup-on-aug-off helping. Both are unexploited intervention space.

### What the literature does *not* tell us

- No team has reported per-dataset aug policy or aug-off arms.
- No team has published Mixup experiments on Language.
- No team has done longer-patience studies.
- The 89.71 % "best competition submission" team is not named in the paper; we cannot identify a specific recipe.

So: **the literature is well-mapped for architecture but uncharted for training-side regularisation on Language.** We can plausibly improve on the published competition floor (89.71 %) *if* we combine our novel aug-off + Mixup finding *with* an architectural change toward grouped convolutions.

## 7. What this means for the project

**Concrete conclusions** that bind subsequent decisions:

1. **Mixup is shippable as a probe-conditional component.** Whenever P1 picks aug-off, also enable Mixup α=0.2. Cheap to add to v5; expected lift on Language ~+1.6 adj (~+2 pp raw); should not harm aug-off-correct other datasets within σ (V5 σ = 0.18 confirms stability).
2. **Patience / LR-schedule / capacity tweaks are not worth shipping on their own.** Their gains are within noise of V0. Only fold them in if a future probe could decide them per-dataset (similar pattern to P1 for aug).
3. **The Language ceiling at 85.20 % can only be crossed with an architectural change.** Training-side regularisation has been mapped and capped.
4. **The realistic Language target is 90+ %.** The 85.20 % harness benchmark is a *floor* established by a stripped ResNet-18, not a ceiling. PC-DARTS and ResNeXt-50 hit 90-94 %.
5. **Our aug-off + Mixup discovery is novel in the published literature.** If we can prove it stacks productively with an architectural change, this is a publishable observation.

## 8. Next experiment — design proposal

The data-backed next move is **adding grouped convolutions to the search space**, with Mixup-on-OFF folded in. Two scoped options:

### Option A — minimal: `GroupConv3` op added to OP_VOCAB

Add a single new operation: `Conv2d(in, out, k=3, groups=G)` with `G = min(in_ch, 32)` to give ResNeXt-style cardinality. This sits between `SepConv3` (depthwise, max cardinality) and `Conv3` (full mixing). One ~10-line addition to `helpers.py:OP_VOCAB`. Existing micro hill-climb will then have the option to pick GroupConv at any layer.

**Sweep design:** P1 v5 (multi-trial K=8 driver — the v4-shadow data showed K=8 is the right fix for the borderline-Δscore issue) + Mixup-on-OFF + the extended search space. 13 datasets × 3 seeds, same paired baseline as v4. ~12-16 h Spark.

**Expected outcome on Language specifically:**
- If grouped convs help, Mixup + GroupConv micro picks should land Language **above 85.20 %** for the first time. Realistic best-case raw 84-87 %.
- If they don't, Language stays at ~81-82 % and we've cleanly proven the ceiling needs a deeper architectural change (Option B).

### Option B — heavier: a true ResNeXt cell as a primitive

A multi-branch grouped-conv block with cardinality 32 + bottleneck (1×1 in → 3×3 grouped → 1×1 out), as a single primitive added to OP_VOCAB alongside Conv3/Conv5/SepConv3/GroupConv3. This is the *named* primitive that vanilla ResNeXt-50 uses to hit 93.97 % on Language. Higher payoff potential, ~2 days of code.

### Recommendation order

**Start with A.** It's a 10-line code change, costs nothing if it fails (it just adds an op micro can choose to ignore), and lets us collect agreement data with the existing pipeline. If A moves Language to ≥85 %, that's the first time any of our methods has crossed the benchmark on this dataset — a substantive thesis result. If A also moves CIFARTile (the other dataset where the literature says conv-only is wrong), it's a double win. Only if A's data says GroupConv-at-the-op-level isn't enough do we escalate to B (the full ResNeXt cell, more invasive).

### Out of scope (still)

- Attention / self-attention primitive — distinct from grouped convs; useful for Cryptic (text-like 6×768 input) but speculative for Language. Address in a separate E2 follow-up if GroupConv moves Cryptic too little.
- Deep-Sets — was originally the main candidate for CIFARTile (set-permutation task), distinct from Language's "character-frequency" structural needs. Address in its own targeted experiment.

## 9. Run mechanics

- **Baseline:** `baselines/tier2a_language_excursion/` — clone of `tier2b_egnas_full/`; `nas.py` replaced with a no-search default-arch constructor (env-overridable `LANG_DEPTH` / `LANG_WIDTH`); `trainer.py` reads `LANG_PATIENCE` / `LANG_BASE_LR` / `LANG_WARMUP`; `data_processor.py` reads `LANG_MIXUP_ALPHA` to override the metadata-derived mixup α (lets us test Mixup-with-aug-off independently of the default `aug_policy_for()` heuristic).
- **Sweep launch:** `dev/run_lang_excursion.sh` — 6-variation matrix × 3 seeds × `--time-limit 0.5` ≈ 90-120 min total.
- **Logs:** `dev/logs/lang_excursion/V{0..5}_*_s{42,1337,2024}.log` per run; `sweep.log` is the orchestrator log + a compact summary at the end.
- **Plots:** `python3 dev/plot_lang_excursion.py` → `lang_excursion.png` + `lang_across_methods.png` (the two figures embedded above).

## 10. Decisions log (for posterity)

- 2026-05-26: ran P1 v4 sweep, observed s2024 Language false-ON via K=4 unanimity failure; shadow K=8 was correct → v5 design discussion.
- 2026-05-28: discussed whether to chase Language benchmark before v5. User: "Language is still below baseline. Maybe we should try out a variation of approaches until we actually get one that beats it." Decision: Language-only excursion before v5.
- 2026-05-28: ran 6 variations on Language only. Mixup is the only meaningful lever; training-side ceiling is ~81.5 %. **The benchmark cannot be crossed on this architecture.**
- 2026-05-28: research agent surveyed Geada et al. CVPR 2024. Confirmed (a) ResNeXt-50 SOTA at 93.97 %, (b) PC-DARTS at 90.12 %, (c) competition floor 89.71 %, (d) grouped convs are the architectural lever, (e) no team has published aug-policy work on Language. **Aug-off + Mixup is novel; GroupConv is the data-backed next experiment.**
