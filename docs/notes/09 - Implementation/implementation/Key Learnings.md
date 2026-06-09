# Key Learnings — the distilled insight from every experiment

> **Status:** living synthesis (2026-06-10) | **Topic:** The handful of findings that actually generalise, pulled out of the full experimental arc (Tier 0 → 1 → Multi-seed → E1 → Tier 2b EGNAS → Tier 2a P1 v1–v7 → v8 train-to-budget → TCN excursion → random arm → SOTA scan + offline validations). If you read one note to understand *what we know*, read this.
> **Related:** [[Implementation Results Overview]], [[Tier 2b - Results]], [[Tier 2a - Results]], [[Tier 2a - TCN Excursion]], [[v10 - Design + Idea Ledger]], [[SOTA Scan - Novel Directions]], [[Multi-Seed - Results]], [[E1 - Results]], [[Run State + Handoff]]

The competition is the **AutoML 2026 NAS Unseen Data Challenge**: the Phase-2/3 datasets are *hidden*, their codenames are values we have never seen, and the method gets a fixed time budget on an *unknown* GPU. Every learning below is shaped by that: **the method must decide everything at runtime from metadata or a cheap probe, identically on every dataset, robust to unknown compute.** No dataset gates, ever.

---

## ⭐ The one big learning: the proxy-reliability wall

**No cheap proxy reliably predicts full-budget accuracy.** Not short-train at any K, not verify-at-2–3×K, not zero-shot proxies. This single fact is the root cause behind almost every failure we hit. It shows up in four different disguises:

1. **Q8 arch-lottery / search-space flatness.** Across 3 seeds the macro search agrees on the final depth **1/13** datasets (`dev/analyze_q8_under_egnas.py`). It's not that the search *converges* to different good archs — the val-acc surface is so *flat* at any feasible search horizon that the argmax is dominated by noise. (This is *why* total σ can stay low despite the chaos: the many different archs all score similarly — except where arch genuinely matters, where the search then picks wrong.)
2. **Exotic ops win short-train and crater at full-train.** v6 Sudoku, smoking gun at *identical* dims: EGNAS-full `D=7 W=48` plain-Conv3 → **90.8 %**, vs v6 `D=7 W=48` with Conv5+attention → **71.0 %** (−20 pp). The micro search picked the exotic ops because they led at K=12; they collapsed by 2 h. Verification at 2× K was *still too short* to catch it.
3. **Recurrence LOSES short-train but WINS full-train.** The mirror image. row-BiLSTM on Language: at the 4–6 epoch micro horizon it is *behind* conv by 1–19 pp and **thrashes wildly** (22 % → 76 % → 47 % → 79 % under the search's crude SGD) — the proxy optimiser literally cannot train it. Yet under the *full* trainer (warmup + cosine + EMA, ~70 epochs) the same arch hits **85.76 %** and beats conv. The proxy is structurally blind to it.
4. **Depth false-strikes.** GameOfLife (a cellular automaton — needs depth) looks terrible shallow-vs-deep at K=12, so depth-grow strikes the deep nets and stays at D=3 (→ 48 %); on the one seed a deep net survived the strike it trained to **99 %**.

**Corollary — the only trustworthy signal is full-convergence training.** Everything reliable we built either (a) uses an *incumbent-kept* search that is robust to the noise rather than trusting it, or (b) makes a decision so cheap and binary it can be probed reliably (see Learning 5). Convergence luckily is *fast* (~5 min, patience-stop) — which is the only reason a full-convergence A/B is even thinkable as a deployment mechanism.

---

## The learnings, in order of how much they shaped the project

### 1. Variance is the dominant signal — always measure ≥3 seeds
Tier-1 single-seed scored **+24.644**; the honest 3-seed figure is **+11.3 ± 8.2** — σ ≈ 100 % of the mean (one seed +0.02, another +19.6). The single-seed champion was a tail draw. **Every conclusion from a single seed has burned us** (the v6 "crossaxis is best" from one seed at 87 %, which cratered to 78 % on another; the s42-only "GameOfLife systematic failure" that the 3rd seed disproved). Rule: no decision on < 3 seeds. ([[Multi-Seed - Results]])

### 2. Architecture — not ensembling — controls variance
E1 (budget-filling ensemble) was a **MIXED** result: it collapsed AddNIST σ (5.45 → 1.00) but was *not* a blanket fix (total σ only ×0.75, GeoClassing regressed). The residual variance is **structural / dataset-level → cannot be averaged away**. EGNAS-full's architecture search, by contrast, collapsed total σ **8.37 → 3.28 (×0.39)**. Variance is a property of the *architecture's fit to the dataset*, and the lever is architecture. ([[E1 - Results]], [[Tier 2b - Results]])

### 3. Simplicity beats sophistication — every "cleverer" iteration lost
The clearest trend in the whole project. EGNAS-full (simple, 3-op vocab, incumbent-kept search) = **+35.51 ± 3.07**. Every attempt to elaborate it lost:
- **v5** (+GroupConv3/ResNeXt): GameOfLife −9.4 pp from a noisy false-positive.
- **v6** (9-op vocab + K=12 + verify + clock-extend + zero-shot): **−12.10 vs EGNAS-full, σ ×2.98** — worse on *both* mean and variance. The rich vocab *poisoned* Sudoku; the aggressive search *amplified* the Q8 lottery.

The richer the search space, the more rope the unreliable proxy has to hang itself. **The winning move is robustness, not search sophistication.** ([[Tier 2a - Results]] §v6)

### 4. The one P1 win that survived: the aug-safety probe on Language (+4.23)
Out of six P1 iterations, exactly one component beats EGNAS-full reliably: the **C-only aug-safety probe** (aug-off + Mixup α=0.2). Language was the project's worst deficit (EGNAS-full −8.32) and the probe lifts it to −4.09, **Δ+4.23**, correct on 2/3 seeds. It is the *only* durable P1 result.

### 5. Blind runtime decisions work when they're cheap + binary; fail when they need full-budget eval
*Why* the aug-probe works and the architecture search doesn't: aug-on/off is a **cheap, binary, reliably-probeable** decision. An **architectural** choice (which op / depth / width) needs full-convergence to judge — which the search can't afford per-candidate. This is the design boundary: lean on cheap binary probes (aug); never trust a cheap proxy for an architectural choice (Learning ⭐).

### 6. Architectural levers are real but hard to deploy
The headroom on the hardest datasets *is* architectural, and we proved two levers:
- **Language → recurrence.** row-BiLSTM is the **first architecture in the project to put Language reliably above benchmark** (85.76 % vs 85.20, all 3 seeds; conv plateaus ~83.8 %). Specifically the *1-D left-to-right sweep* (Language is text rendered along the width) — attention does *not* cross (84.8 %), cross-axis BiLSTM is too unstable (one seed 78 %). Matches Geada et al. (ResNeXt-50 → 94 %, PC-DARTS → 90 % on Language).
- **CIFARTile → non-conv.** Conv search only *just* crosses benchmark (+0.5); the real ~46 pp of headroom needs non-conv (Deep-Sets / E2 territory).

But (Learning ⭐) these only show at full-budget, so **the search can't select them** — deployment needs an expensive full-convergence A/B, which is risky on unknown compute. → they ship only as a **fail-safe, budget-gated A/B opt-in**, or stand as documented thesis findings.

### 7. The macro search mis-prunes at BOTH boundaries — the v7 fix
The short-horizon ranker fails the two macro "kill" decisions symmetrically:
- **Depth false-strike** (GameOfLife): a deep net looks bad at K, gets struck before it can climb. → **verify-of-rejects**: re-train a would-be-struck deeper candidate longer; rescue it if it overtakes.
- **Width false-accept** (Sudoku): a narrow net is within tol at K but craters at full-budget. → **verify-of-accepts**: verify a prune against the wide reference at a longer horizon; reject the mirage.
Both implemented in v7 (`VERIFY_MACRO`, bounded by `MAX_MACRO_VERIFY`), CPU-smoke-confirmed to fire. ([[Tier 2a - Results]] §v7)

### 8. Convergence is fast — a double-edged finding
The nets patience-stop (warmup 20, patience 30) at convergence in **~5 min** on small datasets, often leaving >1 h of budget unused with LR not annealed to 0. **Upside:** makes a full-convergence A/B affordable (the whole recurrence-deployment idea hinges on this). **Possible free lever:** training to LR=0 (more patience / forced anneal) might squeeze the last ~1.7 pp on Language without any architecture change — untested, orthogonal, cheap.

### 9. We held our pipeline to a standard the EGNAS paper never claimed
Reading the actual EGNAS code/paper: their "Spearman 0.91" is **(param count) ↔ (val at incrementally-trained budget) on CIFAR-10**, aggregated over 50 nets — *not* "2-epoch val on Unseen-X predicts 600-epoch test acc". Their setup also differs (4-op binary vocab, hard K=2, 600-epoch final train, test-acc ranking, param-equalised width). Our Q8 problems aren't a contradiction of the paper — we measured a different, harder thing. ([[EGNAS]] §"Our port vs upstream")

### 10. Train-to-budget: the "free lever" from Learning 8, now validated (2026-06)
Early-stop wasted the non-transferable per-dataset budget AND cut the cosine anneal short. Removing it (PATIENCE high, schedule fills the budget, LR to 0): **5 wins / 2 ties / 0 regressions** on frozen archs, and it recovered the v7 MultNIST regression on its own. v8 (v7 + train-to-budget + adaptive stem) seed-42 total: **+48.2 vs v7's +41.94**. The biggest cheap win in the project. ([[Run State + Handoff]])

### 11. The backbone family dominates the search (the TCN excursion)
A FIXED 2-D dilated-conv backbone (TCN-2d, no search at all) beats the entire searched conv pipeline: **+59.2 vs +48.2** (seed 42, full 2h, all 13). Wins 6 / ties 4 / loses 3; CIFARTile alone goes 48 → 72 raw. The receptive field, not the op micro-choice, is the architectural lever. Independently matches the literature (search SPACE beats search strategy; UniRepLKNet's fixed wide-RF net travels across modalities). The 3 losses (Gutenberg, Sudoku, Chesseract) are all small odd encoded inputs. ([[Tier 2a - TCN Excursion]])

### 12. Decisions beat rankings: the abstention gate (offline-validated)
At a 3/5/10-min curve prefix, picking the backbone by the early leaderboard swings from −2.3 to +2.2 vs just fixing TCN-2d. The **gate** (trust the leader only when its margin beats the measured noise, else keep the default) is the only signal that beats the fixed default at EVERY budget (+1.0 to +1.7) and never falls below it. Two honest negatives from the same validation: **TSE is biased across families** (dilated nets drop loss faster regardless of final quality) and **learning-curve extrapolation breaks** on our cosine-inflection curves at small budgets. (`dev/validate_offline_signals.py`, [[SOTA Scan - Novel Directions]])

### 13. Measuring the data sidesteps the proxy wall (the fingerprint)
Local pixel correlation (corr1) on raw tensors, seconds of CPU, no training: **corr1 ≈ 0 identifies exactly the 4 encoded datasets** (Language, Sudoku, Chesseract, Gutenberg) with a 5x gap to the nearest image-like dataset. Those are precisely where the default backbone loses or a specialist wins. And the **random-search arm wins exactly there**: Chesseract +0.90 vs conv +0.30 (all 3 seeds), while losing on normal data. Both confirm the design rule from Learning 5, extended: when you cannot rank candidates, measure the data instead. (`dev/fingerprint_probe.py`)

### 14. Verdicts go stale when the regime changes
The stem, the aug decisions, and the verification machinery were all validated on the CONV backbone with early-stop training. After switching to the dilated backbone + train-to-budget, those verdicts are conjectures again. Phase A re-checks them (stem x dilated, aug x tcn1d, does verification ever fire usefully for the dilated family). Rule: a component verdict is only as durable as the regime it was measured in.

---

## Where this leaves us — the path forward (updated 2026-06-10)

The 2026-06-02 version of this section said: ship v7-conv, recurrence as a gated A/B, try the training-length lever. All three happened and the picture moved on:

1. **The current experimental chain (running unattended):** v9 seed 42 (the dilated backbone INSIDE the search) → Phase A decision-harness tests (backbone-gate arms, TSE-in-search logging, aug pre-filter, regime retests) → then assemble **v10** = measure-gate-train pipeline (fingerprint → aug probe → backbone gate with abstain-to-default → steepness gate → optional search → optional random arm → train-to-budget), full design in [[v10 - Design + Idea Ledger]]. v10 is an EXPERIMENT, not the submission; the 3-seed treatment is reserved for whatever config earns it.
2. **The validated core today:** dilated TCN-2d backbone (Learning 11) + train-to-budget (Learning 10) + aug-probe (Learning 4) + gated decisions (Learning 12), with the random arm for encoded inputs (Learning 13).
3. **v11 direction (parked, deliberate):** the TCN excursion proved one different family was worth +11, so screen MORE families with the same cheap probe protocol, aimed at the encoded datasets the fingerprint flags: state-space models (sequence beyond TCN), MLP-Mixer-style (no spatial prior, fits encoded data), modern ConvNeXt-style blocks (stronger image default).
4. **Stop chasing search sophistication** (Learning 3) still stands; the new corollary is **spend the cleverness on decisions, not on the search** (Learnings 12, 13).
5. **Deadlines:** proposal June 15 (the story: measure first, gate every decision, the proxy-wall map as the contribution). Phase 2 opens July 1 at ~1h per dataset, so a 1h-budget validation pass of the final config is mandatory before packaging.

The meta-point survives every update: **the project's value isn't a single peak number — it's the map of where cheap signals work and where they don't.** That map (Learning ⭐, now with Learnings 12 and 13 as its constructive answer) is the contribution.
