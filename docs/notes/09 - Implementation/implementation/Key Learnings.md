# Key Learnings — the distilled insight from every experiment

> **Status:** living synthesis (2026-06-02) | **Topic:** The handful of findings that actually generalise, pulled out of the full experimental arc (Tier 0 → 1 → Multi-seed → E1 → Tier 2b EGNAS → Tier 2a P1 v1–v7 → LSTM probe). If you read one note to understand *what we know*, read this.
> **Related:** [[Implementation Results Overview]], [[Tier 2b - Results]], [[Tier 2a - Results]], [[Tier 2a - Language Deep Dive]], [[Multi-Seed - Results]], [[E1 - Results]], [[Run State + Handoff]]

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

---

## Where this leaves us — the ideal path forward

The data has converged on a clear strategy (it took the long way, through v5/v6, to earn it):

1. **Ship the robust core: `v7-conv`** = EGNAS-full's proven simple search + 4 conv ops + macro-prune-verify (Learning 7) + the aug-safety probe (Learning 4). Fully compute-robust, no fragile moving parts. Recovers v6's self-inflicted Sudoku/GameOfLife losses; with the Language probe it could *beat* EGNAS-full (the prize: ≈ +40 vs +35.5 if the regressions are eliminated while the +4.23 is kept).
2. **Recurrence as a fail-safe, budget-gated A/B opt-in** (Learning 6 + ⭐ + 8). Train conv to convergence first (guaranteed shippable floor); *only if the measured clock allows*, train the row-BiLSTM variant and keep the better. Currently being **validated**: the A/B must pick *conv* on conv-optimal datasets (Sudoku/AddNIST/CIFARTile) — the guard against over-picking recurrence.
3. **Cheap free wins:** macro-prune-verify; possibly the training-length lever (Learning 8).
4. **Stop chasing search sophistication** (Learning 3). The richer-vocab direction is closed.
5. **Document the architectural findings** (recurrence cracks Language; non-conv is CIFARTile's frontier) as thesis contributions regardless of whether they deploy.

The meta-point: **the project's value isn't a single peak number — it's the map of where cheap signals work and where they don't.** That map (Learning ⭐) is the contribution.
