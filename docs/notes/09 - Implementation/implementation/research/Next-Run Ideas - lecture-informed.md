# Next-Run Ideas — lecture-informed (AutoML Ch6 NAS + Final-Project exam, 2026-06-03)

> **Status:** forward design, saved for the v8 / next-test iteration | **Topic:** three ideas from the AutoML lecture (Ch6 NAS summary + Final-Project exam slides) worth folding into the next run, and three concrete ways to incorporate them. | **Related:** [[Key Learnings]], [[Pipeline v1-v7 - Aug Probe]], [[Run State + Handoff]]

## Why these (the lecture validated our approach + pointed at gaps)

The Ch6 summary independently confirms the core of what we found the hard way:
- **"The search space is the biggest lever; strategy matters less; random/local search is a strong baseline."** = our v6→v7 win (fix the *space*, not the strategy; our incumbent hill-climb ≈ local search).
- **"Zero-cost proxies: weak, inconsistent; #params often competitive."** = verbatim our proxy analysis (ρ≈0.25, params dominant). We reproduced a known result + explained the mechanism on the Unseen suite.
- **DARTS brittle / one-shot supernet ranking-correlation low** = the same proxy-reliability wall → confirms we were right to avoid them.

The **exam** frames the deliverable as a **3-component pipeline: data processor + NAS algorithm + training policy** — and we've barely touched the *data processor*. Grading is a **poster + research question** (proposal due **June 15 9am**; submissions July 1–31, up to 7; poster Aug 6/11), so the methodology *story* counts as much as the leaderboard number.

## The three ideas

**1. Learning-curve (LC) extrapolation — the principled cousin of our verify.** Instead of ranking on noisy val@K, *fit* the candidate's short-train curve (we already log per-epoch val) and extrapolate to final acc. **Caveat (lecture):** "fails for unusual dynamics" — which is *exactly* GameOfLife (flat-then-jump; you can't extrapolate a jump from a flat prefix). Our verify-of-rejects *trains into the jump*, so it's robust precisely where LC fails. → **Methodological win for the poster** (show LC would mis-rank GameOfLife where our re-train verify succeeds). **Refinement value:** use LC as a *cheap first-pass filter*, fall back to the 3×K re-train only for flat/ambiguous curves → cuts the verify cost (our main per-dataset slowness ~75 min/ds) without losing robustness.

**2. Data processor — the untouched third component.** Right now every dataset gets identical processing (per-channel normalise) regardless of structure: Voxel (20ch), Chesseract (12ch), Language (1×24×24 text-as-image), AddNIST (3×28×28) all enter the same conv stem. A **metadata-driven input-handling policy** (decided blind, like the aug-probe) is the most actionable *new* lever and a component the graders explicitly want to see.

**3. Cell-based search space ("transferable across datasets").** The lecture's #1 practical recommendation, and "transferable across datasets" *is* the Unseen Data Challenge's requirement. We use the *least* expressive space (chain-structured, op-per-layer). A cell-based space (search a small cell, repeat it) could generalise better to hidden datasets — but it's a big, risky rewrite. **File as frontier/future**, not the immediate next test.

## Three ways to incorporate them (low → high ambition)

### Way A — "v7.1: cheap wins, keep the proven core" (LOW risk, recommended next TEST)
- **LC-extrapolation as a verify pre-filter** (idea 1): fit a saturating curve (e.g. `a − b·exp(−c·t)`) to each would-be-struck candidate's short-train trajectory. If extrapolated-final is *confidently* below the incumbent → strike, no re-train. If high OR poor-fit/flat → fall back to the existing 3×K re-train verify. **Pure speedup, no behaviour regression** (GameOfLife's flat curves trigger the fallback → still caught).
- **Metadata-driven stem** (idea 2, lightweight): replace the fixed Conv3 stem with one chosen from metadata — high-channel inputs (≥8 ch) → a 1×1 channel-mixing stem (treat planes as features, not spatial); image-like (1–3 ch) → current conv stem. Bolt-on, no search-space change.
- Cell-based: deferred.
- *Effort: small. Risk: low. Keeps v7's +45.21 core; adds speed + one new lever.*

### Way B — "the clean 3-component pipeline" (MEDIUM, poster-optimal)
Build the submission exactly as the exam frames it, each component metadata/probe-driven and individually **ablatable** (great for the poster):
- **Data processor = a probe** (idea 2, full): a cheap runtime measurement picks the input representation per dataset (channel-mix vs spatial stem, resolution/down-sample handling) — generalises the aug-probe pattern to the data component.
- **NAS = v7 macro-prune-verify + LC-extrapolation estimator** (idea 1).
- **Training policy = aug-probe + Mixup** (already have).
- *Effort: medium. Risk: medium (the data-probe is new). Value: a clean, ablatable, defensible 3-component story that maps 1:1 to the grading rubric.*

### Way C — "transferable cell space + smart estimator" (HIGH, research bet)
Pivot the **search space** (the biggest lever per the lecture) to **cell-based** for transferability to unseen data:
- **Space:** a small searchable cell (few ops + a small topology) repeated N times; fixed macro stack; search the cell. More expressive but controlled — keep the cell op-set **conv-only** initially to avoid re-introducing the v6 poisoning.
- **Estimator:** verify + LC-extrapolation (idea 1).
- **Data processor:** metadata stem (idea 2).
- *Effort: high. Risk: high (real rewrite; risks proxy/poisoning if the cell gets too rich). Upside: best potential generalisation to the hidden Phase-2/3 datasets.*

## Recommendation
**Way A as the next test** (cheap wins, validates idea 1's speedup + idea 2's first lever, zero risk to the +45 core). **Frame the poster/proposal as Way B** (the clean 3-component narrative — it's what the rubric rewards). **Keep Way C as documented "future work"** unless Way A's data-stem shows the input-representation lever is large enough to justify the bigger bet.
