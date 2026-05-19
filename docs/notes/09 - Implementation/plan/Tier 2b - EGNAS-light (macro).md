# Tier 2b — EGNAS-light (macro)

> **Status:** macro DONE ([[Tier 2b - Results]]) → **completed to FULL EGNAS (macro + micro) 2026-05-18**; full 11-ds × 3-seed sweep next | **Topic:** The competition's proven core: macro architecture search + micro op-search, scaled to the Phase-2 clock. The first *real* NAS engine in the repo; primary job is to crack the **CIFARTile** ceiling Tier 1 can't move.
> **Related:** [[Tier 2 - EGNAS-Light]] (original algorithm reference), [[Tier 2b - Results]], [[EGNAS]], [[2025 - EVAI]], [[Tier 1 - Augmentation + Op Search]], [[Multi-Seed - Results]], [[E1 - Results]], [[Probe-Driven Self-Configuration]], [[Reliability vs Peak]], [[Run State + Handoff]]

## ⏵ Update 2026-05-18 — completed to FULL EGNAS (macro + micro). Reasoning.

The macro-only diagnostic ([[Tier 2b - Results]]) **passed the make-or-break** (CIFARTile −0.515→+0.167, all 3 seeds, σ↓ — the ceiling no Tier-1 config could move) but with two caveats: the gain was *small*, and the macro argmax-over-noisy-candidates selection was a **Q8 arch-lottery** (s2024 picked D=5 for AddNIST → −2.81).

**Why we did NOT pivot to E2 or polish the ranking (the reasoning):**
- Concluding "conv-search is tapped out, jump to a novel architecture (E2)" from a *deliberately half-built* EGNAS was **premature** — you can't judge the proven 2025-winning method by running half of it. (This was a real over-reach in the earlier recommendation; corrected here.)
- A small "re-rank the finalists" denoiser is low-EV polish on an *incomplete* method.
- The principled move: **finish EGNAS** (add the micro phase), *then* judge it. Highest-EV (completes the proven competition core), lowest novelty-risk.

**What "full" adds, and why micro also fixes the lottery:**
- **Micro op hill-climb:** from the all-Conv3 macro winner, per layer try {Conv3, Conv5, SepConv3}; keep a swap only if it beats the incumbent by `MICRO_MARGIN`. This searches *what each layer computes* (not just depth/width).
- It is **incumbent-kept** — a single noisy short-train cannot hijack the choice (unlike macro's argmax-of-noisy-candidates). So completing the method is *also the principled fix* for the Q8 lottery, not a bolted-on denoiser.
- Budget: macro gets `MACRO_FRACTION` (60 %) of the 30 % search slice, micro the rest; whole-search failure → Tier-1 default (never a new −10).

**Code hygiene / traceability (why a new dir):** `baselines/tier2b_egnas_macro/` is **frozen** as the macro-only comparator; the full method lives in **`baselines/tier2b_egnas_full/`** (only `nas.py` differs). This keeps a clean full-vs-macro attribution on the 3-ds `egnasdiag` and lets us reproduce/trace exactly what micro bought.

**Validation (user decision 2026-05-18):** run the **full 11-dataset × 3-seed sweep directly** (`dev/run_tier2b_full.sh`), paired vs the Tier-1 `full_s*` baseline — for a comprehensive read of *actual* performance rather than another cheap gate. A ~15-min correctness smoke still precedes it (guards against burning ~30 h on never-executed code; *not* the strategic gate being skipped). The "what micro adds" attribution stays precise on the 3-ds `egnasdiag` (macro frozen there).

**E2 is not abandoned — re-ordered:** a relational/non-conv primitive belongs as a candidate **op in EGNAS's search vocabulary** (probe-thesis-consistent: the search discovers when to use it), considered *after* full EGNAS is validated — not as a premature hand-built sidestep.

## Why this, why now

Every prior finding pulled us toward training/data/variance work (Tier 0: overfitting is the binding constraint; Tier 1 ablation: aug is the big lever; Q8: arch choice isn't the variance driver on most datasets). **But that left the competition's actual core — architecture search — unbuilt, and CIFARTile (the canonical +46.7 pp NAS-reward dataset) stuck at Tier-1's −0.5 ± 0.5 ceiling where training-side levers are exhausted.** EGNAS is the proven 2025 winner ([[2025 - EVAI]]). Tier 2b confronts it.

## Locked decisions (2026-05-18)

1. **Macro-only v1** — depth-grow + width-prune. **No micro (op/kernel) hill-climb** in v1 (the original plan's own "optional, skip if tight"; depth/width is exactly what CIFARTile rewards; far less code & clock). Micro = v1.1 if macro shows signal.
2. **Return the single best arch; log the discovered pool.** Single best → clean one-variable test ("does macro-search beat Tier 1?"). Top-k evaluated `(arch, val)` written to disk so Tier-3 ensembling ([[E1 - Results]]'s "emit an ensemble-able pool" lesson) is free later — no extra train cost now.
3. **Built on pure Tier 1** — only `nas.py` changes; `data_processor.py` + `trainer.py` byte-identical. One isolated new variable.

## Algorithm (macro, scaled — full detail in [[Tier 2 - EGNAS-Light]])

```
Init:  D = D_min (≈4 blocks), W = W_max (≈48, = Tier-1 base_width),
       op = Tier-1 default, downsample plan via helpers.plan_downsamples
       (recomputed every depth change so it stays spread).
PHASE 1 depth-grow:  while val improves ≥ τ_grow: append a block.
       Train each candidate on an architecture-aware epoch schedule
       (bigger → more epochs). Stop at D_max or 3 non-improvements.
PHASE 2 width-prune: while val retained: W -= W_step. Stop at W_min / 3 strikes.
Return best (D_f, W_f) fresh → Tier-1 Trainer full-trains it (unchanged).
Log every evaluated (arch, val) → dev/logs/.../<codename>_pool.json.
```
Reuse: `helpers.build_cnn`, `plan_downsamples`; short-train loop pattern from Tier-1 `nas.py:_short_train`. Add to `helpers.py`: the architecture-aware scheduler + stop criteria + pool logger.

## Findings-driven changes vs the original [[Tier 2 - EGNAS-Light]] note

1. **All results ≥3-seed, paired vs the multi-seed `full_s*` Tier-1 baseline.** The original "CIFARTile ≥ +6.0 at 1 h" acceptance criterion was single-seed → dead as evidence (σ≈8).
2. **Verify, don't assume, the architecture-aware schedule.** The paper's Spearman-0.91 ranking claim must be checked under our cuDNN nondeterminism — Q8 showed candidate ranking is a lottery. Log chosen depth/width + candidate vals per run; extend `dev/analyze_search_stability.py` to measure **arch-selection stability across seeds** (is macro-search itself seed-variance-amplifying?).
3. **CIFARTile is the make-or-break acceptance test**, not a side-check. If multi-seed macro-search can't beat Tier-1's CIFARTile ceiling, the "search backend" half of the thesis is in question — surface that result honestly, don't bury it.
4. **Contract hygiene inherited + hard fallback:** clock check between candidates, try/except per candidate (OOM/NaN → drop, continue), and if the *whole* search fails → return the Tier-1 default arch. Never a new −10 ([[Reliability vs Peak]]).

## Validation plan (cheap-first, standing discipline)

1. **CIFARTile diagnostic** — 3 seeds × {CIFARTile + AddNIST (high-var sanity) + GeoClassing (stable, must-not-regress)}, `--time-limit 2.0`, paired vs `full_s*`. **The one question: does macro-search move CIFARTile above Tier-1's −0.5 ceiling, multi-seed?** ~6–9 h Spark.
2. Only if (1) shows real CIFARTile signal → full 11-dataset multi-seed sweep, paired, → total ≥ Tier 1, no regression, no new −10, ≤ budget.

## Files / hooks / success / risk

- `baselines/tier2b_egnas_macro/` = copy of Tier 1; rewrite `nas.py` (EGNAS macro); `helpers.py` += arch-aware scheduler + stop criteria + pool logger; `data_processor.py`/`trainer.py` unchanged.
- Hooks: `EGNAS_DMAX`, `EGNAS_WMIN`, `EGNAS_OFF=1` (→ Tier-1 random op-search, for A/B). Drivers: `dev/run_egnas_diagnostic.sh`, later `dev/run_tier2b.sh`.
- **Success:** CIFARTile multi-seed mean beats Tier-1's −0.5 ± 0.5; full sweep ≥ Tier 1 overall with no regression; arch-selection stability characterised; ≤ clock; zero new failure modes.
- **Risks:** (a) macro-search may not beat Tier 1 on most datasets — *acceptable*, CIFARTile is the test that matters; (b) macro-search may itself be a seed-lottery (Q8) — the multi-seed harness measures this; (c) depth-grow trains many candidates → budget pressure — conservative `D_max` + inherited clock hygiene; (d) if CIFARTile doesn't move, that is itself a thesis-critical finding to report, not a failure to hide.
