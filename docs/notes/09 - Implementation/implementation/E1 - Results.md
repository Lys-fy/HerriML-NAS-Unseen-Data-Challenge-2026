# E1 — Results

> **Status:** ✅ complete (3 seeds) — **mechanism validated on its target (AddNIST), but NOT a blanket reliability layer.** Conditionally useful; one regression; headline gain partly noise. | **Topic:** Does the E1 budget-filling diverse-restart ensemble collapse the σ≈8 single-run variance the multi-seed sweep exposed? The reasoning that led here, what the test does, and the (honest, mixed) verdict.
> **Related:** [[Multi-Seed - Results]], [[Tier 1 - Results]], [[Probe-Driven Self-Configuration]] (§E1), [[Reliability vs Peak]], [[Threats to Validity]], [[Implementation Results Overview]], [[Run State + Handoff]]

## Why this test exists — the reasoning chain

1. **Tier 1 looked great at one seed (+24.644)** — see [[Tier 1 - Results]].
2. **The multi-seed sweep ([[Multi-Seed - Results]]) demolished that number.** 3 seeds of the *same recipe* → **+8.4 ± 8.4 internal** (one seed +0.02, another +16.75). Seed variance is ≈100 % of the signal. Also: AddNIST/MultNIST were never "fixed" (±5.4 / ±3.0 — a per-run coin-flip); the only seed-*stable* signal is the aug double-edge **direction** (Language aug-cost −6.82 ± 1.88; aug-off a dead −10 floor on AddNIST/MultNIST — the **aug-off arm** result).
3. **Strategic consequence (the binding fact).** Phase 2 scores us on **one run**. A single run of our recipe is a draw from σ≈8. Per [[Reliability vs Peak]] and [[Threats to Validity]] §4/§7, that variance — not peak — is the dominant risk, and Threats §4 *already prescribes* "ensemble to reduce variance." You also can't measure whether any later probe/Tier-2 change helped while the noise floor is ±8.
4. **Step 0 / Q8 — is the search the variance source?** (`dev/analyze_search_stability.py`, mines the 54-run logs, zero compute.) Finding: the random op-search picks a **different architecture every seed for 10/11 datasets** — but that instability is **universal and not the discriminating cause**: GeoClassing is equally search-unstable yet rock-stable downstream (σ 0.29). The AddNIST/MultNIST variance is genuine **raw-accuracy instability** (AddNIST raw σ ≈5.5 pp) amplified 1:1 by the ~90 % benchmark scoring slope. So the cure must average over *training-process* variance, not just architecture choice.
5. **⇒ E1.** Ensembling is the structural fix for raw-accuracy variance, it converts otherwise-wasted clock+VRAM into robustness, and it is needed by whatever backbone ships (Tier 1 *or* Tier 2). Promoted from optional → necessary; this test validates it.

## What E1 is (mechanism — full design in [[Probe-Driven Self-Configuration]] §E1)

`baselines/tier1e1_ensemble/` — strict superset of Tier 1:

- **Member 0 = the chosen searched arch, trained with Tier-1's exact floor.** No leftover clock ⇒ bit-for-bit Tier-1 behaviour ⇒ **E1 cannot regress** (the load-bearing safety property).
- **Extra members fill only otherwise-wasted slack:** seed restarts of the chosen arch (the dominant variance axis per Step 0) interleaved with the decorrelated searched candidate archs NAS hands over via `model._e1`.
- **predict() averages softmax** across members (never select-by-val — no test labels → winner's curse). Peak VRAM = one model (members rebuilt one at a time). Hard clock guard drops members rather than overrun the −10 cliff.
- **Smoke-passed** (2026-05-16, Chesseract+Sudoku): multi-member path, averaged predict on two shapes, reserve/clock guards, and the no-slack→1-member≡Tier-1 fallback all verified, zero exceptions.

## What the current test does

`dev/run_e1_validation.sh`, launched 2026-05-16 22:25 CEST, nohup'd on Spark.

- **Matrix:** e1 arm × **{AddNIST, CIFARTile, GeoClassing, Language, MultNIST}** × seeds **42/1337/2024** = 15 runs, `--time-limit 2.0`.
- **Paired Tier-1 baseline is reused, not re-run** — E1's member 0 *is* the Tier-1 model, so the multi-seed `full_s*` logs for these 5 datasets/seeds at the same budget are the exact paired control (saves ~half the Spark budget).
- **Dataset choice:** the high-variance bombs (AddNIST σ 5.45, MultNIST 3.00), the architectural ceiling (CIFARTile — sanity that E1 ≠ free lunch), the seed-stable aug-regression (Language — must not get worse), and the rock-stable win (GeoClassing — must not regress).
- **Success criteria:** per-dataset σ_adj shrinks substantially (headline target **AddNIST σ 5.45 → <2**) **AND** mean ≥ Tier-1 mean **AND** no regression on GeoClassing/Language **AND** no new −10. Answers thesis **Q4**.
- **Analysis is pre-written + tested:** `dev/aggregate_e1.py` (σ-collapse + auto-verdict) and `dev/plot_tier_results.py --e1`.

## Results — ✅ complete (3 seeds, sweep 2026-05-16 → 2026-05-18)

`dev/aggregate_e1.py` → `dev/logs/e1val/_e1_aggregate.json`. Internal adj, mean ± std over seeds 42/1337/2024. Tier-1 column = the paired `full_s*` baseline (same datasets/seeds/budget).

![[e1.png]]

| dataset | Tier-1 | E1 | Δmean | σ T1→E1 | ×ratio | members | read |
|---|---:|---:|---:|---:|---:|---:|---|
| **AddNIST** | −0.38 ± 5.45 | **+2.75 ± 1.00** | **+3.12** | 5.45→1.00 | **0.18** | 4.3 | ✅ the win |
| CIFARTile | −0.52 ± 0.52 | +0.10 ± 0.32 | +0.62 | 0.52→0.32 | 0.60 | 3.3 | ✅ clean small |
| GeoClassing | +7.68 ± 0.29 | +7.34 ± 0.22 | **−0.34** | 0.29→0.22 | 0.75 | 2.0 | ❌ regressed |
| Language | −9.38 ± 1.08 | −4.64 ± 2.99 | +4.74 | 1.08→**2.99** | **2.78** | 6.0 | ⚠ σ↑, noise |
| MultNIST | −0.11 ± 3.00 | +1.52 ± 2.31 | +1.62 | 3.00→2.31 | 0.77 | 4.3 | ◐ partial |
| **Total (5)** | **−2.70 ± 8.63** | **+7.07 ± 6.45** | **+9.76** | 8.63→6.45 | **0.75** | | mixed |

Per-seed E1 totals: **+12.56 / +8.67 / −0.04**. Mechanically flawless across all 15 runs (ensemble size scales with slack, `averaged N/N` every dataset, zero −10, no crashes).

### Honest verdict — mechanism validated, NOT a blanket fix

**Seed 42 alone (+12.56) would have declared a clean triumph. The 3-seed discipline caught a mixed, partly-negative truth.**

✅ **Where it works, it works well.** AddNIST — the exact high-variance bomb E1 targets — σ **5.45 → 1.00 (×0.18, −82 %)** *and* mean −0.38 → **+2.75**. CIFARTile: small clean win, no downside. When there is real slack and the variance is decorrelatable raw-accuracy noise, E1 does exactly what the thesis predicted.

❌ **But it fails its actual objective and has a real flaw:**

1. **Total variance is NOT collapsed.** E1 total σ 6.45 vs Tier-1 8.63 — only ×0.75; per-seed totals still span −0.04 … +12.56. A Phase-2 *single run* of E1 is still a lottery → the [[Reliability vs Peak]] / [[Threats to Validity]] §7 goal (turn the one-shot draw into a stable number) is **not met at submission level**. The residual variance is therefore largely **structural / dataset-level**, not per-run training noise that averaging removes — a direct argument *for* the probe/Tier-2 direction, not "ensemble harder".
2. **GeoClassing regressed −0.34** (violates the must-not-regress criterion). Root cause is mechanical and visible (members = 2.0): when m0 consumes ~the full 2 h, the lone extra member is *under-trained*, and averaging a strong solo m0 with a weak r1 dilutes it. **Cheap fix exists** (below) — carried as a Tier-3 design note, not iterated standalone.
3. **Language's +4.74 is the least trustworthy number** and the single largest contributor to the total mean gain: σ *tripled* (1.08 → 2.99, ×2.78) — the opposite of collapse. Language's real problem is augmentation damage (P1's job), not variance; ensembling 6 aug-damaged members is ensemble luck on a fast-early-stopping dataset, not a stable fix. **Do not credit E1 with "fixing Language."**

**Decomposition of the +9.76:** AddNIST +3.12 (trustworthy) · Language +4.74 (noise-inflated) · MultNIST +1.62 (σ still ±2.3, target <1.5 missed) · CIFARTile +0.62 · GeoClassing −0.34. Strip Language → ~+5 over 5 datasets: real but modest, with submission-level σ essentially unchanged.

### Carry-forward (Tier-3 design note, not a standalone re-run)

- **Member-quality gate:** only fold a trained member into the softmax average if its val-acc ≥ m0_val − ε. Fixes the GeoClassing dilution; turns E1 from blanket → conditional.
- **Slack gate:** only ensemble when real slack exists (m0 early-stopped well under budget). On budget-saturating datasets (GeoClassing) E1 should ≈ no-op.
- E1 stays in the toolbox as a **probe-gated reliability component for AddNIST-class high-variance datasets** ([[Tier 3 - EGNAS + Reliability]]) — not a default wrapper.

### Decision (2026-05-18)

Recorded honest verdict; **proceeding to Tier 2**, designed so the search *emits an ensemble-able pool* (not a single argmax arch) — the residual structural variance is exactly what the probe/Tier-2 approach exists to address. E1 is not re-run now; its gate is a Tier-3 note. See [[Probe-Driven Self-Configuration]] §E1 and [[Run State + Handoff]].
