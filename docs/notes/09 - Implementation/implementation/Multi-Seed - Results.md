# Multi-Seed — Results

> **Status:** current | **Topic:** The 3-seed variance + augmentation-effect sweep of the Tier 1 recipe. Quantifies how much of every single-seed delta is noise, and confirms which findings are seed-stable. **This note supersedes the point estimates in [[Tier 1 - Results]] as the trustworthy numbers.**
> **Related:** [[Tier 1 - Results]], [[Probe-Driven Self-Configuration]], [[Threats to Validity]], [[Reliability vs Peak]], [[Implementation Results Overview]], [[Run State + Handoff]]

Run 2026-05-15 → 2026-05-16, DGX Spark (NVIDIA GB10, torch 2.11.0+cu128). Code: `baselines/tier1_aug_op_search/` (`full` config **is** the Tier 1 recipe). Driver `dev/run_multiseed.sh`, aggregation `dev/aggregate_multiseed.py`, plot `dev/plot_tier_results.py --multiseed`.

**Matrix:** `full ×11 datasets ×3 seeds` + `aug-off ×7 aug-firing datasets ×3 seeds` = 54 runs, `--time-limit 2.0` (matches the Tier 0/1 sweeps so single-vs-multi-seed is apples-to-apples). Seeds 42 / 1337 / 2024. **Unit of analysis = mean ± std across the 3 seeds per (config, dataset)** — never a single number. Runs are non-reproducible even at a fixed seed (`cudnn.benchmark=True`).

Two scoring lenses, always paired: **shipped** (what the grader computes; Sudoku/Sokoto `benchmark=0.0` so its adj is just raw/10) and **internal** (Sokoto→70.95 via `dev/internal_benchmarks.py`; the honest tier-to-tier story). Internal is the default below.

![[multiseed.png]]

## A. Seed variance — the headline

| config | N ds | internal total (mean ± std) | min | max | spread | per-seed |
|---|---:|---:|---:|---:|---:|---|
| **full** (Tier 1 recipe) | 11 | **+8.39 ± 8.37** | +0.02 | +16.75 | **16.73** | +16.75, +8.40, +0.02 |
| aug-off | 7 | −21.80 ± 4.23 | −26.15 | −17.69 | 8.46 | −21.55, −17.69, −26.15 |

Shipped-scoring totals (grader's view): `full` = **+11.27 ± 8.21** (seeds +19.59, +11.06, +3.16); `aug-off` = −21.80 ± 4.23.

**The std is the size of the mean.** One seed of the *exact same recipe* scored a near-break-even **+0.02 internal**; another scored **+16.75**. This is the strongest possible empirical statement that single-seed deltas in this setting are noise.

**Reconciling [[Tier 1 - Results]]'s +24.644:** that headline was **one seed, and it sits above the maximum of all three re-runs** (shipped max here = +19.59; internal +21.7 vs new internal max +16.75). It was a favourable tail draw, not the central tendency. The honest Tier 1 number is **≈ +11 shipped / ≈ +8 internal, ± ~8**. The single-seed caveat already in that note was correct; this sweep gives it a measured magnitude.

## B. Per-dataset stability (`full`, internal adj, mean ± std over 3 seeds)

| dataset | adj mean ± std | spread | per-seed | class |
|---|---:|---:|---|---|
| GeoClassing | **+7.68 ± 0.29** | 0.53 | +7.34, +7.87, +7.83 | rock-stable win |
| Sudoku | **+5.94 ± 0.34** | 0.68 | +6.00, +6.25, +5.57 | rock-stable win |
| Myofibre | +1.44 ± 0.55 | 1.11 | +1.98, +0.87, +1.48 | stable small win |
| Voxel | +1.38 ± 0.32 | 0.58 | +1.60, +1.02, +1.54 | stable small win |
| Gutenberg | +0.92 ± 0.13 | 0.24 | +0.86, +0.83, +1.07 | stable, ~neutral |
| GameOfLife | +0.91 ± 0.50 | 0.90 | +0.65, +0.59, +1.49 | stable, ~neutral |
| Chesseract | +0.50 ± 0.19 | 0.37 | +0.29, +0.53, +0.67 | stable, ~neutral |
| MultNIST | −0.11 ± **3.00** | 5.98 | +2.77, +0.12, −3.21 | **high-variance** |
| AddNIST | −0.38 ± **5.45** | **10.50** | +4.04, +1.29, **−6.46** | **high-variance** |
| CIFARTile | −0.52 ± 0.52 | 1.02 | −0.64, −0.96, +0.06 | stable ceiling |
| Language | **−9.38 ± 1.08** | 1.86 | −8.14, −10.00, −10.00 | **stable failure** |

Variance is **dataset-specific**: 8 of 11 datasets are tight (std < 0.6); the entire seed-spread of the total comes from **AddNIST and MultNIST** (std 5.4 / 3.0). These are not "fixed" — with the full recipe they are a coin-flip between solidly positive and deeply negative.

## C. Augmentation effect — `full` − `aug-off`, shared 7, internal adj

| dataset | full | aug-off | Δ (full − augoff) |
|---|---:|---:|---:|
| MultNIST | −0.11 ± 3.00 | **−10.00 ± 0.00** | **+9.89 ± 3.00** |
| AddNIST | −0.38 ± 5.45 | **−10.00 ± 0.00** | **+9.62 ± 5.45** |
| GeoClassing | +7.68 ± 0.29 | +3.10 ± 1.98 | +4.58 ± 1.72 |
| Myofibre | +1.44 ± 0.55 | −1.85 ± 1.38 | +3.29 ± 1.51 |
| CIFARTile | −0.52 ± 0.52 | −1.24 ± 0.27 | +0.73 ± 0.73 |
| Gutenberg | +0.92 ± 0.13 | +0.76 ± 0.16 | +0.16 ± 0.24 |
| **Language** | −9.38 ± 1.08 | **−2.56 ± 2.03** | **−6.82 ± 1.88** |
| **TOTAL (7 ds)** | **−0.34 ± 8.74** | **−21.80 ± 4.23** | **+21.46** |

On AddNIST/MultNIST, aug-off is a **dead −10.00 ± 0.00 floor every seed** — augmentation is the difference between functioning and total failure. **Language is the sole large reversal** and it is seed-stable (Δ −6.82 ± 1.88, negative all 3 seeds). Aug is worth **+21.5 net** over the 7.

## What this changes vs. the [[Tier 1 - Results|single-seed Tier 1 beliefs]]

1. **"+25 swing / +24.644" → a tail draw.** Honest Tier 1 ≈ +11 ± 8 shipped. The qualitative Tier-0→Tier-1 jump (escaping the −10 floors) is real and large; the precise total was lucky.
2. **AddNIST/MultNIST were never "fixed."** The single-seed note read their −10→+1.9 as a solved problem. They are high-variance (AddNIST −0.38 ± 5.45) — sometimes +4, sometimes −6.5. This is a *separate* failure mode from the aug question.
3. **The Language regression is confirmed and seed-stable** (−6.82 ± 1.88 aug-cost, negative every seed) — not noise. The aug double-edge is real; its *direction* is rock-solid even though the AddNIST *magnitude* is noisy.
4. **Aug is overwhelmingly net-positive and sharply targeted** (+21.5 / 7 ds; only Language negative). The metadata-blind double-edge is the central problem, exactly as stated — now with the target precisely localised.

## Methodological consequence (feeds [[Probe-Driven Self-Configuration]] §methodology)

- **Single-seed deltas are dead as evidence.** Any claim that drives a frozen probe threshold or a write-up number must be ≥3-seed mean ± std. The +4.9 oracle-aug and ~+30.3 projections in [[Tier 1 - Results]] must be re-expressed with this spread.
- **Ensemble, not select-by-val.** With AddNIST's per-run outcome a ±5.4 coin-flip, picking one model by validation accuracy inherits that variance. A budget-filling **diverse-restart ensemble** ([[Probe-Driven Self-Configuration]] E1) collapses it — **E1 is promoted from optional to necessary**, this sweep is its empirical justification.
- **The P1 aug-safety probe's job is now quantified.** It must detect the *one* Language-type case without false-positiving on the six where aug ranges neutral-to-+9.9. False-negative cost (turning aug off on AddNIST) ≈ −9.6; this is the empirical basis for the **asymmetric keep-aug fail-safe** (the bar to *disable* aug must be very high).
- **Probe targets must be seed-stable to be probeable.** The probe can only chase signals like the Language reversal (Δ −6.82 ± 1.88, consistent) — not AddNIST's sign, which is seed-noise, not a property of the data.

## Reproduce

```bash
rsync -a spark:'~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/logs/multiseed*' dev/logs/
python3 dev/aggregate_multiseed.py            # tables + dev/logs/multiseed/_aggregate.json
python3 dev/plot_tier_results.py --multiseed  # docs/notes/.../implementation/multiseed.png
```
