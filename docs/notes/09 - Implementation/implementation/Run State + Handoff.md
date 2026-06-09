# Run State + Handoff

> **Status:** live operational state (updated 2026-06-02) | **Topic:** Everything needed to resume work after a context reset — Spark state, experiment results, how to pull/aggregate/plot/notate results, the next-action queue.
> **Related:** [[Key Learnings]] (read FIRST — the distilled insight), [[Probe-Driven Self-Configuration]] (plan), [[Tier 2a - Results]], [[Tier 2b - Results]], [[Implementation Results Overview]]

This is the **resume document**. A fresh session should: (1) read **[[Key Learnings]]** for the distilled what-we-know, (2) read this for operational state, (3) `ssh spark` and re-check live — never trust a stale snapshot.

## ⟵ CURRENT STATE (2026-06-09)

- **v8 = the consolidated config, seed-42 verdict +48.2** (v7 +41.94 core + train-to-budget + adaptive stem; micro dropped). `baselines/tier2a_p1_augsafety_v8/`. The big lever was **train-to-budget** (no early-stop, train the final model to the full per-dataset budget and anneal LR to 0): validated 5 wins / 2 ties / 0 regress on frozen archs, and it recovered the v7 MultNIST regression without the (dropped, harmful) micro-fix. One weak spot: Language −1.03 under live search.
- **✅ TCN excursion COMPLETE — the dilated-conv backbone wins.** Full writeup: **[[Tier 2a - TCN Excursion]]**. Fixed TCN-2d (2-D dilated conv) beats the **searched** conv backbone by **+10.9 total adj** (conv +48.2 to TCN-2d-fixed +59.2, seed 42), winning 6 / tying 4 / losing 3. Losses (Gutenberg, Sudoku, Chesseract) are all small odd-shaped logic inputs. TCN-1d (1-D sequence) is a specialist (Cryptic/Language/Voxel), below conv overall. Backbone bake-off is fragile (proxy wall at backbone level). Plots `images/tier2a/tier2a_tcn_curves_grid.png`, `images/language/lang_tcn_curves.png`.
- **IN FLIGHT (2026-06-10, fully unattended chain):** **v9 seed 42 ONLY** (the dilated TCN-2d block searched inside v8; `baselines/tier2a_p1_augsafety_v9/`, micro off, EGNAS_DMAX=8). A watcher (`/tmp/v9_s42_only.sh` on spark) auto-cancels seeds 1337/2024 the moment s1337 would start; the 3-seed treatment is reserved for the eventual submission config. Behind it, **Phase A auto-queues** (`/tmp/phaseA_queue.sh` → `dev/run_phaseA.sh`, ~10h): backbone-gate arms (13 ds x 3 arms), TSE-in-search logging (4 ds), aug pre-filter (`dev/test_aug_prefilter.py`), and the **regime retests** (stem x dilated on Voxel/Chesseract via `PROBE_TCN_STEM`, aug on/off x tcn1d on Language). Free retests from v9 logs afterwards: does verification ever fire for the dilated family, does width-prune / the D-cap bind. Early v9 signal: AddNIST D=8 96.2 val, CIFARTile D=5 76.3 val (above the fixed TCN-2d's 72.2 test).
- **Plan docs:** [[v10 - Design + Idea Ledger]] (measure-gate-train pipeline, Phase A details, v11 shortlist: SSM / MLP-Mixer / ConvNeXt-block / RandWire screening aimed at the encoded datasets). [[Key Learnings]] updated with Learnings 10-14 (train-to-budget, backbone family dominates, abstention gate, fingerprint, regime-staleness).
- **Random-search arm DONE (9/9):** wins Chesseract on all 3 seeds (+0.90 mean vs conv +0.30), ties Sudoku with worse variance, loses Gutenberg. Verdict: keep as a budget-gated arm for encoded/weird inputs only.
- **SOTA scan + offline validations DONE:** 19 paper notes in `08 - References/papers/`, synthesis in [[SOTA Scan - Novel Directions]]. Offline: the abstain-to-default backbone gate beats the fixed default at every budget; TSE fails cross-family; LC extrapolation fails early. Fingerprint probe (`dev/fingerprint_probe.py`, CPU): corr1 near zero cleanly detects the 4 encoded datasets (Language/Sudoku/Chesseract/Gutenberg), exactly where specialists win.
- **v10 design + full idea verdicts:** [[v10 - Design + Idea Ledger]] (ship list, Phase A tests, rejected list, the measure-gate-train pipeline, overfitting guards).
- **Live status:** `ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && pgrep -af mock_harness|grep -v pgrep; ls dev/logs/random_search/*.done dev/logs/tier2a_p1_v9/*.done 2>/dev/null|wc -l"`

### Older state (2026-06-02)

- **v6 P1 sweep DONE — verdict: net regression.** EGNAS-full +35.51 ± 3.07 → +P1 v6 +23.41 ± 9.17 (internal, 13 ds × 3 seeds), Δmean **−12.10**, σ **×2.98**. Rich vocab poisoned Sudoku (−20 pp), macro-prune cratered GameOfLife (depth lottery). Only win = Language aug-probe +4.23. Full deep analysis + 3-seed figures in [[Tier 2a - Results]].
- **LSTM Language probe DONE — row-BiLSTM cracks Language.** First arch in the project reliably above benchmark: 85.76 % (all 3 seeds > 85.20). Attention doesn't cross (84.8 %); cross-axis unstable (one seed 78 %). Figure `images/language/lstm_lang_probe.png`.
- **Micro-vocab for recurrence = DEAD.** Short-train test proved the search's crude SGD can't even train an LSTM (thrashes 22↔79 %); recurrence only performs under the full trainer. → recurrence must be deployed via a full-convergence A/B, not the search.
- **v7 BUILT + GPU-smoke-PASSED.** `baselines/tier2a_p1_augsafety_v7/` = EGNAS-full clean search + 4 conv ops + macro-prune-verify (depth verify-of-rejects + width verify-of-accepts) + aug-probe. Smoke: Chesseract +0.379, search/train/predict all clean, exit 0. Sweep script `dev/run_tier2a_p1_v7.sh`. Verify knobs (`VERIFY_K_MULT=3`, `MAX_MACRO_VERIFY=5`) still need GPU calibration on GameOfLife+Sudoku.
- **✅ v7 macro-verify CALIBRATION PASSED (2026-06-02) — both v6 failures cured, both seeds.** `dev/logs/v7_calib/`. **GameOfLife: 99.94 / 99.81 %** (v6 ~48 % — was a lottery; depth verify-of-rejects rescued false-strikes at D=5/7/8/9 → grew DEEP → reliably SOLVED). **Sudoku: 92.61 / 93.38 %** (v6 67-71 %; depth-verify grew D=9 + width stayed W=48 → beats EGNAS-full's ~91 %). The headline fix works exactly as designed.
- **✅ Recurrence A/B validation (partial, stopped early — enough signal).** `dev/logs/recur_val/`. On AddNIST (where 5-block conv is strong, 92.5 %), **row-BiLSTM loses to conv by ~7 pp (85.3 %)** → the blind A/B correctly picks conv on conv-optimal data. Sudoku was a confounded "tie" (5-block under-powers conv on a depth-needing task — design flaw, not a real LSTM win). CIFARTile skipped. Conclusion: the A/B does NOT over-pick recurrence.
- **✅ FULL v7 SWEEP DONE — VERDICT: v7 BEATS EGNAS-full.** 3-seed × 13-ds. **EGNAS-full +35.51 ± 3.07 → +P1 v7 +41.94 ± 3.45 (internal), Δmean +6.43, σ ×1.12.** Best documented result in the project. Per-seed v7 [43.24, 38.03, 44.54]. **The win is dominated by Language Δ+7.34** (aug-probe OFF all 3 seeds → −8.32 → −0.98) — i.e. the verify's real job was making archs stable enough that the aug-probe gain isn't eaten by arch variance (which killed it in v6). GameOfLife Δ+0.65 (better + tighter, σ 2.28→1.95); small gains Cryptic/Sudoku/CIFARTile. **One regression: MultNIST −1.15** (v7 deep-plain-conv < EGNAS's MultNIST arch). Aggregator `dev/aggregate_tier2a_p1_v7.py` → `_tier2a_p1_v7_aggregate.json`; figures `--tier2a-p1-v7[-curves]` + per-seed. **KEY INSIGHT: aug-probe is the engine; the LSTM's marginal value shrank to ~+1.4 on Language** (already near-benchmark via conv+aug-off), so the next-run priority shifts toward the **channel-embed/data-preprocessing** lever + MultNIST recovery, LSTM as a small top-up.

**Plan (see [[Key Learnings]] §"ideal path"):** `v7-conv` is the submission core (now validated). Recurrence = a fail-safe, budget-gated, conv-first convergence-A/B opt-in (validated it picks conv on conv-optimal data; still TODO to wire into v7). Next when sweep lands: aggregate + verdict + notes + commit; then decide whether to add the recurrence A/B.

**Lecture-informed next-run ideas (2026-06-03) → [[Next-Run Ideas - lecture-informed]]:** AutoML Ch6 + exam validated our findings (space > strategy; ZCPs weak/params competitive) and flagged 3 ideas — (1) LC-extrapolation as a cheap verify pre-filter, (2) the untouched **data-processor** component (metadata-driven stem/input handling), (3) cell-based transferable space (frontier). Three incorporation paths (A=v7.1 cheap wins / B=clean 3-component poster framing / C=cell-space bet). **PROPOSAL DUE June 15 9am** (research question + steps + eval → a.winje@/m.wever@, supervisors Lindauer/Wever/Winje). Rec: Way A next test, Way B poster framing.

**Live status:** `ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && pgrep -af mock_harness | grep -v pgrep; grep -hE 'raw=|Total score' dev/logs/tier2a_p1_v7/p1v7_s42.log | tail -15"`

## Spark (the only place torch runs — local box has no GPU)

- SSH host alias: `spark` (key `~/.ssh/id_ed25519_dgx`, user `lysfy`, NVIDIA GB10).
- Repo on Spark: `~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/`
- Datasets: `~/local/nas-comp-datasets/` (symlinked as `datasets/`)
- venv: `.venv/bin/python` (torch 2.11.0+cu128). Dev loop = rsync local→spark, run via ssh, results rsync back.

## ✅ EGNAS-macro CIFARTile diagnostic — COMPLETE (2026-05-18, 3 seeds). Verdict: **PASS + 2 caveats**

Authoritative writeup: **[[Tier 2b - Results]]**. Make-or-break **passed**: CIFARTile **−0.515±0.521 → +0.167±0.269** (all 3 seeds above benchmark, σ↓) — the ceiling Tier-1 couldn't move; EGNAS beats Tier-1 on the 3-ds mean (+6.79→+10.05). **But:** (1) the CIFARTile gain is *small* (real frontier lever = E2/Deep-Sets), (2) the macro search is itself a **Q8 arch-lottery** (s2024 picked D=5 for AddNIST → −2.81). EGNAS-macro is validated & kept but was **not a finished engine**.

**→ Superseded 2026-05-18: completed to FULL EGNAS (macro + micro).** Reasoning: judging the proven 2025-winning method by *half* of it (macro-only) was premature; a re-rank denoiser is low-EV polish on an incomplete method. Instead **finished the method** — added a micro per-layer op hill-climb ({Conv3,Conv5,SepConv3}), which is *incumbent-kept* so it is also the **principled fix for the macro Q8 lottery** (a noisy fluke can't hijack a kept incumbent), not a bolted-on denoiser. Macro-only kept **frozen** as comparator: `baselines/tier2b_egnas_macro/` (untouched); full method = **`baselines/tier2b_egnas_full/`** (only `nas.py` differs). Full reasoning in [[Tier 2b - EGNAS-light (macro)]] §Update 2026-05-18. **User decision: run the full 11-ds × 3-seed sweep directly** (`dev/run_tier2b_full.sh`) for a comprehensive performance read — a ~15-min correctness smoke precedes it (guards 30 h-on-untested-code, not the strategic gate). E2 re-ordered to *after* full-EGNAS validation, as a micro-vocabulary op.

**⚠ Gated-launcher bug postmortem (2026-05-18→19):** `dev/run_tier2b_full_gated.sh` waited on `pgrep -f "predictions-egnasfullsmoke"`. A leftover background *monitor* had that exact string in its own argv → the two waiters perpetually matched each other; the smoke actually finished overnight but neither could detect it → **deadlock, ~10 h overnight window lost, sweep did not auto-launch.** Lesson: never wait on a `pgrep -f` pattern that other helper processes' command lines also contain — match the specific script/python (`pgrep -af run_tier2b_full.sh`) or use a PID file. The gated script is otherwise sound but **don't reuse its pgrep pattern**.

**✅ Smoke verified PASSED** (read directly): full EGNAS macro+micro ran end-to-end, micro hill-climb fired correctly (`micro-changed 4/9 ops`), no traceback, Total +8.816, predictions written. Gate satisfied.

**✅ FULL SWEEP COMPLETE 2026-05-20 11:08 — verdict in [[Tier 2b - Results]].** Headline: Tier-1 +8.39 ± 8.37 → **EGNAS-full +27.99 ± 3.28** (internal adj, 11 ds, 3 seeds) — **Δmean +19.60, σ ×0.39 collapse**, s2024-cratering pattern broken. 6 big wins, 2 honest regressions (Myofibre −0.96, Chesseract −0.37), Language still aug-bound (P1). E1 superseded for variance control — full EGNAS does it via architecture, not ensembling. Aggregator `dev/aggregate_tier2bfull.py` → `_tier2bfull_aggregate.json`; figure `tier2bfull.png` via `--tier2bfull`.

**🔄 Windspeed + Cryptic follow-on RUNNING (launched 2026-05-20 ~12:20, ~12 h ETA).** `dev/run_tier2b_newdatasets.sh` (PID was 375475), `tier2b_egnas_full × {Windspeed, Cryptic} × seeds 42/1337/2024`, paired-Tier-1 N/A for these 2 (not in multiseed). On landing: extend `aggregate_tier2bfull.py` (or a sibling) to include them, regenerate `tier2bfull.png`, append a §Windspeed+Cryptic block to the Full-EGNAS results note.

**Check command (FIXED — specific match, not the buggy pattern):**
```bash
ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && \
  grep -E 'config=|started:|finished:|Total score|COMPLETE' dev/logs/tier2bfull-all.log; \
  for s in 42 1337 2024; do f=dev/logs/tier2bfull/t2bfull_s\$s.log; [ -f \"\$f\" ] && printf 's%s ' \$s && (grep 'Total score' \"\$f\"|tail -1||echo running); done; \
  echo '— active —'; tail -n 3 \"\$(ls -t dev/logs/tier2bfull/t2bfull_s*.log|head -1)\"; \
  pgrep -af run_tier2b_full.sh | grep -v pgrep | head -1"
```
`pgrep` empty + `COMPLETE` = done. Per-seed logs `dev/logs/tier2bfull/t2bfull_s<seed>.log` (scores + `EGNAS full done` + macro/micro trace); pool JSONs `…/t2bfull_s<seed>/<codename>_egnas_pool.json`. On landing: aggregate 11-ds vs Tier-1 `full_s*` (reuse `aggregate_egnas` pattern, extend to 11 ds), check Q8 arch-stability now incumbent-kept (vs macro-only `egnasdiag`), honest verdict → `Tier 2b - Results.md` update + commit. Macro-vs-full attribution stays precise on the 3-ds `egnasdiag`.

**Live status:**
```bash
ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && \
  grep -E 'config=|started:|finished:|Total score|COMPLETE' dev/logs/egnasdiag-all.log; \
  echo '— active —'; tail -n 4 \"\$(ls -t dev/logs/egnasdiag/*.log | head -1)\"; \
  pgrep -af run_egnas_diagnostic | grep -v pgrep | head -1"
```
`pgrep` empty + `COMPLETE` = done. Per-seed logs `dev/logs/egnasdiag/egnas_s<seed>.log` (scores + `EGNAS macro done: best D=/W=` + per-arch search trace); pool JSONs `dev/logs/egnasdiag/egnas_s<seed>/<codename>_egnas_pool.json`.

**On landing:** pull (`rsync -a spark:'~/projects/.../dev/logs/egnasdiag*' dev/logs/`); compare CIFARTile mean±std (3 seeds) vs Tier-1 `full_s*` CIFARTile (−0.515 ± 0.521); also AddNIST (high-var sanity) + GeoClassing (must-not-regress). **Decision gate:** if CIFARTile beats the ceiling multi-seed → full 11-dataset sweep + write `Tier 2b - Results.md`, fold into thesis. If not → that is a thesis-critical finding (architecture has little headroom here) → consolidate/ship a reliable Tier-1-class submission instead of chasing peak. Either way: aggregate (extend `aggregate_e1.py` pattern), measure arch-selection stability across seeds (Q8 — does macro-search pick consistent D/W?), commit the Tier 2b batch (no AI attribution).

## ✅ E1 validation — COMPLETE (2026-05-16 → 2026-05-18, 3 seeds). Verdict: **MIXED**

Authoritative writeup: **[[E1 - Results]]**. Headline: the mechanism is validated on its target (**AddNIST σ 5.45→1.00 ×0.18 AND mean −0.38→+2.75**), but E1 is **NOT a blanket reliability layer** — total submission σ only ×0.75 (per-seed totals still −0.04…+12.56), **GeoClassing regressed −0.34** (under-trained member dilutes a budget-saturating solo m0), and Language's +4.74 is noise-inflated (σ ×2.78). Decisive takeaway: residual variance is **structural/dataset-level → cannot be ensembled away → reinforces the probe/Tier-2 direction**. E1 → conditional, probe-gated component for AddNIST-class datasets with a member-quality + slack gate (a **Tier-3 design note**, not re-run now). Decision 2026-05-18: **proceed to Tier 2**, designed so the search emits an *ensemble-able pool*. The live-status / "when it lands" blocks below are historical (done).

**Live status command** (run on resume):
```bash
ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && \
  grep -E 'config=|started:|finished:|Total score|COMPLETE' dev/logs/e1val-all.log; \
  echo '— active —'; tail -n 3 \"\$(ls -t dev/logs/e1val/*.log | head -1)\"; \
  pgrep -af run_e1_validation | grep -v pgrep | head -1"
```
`pgrep` empty + `COMPLETE` banner = done. Outputs: `dev/logs/e1val-all.log` (timeline), `dev/logs/e1val/e1_s<seed>.log` (scores + `E1: ensemble of N`/`E1 predict: averaged` banners + per-epoch), `dev/logs/e1val/e1_s<seed>/<codename>[.<tag>].csv` (member curves; `.r*`=seed restart, `.a*`=arch member), `dev/predictions-e1val-e1_s<seed>/`.

**Smoke (passed, 2026-05-16):** Chesseract+Sudoku at 0.08 h — multi-member path, averaged predict on 12×8×8 and 9×9, reserve/clock guards, and the no-slack→1-member≡Tier-1 fallback all verified, zero exceptions.

**When it lands (tools pre-written + tested 2026-05-16):**
```bash
rsync -a spark:'~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/logs/e1val*' dev/logs/
python3 dev/aggregate_e1.py            # σ-collapse vs Tier-1 + auto-verdict + _e1_aggregate.json
python3 dev/plot_tier_results.py --e1  # → docs/notes/.../implementation/e1.png
```
`dev/aggregate_e1.py` reuses `aggregate_multiseed`'s parser, pairs `e1_s*` vs the `full_s*` Tier-1 baseline subset (5 datasets), prints per-dataset mean±std, **σ-collapse** (target AddNIST σ_adj 5.45 → <2), Δmean, ensemble-size, and a pass/fail verdict. Then write `E1 - Results.md` (embed `e1.png`), fold into [[Probe-Driven Self-Configuration]] §E1 + [[Multi-Seed - Results]], commit the E1 batch. Success = σ shrinks substantially AND mean ≥ Tier-1 mean AND no regression on GeoClassing/Language/no new −10. **Then** present the Tier-2-vs-next decision with data.

## ✅ Multi-seed sweep — COMPLETE (2026-05-15 13:11 → 2026-05-16 15:23 CEST, ~26 h)

All 54 runs landed, pulled + aggregated + notated → **[[Multi-Seed - Results]]** is the authoritative finding. Headline: Tier 1 recipe = **+11.3 ± 8.2 shipped / +8.4 ± 8.4 internal** over 3 seeds (seed variance ≈ the whole signal; the old single-seed +24.644 was a tail draw). It is the **paired Tier-1 baseline for the E1 sweep above**.

Original matrix (template for future sweeps): `full ×11 ×3 seeds` + `augoff ×7 ×3 seeds` = 54 runs, `--time-limit 2.0`, seeds `42 1337 2024`, `nohup`'d, ~26 h wall. Driver `dev/run_multiseed.sh`.

**Live status command** (retained for the *next* long run; `pgrep` empty = done):
```bash
ssh spark "cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && \
  grep -E 'config=|finished:|Total score' dev/logs/multiseed-all.log; \
  echo '— active —'; tail -n 3 \"\$(ls -t dev/logs/multiseed/*.log | head -1)\"; \
  pgrep -af mock_harness | grep -v pgrep | head -1"
```
If `pgrep` is empty → sweep finished (or died — check `multiseed-all.log` tail for the completion banner).

Outputs on Spark:
- `dev/logs/multiseed-all.log` — runner timeline + end-of-run per-config score grid
- `dev/logs/multiseed/<cfg>_s<seed>.log` — per-sweep log (has `raw=…benchmark=…adj=…` scoring lines + per-epoch lines)
- `dev/logs/multiseed/<cfg>_s<seed>/<codename>.csv` — per-epoch curves (epoch,lr,train_loss,val_acc,val_acc_ema,best_val_acc,epoch_time_s,clock_remaining_s)
- `dev/predictions-multiseed-<cfg>_s<seed>/<codename>.npy` + `_stats.pkl`

## How to cumulate results + run stats

1. **Pull from Spark:**
   ```bash
   rsync -av spark:'~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/dev/logs/multiseed*' dev/logs/
   ```
2. **Scores** are parsed from the sweep logs (`raw=`/`benchmark=`/`adj=` lines) — same regex as `dev/plot_tier_results.py:parse_scores`. For multi-seed the unit of analysis is **mean ± std across the 3 seeds per (config, dataset)** — never a single number (runs are non-reproducible even at fixed seed: `cudnn.benchmark=True`; see [[Tier 1 - Results]] single-seed caveat).
3. **Sudoku** ships placeholder `benchmark=0`. Use `dev/internal_benchmarks.py` (`adj_score_internal`, override Sokoto=70.95) for any honest tier-to-tier or multi-seed total. mock_harness itself stays faithful to shipped values (that's what the grader does).
4. **Aggregation script: DONE** — `dev/aggregate_multiseed.py` reads the 6 sweep logs → per-(config,dataset) mean±std adj (shipped + internal), full−augoff delta, per-seed scatter; writes `dev/logs/multiseed/_aggregate.json`. Re-run both: `python3 dev/aggregate_multiseed.py && python3 dev/plot_tier_results.py --multiseed`.

## How to plot + notate

- Plotter: `dev/plot_tier_results.py` (per-tier `--tier {tier0,tier1}` + `--ablation`). Uses `matplotlib.use("Agg")` (a Gdk segfault-on-exit is benign — PNGs are written before it). Outputs go to `docs/notes/09 - Implementation/implementation/`.
- Multi-seed plot mode: **DONE** — `dev/plot_tier_results.py --multiseed` reads `_aggregate.json` → 3-panel `multiseed.png` (per-dataset mean±std + seed dots; full-vs-augoff ±std; per-config seed-total scatter). Embedded in [[Multi-Seed - Results]].
- **Notation convention:** results live in `docs/notes/09 - Implementation/implementation/<Tier N> - Results.md`, graphs embedded with Obsidian `![[name.png]]` (resolves by basename across the vault). Plan notes in `…/plan/`. Every results note carries the single-seed/variance caveat where relevant. Methodology rule (survey Sec 9.1): probe thresholds frozen on held-out data, never tuned on eval.

## Next-action queue (priority order)

1. ~~Aggregate + variance finding + Multi-Seed note + fold caveats~~ **✅ DONE 2026-05-16** — [[Multi-Seed - Results]]; Tier 1 caveat + thesis methodology + Overview folded in; committed.
2. ~~Step 0 (Q8 search-stability) + implement E1 + smoke~~ **✅ DONE 2026-05-16** — `dev/analyze_search_stability.py` (search instability universal but not the discriminating cause → E1 ensembles restarts *and* candidate archs); `baselines/tier1e1_ensemble/` built, smoke-passed.
3. ~~E1 validation + writeup + fold + commit~~ **✅ DONE 2026-05-18** — [[E1 - Results]] (MIXED verdict: mechanism validated on AddNIST, not a blanket fix, GeoClassing regression, Language noise; residual variance structural → argues for Tier 2). Folded into thesis §E1 + Overview; aggregator verdict-line de-rosed; E1 batch committed (see Git state). Decision: proceed to Tier 2.
4. ~~Implement Tier 2b EGNAS-macro + CIFARTile diagnostic~~ **✅ DONE 2026-05-18** — [[Tier 2b - Results]] (make-or-break PASSED: CIFARTile −0.515→+0.167 all 3 seeds, σ↓; beats Tier-1 on the 3-ds mean). Caveats: small CIFARTile gain (→E2), and macro search is a Q8 arch-lottery. Folded into thesis Q8 + Overview; committed.
5. ~~Full EGNAS 11-ds × 3-seed sweep + verdict~~ **✅ DONE 2026-05-20** — [[Tier 2b - Results]]: +19.60 mean, σ ×0.39 collapse, s2024-pattern broken. `aggregate_tier2bfull.py` + `--tier2bfull` plot. Variance fixed by architecture; E1 superseded.
6. ~~Windspeed+Cryptic follow-on EGNAS-full sweep~~ **✅ DONE 2026-05-20 15:18** — folded into [[Tier 2b - Results]] §Windspeed+Cryptic; Windspeed mean +0.655 adj (beats 2025 winner's +0.449), Cryptic 68.7 % raw (placeholder benchmark — see headline caveat). Aggregator + curves figure now cover 13 ds.
7. ~~Tier 2a / P1 v2-v4 sweeps + writeup~~ **✅ DONE 2026-05-28** — full arc lives in [[Tier 2a - Results]]. v4 shadow data shows **K=8 is the lever, not multi-trial** (driver A's K=4 unanimity fails on borderline-Δscore Language s2024; B + C, both K=8, were both correct). Net v4 −4.32 vs EG due to EGNAS-internal variance on skip-probe datasets, but **Language is reliably +3.72 Δmean**. Q8-under-EGNAS analysis (`dev/analyze_q8_under_egnas.py`) folded into [[Tier 2b - Results]] §Q8 — σ-collapse is search-space *flatness*, not arch *convergence* (0/13 cross-seed depth agreement).
8. ~~Language Excursion + Literature Review~~ **✅ DONE 2026-05-28** — [[Tier 2a - Language Deep Dive]]. Six training-side variations × 3 seeds × Language only. **Training-side ceiling is ~81.5 % raw**; Mixup is the only lever (+1.6 adj); patience/LR/capacity give ±0.4-0.8. **Geada et al. arXiv 2404.02189**: ResNeXt-50 hits 93.97 % on Language, PC-DARTS 90.12 %, best published competition 89.71 %. Architectural lever is **grouped convolutions** — not in our current OP_VOCAB. Aug-off + Mixup on Language is **novel in the published literature**.
9. ~~P1 v5 build~~ **✅ BUILT but ABORTED 2026-05-28**. Driver A→C swap + Mixup-on-OFF + GroupConv3 + ResNeXt in OP_VOCAB. Sweep killed during s1337 — s42 partial data shows net negative: probe correct, but the new ops produced a **GameOfLife −9.4 pp regression** because the K=2-6 short-train val_acc ranker picked ResNeXt @ L0 on noisy signal that didn't hold up at full training. Code preserved at `baselines/tier2a_p1_augsafety_v5/` for ablation.
10. ~~Read the actual EGNAS paper + code~~ **✅ DONE 2026-05-28**. github.com/siddikui/Efficient-Macro-Micro-NAS + arXiv:2502.03553. Found systematic methodology mismatch in our port (4-op vocab vs our 7+, K=2 fixed vs our 2-6 depth-scaled, NO verification of accepts, 600-epoch final training vs our ~70 at 2h budget, param-fairness via width shrink we don't enforce). The Q8 problems we measured aren't a contradiction of the paper — their "Spearman 0.91" was for a different measurement (param count vs incrementally-trained val on CIFAR-10). See [[EGNAS]] §"Our port vs upstream".
11. **✅ DONE 2026-06-01 — VERDICT: net regression.** **P1 v6 — Q8-stabilised methodology + 4 more architectural primitives + zero-shot proxies** (task #49). Full 3-seed sweep complete. **EGNAS-full +35.51 ± 3.07 → +P1 v6 +23.41 ± 9.17 (internal) → Δmean −12.10, σ ×2.98 — WORSE on mean AND variance.** Root cause: rich 9-op vocab POISONS plain-conv-optimal datasets (Sudoku −20 pp: same D=7 W=48, EGNAS plain-Conv3 91% vs v6 Conv5+attn 71% — exotic ops win noisy K=12 short-train, crater at full train; verify@K=24 too short to catch). Macro prune craters depth/width tasks (GameOfLife depth-lottery; Sudoku width-prune to 16). ONLY win = Language aug-probe +4.23 (the one durable P1 component). **Prize: recover the self-inflicted regressions + keep Language → ≈+40 > EGNAS +35.5.** Full deep analysis in [[Tier 2a - Results]] §"3-seed deep analysis". Figures regenerated (3-seed). **→ v7 pivot: SIMPLIFY (EGNAS-full + Language probe + macro-prune-verify), exotic ops tested out-of-band via FIXED_ARCH, NOT in live search.** Built tonight 2026-05-28 ~22:00; smoke launched. Bundles 6 changes: (a) drop A+B probes — only C drives; (b) OP_VOCAB to 9 swappable ops adding **Conv7, DilatedConv3, SEConv3, SpatialSelfAttn2D**; (c) `MAX_SHORT_EPOCHS` 6→12; (d) verification-of-accepts (re-train accepts at 2× K, revert if doesn't hold); (e) clock-extend (NAS time refunded to clock so trainer always sees full 2h; per-dataset wall-clock = 2h + nas_duration); (f) zero-shot proxies (NASWOT, Synflow, SNIP, ZenScore) computed per candidate, logged as shadow signal for post-sweep Spearman analysis. Code: `baselines/tier2a_p1_augsafety_v6/` + `zeroshot.py` + `dev/run_tier2a_p1_v6.sh`. Sweep ~70 h wall-clock with early-stop. **What we expect to learn:** per-op survival rate by dataset; whether verification catches Q8 false-positives; whether zero-shot proxies transfer to Unseen suite (would enable v7 with 10-1000 × faster search if τ ≥ 0.5); cleaner Language verdict with K=12 and probe C.
12. **P1 v7 — scope LOCKED with user 2026-05-30; ambition (full sweep vs targeted) decided after v6 3-seed verdict.** s42 reframed the priority: the **macro depth bug is the headline (+~10 pts)**, the LSTM is secondary. Six changes (full design + build status in [[Tier 2a - Results]] §"v7 locked scope"):
    1. **Macro depth fix — val-slope + verification-of-rejects (HEADLINE).** Depth-grow strikes deep candidates on absolute `val@K`; deep nets on iterated-computation tasks (GameOfLife) look bad at K=12, brilliant at full train. Fix: judge on `val@K + λ·slope`, and re-train would-be-struck deeper nets at 2-3× K before striking (symmetric twin of v6 verify-of-accepts). Moderate `nas.py` refactor (thread per-epoch val through `_short_train→_eval_arch`). **Deferred to implement+validate when v6 lands** — wants GameOfLife epoch-curves to calibrate λ/verify-K, and a free GPU to validate. Cures the v5+v6 GameOfLife regression.
    2. **Recurrent op family — row-BiLSTM + cross-axis BiLSTM. ✅ BUILT + smoke-passed 2026-05-30.** `baselines/tier2a_p1_augsafety_v7/` (copy of v6 + 2 ops in OP_VOCAB + `FIXED_ARCH` hook in nas.py). Full-budget Language A/B ready: `dev/run_lstm_language_probe.sh` (6 fixed 5-block @ W=24 archs, run AFTER stopping v6 sweep post-s1337). Tests LSTM-HEAVY archs not 1-layer swaps — s42 proved 1-layer op-wins don't translate (attention won Language micro, final gain ≈ 0).
    3. **Prune OP_VOCAB to survivors** (gated on v6 3-seed attribution) — cut dead ops (likely SepConv3/SEConv3/DilatedConv3) to ~5+LSTM; 9-op dilution is costing GeoClassing/MultNIST.
    4. **Param-fairness in micro** (gated, then implement) — match params per op (width-shrink, like upstream EGNAS); v6 proxy data shows `params` is the top-Spearman confound (+0.29).
    5. **Proxy-rescue for depth** (gated on v6 proxy τ ≥ 0.4) — use Synflow/Zen to rescue deep candidates the short-train ranker kills; proxies' first decision role. Dropped if τ < 0.4.
    6. **NASWOT sigmoid/tanh hook + per-op K-multiplier** (LSTM = 1.5× K) — fair judging of recurrent ops.
    **Build status:** #2 done+ready; #1/#3/#4/#5/#6 documented + deferred to when v6's verdict + attribution + proxy-τ land (~48 h). `FIXED_ARCH`/`FIXED_WIDTH` hook is the reusable full-budget A/B tool for all v7 dev.
    **Next-action when GPU frees (UPDATED 2026-06-01):** the original "stop before s2024" plan is SUPERSEDED — by the time s1337 landed, s2024 was already ~42 % done, so user decided to **let s2024 finish** (clean 3-seed verdict; AddNIST swings 7 pts across seeds so the variance estimate is worth it). When s2024 completes (~13 h from 2026-06-01 07:30), the GPU frees: `ssh spark` then `nohup bash dev/run_lstm_language_probe.sh > dev/logs/lstm_lang_probe/probe.log 2>&1 &` (~12 h, 6 arms × 2 h). v7 code synced to spark.
    **2-seed verdict so far (s42 / s1337, shipped adj total):** +26.66 / +24.80 (mean ~+25.7). Consistent story: GameOfLife reliably broken (+0.03/+0.09 — depth bug, 2 seeds agree on the WRONG shallow arch), Language reliably below benchmark (−6.51/−4.53 — structural), AddNIST high-variance Q8 lottery (+5.09/−1.94, 9×48→95% vs 5×32→88%), rest solid (Sudoku +8, MultNIST +7, Cryptic +6.8, Voxel +2.6). **Q8 cross-seed arch agreement = 1/8** (only GameOfLife, the broken one) — confirms search-space flatness; single-seed op-attribution is unreliable (Language picked attn on s42, Conv5 on s1337).
13. **E2 (non-conv / Deep-Sets)** for CIFARTile — confirmed: macro+micro conv only crosses benchmark (+0.54), real headroom (~+46 pp) needs non-conv. Belongs as a micro-vocab op.
9. Non-ML prerequisites (user-owned): Sudoku benchmark email (still unpublished as of 2026-05-19 figshare-API check), Myofibre unconfirmed benchmark; x86 cross-verify.
6. **Tier 2a / Probe v1** — P1 aug-safety only. Design locked ([[Tier 2a - Probe v1]]) but **deferred behind Tier 2b**.
7. **E3** architecture×dataset gold grid (thesis validation harness). **E2** (non-conv / Deep-Sets) — confirmed CIFARTile's real frontier lever by [[Tier 2b - Results]].
8. Non-ML prerequisites (long lead, user-owned): organiser email re Sudoku placeholder + **Myofibre unconfirmed benchmark** (Cryptic/Windspeed DOIs **RESOLVED 2026-05-19** — were just published late, now on Edinburgh DataShare CC-BY-4.0; Windspeed = 12×32×32 → ~21 ordered Beaufort bins + temporal shift, Cryptic = 6×768 BERT); x86 cross-verify ([[Threats to Validity]] §1) — *user will do soonish*.

## "Questions we want answered"

Not duplicated here — they live in [[Probe-Driven Self-Configuration]] §"Questions we want answered" (8-item priority list). Q2 (variance scale) **answered** ([[Multi-Seed - Results]]); Q8 (search variance-amplifying?) **answered** — Step 0: Tier-1 op-search instability universal but not the *downstream* driver; **[[Tier 2b - Results]]: EGNAS-macro definitively IS arch-variance-amplifying** (s2024 D=5/AddNIST → −2.81; arch-aware Spearman-0.91 fails at our budget) → arch-selection stabilisation is the active next action; Q4 **answered** ([[E1 - Results]]: partial — yes AddNIST-class, no at submission level; residual variance structural); E3 answers Q3/Q5; partial Q1/Q6 pending Tier 2a.

## Git state

- `main` — pristine.
- `baseline_approaches` (pushed) — Tier 0+1 impl, results notebase, thesis adoption + NAS survey. 2 commits.
- `exploration/probe-self-config` (this branch) — thesis P1 redesign + E1/E2/E3 + arch menu + questions, `BASELINE_SEED` env var, `dev/run_multiseed.sh`, handoff doc.
  - `c69e9d9` (2026-05-16) — multi-seed results (aggregate_multiseed, --multiseed, Multi-Seed note, Tier1/Overview/thesis fold-in, _aggregate.json).
  - `47eba55` (2026-05-18) — E1 validation batch: `baselines/tier1e1_ensemble/`, `dev/analyze_search_stability.py`, `dev/aggregate_e1.py`, `dev/run_e1_validation.sh`, `--e1` plot, `E1 - Results.md`+`e1.png`, thesis §E1/Overview/Plan-Overview fold-in, `_e1_aggregate.json`.
  - `45c939f` (2026-05-18) — Tier 2b batch: `baselines/tier2b_egnas_macro/` (only `nas.py` differs from Tier 1), `dev/aggregate_egnas.py`, `dev/run_egnas_diagnostic.sh`, `Tier 2b - Results.md`, plan notes (`Tier 2a/2b`), thesis Q8 + Overview + Plan-Overview fold-in, `_egnas_aggregate.json`.
  - **All commits LOCAL ONLY — `exploration/probe-self-config` is NOT pushed.** `.obsidian/workspace.json` is local UI noise — never stage it. Commit-msg rule: no AI/Claude/Co-Authored-By attribution.
