# Run State + Handoff

> **Status:** live operational state (2026-05-16) | **Topic:** Everything needed to resume work after a context reset — Spark state, experiment results, how to pull/aggregate/plot/notate results, the next-action queue. **Multi-seed sweep COMPLETE → [[Multi-Seed - Results]] is the authoritative finding.**
> **Related:** [[Probe-Driven Self-Configuration]] (plan + "Questions we want answered"), [[Tier 1 - Results]], [[Implementation Results Overview]]

This is the **resume document**. A fresh session should: (1) read [[Probe-Driven Self-Configuration]] for the thesis/plan/open-questions, (2) read this for operational state, (3) `ssh spark` and run the live-status command below — never trust a stale snapshot, always re-check live.

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

**🔄 FULL SWEEP RUNNING — relaunched directly 2026-05-19 09:13 CEST** (no re-gate needed; smoke already verified). `nohup bash dev/run_tier2b_full.sh` (was sweep PID 337030 / python 337037), seed 42 / 11 datasets, `baselines/tier2b_egnas_full`, `--time-limit 2.0`, paired vs `multiseed/full_s*`. ~28-32 h → **ETA ~late 2026-05-20 / early 05-21**. Stale waiters killed; only the sweep runs.

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
5. **← ACTIVE HEAD. FULL EGNAS (macro + micro) built — running the full 11-ds × 3-seed sweep.** `baselines/tier2b_egnas_full/` (macro frozen at `tier2b_egnas_macro/` as comparator). Micro = per-layer op hill-climb, incumbent-kept ⇒ subsumes the old "stabilise arch-selection" item (the principled lottery fix). py_compile OK. Next: rsync → ~15-min correctness smoke (2 fast ds, tight clock) → `nohup bash dev/run_tier2b_full.sh` (~28-32 h), paired vs `multiseed/full_s*`. On landing: aggregate (reuse `aggregate_multiseed`/`aggregate_egnas` pattern over 11 ds) vs Tier-1; check Q8 arch-stability now incumbent-kept; honest verdict + commit.
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
