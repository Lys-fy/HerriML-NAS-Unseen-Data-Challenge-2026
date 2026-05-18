# Run State + Handoff

> **Status:** live operational state (2026-05-16) | **Topic:** Everything needed to resume work after a context reset — Spark state, experiment results, how to pull/aggregate/plot/notate results, the next-action queue. **Multi-seed sweep COMPLETE → [[Multi-Seed - Results]] is the authoritative finding.**
> **Related:** [[Probe-Driven Self-Configuration]] (plan + "Questions we want answered"), [[Tier 1 - Results]], [[Implementation Results Overview]]

This is the **resume document**. A fresh session should: (1) read [[Probe-Driven Self-Configuration]] for the thesis/plan/open-questions, (2) read this for operational state, (3) `ssh spark` and run the live-status command below — never trust a stale snapshot, always re-check live.

## Spark (the only place torch runs — local box has no GPU)

- SSH host alias: `spark` (key `~/.ssh/id_ed25519_dgx`, user `lysfy`, NVIDIA GB10).
- Repo on Spark: `~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/`
- Datasets: `~/local/nas-comp-datasets/` (symlinked as `datasets/`)
- venv: `.venv/bin/python` (torch 2.11.0+cu128). Dev loop = rsync local→spark, run via ssh, results rsync back.

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
4. **← ACTIVE HEAD.** Implement **Probe v1 / Tier 2a** — P1 aug-safety **only** (P3-lite deferred: multi-seed shows GameOfLife stable +0.91±0.50, its single-seed regression was noise). Asymmetric keep-aug fail-safe; only seed-stable target is the Language-type aug-regression (aug-off costs −9.6 on AddNIST). Validate against the multi-seed aug-oracle (already have it). New dir `baselines/tier2a_probe_v1/`. **Design Tier 2 so search emits an ensemble-able pool, not a single argmax arch** (E1's residual-structural-variance finding). Carry the E1 member-quality + slack gate as a [[Tier 3 - EGNAS + Reliability]] design note.
5. **E3** architecture×dataset gold grid (thesis validation harness — proves/kills the probe approach).
6. Non-ML prerequisites (long lead): email organisers re Sudoku placeholder + **Myofibre unconfirmed benchmark** + broken Cryptic/Windspeed DOIs; x86 cross-verify with eval-pinned env ([[Threats to Validity]] §1).
7. E2 (non-conv primitive — Deep-Sets head first for CIFARTile).

## "Questions we want answered"

Not duplicated here — they live in [[Probe-Driven Self-Configuration]] §"Questions we want answered" (8-item priority list). Q2 (variance scale) **answered** ([[Multi-Seed - Results]]); Q8 (search variance-amplifying?) **answered** (Step 0: instability universal, *not* the discriminating cause — `dev/analyze_search_stability.py`); Q4 (does ensembling convert slack→robust gain?) **answered** ([[E1 - Results]]: partially — yes for AddNIST-class, no at submission level; residual variance is structural); E3 answers Q3/Q5; partial Q1/Q6 pending Tier 2a.

## Git state

- `main` — pristine.
- `baseline_approaches` (pushed) — Tier 0+1 impl, results notebase, thesis adoption + NAS survey. 2 commits.
- `exploration/probe-self-config` (this branch) — thesis P1 redesign + E1/E2/E3 + arch menu + questions, `BASELINE_SEED` env var, `dev/run_multiseed.sh`, handoff doc.
  - **Commit `c69e9d9` (2026-05-16) — multi-seed results: LOCAL ONLY, NOT pushed.** `dev/aggregate_multiseed.py`, `--multiseed` plot mode, `Multi-Seed - Results.md`, `multiseed.png`, Tier 1/Overview/thesis fold-in, `dev/logs/multiseed/_aggregate.json` force-added (raw `dev/logs/` gitignored, lives on Spark).
  - **UNCOMMITTED E1 batch** (commit when the validation sweep lands, with the results): `baselines/tier1e1_ensemble/` (new), `dev/run_e1_validation.sh` (new), `dev/analyze_search_stability.py` (new), and this handoff update. `.obsidian/workspace.json` is local UI noise — never stage it. Commit-msg rule: no AI/Claude/Co-Authored-By attribution.
