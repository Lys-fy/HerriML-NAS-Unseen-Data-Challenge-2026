# Run State + Handoff

> **Status:** live operational state (2026-05-16) | **Topic:** Everything needed to resume work after a context reset — Spark state, experiment results, how to pull/aggregate/plot/notate results, the next-action queue. **Multi-seed sweep COMPLETE → [[Multi-Seed - Results]] is the authoritative finding.**
> **Related:** [[Probe-Driven Self-Configuration]] (plan + "Questions we want answered"), [[Tier 1 - Results]], [[Implementation Results Overview]]

This is the **resume document**. A fresh session should: (1) read [[Probe-Driven Self-Configuration]] for the thesis/plan/open-questions, (2) read this for operational state, (3) `ssh spark` and run the live-status command below — never trust a stale snapshot, always re-check live.

## Spark (the only place torch runs — local box has no GPU)

- SSH host alias: `spark` (key `~/.ssh/id_ed25519_dgx`, user `lysfy`, NVIDIA GB10).
- Repo on Spark: `~/projects/HerriML-NAS-Unseen-Data-Challenge-2026/`
- Datasets: `~/local/nas-comp-datasets/` (symlinked as `datasets/`)
- venv: `.venv/bin/python` (torch 2.11.0+cu128). Dev loop = rsync local→spark, run via ssh, results rsync back.

## ✅ Multi-seed sweep — COMPLETE (2026-05-15 13:11 → 2026-05-16 15:23 CEST, ~26 h)

All 54 runs landed, pulled + aggregated + notated → **[[Multi-Seed - Results]]** is the authoritative finding. Headline: Tier 1 recipe = **+11.3 ± 8.2 shipped / +8.4 ± 8.4 internal** over 3 seeds (seed variance ≈ the whole signal; the old single-seed +24.644 was a tail draw). **Spark is now free for the next run.**

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

1. ~~Aggregate + variance finding + Multi-Seed note + fold caveats~~ **✅ DONE 2026-05-16** — [[Multi-Seed - Results]] written; Tier 1 caveat + thesis methodology + Overview folded in; committed.
2. **← ACTIVE HEAD OF QUEUE.** Implement **Probe v1 / Tier 2a** (P1 aug-safety + P3-lite reg-strength) per [[Probe-Driven Self-Configuration]] — dynamic racing stop on trajectory-shape, asymmetric fail-safe (keep-aug default). New baseline dir `baselines/tier2a_probe_v1/`. The multi-seed result sharpens the spec: P1's only seed-stable target is the Language-type aug-regression; aug-off costs −9.6 on AddNIST so the bar to disable aug must be high.
3. **E3** architecture×dataset gold grid (the thesis validation harness — highest strategic priority; proves/kills the probe approach).
4. Non-ML prerequisites (long lead time, start now): email organisers re Sudoku placeholder benchmark + broken Cryptic/Windspeed DOIs; x86 cross-verify with eval-pinned env ([[Threats to Validity]] §1, `docs/environment.md`).
5. E1 (budget-filling diverse-restart ensemble), E2 (non-conv primitive — Deep-Sets head first for CIFARTile).

## "Questions we want answered"

Not duplicated here — they live in [[Probe-Driven Self-Configuration]] §"Questions we want answered" (8-item priority list). The multi-seed run answers Q2 (variance scale) and partially Q1/Q6; E3 answers Q3/Q5; E1 answers Q4.

## Git state

- `main` — pristine.
- `baseline_approaches` (pushed) — Tier 0+1 impl, results notebase, thesis adoption + NAS survey. 2 commits.
- `exploration/probe-self-config` (this branch) — thesis P1 redesign + E1/E2/E3 + arch menu + questions, `BASELINE_SEED` env var, `dev/run_multiseed.sh`, handoff doc. **+ multi-seed results commit (2026-05-16):** `dev/aggregate_multiseed.py`, `--multiseed` plot mode, `Multi-Seed - Results.md`, `multiseed.png`, the Tier 1 / Overview / thesis variance fold-in, and `dev/logs/multiseed/_aggregate.json` force-added (the small distilled artifact — raw `dev/logs/` stays gitignored and lives on Spark; rerun `rsync … && python3 dev/aggregate_multiseed.py` to regenerate).
