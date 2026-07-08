> **Status:** drafting | **Topic:** Phase 1, 2, 3 — what happens in each, key dates.
> **Related:** [[Competition Overview]], [[Submission Contract]], [[Phase 2 Submission Strategy]]

## ⭐ ORGANISER CONFIRMATIONS (David Towers, Discord, 2026-07-06/08 — OVERRIDES the sections below where they conflict)

1. **Time limit is PER DATASET** (not pooled). `TIME_LIMIT` env is deprecated; the limit comes from dataset metadata (`main.py` line 110). Rely on `metadata["time_remaining"]` / the clock — our design already complies (clock-fraction budgets everywhere).
2. **Phase 3 final evaluation: 24 hours total across THREE datasets, ~8 hours each.** Only 3 hidden finals (2025 precedent: GameOfLife, Wind, Cryptic). The ~8h/dataset operating point is exactly our long-tier test regime.
3. **Phase 2 (started July 1): 7 submissions, DUMMY data, short limit (30 min, possibly 1h), evaluated MANUALLY.** Purpose is only "does the code run in our environment".
4. **Submission is by EMAIL**: d.towers2@ncl.ac.uk or nas-competition-contact@newcastle.ac.uk (no portal).
5. **The LAST successful Phase-2 submission advances to Phase 3** — so the final email before July 31 must be the best package, and every earlier submission is a free smoke test.
6. Eval hardware (inferred, not confirmed): Newcastle's HPC is **Comet** (Rocket decommissioned 2025) — GPU nodes are **H100 96GB PCIe (4/node, limited/paid)** and more numerous **L40S** GPU-S nodes. So the grader most likely runs an H100 or L40S — both bracketed by our 2h hardware matrix (H100 measured directly; L40S ≈ A100-class for our small nets).

Three phases. Phase 1 is local development. Phase 2 is timed runs on the organisers' servers against hidden datasets. Phase 3 is **not a separate submission** — it re-runs the last working Phase-2 entry with a (longer, unspecified) wall-clock budget.

## Timeline (2026 edition)

| Phase | When | Where | Datasets | Submission cap | Wall-clock per dataset |
|-------|------|-------|----------|----------------|-----------------------|
| Phase 1 | Open now (April 2026) | Your machines | 13 public (see [[Datasets Overview]]) | Unlimited | Whatever you set locally — use [[Mock Harness]] |
| Phase 2 | **July 1 – July 31** | Organisers' servers | Hidden | **≤7** | **1 hour, explicitly "just testing the code works"** (a smoke test, not the scored run) |
| Phase 3 | **August 1+** | Organisers' servers | Same hidden datasets as Phase 2 | 0 (auto-promoted) | **"An unknown amount of runtime"** — the pipeline must adapt to whatever the clock says |

### Runtime rules confirmed from the 2026 starter kit (checked 2026-06-10, source: starter-kit README + `evaluation/main.py`)

- The per-dataset time limit now comes from the **metadata file** (`time_limit` key, in hours). The starter kit's code default is **30 minutes** per dataset; the old global `TIME_LIMIT = 12` constant is commented out. Consequence: the pipeline must behave sanely at ANY budget from 30 min to many hours, so every internal allocation (probes, search, training) must be a FRACTION of the clock, never an absolute number of minutes.
- **New in 2026: a hard termination watchdog**, separate from the clock object, kills the submission once it exceeds the limit. Keep the safety margin so predictions are written before the axe falls.
- The organisers run their own `main.py` and `score.py`, and **"any attempts to extend your time limit will result in disqualification."** Anything resembling clock manipulation (e.g. the v6 exploration-only `EXTEND_CLOCK_FOR_NAS` flag) must NEVER be in a shipped submission. Packaging checklist item.

## Phase 1 — Development

- Build and iterate locally against the 13 public datasets.
- Run [[Mock Harness]] (`dev/mock_harness.py`) to mirror how the grader will invoke your code.
- Any number of iterations. No scoring against the leaderboard.
- Two of the public datasets — [[Cryptic]] and [[Windspeed]] — are unreachable because the starter-kit DOIs are broken. Develop without them; coordinate with [[Datasets Overview]] researchers if/when fixed.

## Phase 2 — "Validation" / smoke test

- Submit the 4-file package (see [[Submission Contract]]) via the organisers' upload portal.
- Each submission is run on **all hidden Phase-2 datasets** with **~1 hour per dataset**. Concretely: a per-dataset [[Clock and Time Budget]] of 1 h.
- **Hard cap of 7 submissions over the whole month.** Treat this as your debugging budget. See [[Phase 2 Submission Strategy]] for how to allocate them.
- Crashes, timeouts, OOM and shape violations on a dataset score `-10` for that dataset (see [[Failure Modes]]).
- The organisers track which entry is your **last working submission**. That entry — not your highest-scoring entry, not your most recent entry — is what advances.

## Phase 3 — Final scored run

- Begins **August 1**.
- The last working Phase-2 submission is **auto-promoted**. No new upload, no new code.
- Same hidden datasets as Phase 2.
- **Wall-clock per dataset is unspecified** ("an unknown amount of runtime"). The pipeline must read `clock.check()` and adapt — do not hard-code a budget.
- This is the run that determines final standings.

## Implications for our schedule

- Phase 1 ends *implicitly* when Phase 2 opens July 1. Aim to have a stable, reliability-first submission by **late June 2026** so the first Phase-2 run is not the first time anyone has executed our code on a fresh machine.
- Reserve at least 2 of the 7 Phase-2 submissions for **late-month bugfixes**. The temptation to push improvements early is the easiest way to land on submission #7 with a regression.
- Anything that improves Phase-3-only behaviour (e.g. exploits longer wall-clock) needs to already be in the Phase-2 pipeline — there is no Phase-3 push. See [[Search vs Train Split]].

Finalists are invited to **present at AutoML 2026** (in person or virtually).
