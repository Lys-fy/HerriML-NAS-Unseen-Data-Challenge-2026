> **Status:** drafting | **Topic:** Phase 1, 2, 3 — what happens in each, key dates.
> **Related:** [[Competition Overview]], [[Submission Contract]], [[Phase 2 Submission Strategy]]

Three phases. Phase 1 is local development. Phase 2 is timed runs on the organisers' servers against hidden datasets. Phase 3 is **not a separate submission** — it re-runs the last working Phase-2 entry with a (longer, unspecified) wall-clock budget.

## Timeline (2026 edition)

| Phase | When | Where | Datasets | Submission cap | Wall-clock per dataset |
|-------|------|-------|----------|----------------|-----------------------|
| Phase 1 | Open now (April 2026) | Your machines | 13 public (see [[Datasets Overview]]) | Unlimited | Whatever you set locally — use [[Mock Harness]] |
| Phase 2 | **July 1 – July 31** | Organisers' servers | Hidden | **≤7** | ~1 hour |
| Phase 3 | **August 1+** | Organisers' servers | Same hidden datasets as Phase 2 | 0 (auto-promoted) | Unspecified (longer than Phase 2) |

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
