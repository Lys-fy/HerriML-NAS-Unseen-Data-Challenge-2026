> **Status:** drafting | **Topic:** How to spend the 7 Phase-2 submission attempts.
> **Related:** [[Phases and Timeline]], [[Rules]], [[Reliability vs Peak]], [[Threats to Validity]], [[Testing Strategy]], [[Submission Packaging]], [[Score-Efficient Targeting]]

**Hard constraint**: 7 submissions in Phase 2 (1–31 July). The **last working** Phase-2 submission auto-promotes to Phase 3 — there is no separate Phase-3 push (see [[Phases and Timeline]] / [[Rules]]). This shapes the cadence: the *last* submission is the one that gets scored, so the strategy is "make sure attempt #7 (or earlier if necessary) is our best confirmed working version".

## Recommended cadence

| Attempt | Purpose | Timing (rough) | Risk tolerance |
|--:|---|---|---|
| 1 | **Smoke test.** Confirm the package imports, runs, and produces *any* score. Use the most defensive submission we have (reliable mediocre — see [[Reliability vs Peak]]). | Day 1–2 | Very low |
| 2 | **Calibration.** Adjust based on attempt-1 leaderboard signal. Tune [[Budget Allocation\|budget allocation]] per known dataset codename. | Day 4–7 | Low |
| 3 | **First real attempt.** Full pipeline including [[Submission Contract]] doing actual search. | Day 10–14 | Medium |
| 4 | **Iteration on attempt-3.** Fix anything that scored worse than expected. | Day 17–20 | Medium |
| 5 | **Push attempt.** Deploy the most ambitious working variant. Use only if attempts 1–4 are stable. | Day 22–25 | Higher |
| 6 | **Recovery slot (reserve).** Ship only if attempt 5 went *backwards*. Otherwise hold. | Day 27–28 | Match the best earlier attempt |
| 7 | **Final reserve.** Conservative re-submit of best-known-working code. Anchors Phase 3. | Day 30 | Lowest possible |

Three principles encoded above:

1. **The first attempt is a smoke test, not a competition entry.** A −30 result on attempt 1 (failure on every dataset due to a packaging bug — see [[Submission Packaging]]) is a one-attempt loss; same bug on attempt 7 forfeits the competition.
2. **The last working submission is what counts.** Attempts 5–7 should be progressively *less* aggressive, not more. End on a confirmed-good baseline; don't let the timer run out on a broken final push.
3. **Hold reserves.** Attempts 6 and 7 are insurance. Don't burn them in week 2.

## What gates each submission

Before any attempt, run the full [[Testing Strategy]] matrix:

- Tier 1 — full sweep on 11 public datasets, score logged.
- Tier 2 — stress tests at `--time-limit 0.05`.
- Tier 3 — low-VRAM / CPU-only path.
- Tier 5 — zipped-archive verification.

If any tier fails, the attempt is not allowed to ship. Period. The 7-attempt cap means each wasted attempt is ~14 % of our budget.

## Per-attempt checklist

```
[ ] Tier 1 score is non-negative AND >= last-shipped attempt's local score
[ ] Tier 2 (3-min clock) passes 3/3 datasets without timeout
[ ] Tier 3 CPU path completes (proves no CUDA-only assumptions)
[ ] Tier 5 zipped archive imports cleanly in fresh tmp dir
[ ] Diff against organisers' current evaluation/main.py — no contract drift
[ ] Submission archive is < 200 MB
[ ] git tag this submission (e.g. phase2-attempt-3) before uploading
[ ] After upload: record the leaderboard score per dataset in dev/logs/phase2-results.md
```

The git-tag step matters: if attempt 5 is the best, attempt 6 needs to be a strict superset of attempt 5's code, not a divergent branch. We need to be able to bisect across attempts.

## Reading the Phase 2 leaderboard

Each attempt produces per-dataset adjusted scores. Key signals:

- **A −10 anywhere** → packaging or contract bug. Halt attempts, fix, re-test the full Tier 1–5 matrix before next attempt.
- **A score lower than ResNet-18 benchmark** ([[Score-Efficient Targeting]]) → the model didn't even train. Likely a Trainer bug or budget collapse.
- **Cliff between two similar datasets** (e.g. AddNIST ok, MultNIST −10) → shape-specific bug. Inspect input handling.
- **Consistent ~+0 across all datasets** → submission is clearing benchmark but not learning. Something wrong with [[Search vs Train Split]] or training schedule.

After each attempt, write a short post-mortem in `dev/logs/phase2-attempt-N.md` covering: what we expected, what we got, hypothesis for any deltas, what we'll change next.

## Don't do this

- **Don't submit on the deadline.** Network outages, server queues. Aim for ≥24 h before midnight 31 July.
- **Don't submit a code change that wasn't tested in [[Mock Harness]] within the last 24 h.** Stale binaries / accidentally edited file = −30.
- **Don't optimise an attempt-7 submission for theoretical peak.** [[Reliability vs Peak]]: the reproducibility risk on hidden datasets is already the dominant variance source. Submit the most consistent good variant, not the highest-ever-observed.
- **Don't rely on a Phase-2-submitted-but-not-yet-graded result.** Server-side scoring may be delayed. Plan as if attempt N's leaderboard appears the day before attempt N+1's deadline.

## Phase 3 transition

The last working Phase 2 submission auto-runs in Phase 3 on the same hidden datasets. **No code edits between phases.** This means:

- Attempt 7 (or the last working one) **is** our Phase 3 entry.
- We will not get a chance to fix anything between Phase 2 and Phase 3.
- The 1 h Phase 2 clock may not match Phase 3's "unknown amount of runtime" — code must be clock-adaptive ([[Clock and Time Budget]] / [[Threats to Validity]]).

The strategic implication: **conservative final attempt** beats a brilliant-but-fragile one. Phase 3 is when [[Reliability vs Peak]] cashes in.
