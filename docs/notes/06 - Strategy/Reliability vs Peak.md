> **Status:** drafting | **Topic:** Why avoiding -10 dominates pursuing +10. Expected-score math.
> **Related:** [[Failure Modes]], [[Scoring]], [[Clock and Time Budget]], [[Mock Harness]], [[Budget Allocation]], [[Threats to Validity]]

**The dominant strategic fact**: under the [[Scoring]] formula, the gap between a successful submission and a failed one is **+10 to −10 = 20 points**, while the gap between a mediocre success and a perfect success is at most **+10 points**. One failure costs as much as two perfect runs. Reliability is therefore worth roughly 2× the marginal effort of pushing peak accuracy.

## Expected-score math

Let `p` be the probability our submission survives a given dataset (no timeout, no OOM, no crash, no shape-assertion violation — see [[Failure Modes]]). Let `s_success` be the conditional adjusted score given success. Then the expected per-dataset contribution is:

```
E[score] = p * s_success + (1 - p) * (-10)
        = p * (s_success + 10) - 10
```

The leverage of `p` is `(s_success + 10)`. Even a barely-positive submission (`s_success = 0`) earns `+10p − 10` — at `p = 0.9` that's `−1`, at `p = 0.5` it's `−5`. **A reliable mediocre solution beats a fragile excellent one** until `p` crosses ~0.95. Concretely:

| Strategy | s_success | p | E[score] |
|----------|----------:|---:|---------:|
| Fragile peak | +9.0 | 0.70 | +0.30 |
| Reliable mediocre | +3.0 | 0.99 | +2.87 |
| Reliable strong | +6.0 | 0.99 | +5.84 |

A fragile +9 submission is worse than a reliable +3 submission. This is the single most important table on the team.

## Why the cliff is so steep

- The −10 floor is symmetric with the +10 cap, so any failure erases an entire perfect dataset elsewhere. In 2025, **EVAI's +9.998 on GameOfLife** would have been wiped by a single failure on Cryptic — they would have lost the competition (see [[2025 - EVAI]]).
- The grader catches every exception in `run_submission` and converts it to −10 (see [[Mock Harness]] for the exact handler). There is no partial credit, no graceful degradation hook.
- The [[Clock and Time Budget]] is per-dataset and independent — going over by one second on dataset A doesn't hurt B, but it _does_ cost the entire 20-point swing on A.

## Implications for design

1. **Always ship a fallback model.** If [[Submission Contract]] fails or times out, the [[Submission Contract]] should still receive *something* trainable — a small ResNet-style baseline, hardcoded. The 2024 paper's ResNet-18 column (see Geada Table 1) shows ResNet-18 beats benchmark on most datasets out of the box.
2. **Budget defensively, not optimally.** Reserve ~5 % of the per-dataset clock as buffer for prediction + I/O + grader overhead. See [[Budget Allocation]] for ratios.
3. **Use grace, not bravery, on the clock.** The mock harness logs a 1-minute grace window after `train()` but still calls it a fail at the contract level. Treat any deadline as hard.
4. **Test the failure paths first.** Stress-test with `--time-limit 0.05` (3 minutes) before optimising for accuracy. A submission that gracefully short-circuits under tight clocks is more valuable than one that wins under generous clocks. See [[Testing Strategy]].
5. **Pre-validate every submission.** Run [[Dev Loop]] across all 11 public datasets before any [[Phase 2 Submission Strategy|Phase 2 attempt]]. The 7-attempt cap makes each submission expensive — a silent crash on dataset 4 of 13 is catastrophic.

## What to optimise after reliability is solved

Only after `p ≥ 0.95` across all known dataset shapes does pushing `s_success` start to dominate. At that point, the slope formula in [[Score-Efficient Targeting]] tells us *which* datasets to push — not all peak gains are worth equal effort. See [[Search vs Train Split]] for how we then re-allocate compute.

The principle reduces to: **the first nine points come from not crashing**. Spend the development budget accordingly.
