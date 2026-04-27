# Strategy + Workflow dossier (sections 06 + 07)

> Owner: `strategy-workflow`. Notes filled: 5 in `06 - Strategy/`, 4 in `07 - Workflow/`. Frontmatter pattern preserved across all 9.

## Key facts established

- **Reliability dominates peak by ~2× leverage**: −10 floor and +10 cap make a single failure cost as much as two perfect runs. A reliable mediocre submission (s=+3, p=0.99) beats a fragile peak submission (s=+9, p=0.70). Crossover at p ≈ 0.95.
- **Default budget split is 30 % NAS / 65 % Trainer / 5 % buffer** per dataset. Justified by [[EGNAS]]'s 50-epoch retrain default and the per-dataset clock semantics.
- **Per-dataset regimes** keyed off `metadata['benchmark']`: Regime A (b ≥ 85) 15/80, Regime B (40 ≤ b < 70, NAS-rewarding) 50/45, Regime C (b < 40, noise floor) 10/85, Regime D (default) 30/65.
- **Slope formula `10 / (100 − benchmark)`** — points per 1 % raw gain. High-benchmark datasets reward more. Built a per-dataset slope table from the Geada Table 1 numbers in [[Papers]].
- **CIFARTile is the single biggest NAS-vs-CNN gap** in the public 13 (Geada: 51 % CNN vs 92 % PC-DARTS = +41 %). Justifies Regime B existence.
- **Gutenberg and Windspeed are noise-floor traps**: shallow slope, near-zero achievable headroom. Strategy is "don't optimise, just clear benchmark".
- **Phase 2 cadence**: smoke-test (attempt 1), calibration (2), real attempts (3–4), push (5), reserves (6–7). Last working submission auto-promotes to Phase 3, so attempt 7 should be progressively *less* aggressive, not more.
- **Mock harness mirror is faithful**: `dev/mock_harness.py` matches `evaluation/main.py` in import shape, Clock class, test loader assertions, and try/except wrapper. Local pass implies high probability of grader pass.
- **Tiered testing matrix** documented (Tier 1 integration, Tier 2 stress-clock, Tier 3 low-VRAM, Tier 4 randomness, Tier 5 zip-package verify). Tier 5 is non-negotiable before each Phase-2 attempt.
- **Submission packaging**: `cd submission && zip -r ../submission.zip . -x '*.pyc' '__pycache__/*' '.*'`. Files at top level of archive (no parent folder); `main.py`/`score.py` excluded since the grader overwrites them.

## Wikilinks used (cross-section, need to resolve)

From section 01 — Competition: [[Phases and Timeline]], [[Scoring]], [[Rules]], [[Overview]].
From section 02 — Framework: [[Submission Contract]], [[Pipeline Flow]], [[Clock and Time Budget]], [[Failure Modes]], [[Mock Harness]].
From section 03 — Datasets: [[CIFARTile]], [[Gutenberg]], [[GameOfLife]], [[Cryptic]], [[Windspeed]].
From section 04 — Approaches: [[EGNAS]], [[Predictor-Guided Evolutionary]], [[DARTS Family]].
From section 05 — Past Winners: [[2025 - EVAI]], [[2025 - Sichuan teams]].
From section 08 — References: [[Papers]].
Also referenced as plain term once: [[NAS]], [[Trainer]], [[DataProcessor]] — these are *class* names from the contract, not note basenames. Confirmed not in the inventory; rendered as wikilinks anyway since several stubs already use that shorthand. **Flag for consolidator**: replace with plain text or create stubs if the team prefers.

## Open questions

- **Submission size limit**: organisers haven't published a hard number. Working assumption: <200 MB safe, >500 MB risky. Worth asking on the official channel before [[Phase 2 Submission Strategy|Phase 2]].
- **Phase 3 wall-clock**: officially "unknown amount of runtime". Strategy notes assume tighter-than-Phase-2 to be safe, but a confirmed value would let us tune the [[Budget Allocation]] defaults.
- **Hidden Phase 2 dataset shapes**: by design unknown, but if any historical hint exists from the organisers (input-shape ranges, channel counts), it would calibrate the Regime D default.
- **Does the eval environment have `sklearn`?** The starter kit pins the deprecated `sklearn` package name — `mock_harness.py`'s `--score` mode imports `sklearn.metrics.accuracy_score`. If the eval env doesn't have it, our Trainer must not import `sklearn` either.

## Conflicts with PRIOR_FINDINGS.md or sibling sections

- **No conflicts identified** with `PRIOR_FINDINGS.md`. The strategy notes consume its facts (slope table, 2025 leaderboard, public-13 benchmarks) verbatim.
- **Potential overlap with section 04** (Approaches) on EGNAS budget defaults: the [[Budget Allocation]] note cites EGNAS's 50-epoch retrain, while [[EGNAS]] (section 04, owned by `approaches-researcher`) covers the same fact. Resolution: section 06 wikilinks to [[EGNAS]] and only states the budget implication, not the algorithmic detail. No duplication.
- **Potential overlap with section 02** (Framework) on failure modes: [[Reliability vs Peak]] enumerates failure outcomes briefly, but defers the full taxonomy to [[Failure Modes]]. Verified the note text just summarises and links.

## Notes for the consolidator

- Two of the strategy notes (`Reliability vs Peak`, `Score-Efficient Targeting`) include numerical tables computed from `PRIOR_FINDINGS.md`'s data. If `winners-references` updates the 2025 leaderboard or benchmark figures, these tables should be re-checked.
- [[NAS]], [[Trainer]], [[DataProcessor]] used as wikilinks in a few notes (typical Obsidian shorthand for the contract classes). These do **not** correspond to notes in `SECTION_OWNERSHIP.md`. Either (a) treat as plain prose, (b) point them at [[Submission Contract]], or (c) create one-line stubs. Recommend (b).
- The `Gutenberg` benchmark in `PRIOR_FINDINGS.md` table 7 says ResNet-18 = 50.0 %, AlexNet = 50.0 %, but the Geada CVPR Table 1 (verified via PDF read) shows ResNet-18 = 40.98 %, AlexNet = 45.53 %. **Conflict**. The notes here cited the Geada Table 1 values explicitly. Flagging for `datasets-researcher` and the consolidator.
- All 9 stubs are in `drafting` status — flip to `current` after consolidation review.
