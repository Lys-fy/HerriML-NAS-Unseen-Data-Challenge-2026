# Section dossier — 01 Competition + 02 Framework

> Owner: `competition-framework`. Covers `docs/notes/01 - Competition/` (4 notes) and `docs/notes/02 - Framework/` (5 notes).

## Key facts established (beyond `PRIOR_FINDINGS.md`)

- **`Clock` is `time.perf_counter`-based, monotonic.** Verified by reading `dev/mock_harness.py` (mirror of `evaluation/main.py`). Means our inner-loop time checks are unaffected by NTP drift / system clock changes — we can rely on `clock.check()` for fine-grained training-loop decisions.
- **Pre-stage check fires before each of `process / search / train / predict`**, but the grader does **not** interrupt *inside* a stage. Documented this explicitly in [[Pipeline Flow]] and [[Failure Modes]] — it's the easiest gotcha (a long DataProcessor cannot be aborted; it just leaves no time for the rest).
- **The grace window is exactly 1 minute and only between `train()` and `predict()`.** All other stage transitions are non-grace. Recorded in [[Clock and Time Budget]].
- **`device = cuda if available else cpu` is selected by the grader between `NAS.search()` and `Trainer.__init__`.** Therefore `NAS.search` runs without an explicit device parameter — anything we put on a device inside `search()` may need to be moved before being handed to `Trainer`. Captured in [[Pipeline Flow]].
- **`metadata["time_remaining"]` is refreshed by the grader before each stage** but inner loops should use `clock.check()` directly (the metadata value is stale during a long stage).
- **Failure semantics are independent across datasets.** A `-10` on dataset A does not propagate. Captured in [[Failure Modes]].
- **Slope-by-benchmark table** (0.2 / 0.33 / 1.0 / 2.0 pts per +1% raw at benchmarks 50 / 70 / 90 / 95) reproduced in [[Scoring]] with the practical reading: low-benchmark datasets (Gutenberg-style) give near-zero upside per minute; high-benchmark datasets reward squeezing the last percent.
- **The "last working Phase-2 entry is auto-promoted to Phase 3"** rule — re-emphasised in [[Phases and Timeline]]. Implication: Phase 2 submission #7 is your **last chance** to land a stable entry, not your best chance to push improvements.

## Wikilinks used pointing to other sections

Verified against `SECTION_OWNERSHIP.md`. All targets exist:

- 03 - Datasets: `[[Overview]]` (section-03 file), `[[AddNIST]]`, `[[MultNIST]]`, `[[Voxel]]`, `[[GeoClassing]]`, `[[CIFARTile]]`, `[[Chesseract]]`, `[[Gutenberg]]`, `[[Cryptic]]`, `[[Windspeed]]`.
- 04 - Approaches: `[[EGNAS]]`, `[[Approaches]]` *(NOTE: there is no `Approaches.md` — only `04 - Approaches/Overview.md`. Used `[[EGNAS]]` and bare-text "approaches" elsewhere; flagging in conflicts below.)*
- 05 - Past Winners: `[[2025 - EVAI]]`.
- 06 - Strategy: `[[Reliability vs Peak]]`, `[[Score-Efficient Targeting]]`, `[[Budget Allocation]]`, `[[Search vs Train Split]]`, `[[Threats to Validity]]`.
- 07 - Workflow: `[[Submission Packaging]]`, `[[Phase 2 Submission Strategy]]`, `[[Dev Loop]]`, `[[Testing Strategy]]`.
- 08 - References: `[[Papers]]`, `[[Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf]]` *(this is a PDF filename, not a note name — flagging below).*

## Conflicts noticed / wikilinks to fix

- **`[[Approaches]]`** — I referenced this in [[Pipeline Flow]] ("highly approach-dependent ... See [[Approaches]]"). There is no `Approaches.md`; the section-04 index is `Overview.md`, which collides with `01 - Competition/Overview.md` and `03 - Datasets/Overview.md` since wikilinks resolve by basename. The collision is real — Obsidian will pick one arbitrarily. **Suggest: consolidator either renames the three Overviews to be section-prefixed or section-04 owner adds an `Approaches.md` redirect.**
- **`[[Geada-2024_Insights-Unseen-NAS-Datasets_CVPR.pdf]]`** — referenced in [[Submission Contract]] and [[Overview]]. This is a PDF in `docs/papers/`, not a note. Obsidian renders PDF wikilinks fine but if the team prefers note-only links, swap to `[[Papers]]` (section 08) which should list the PDF.
- **Three colliding `Overview.md` notes** (sections 01, 03, 04). Bare `[[Overview]]` is ambiguous. I used relative-disambiguating prose ("`[[Overview]]` (section-03 file)") in one place — not great. Consolidator should decide on a section-prefixed naming convention or accept the collision.
- **No conflicts with `PRIOR_FINDINGS.md`.** All numeric facts (timing, slope table, formula, leaderboard, contract) reproduce the prior findings verbatim. Where I added detail, it came from reading `dev/mock_harness.py` directly.

## Open questions

- **Is `helpers.py` actually required?** I asserted yes in [[Submission Contract]] ("must exist even if empty"). Source: starter kit imports from it. Verify against the current starter kit `evaluation/main.py` — the prior findings list it as one of the 4 files but don't confirm import-from-helpers.
- **What exactly is the Phase-3 wall-clock budget?** Prior findings say "unknown amount of runtime" — flagged as `unspecified` in [[Phases and Timeline]] and [[Clock and Time Budget]]. Worth emailing the organisers (`nas-competition-contact@newcastle.ac.uk`) once we have a stable Phase-2 entry.
- **Phase-2 datasets list.** Not in prior findings, not on the public site. Datasets used in 2025 (GameOfLife, Windspeed, Cryptic) are unlikely to be reused verbatim but may overlap. Worth asking the **datasets researcher** whether they have any signal on what the 2026 hidden set looks like.
- **Memory budget.** No specific RAM/VRAM number is published. Mentioned in [[Failure Modes]] and [[Rules]] but the prior-findings note 9 ("Eval-environment unknowns") confirms this is genuinely unspecified — not just missing from our notes.
- **Whether internal `try/except` is ever rewarded over a `-10`.** Discussed in [[Submission Contract]] and [[Failure Modes]]: it only helps if you have a known-good fallback. **Strategy section may want to debate whether to ship a hardcoded ResNet-18 baseline as a fallback inside `NAS.search`.**

## Coordination notes for siblings

- **datasets-researcher (#2):** I cite [[Cryptic]] and [[Windspeed]] as "broken DOIs" in [[Phases and Timeline]] and [[Mock Harness]]. Please confirm in those dataset notes; flag if there's been any update since the prior findings.
- **strategy-workflow (#5):** [[Clock and Time Budget]] establishes the in-stage time-check pattern; [[Pipeline Flow]] sketches the per-stage budget split (5 / 10–50 / 40–80 / <1 percent). [[Search vs Train Split]] and [[Budget Allocation]] should plug into these — happy to refine the split in [[Pipeline Flow]] after seeing your numbers.
- **approaches-researcher (#3):** [[Pipeline Flow]] pre-cites "predictor-based and zero-cost are minutes; full retrain-per-candidate is the budget" — please tighten in your method notes if those rough orders are wrong.
- **winners-references (#4):** I link `[[2025 - EVAI]]` from [[Overview]]. Please make sure that note exists with that exact basename.

## Deliverables status

- 4 / 4 section-01 stubs filled (`Overview`, `Phases and Timeline`, `Scoring`, `Rules`).
- 5 / 5 section-02 stubs filled (`Submission Contract`, `Pipeline Flow`, `Clock and Time Budget`, `Failure Modes`, `Mock Harness`).
- Frontmatter blockquote pattern preserved in all 9; status promoted from `stub` to `drafting`.
- No notes touched outside sections 01 and 02. No commits, no pushes.
