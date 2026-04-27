> **Status:** drafting | **Topic:** Submission limits, prohibited actions, contact, mailing list.
> **Related:** [[Phases and Timeline]], [[Submission Packaging]]

## Submission limits

| Limit | Value |
|-------|------:|
| Phase-2 submissions per team | **7** over the month (Jul 1 – Jul 31) |
| Submissions per Phase-3 promotion | 0 — the last working Phase-2 entry is **auto-promoted** |
| Teams per submitter | 1 (one submission per team) |

Multiple teams from the same institution are allowed **only if** their member sets are disjoint *and* their approaches are clearly distinct. The organisers reserve the right to ask for justification.

## Hard prohibitions

These score `-10` on the affected dataset (failure) or, for the last bullet, **disqualify the team**:

- **Time overruns.** Code must respect the `clock` object passed into every stage. See [[Clock and Time Budget]] and [[Failure Modes]]. The grader marks the dataset failed if `clock.check()` is consumed before predictions are written.
- **Memory overruns.** The submission must "be adaptable to the amount of memory available." Out-of-memory crashes from RAM or VRAM are caught by the grader's `try/except` and score `-10` (see [[Failure Modes]]).
- **Modifying the test loader.** The grader asserts `not isinstance(test_loader.sampler, RandomSampler)` and `not test_loader.drop_last`. A violation crashes the submission immediately. See [[Submission Contract]].
- **Including `main.py` or `score.py`** in your submission package. They are overwritten by the grader, but the upload should be the 4 expected files only — see [[Submission Packaging]].
- **Attempting to download or otherwise access the held-out final-stage datasets.** The organisers list this explicitly as **disqualifying**.

## Eligibility and presentation

- Finalists are expected to **present at AutoML 2026** (in person or virtually).
- Past finalists provide a short methods write-up; expect the same in 2026.

## Contact and mailing list

- Email: **`nas-competition-contact@newcastle.ac.uk`** (Newcastle University organising team).
- Site / signup: <https://www.nascompetition.com/>.
- Use the contact email for: clarification on dataset DOIs (e.g. the broken [[Cryptic]] / [[Windspeed]] links), submission-portal issues, and questions about Phase-3 wall-clock policy.

## Things that are *not* prohibited (and worth knowing)

- **Pre-trained weights.** Not banned by the rules, but the submission is invoked from a clean environment — anything you ship in the package must be loaded from a file you bundled, not downloaded at runtime (no internet access on the eval server).
- **Larger models that fit the wall-clock and memory.** The competition rewards what works, not minimality. There is no model-size cap or FLOPs cap announced.
- **Custom dataloaders, augmentation policies, ensembles.** All are allowed inside the 4-file [[Submission Contract]].

## Internal team rule reminder

Multiple teams from the same institution are allowed only with disjoint members and distinct approaches — relevant only if other University of Freiburg / project-area students enter independently. Coordinate before assuming we're the only entry from our institution.
