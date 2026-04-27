> **Status:** drafting | **Topic:** How to zip submission/, what to exclude, packaging gotchas.
> **Related:** [[Submission Contract]], [[Rules]], [[Pipeline Flow]], [[Dev Loop]], [[Testing Strategy]]

The grader unzips our submission, copies its own `main.py` and `score.py` over the top, and runs `evaluation/main.py`. Anything broken about the package shape becomes −10 across all datasets — a catastrophic single-mistake mode (see [[Reliability vs Peak]]).

## The exact zip command

```bash
cd submission && zip -r ../submission.zip . -x '*.pyc' '__pycache__/*' '.*'
```

That produces `submission.zip` at the repo root with the four required files at the **top level** of the archive (no parent folder). The grader's importer expects `from nas import NAS`, etc. — see [[Pipeline Flow]] for the import shape.

Verify before uploading:

```bash
unzip -l submission.zip
# Should list, at the top level:
#   data_processor.py
#   nas.py
#   trainer.py
#   helpers.py
# (plus any extra modules you added under submission/)
```

If you see `submission/data_processor.py` instead of `data_processor.py`, the zip will fail to import on the grader. The `cd submission && zip ../submission.zip .` form avoids this.

## Required files (the [[Submission Contract]])

| File | Class | Purpose |
|---|---|---|
| `data_processor.py` | `DataProcessor` | Returns `(train_loader, valid_loader, test_loader)` |
| `nas.py` | `NAS` | Returns a PyTorch model from `.search()` |
| `trainer.py` | `Trainer` | `.train()` returns trained model, `.predict()` returns list of ints |
| `helpers.py` | (any) | Free for utility code |

Filenames and class names are **hardcoded** in the grader's `evaluation/main.py` — see [[Submission Contract]]. Don't rename, don't refactor into packages, don't move classes to other modules without re-exporting.

## What to exclude

- **`__pycache__/`, `*.pyc`** — pollutes the archive, may shadow imports.
- **`.DS_Store`, `Thumbs.db`** — OS metadata.
- **Hidden files (`.*`)** — including any `.env` or local config.
- **`main.py`, `score.py`** — these get overwritten by the grader, so shipping them is wasted bytes (and confusing if a teammate reads the zip).
- **Any dataset files, model checkpoints, or pretrained weights >100 MB** — the eval environment may not allow large packages, and we want to fit comfortably below any size limit (organisers haven't published one explicitly — verify before each [[Phase 2 Submission Strategy|Phase 2 attempt]]).

## Forbidden contents

- **No internet calls.** The eval environment is presumed to be sandboxed; any `urllib`/`requests` call to download weights or data times out and likely crashes. See [[Failure Modes]].
- **No download attempts of the final-stage datasets.** Disqualifying per [[Rules]].
- **No reading from absolute filesystem paths.** Use only paths derived from the function arguments — anything else won't exist on the grader's box.
- **No imports outside the starter-kit pin set.** The starter kit's `requirements.txt` is the floor; the actual eval pin is unspecified but presumed to be a similar PyTorch environment. Adding `lightning`, `timm`, `transformers`, etc. is a risk — see [[Threats to Validity]].

## Pretrained weights

If we want to ship pretrained weights (e.g. a small ImageNet-warmstart for [[Submission Contract]]), they must be:

1. Inside the zip (the eval env has no internet).
2. Loaded via a path **relative to `__file__`** — the grader unzips us into an unknown location:

```python
import os
WEIGHTS = os.path.join(os.path.dirname(__file__), 'pretrained.pth')
state_dict = torch.load(WEIGHTS, map_location='cpu')
```

Beware size: a single ResNet-50 checkpoint is ~100 MB. Multiple checkpoints inflate the archive fast.

## Verifying the package

Before uploading, run [[Mock Harness]] one more time **with the unzipped archive as `submission/`**:

```bash
# In a fresh tmp dir
cd /tmp && mkdir verify && cd verify
unzip ~/repo/submission.zip -d submission/
# Then symlink or override SUBMISSION_DIR
DATASETS_DIR=~/local/nas-comp-datasets python ~/repo/dev/mock_harness.py Sudoku --time-limit 0.05
```

This catches packaging bugs (missing files, wrong layout) that a normal in-tree run masks because `submission/` is already on the path. See [[Testing Strategy]] for a checklist.

## Submission-size norms

The organisers have not published a hard limit. Historical hints from prior NAS competitions: assume **<200 MB** is safe, treat anything over **500 MB** as risky. If the archive is above 200 MB, audit:

- Are there pretrained weights we could prune or quantise?
- Are there checkpoints accidentally included?
- Is there a `.git/` directory inside `submission/` (`zip -x '.*'` excludes this)?

Smaller archives also upload faster, which matters during the [[Phase 2 Submission Strategy|7-attempt window]] when network hiccups can eat a full attempt.
