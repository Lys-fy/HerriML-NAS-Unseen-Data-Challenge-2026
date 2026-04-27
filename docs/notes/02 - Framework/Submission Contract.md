> **Status:** drafting | **Topic:** Fixed file names, class signatures, asserted constraints.
> **Related:** [[Pipeline Flow]], [[Failure Modes]], [[Mock Harness]]

The grader (`evaluation/main.py` from the [starter kit](https://github.com/Towers-D/NAS-Comp-Starter-Kit)) imports our submission **by exact filename and class name**. Any deviation breaks the import and scores `-10` on every dataset.

## File layout

The submission is a flat folder with **exactly four files**:

```
submission/
  data_processor.py    # exposes class DataProcessor
  nas.py               # exposes class NAS
  trainer.py           # exposes class Trainer
  helpers.py           # any shared utilities, must be importable as `helpers`
```

`main.py` and `score.py` are **overwritten by the grader** and must not be in the package — see [[Submission Packaging]] in section 07. Any extra modules must be siblings of the four files (not in subfolders) so the grader's `sys.path` picks them up.

## Class signatures (hardcoded in `evaluation/main.py`)

```python
class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock): ...
    def process(self) -> tuple[DataLoader, DataLoader, DataLoader]: ...
        # returns (train_loader, valid_loader, test_loader)

class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock): ...
    def search(self) -> torch.nn.Module: ...

class Trainer:
    def __init__(self, model, device, train_loader, valid_loader, metadata, clock): ...
    def train(self) -> torch.nn.Module: ...
    def predict(self, test_loader) -> list[int]: ...
```

The grader passes the **same `clock` object** through all three classes. Re-use it; do not construct a new clock. See [[Clock and Time Budget]].

## `metadata` dict — what's inside

Set by the organisers per dataset, augmented at each stage:

| Key | Type | Notes |
|-----|------|-------|
| `num_classes` | int | Required for the final classifier head |
| `codename` | str | Stable per dataset; used for prediction filenames |
| `input_shape` | tuple | `(C, H, W)` after any of *our* `DataProcessor.process()` reshaping. The raw `train_x` may not match this until processed |
| `benchmark` | float | Public CNN-average baseline, in 0–100 percent. Drives the [[Scoring]] formula |
| `time_limit` | float | Hours, default 0.5. Per-dataset wall-clock budget |
| `time_remaining` | float | Set by the grader before each stage. Also re-readable as `clock.check()` |

Datasets are **4D image tensors** `(N, C, H, W)`. Even datasets that started as 1D (e.g. [[Voxel]]) are reshaped to a 4D image-shaped representation upstream.

## Asserted constraints

The grader asserts the following on the **test loader** returned by `DataProcessor.process()`:

```python
assert not isinstance(test_loader.sampler, RandomSampler)   # no shuffling
assert not test_loader.drop_last                            # no dropped batches
```

Violation crashes the entire submission for that dataset. Effects:

- **Order is load-bearing.** The test labels live on the grader side, indexed by the same fixed order. A shuffled or partial test loader produces predictions in the wrong order — even if accuracy was high pre-shuffle, scoring against the held-out `test_y` will be near-random.
- The train / valid loaders may shuffle and drop_last as you wish.

## Failure model the contract assumes

- **Any uncaught exception** in any stage is caught by the grader's outer `try/except`, the dataset is marked failed, and we score `-10` for that dataset (see [[Failure Modes]]).
- Internal `try/except` is therefore *defensive padding only* — it does not save us from a `-10` if recovery is impossible. Use it only to recover into a working fallback (e.g. ditching NAS and returning a known-good baseline architecture).

## Things often forgotten

- `helpers.py` must exist even if empty. The starter kit imports from it.
- The 4 files must work with the **starter kit's `requirements.txt`** in spirit, but that file is stale (pins `torch==1.10.1+cu113`). The actual eval environment is unspecified — write code against modern, widely-supported PyTorch idioms and avoid version-locked APIs. See [[Threats to Validity]].
- Do not rely on internet access at runtime. Bundle any pretrained weights as files inside the submission package.
