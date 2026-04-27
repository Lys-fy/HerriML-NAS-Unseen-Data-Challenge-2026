# HerriML — NAS Unseen Data Challenge 2026

Team submission for the [5th NAS Unseen-Data Competition](https://www.nascompetition.com/) at AutoML 2026.

The competition hands our pipeline three unseen 4D-image classification datasets at evaluation time. The pipeline must process the data, search for an architecture, train it, and produce predictions — all under a fixed time budget per dataset. Score per dataset is `(accuracy − benchmark) × scale`, capped at +10 / −10. Crashes, OOMs, and timeouts each score −10.

## Quick start

```bash
git clone git@github.com:Lys-fy/HerriML-NAS-Unseen-Data-Challenge-2026.git
cd HerriML-NAS-Unseen-Data-Challenge-2026

# 1. Download the public datasets (~20 GB) — idempotent, can be re-run safely.
./scripts/download_datasets.sh                       # all 11 datasets
./scripts/download_datasets.sh Sudoku GameOfLife     # filter by name

# 2. Run the local mock grader against the code in submission/.
python dev/mock_harness.py Sudoku --score

# 3. When ready to submit, build the zip.
cd submission && zip -r ../submission.zip *
```

`datasets/` is gitignored. By default it's a symlink to a local-only directory (e.g. `~/local/nas-comp-datasets/`). Override the destination with `DATASETS_DIR=...`.

## Layout

| Path | What lives here |
|------|-----------------|
| `submission/` | The four files that get zipped: `data_processor.py`, `nas.py`, `trainer.py`, `helpers.py`. Filenames and class signatures are fixed by the grader. |
| `dev/` | Local-only tooling. `mock_harness.py` mirrors the grader's `evaluation/main.py`. |
| `scripts/` | Build helpers. `download_datasets.sh` pulls the public datasets from figshare. |
| `docs/` | Team notes (e.g. `environment.md` flags the dev/eval architecture split). |
| `Usefull_docs/` | Reference material: papers, past-winner approaches, dataset notes. |
| `datasets/` | Public datasets. **Gitignored**, symlinked to a local path outside the repo. |

## The submission contract

The grader copies `submission/*` next to its own `evaluation/main.py` and runs each dataset through:

```
DataProcessor → process()         → (train_loader, valid_loader, test_loader)
NAS           → search()          → PyTorch model
Trainer       → train()           → trained model
Trainer       → predict(test_loader) → list of int labels
```

A `clock` object is passed to every stage; `clock.check()` returns seconds remaining for the current dataset. Time and memory failures score −10. Phase 2 allows up to 7 submissions; the last working Phase 2 submission becomes the Phase 3 entry unchanged. Detailed contract notes (allowed deviations, fixed signatures, asserted constraints) are in `Usefull_docs/README.md`.

## Dev workflow

1. **Edit `submission/`.** Extra modules can live alongside the four required files; the grader copies the whole folder.
2. **Test locally** with `python dev/mock_harness.py <Dataset> --score`. Same call sequence as the real grader, plus scoring against local `test_y.npy`.
3. **Stress-test** with `--time-limit 0.05` (3 min) to verify graceful exits, and on different hardware to catch arch-specific bugs.
4. **Cross-verify on x86** before any Phase 2 push (see below).

`mock_harness.py` deliberately replicates the grader's failure modes (clock exits, test-loader assertions, blanket exception → −10). Passing here is the best signal we have that Phase 2 will pass.

## Hardware split

| Machine | Role |
|---------|------|
| **DGX Spark** (Grace ARM64, 128 GB unified) | Primary dev and training. Modern PyTorch + CUDA 12.8+. |
| **M75q** (x86 + AMD iGPU) | Cross-verification before submission. The eval server is presumed x86. |

Final submissions must run cleanly on both. Details in `docs/environment.md`.

## References

- `Usefull_docs/` — past-winner papers, 2025 leaderboard analysis, dataset notes.
- [Starter kit](https://github.com/Towers-D/NAS-Comp-Starter-Kit) — clone separately if you want the official Makefile pipeline. Not vendored here.
- [Public datasets repo](https://github.com/Towers-D/NAS-Unseen-Datasets) — organisers' canonical dataset descriptions.
- [Competition site](https://www.nascompetition.com/) — official rules and timeline.
