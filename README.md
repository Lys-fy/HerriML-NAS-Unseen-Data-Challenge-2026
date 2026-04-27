# HerriML — NAS Unseen Data Challenge 2026

Team submission for the [5th NAS Unseen-Data Competition](https://www.nascompetition.com/).

## Layout

| Path | What lives here |
|------|-----------------|
| `submission/` | The code that gets zipped and submitted. `DataProcessor`, `NAS`, `Trainer`. |
| `dev/` | Local-only dev tooling (mock harness, validation scripts, benchmarks). Not zipped. |
| `scripts/` | Download / build / eval helpers. |
| `docs/` | Team notes, plan, decisions. |
| `datasets/` | Public competition datasets. **Gitignored.** Lives on the Spark / NAS, not in git. |

The `evaluation/` and `submission_example/` directories from the official starter kit are **not** vendored here; clone [NAS-Comp-Starter-Kit](https://github.com/Towers-D/NAS-Comp-Starter-Kit) separately if you want to reproduce the organisers' Makefile pipeline.

## Setup

Targets: Python 3.x + PyTorch. Final environment will be pinned later. Primary compute is the team's DGX Spark (ARM64); cross-verify on x86 (M75q) before each submission.

## Submitting

```
cd submission && zip -r ../submission.zip *
```

`submission.zip` is gitignored — never commit a built submission.
