# Tier 0 — Hardcoded CNN (code)

Zero-search fixed CNN, sized at runtime from `metadata`. Validates the harness
end-to-end on every public dataset shape; doubles as the in-process fallback
later tiers return from `NAS.search()` on exception.

**Documentation lives in the Obsidian notebase, not here:**

- Plan / design rationale → `docs/notes/09 - Implementation/plan/Tier 0 - Hardcoded CNN.md`
- Measured results + graphs → `docs/notes/09 - Implementation/implementation/Tier 0 - Results.md`

## Files

| File | Role |
|---|---|
| `data_processor.py` | numpy → normalised tensors → DataLoaders; `_to_nchw` shape shim |
| `nas.py` | returns one fixed CNN sized from `metadata` (no search) |
| `trainer.py` | SGD + cosine LR (T_max projected from epoch 0), clock-aware, best-val checkpoint, CSV logging |
| `helpers.py` | adaptive-downsample planner + ConvBlock + `build_fixed_cnn` |

## Run

```bash
# 3-min smoke
python dev/mock_harness.py --submission-dir baselines/fixed_cnn Sudoku --time-limit 0.05 --score

# full sweep at 2 h/dataset, with per-epoch CSV logging (run on DGX Spark)
BASELINE_LOG_DIR=dev/logs/training-2h \
  python dev/mock_harness.py --submission-dir baselines/fixed_cnn \
    --time-limit 2.0 --predictions-dir dev/predictions-2h --score
```

Headline: total adj **−0.355**, 6/11 datasets ≥ benchmark, 2 at the −10 floor
(AddNIST/MultNIST — overfit, no regularization). Full breakdown in the
implementation note above.
