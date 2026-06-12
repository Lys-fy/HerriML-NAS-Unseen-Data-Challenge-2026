# Tier 1 — Augmentation + Random Op Search (code)

Tier 0 + the regularization it deliberately skipped (label smoothing,
DropPath, EMA, metadata-only label-preserving augmentation) + residual
ConvBlocks at base_width 48 + a first random op-type search layer.

**Documentation lives in the Obsidian notebase, not here:**

- Plan / design rationale → `docs/notes/09 - Implementation/plan/Tier 1 - Augmentation + Op Search.md`
- Measured results + ablation + graphs → `docs/notes/09 - Implementation/implementation/Tier 1 - Results.md`

## Files

| File | Role |
|---|---|
| `data_processor.py` | adds `aug_policy_for(metadata)` + `AugmentedTensorDataset`; sets `metadata['mixup_alpha']` |
| `nas.py` | random op-type search (K=8, 3-epoch short-train, best by val_acc), 30/65/5 split |
| `trainer.py` | label smoothing, EMA, warmup+cosine LR, NaN recovery, channels_last, CSV logging |
| `helpers.py` | residual `ConvBlock`+DropPath, `OP_VOCAB`, metadata-only `aug_policy_for`, `ModelEMA` |

## Ablation env-var knobs

The same code runs the single-feature ablations via environment variables
(defaults preserve normal Tier 1 behavior):

| Env var | Effect |
|---|---|
| `TIER1_SEARCH_OFF=1` | skip op-search, use fixed all-Conv3 |
| `TIER1_EMA_OFF=1` | disable EMA (live model only) |
| `TIER1_AUG_OFF=1` | force empty aug policy regardless of metadata |

Driver: `dev/run_tier1_ablations.sh`.

## Run

```bash
# full sweep at 2 h/dataset (run on DGX Spark)
BASELINE_LOG_DIR=dev/logs/training-tier1-2h \
  python dev/mock_harness.py --submission-dir baselines/aug_op_search \
    --time-limit 2.0 --predictions-dir dev/predictions-tier1-2h --score
```

Headline: total adj **+24.644** (Tier 0 was −0.355), 9/11 ≥ benchmark, 0
floors. One regression (Language −5.7, aug breaks encoded text). Ablation
shows search worth +5.9, EMA +3.2, augmentation double-edged. Full analysis
in the implementation note above.
