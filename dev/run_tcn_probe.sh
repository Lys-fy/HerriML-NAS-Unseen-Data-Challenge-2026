#!/bin/bash
# TCN excursion (colleague's suggestion): how good is a Temporal Convolutional
# Network across ALL 13 datasets, vs the v8 conv baseline (seed 42)?
#   tcn1d — true sequence TCN: reshape C×H×W → (C, L=H*W), 1-D dilated convs.
#           Expected strong on sequential (Language/Cryptic), weak on images.
#           Unlike the row-BiLSTM it trains under plain SGD → searchable.
#   tcn2d — 2-D dilated-conv TCN-style net (exponential 2-D dilation + residual,
#           spatial kept). A fair general-architecture test.
# Fixed arch (no search), train-to-budget (TRAIN_PATIENCE=999/MAX=600), 2h to
# match the v8 seed-42 conv numbers. 1 seed (42). Sequential datasets first.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tcn_probe
LOG=dev/logs/tcn_probe; mkdir -p "$LOG"
DATASETS=(Language Cryptic AddNIST CIFARTile Chesseract GameOfLife GeoClassing Gutenberg MultNIST Myofibre Sudoku Voxel Windspeed)
SEED=42
for ds in "${DATASETS[@]}"; do
  for kind in tcn1d tcn2d; do
    tag="${kind}_${ds}_s${SEED}"; [ -f "$LOG/$tag.done" ] && continue
    echo; echo "── $tag ──  $(date -Iseconds)"
    env BASELINE_SEED="$SEED" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
        PROBE_ARCH="$kind" FIXED_WIDTH=48 \
        TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-tcn-$tag" \
            --time-limit 2.0 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1 || echo "(no score)"
  done
done
echo; echo "=== TCN PROBE SUMMARY (raw %, tcn1d | tcn2d) — compare vs v8 conv seed42 ==="
for ds in "${DATASETS[@]}"; do
  a=$(grep -hoE 'raw=[0-9.]+' "$LOG/tcn1d_${ds}_s${SEED}.log" 2>/dev/null|head -1|cut -d= -f2)
  b=$(grep -hoE 'raw=[0-9.]+' "$LOG/tcn2d_${ds}_s${SEED}.log" 2>/dev/null|head -1|cut -d= -f2)
  printf '  %-11s tcn1d=%-8s tcn2d=%-8s\n' "$ds" "${a:-...}" "${b:-...}"
done
echo "=== TCN probe done $(date -Iseconds) ==="
