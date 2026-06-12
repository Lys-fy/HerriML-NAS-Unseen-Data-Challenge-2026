#!/bin/bash
# Representation bake-off — the data-preprocessing arm of the "let the data
# choose its interpretation" extension to v7.
#
# Tests the CHANNEL-EMBED stem (1x1 channel-mix front → treat input as feature
# planes) vs the default CONV3 stem (treat input as a 2D image), at MATCHED
# body (same all-Conv3 arch at v7's found depth, only the stem differs), full
# budget, on the datasets whose channels are semantic feature planes:
#   Chesseract (12ch piece-types, v7 flat ~57.8%) — most upside
#   Voxel      (20ch slices)
#   Windspeed  (12ch frames)
#   MultNIST   (the v7 -1.15 regression — does a different front-end recover it?)
# 2 seeds. The row-BiLSTM (sequence) arm is already validated (LSTM probe:
# Language 85.76 > conv 83.83); not re-run here.
#
# Output: per-dataset conv3-stem vs channel_embed-stem → does interpreting the
# input as feature planes beat interpreting it as an image? Feeds the fail-safe
# representation A/B (conv-first, budget-gated, metadata-selects-the-arm).

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/tcn_probe
LOG=dev/logs/repr_bakeoff
mkdir -p "$LOG"

# dataset : body ops (all-Conv3 @ v7-found depth)
declare -A OPS=(
    [Chesseract]="Conv3,Conv3,Conv3,Conv3"
    [Voxel]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
    [Windspeed]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
    [MultNIST]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
)
DATASETS=(Chesseract Voxel Windspeed MultNIST)
SEEDS=(42 1337)
WIDTH=48

echo "=== representation bake-off (channel-embed vs conv3 stem) — start $(date -Iseconds) ==="
for ds in "${DATASETS[@]}"; do
  for stem in conv3 channel_embed; do
    for seed in "${SEEDS[@]}"; do
      tag="${ds}_${stem}_s${seed}"
      echo; echo "── $tag  ops=[${OPS[$ds]}] stem=$stem ──  $(date -Iseconds)"
      env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
          FIXED_ARCH="${OPS[$ds]}" FIXED_WIDTH="$WIDTH" FIXED_STEM="$stem" \
          EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
          "$VENV" -u dev/mock_harness.py \
              --submission-dir "$SUB" \
              --predictions-dir "dev/predictions-bakeoff-$tag" \
              --time-limit 2.0 --score "$ds" \
          > "$LOG/$tag.log" 2>&1
      printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1 || echo "(no score)"
    done
  done
done

echo; echo "=== BAKE-OFF SUMMARY (conv3 vs channel_embed, raw acc) ==="
for ds in "${DATASETS[@]}"; do for seed in "${SEEDS[@]}"; do
  c=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_conv3_s${seed}.log" 2>/dev/null | head -1 | cut -d= -f2)
  e=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_channel_embed_s${seed}.log" 2>/dev/null | head -1 | cut -d= -f2)
  printf '  %-11s s%-5s conv3=%-8s channel_embed=%-8s\n' "$ds" "$seed" "${c:-?}" "${e:-?}"
done; done
echo "=== bake-off done $(date -Iseconds) ==="
