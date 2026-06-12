#!/bin/bash
# A6: do the conv-arm-derived aug decisions transfer to the TCN backbones?
# A5b proved they do NOT on Language (tcn1d: aug ON beats the probe's OFF by
# +2.7). Re-check every dataset that v10 routes to tcn2d and where aug
# applies: direct A/B (aug on vs off), tcn2d backbone, 0.3h, seed 42.
# (Gutenberg/Sudoku/Chesseract route to conv, where the original probe
# decisions were derived and validated, so they are out of scope.)
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tcn_probe
LOG=dev/logs/phaseA6; mkdir -p "$LOG"
for ds in AddNIST CIFARTile GeoClassing MultNIST Myofibre Language; do
  for aug in on off; do
    tag="${ds}_${aug}"; [ -f "$LOG/$tag.done" ] && continue
    if [ "$aug" = off ]; then AUGENV="TIER1_AUG_OFF=1"; else AUGENV="TIER1_AUG_OFF=0"; fi
    echo "-- A6 $tag --  $(date -Iseconds)"
    env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/$tag" \
        PROBE_ARCH=tcn2d $AUGENV \
        TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-A6-$tag" \
            --time-limit 0.3 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/$tag.log"|head -1
  done
done
echo "=== A6 SUMMARY (raw %, on vs off; conv-probe said ON for all but Language) ==="
for ds in AddNIST CIFARTile GeoClassing MultNIST Myofibre Language; do
  a=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_on.log" 2>/dev/null|head -1|cut -d= -f2)
  b=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_off.log" 2>/dev/null|head -1|cut -d= -f2)
  printf '  %-11s on=%-8s off=%-8s\n' "$ds" "${a:-...}" "${b:-...}"
done
echo "=== A6 done $(date -Iseconds) ==="
