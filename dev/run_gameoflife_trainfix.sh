#!/bin/bash
# GameOfLife training-side fix (B2): the deep GoL net's val oscillates 70<->99%
# every epoch; v7 s1337 patience-early-stopped at ep78 (best 82.3%) before it
# caught a high swing, while s42/s2024 trained ~150 ep and caught 99%. So it's
# a TRAINING-policy issue, not the search. Fixed DEEP arch (D=9 all-Conv3, the
# arch v7 grows) to isolate training. Test 3 fixes vs baseline, 3 seeds:
#   baseline   — patience 30, lr 0.025 (reproduce the bug)
#   patience80 — let it run longer to catch high-oscillation epochs
#   lr_half    — halve LR to DAMPEN the oscillation (the principled fix)
# Win = s1337 recovers to ~99 AND variance across seeds collapses.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/fixed_arch_probe
LOG=dev/logs/gol_trainfix
mkdir -p "$LOG"
ARCH="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"   # D=9 deep, all-Conv3

# variant : extra env
declare -A VENV_ENV=(
    [baseline]="TRAIN_PATIENCE=30 TRAIN_LR_SCALE=1.0"
    [patience80]="TRAIN_PATIENCE=80 TRAIN_LR_SCALE=1.0"
    [lr_half]="TRAIN_PATIENCE=30 TRAIN_LR_SCALE=0.5"
)
ORDER=(baseline patience80 lr_half)
SEEDS=(42 1337 2024)

echo "=== GameOfLife train-fix — start $(date -Iseconds) ==="
for v in "${ORDER[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="${v}_s${seed}"
    echo; echo "── $tag  (${VENV_ENV[$v]}) ──  $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
        FIXED_ARCH="$ARCH" FIXED_WIDTH=48 FIXED_STEM=conv3 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        ${VENV_ENV[$v]} \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-golfix-$tag" \
            --time-limit 2.0 --score GameOfLife \
        > "$LOG/$tag.log" 2>&1
    printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1
  done
done

echo; echo "=== GAMEOFLIFE TRAIN-FIX SUMMARY (raw %) ==="
printf '%-12s %8s %8s %8s\n' variant s42 s1337 s2024
for v in "${ORDER[@]}"; do
  printf '%-12s' "$v"
  for seed in "${SEEDS[@]}"; do
    r=$(grep -hoE 'raw=[0-9.]+' "$LOG/${v}_s${seed}.log" 2>/dev/null|head -1|cut -d= -f2)
    printf ' %8s' "${r:-...}"
  done; echo
done
echo "=== done $(date -Iseconds) ==="
