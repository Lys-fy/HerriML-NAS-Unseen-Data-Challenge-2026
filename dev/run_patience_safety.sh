#!/bin/bash
# Patience safety-check: does global patience 30->80 HURT any non-GameOfLife
# dataset (overfit from training a converged net longer)? Fixed v7-found archs
# (isolates the training-patience effect), patience 30 vs 80, 2 seeds.
#   AddNIST  (fast-converging digit task — overfit risk)
#   Language (aug-off; converges ~84%)
#   Sudoku   (medium)
# Safe to fold patience-80 into v7 IFF patience80 >= patience30 (within noise)
# on all three. If it hurts any -> go ADAPTIVE (more patience only on oscillating
# curves) instead of global.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/tcn_probe
LOG=dev/logs/patience_safety
mkdir -p "$LOG"

declare -A ARCH=(
    [AddNIST]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
    [Language]="Conv3,Conv3,Conv3,Conv3,Conv7,Conv3"
    [Sudoku]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
)
DATASETS=(AddNIST Language Sudoku)
SEEDS=(42 1337)

echo "=== patience safety-check — start $(date -Iseconds) ==="
for ds in "${DATASETS[@]}"; do
  for pat in 30 80; do
    for seed in "${SEEDS[@]}"; do
      tag="${ds}_p${pat}_s${seed}"
      echo; echo "── $tag ──  $(date -Iseconds)"
      env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
          FIXED_ARCH="${ARCH[$ds]}" FIXED_WIDTH=48 FIXED_STEM=conv3 \
          TRAIN_PATIENCE="$pat" \
          EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
          "$VENV" -u dev/mock_harness.py \
              --submission-dir "$SUB" \
              --predictions-dir "dev/predictions-patsafe-$tag" \
              --time-limit 2.0 --score "$ds" \
          > "$LOG/$tag.log" 2>&1
      printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1
    done
  done
done

echo; echo "=== PATIENCE SAFETY SUMMARY (raw %, p30 vs p80) ==="
for ds in "${DATASETS[@]}"; do for seed in "${SEEDS[@]}"; do
  a=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_p30_s${seed}.log" 2>/dev/null|head -1|cut -d= -f2)
  b=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_p80_s${seed}.log" 2>/dev/null|head -1|cut -d= -f2)
  printf '  %-9s s%-5s p30=%-8s p80=%-8s\n' "$ds" "$seed" "${a:-...}" "${b:-...}"
done; done
echo "=== done $(date -Iseconds) ==="
