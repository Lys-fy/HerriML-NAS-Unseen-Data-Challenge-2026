#!/bin/bash
# Budget validation (part 2): early-stop vs full-budget on the datasets NOT in
# the overnight test, to confirm "train-to-budget, no early-stop" is a NET win
# (and crucially does NOT overfit/hurt anything). Fixed v7-found archs, 2 seeds.
# Overnight already showed: GameOfLife/AddNIST/Language all WIN with full-budget.
# These add: CIFARTile, Sudoku, GeoClassing, MultNIST, Cryptic, Gutenberg,
# Myofibre, Voxel, Chesseract, Windspeed.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/fixed_arch_probe
LOG=dev/logs/budget_validate; mkdir -p "$LOG"
# dataset : v7-found arch (depth from the v7 sweep; all-Conv3 unless noted)
declare -A ARCH=(
  [CIFARTile]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [Sudoku]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [GeoClassing]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [MultNIST]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [Gutenberg]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [Myofibre]="Conv3,Conv3,Conv3,Conv3,Conv3"
  [Voxel]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [Chesseract]="Conv3,Conv3,Conv3,Conv3"
)
declare -A VAR=( [earlystop]="TRAIN_PATIENCE=30 TRAIN_MAX_EPOCHS=200" [fullbudget]="TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600" )
for ds in CIFARTile Sudoku GeoClassing MultNIST Gutenberg Myofibre Voxel Chesseract; do
  for v in earlystop fullbudget; do
    for seed in 42 1337; do
      tag="${ds}_${v}_s${seed}"; [ -f "$LOG/$tag.done" ] && continue
      echo; echo "── $tag (${VAR[$v]}) ──  $(date -Iseconds)"
      env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
          FIXED_ARCH="${ARCH[$ds]}" FIXED_WIDTH=48 FIXED_STEM=conv3 \
          EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 ${VAR[$v]} \
          "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
              --predictions-dir "dev/predictions-budval-$tag" \
              --time-limit 2.0 --score "$ds" > "$LOG/$tag.log" 2>&1
      touch "$LOG/$tag.done"
      printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/$tag.log"|head -1
    done
  done
done
echo; echo "=== BUDGET VALIDATE SUMMARY (raw %, earlystop -> fullbudget) ==="
for ds in CIFARTile Sudoku GeoClassing MultNIST Gutenberg Myofibre Voxel Chesseract; do for s in 42 1337; do
  a=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_earlystop_s${s}.log" 2>/dev/null|head -1|cut -d= -f2)
  b=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_fullbudget_s${s}.log" 2>/dev/null|head -1|cut -d= -f2)
  printf '  %-11s s%-5s  es=%-8s  fb=%-8s\n' "$ds" "$s" "${a:-...}" "${b:-...}"
done; done
echo "=== done $(date -Iseconds) ==="
