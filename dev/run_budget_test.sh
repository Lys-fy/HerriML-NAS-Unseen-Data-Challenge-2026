#!/bin/bash
# Early-stop vs FULL-BUDGET training (the principled version of the arbitrary
# "patience 30 vs 80"). Per-dataset budget is fixed and non-transferable, so
# early-stopping just leaves budget on the table + cuts the cosine LR anneal
# short (under-polished, inconsistent stop times -> not comparable).
#   earlystop  : patience 30, max 200 ep  (current default)
#   fullbudget : patience 999 (no early-stop), max 600 ep -> trains to the
#                budget-filling schedule, LR anneals to 0
# Fixed v7 archs (isolate training from search). GameOfLife = target (does
# full-budget reliably catch the 99% the oscillation hides?); AddNIST/Language
# = controls (does full-budget help or hurt converged datasets / overfit?).
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tcn_probe
LOG=dev/logs/budget_test; mkdir -p "$LOG"
declare -A ARCH=(
  [GameOfLife]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [AddNIST]="Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3,Conv3"
  [Language]="Conv3,Conv3,Conv3,Conv3,Conv7,Conv3"
)
declare -A VAR=(
  [earlystop]="TRAIN_PATIENCE=30 TRAIN_MAX_EPOCHS=200"
  [fullbudget]="TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600"
)
for ds in GameOfLife AddNIST Language; do
  for v in earlystop fullbudget; do
    for seed in 42 1337; do
      tag="${ds}_${v}_s${seed}"
      [ -f "$LOG/$tag.done" ] && continue
      echo; echo "── budget $tag  (${VAR[$v]}) ──  $(date -Iseconds)"
      env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
          FIXED_ARCH="${ARCH[$ds]}" FIXED_WIDTH=48 FIXED_STEM=conv3 \
          EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
          ${VAR[$v]} \
          "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
              --predictions-dir "dev/predictions-budget-$tag" \
              --time-limit 2.0 --score "$ds" > "$LOG/$tag.log" 2>&1
      touch "$LOG/$tag.done"
      ep=$(grep -hcE '^  epoch' "$LOG/$tag.log")
      printf '   %s  ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/$tag.log"|head -1; echo "      (trained $ep epochs)"
    done
  done
done
echo; echo "=== BUDGET SUMMARY (raw % | epochs) ==="
for ds in GameOfLife AddNIST Language; do for s in 42 1337; do
  for v in earlystop fullbudget; do
    r=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_${v}_s${s}.log" 2>/dev/null|head -1|cut -d= -f2)
    e=$(grep -hcE '^  epoch' "$LOG/${ds}_${v}_s${s}.log" 2>/dev/null)
    printf '  %-11s s%-5s %-11s raw=%-8s ep=%s\n' "$ds" "$s" "$v" "${r:-...}" "${e:-0}"
  done
done; done
echo "=== done $(date -Iseconds) ==="
