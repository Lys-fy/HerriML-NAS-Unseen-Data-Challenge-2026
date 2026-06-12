#!/bin/bash
# v7 macro-verify CALIBRATION — the two datasets v6 broke, where the new
# depth/width verification has to prove itself:
#   GameOfLife (depth verify-of-rejects must rescue deep nets → solve it, not
#              stay at D=3/48%);
#   Sudoku     (width verify-of-accepts must reject the W=16 mirage → stay wide
#              ~90%, not crater to ~67-71% like v6).
# Waits for the recurrence A/B to finish first (shares the GPU), then runs
# v7 on these 2 datasets × 2 seeds @ full 2h. Watch the logs for RESCUED /
# REJECTED lines and the final raw scores vs v6 (GoL ~48, Sudoku ~67-71) and
# EGNAS-full (GoL up to 99 lottery, Sudoku ~91).
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/v7
LOG=dev/logs/v7_calib
mkdir -p "$LOG"

echo "=== v7 calibration — waiting for recurrence A/B to finish ($(date -Iseconds)) ==="
while pgrep -f run_recurrence_validation >/dev/null 2>&1; do sleep 120; done
# also wait out any lingering mock_harness from the A/B
while pgrep -f 'predictions-recur' >/dev/null 2>&1; do sleep 60; done
echo "=== A/B done — starting v7 calibration $(date -Iseconds) ==="

for seed in 42 1337; do
  for ds in GameOfLife Sudoku; do
    tag="${ds}_s${seed}"
    echo; echo "── v7 calib: $tag ──  $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 VERIFY_MACRO=1 \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-v7calib-$tag" \
            --time-limit 2.0 --score "$ds" \
        > "$LOG/$tag.log" 2>&1
    printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1
    grep -hE 'RESCUED|REJECTED|EGNAS full done' "$LOG/$tag.log" | sed 's/^/      /'
  done
done

echo; echo "=== v7 CALIBRATION SUMMARY ($(date -Iseconds)) ==="
echo "(v6 was: GameOfLife ~48%, Sudoku ~67-71%; EGNAS-full: GoL up to 99 lottery, Sudoku ~91%)"
for ds in GameOfLife Sudoku; do for seed in 42 1337; do
  printf '  %-11s s%-5s ' "$ds" "$seed"
  grep -hoE 'raw=[0-9.]+%' "$LOG/${ds}_s${seed}.log" 2>/dev/null | head -1
done; done
echo "=== v7 calibration done $(date -Iseconds) ==="
