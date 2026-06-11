#!/bin/bash
# v10 smoke: one dataset per pipeline path.
#   Language 0.3h : encoded full path (3-arm gate, family aug probe,
#                   depth incumbent, random arm)
#   Voxel    0.3h : image-like path (skip bake-off, channel-embed stem,
#                   depth search; aug n/a at 20ch)
#   Sudoku  0.15h : low-budget COLLAPSE path (encoded + <35 min ->
#                   conv default, probes skipped)
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/v10
LOG=dev/logs/v10_smoke; mkdir -p "$LOG"
run() {
  local ds=$1 tl=$2
  echo; echo "-- v10 smoke $ds (tl=$tl) --  $(date -Iseconds)"
  env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/$ds" \
      "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
          --predictions-dir "dev/predictions-v10smoke-$ds" \
          --time-limit "$tl" --score "$ds" > "$LOG/$ds.log" 2>&1
  echo "  exit=$?"
  grep -hE 'v10 |raw=.*adj=|Traceback|Error' "$LOG/$ds.log" | head -14
}
run Language 0.3
run Voxel 0.3
run Sudoku 0.15
echo "=== V10 SMOKE DONE $(date -Iseconds) ==="
