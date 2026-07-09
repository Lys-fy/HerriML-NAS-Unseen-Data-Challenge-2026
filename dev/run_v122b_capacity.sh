#!/usr/bin/env bash
# v12.2b overnight: CAPACITY-ARM WIDTH LADDER (W96 vs W128), zero new code.
# 2 widths x {CIFARTile, Chesseract, Sudoku} @2h ~ 9h -> fits a 10h allocation.
# Fast/K>=3 datasets so the capacity seed always gets a slot. Compare the
# "capacity seed" val + selection outcome across widths (and vs the no-cap-arm
# v12.1 tier rows). Run inside an salloc'd shell with conda env active:
#   conda activate nas && cd <repo> && git pull && bash dev/run_v122b_capacity.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-/bigwork/$USER/nas-comp-datasets}

for W in 96 128; do
  echo "[v122b] $(date) capacity width=$W @2h"
  export BASELINE_LOG_DIR=$PWD/dev/logs/v122b_w$W
  V12_CAP_ARM_WIDTH=$W python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
      --predictions-dir dev/predictions_v122b_w$W --time-limit 2.0 --score \
      CIFARTile Chesseract Sudoku 2>&1 | tee v122b_w${W}_2h.log
done
echo "[v122b] $(date) CAPACITY LADDER DONE"
