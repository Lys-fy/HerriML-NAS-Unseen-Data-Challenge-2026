#!/usr/bin/env bash
# Overnight equivalence verification of the SHIPPED submissions vs the validated
# baselines. Paired runs (same seed, same budget, full-pipeline 1.5h > collapse
# threshold), then compare predictions + scores + decisions. ~10.5h.
#   Chesseract: v13 twice (run-noise floor) + clean-v2 (equivalence, encoded
#               gate + K-seeds + saturation + top-ups path)
#   Cryptic:    v13 vs clean-v2 (token-family path)
#   Sudoku:     v12_1 vs clean-v1 (v1 path)
# Launch (spark):
#   nohup bash dev/verify_submissions.sh > ~/local/verify_v1v2/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
export BASELINE_SEED=42
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/verify_v1v2
mkdir -p "$OUT"
H=1.5

run() {  # <label> <submission-dir> <dataset> <hours>
  echo "[verify] $(date) $1 : $2 $3 @$4h"
  BASELINE_LOG_DIR=$OUT/dec_$1 \
    $PY -u dev/mock_harness.py --submission-dir "$2" \
    --predictions-dir "$OUT/pred_$1" --time-limit "$4" --score "$3" \
    2>&1 | tee "$OUT/$1.log"
}

run v13a_chess    baselines/v13    Chesseract $H
run v13b_chess    baselines/v13    Chesseract $H
run v2_chess      submissions/v2   Chesseract $H
run v13_cryptic   baselines/v13    Cryptic    $H
run v2_cryptic    submissions/v2   Cryptic    $H
run v121_sudoku   baselines/v12_1  Sudoku     $H
run v1_sudoku     submissions/v1   Sudoku     $H

echo "[verify] $(date) ===== COMPARISON ====="
$PY dev/compare_equiv.py "$OUT" | tee "$OUT/VERDICT.txt"
echo "[verify] $(date) DONE"
