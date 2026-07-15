#!/usr/bin/env bash
# Extension (+~4h): waits for verify_submissions.sh to finish, then adds a
# CIFARTile equivalence pair (scaler / longer-training path) at 2h each, and
# re-runs the full comparison so VERDICT.txt includes it.
# Launch (spark):
#   nohup bash dev/verify_extend.sh > ~/local/verify_v1v2/extend.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
export BASELINE_SEED=42
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/verify_v1v2
mkdir -p "$OUT"

echo "[extend] $(date) waiting for verify_submissions.sh to finish"
while pgrep -f verify_submissions.sh > /dev/null; do sleep 120; done
echo "[extend] $(date) main battery done -> CIFARTile equivalence pair @2h"

run() {  # <label> <submission-dir> <dataset> <hours>
  echo "[extend] $(date) $1 : $2 $3 @$4h"
  BASELINE_LOG_DIR=$OUT/dec_$1 \
    $PY -u dev/mock_harness.py --submission-dir "$2" \
    --predictions-dir "$OUT/pred_$1" --time-limit "$4" --score "$3" \
    2>&1 | tee "$OUT/$1.log"
}

run v13_cifar   baselines/v13    CIFARTile 2.0
run v2_cifar    submissions/v2   CIFARTile 2.0

echo "[extend] $(date) ===== FINAL COMPARISON ====="
$PY dev/compare_equiv.py "$OUT" | tee "$OUT/VERDICT.txt"
echo "[extend] $(date) DONE"
