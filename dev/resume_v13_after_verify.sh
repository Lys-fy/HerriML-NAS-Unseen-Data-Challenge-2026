#!/usr/bin/env bash
# Resume the v13 validation run AFTER the overnight equivalence verification
# (+extension) finishes. Leg 1 (chess8_sat -> 67.41) already done; this runs
# legs 2-6. Same Sunday-20:00 hard guard so the box is free for packaging Monday.
#   2  CIFARTile @8h SAT=1  (over-stop risk on a scaler)
#   3  Sudoku    @8h SAT=1  (over-stop risk on a small board)
#   4  Chesseract @8h SAT=0 (attribution: reproduce the ~63 inversion)
#   5  Cryptic   @8h SAT=1  (token family at 8h)
#   6  full 16   @2h SAT=1  (broad regression)
# Launch (spark):
#   nohup bash dev/resume_v13_after_verify.sh > ~/local/v13_test/resume.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v13_test
mkdir -p "$OUT"
DEADLINE=$(date -d "2026-07-19 20:00" +%s)   # Sunday 20:00, hard stop

echo "[v13-resume] $(date) waiting for overnight verification to finish"
while pgrep -f "verify_extend.sh" > /dev/null || pgrep -f "verify_submissions.sh" > /dev/null; do
  sleep 120
done
echo "[v13-resume] $(date) verification done -> resuming v13 legs 2-6"

guard() {  # skip a leg if it cannot finish a --time-limit=$1 h run by DEADLINE
  local need_h=$1 name=$2
  local now; now=$(date +%s)
  local finish; finish=$(awk -v n="$now" -v h="$need_h" 'BEGIN{print int(n + h*3600 + 900)}')
  if [ "$finish" -gt "$DEADLINE" ]; then
    echo "[v13] $(date) SKIP $name -- would finish past Sunday 20:00 deadline"
    return 1
  fi
  return 0
}

run() {  # run <name> <time_h> <SAT> <dataset...>
  local name=$1 th=$2 sat=$3; shift 3
  guard "$th" "$name" || return 0
  echo "[v13] $(date) START $name (${th}h, SATURATE=$sat): $*"
  BASELINE_LOG_DIR=$OUT/curves_$name V13_SATURATE=$sat \
    $PY -u dev/mock_harness.py --submission-dir baselines/v13 \
    --predictions-dir "$OUT/pred_$name" --time-limit "$th" --score \
    "$@" 2>&1 | tee "$OUT/${name}.log"
}

run cifar8_sat  8.0 1 CIFARTile
run sudoku8_sat 8.0 1 Sudoku
run chess8_off  8.0 0 Chesseract
run cryptic8    8.0 1 Cryptic
run tier16      2.0 1 AddNIST CIFARTile Cauchy Chesseract Cryptic GameOfLife \
                      GeoClassing Gutenberg Language MultNIST Myofibre Sedol \
                      Sloan Sudoku Voxel Windspeed

echo "[v13-resume] $(date) VALIDATION RUN DONE"
