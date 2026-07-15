#!/usr/bin/env bash
# v13 validation run. v13 = v12_1 + two changes, both DATASET-AGNOSTIC:
#   (1) gate tie-break default OFF (v12.4: OFF 65.42 > ON 63.35 on Chesseract)
#   (2) saturation early-stop: patience anchor advances only on a gain that
#       clears the run's OWN measured val noise; saturated seeds release the
#       slice. Nothing keyed to board size or the known 16 datasets.
#
# GOAL: prove v13 does not regress ANYWHERE and recovers the v12.4 T3 inversion.
# Ordered so the highest-risk unknowns resolve first (by ~Thu), broad
# regression fills the weekend. HARD deadline guard: nothing starts past
# Sunday 20:00 -- the box is clean for packaging Monday.
#
#   1  Chesseract @8h SAT=1  -- the inversion case; must recover toward 67.0
#   2  CIFARTile  @8h SAT=1  -- RISK: scaler that WANTS long training (~90);
#                              the fix must NOT over-stop it
#   3  Sudoku     @8h SAT=1  -- RISK: small board that does NOT overfit (~98);
#                              another over-stop probe
#   4  Chesseract @8h SAT=0  -- attribution: should reproduce the ~63 inversion
#   5  Cryptic    @8h SAT=1  -- token family still ~77 at 8h under v13
#   6  full 16    @2h SAT=1  -- broad regression backbone vs the v12.3 2h tier
# Launch (spark):
#   cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && \
#   nohup bash dev/run_v13_test.sh > ~/local/v13_test/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v13_test
mkdir -p "$OUT"
DEADLINE=$(date -d "2026-07-19 20:00" +%s)   # Sunday 20:00, hard stop

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

run chess8_sat  8.0 1 Chesseract
run cifar8_sat  8.0 1 CIFARTile
run sudoku8_sat 8.0 1 Sudoku
run chess8_off  8.0 0 Chesseract
run cryptic8    8.0 1 Cryptic
run tier16      2.0 1 AddNIST CIFARTile Cauchy Chesseract Cryptic GameOfLife \
                      GeoClassing Gutenberg Language MultNIST Myofibre Sedol \
                      Sloan Sudoku Voxel Windspeed

echo "[v13] $(date) VALIDATION RUN DONE"
