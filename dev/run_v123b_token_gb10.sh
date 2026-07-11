#!/usr/bin/env bash
# v12.3b GB10 battery -- TOKEN FAMILY integrated pipeline (~16h), queued
# behind run_v123_gb10.sh:
#   A (2h): Cryptic @2h -- token gate should trigger (1x6x768, ratio 128),
#      win decisively, and the budget planner fills the whole window
#      (no 400-epoch cap, no patience: schedule = plan, up to CAP_EPOCHS).
#      Bar: excursion 77.3 single / 78.7 ens; pipeline plateau 72-73.8.
#   B (8h): Cryptic @8h -- does the token family scale with budget the way
#      the conv family never did on this dataset?
#   C (6h): regression sweep @2h x {Chesseract, Gutenberg, Volga} -- none
#      match the shape trigger; decisions logs must show NO token_gate
#      stage and scores in the usual band.
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v123b_token
mkdir -p "$OUT"

echo "[v123b] $(date) A: Cryptic @2h token-integrated pipeline"
BASELINE_LOG_DIR=$OUT/curves_a \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_a" --time-limit 2.0 --score \
  Cryptic 2>&1 | tee "$OUT/a_cryptic_2h.log"

echo "[v123b] $(date) B: Cryptic @8h token-integrated pipeline"
BASELINE_LOG_DIR=$OUT/curves_b \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_b" --time-limit 8.0 --score \
  Cryptic 2>&1 | tee "$OUT/b_cryptic_8h.log"

echo "[v123b] $(date) C: no-trigger regression sweep @2h"
BASELINE_LOG_DIR=$OUT/curves_c \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_c" --time-limit 2.0 --score \
  Chesseract Gutenberg Volga 2>&1 | tee "$OUT/c_regression_2h.log"

echo "[v123b] $(date) TOKEN BATTERY DONE"
