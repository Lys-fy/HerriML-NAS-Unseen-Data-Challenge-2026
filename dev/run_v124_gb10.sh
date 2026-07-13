#!/usr/bin/env bash
# v12.4 GB10 battery (~19h). Four open questions, in value order.
#
#   T0 (2h)  Voxel @2h -- completes the v12.3 no-trigger regression sweep.
#            (Leg C passed "Volga" -- the CODENAME -- so the harness skipped
#            it: "no metadata file". Chesseract + Gutenberg already clean.)
#
#   T1 (3h)  THE 30-MINUTE RUNG, never once tested. Phase 2 evaluates on dummy
#            data at ~30min/dataset, and our lowest validated budget is 2h. If
#            the planner mis-plans, collapses, or skips the token gate at 0.5h
#            we fail the SUBMISSION GATE, not just a score. 6 sets x 0.5h.
#            Watch: collapse path, K, projected epochs, and whether Cryptic's
#            token gate still fires when probes are 4x shorter.
#
#   T2 (6h)  GATE TIE-BREAK A/B, 3 more OFF rolls on Chesseract @2h. Pooled so
#            far: OFF 64.07/66.20/66.60 (mean 65.6) vs ON 62.64/63.62/65.00/
#            62.15 (mean 63.3). The tie-break resolves to tcn1d -- best val,
#            worst test -- and once dropped the alt hedge entirely. If n=6 OFF
#            holds ~65.5, V12_GATE_TIEBREAK flips to 0 by default: on this
#            dataset the family race is a val->test MISMATCH, so buying probe
#            resolution buys the wrong answer, and the hedge is what pays.
#
#   T3 (8h)  Chesseract @8h, tie-break OFF -- does the 8-seed top-up roster
#            wash out gate variance at the Phase-3 operating point? (Bar: 67.01
#            on A100 @8h with v12.1, which had no tie-break.)
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v124_gb10
mkdir -p "$OUT"

echo "[v124] $(date) T0: Voxel @2h -- regression sweep completion"
BASELINE_LOG_DIR=$OUT/curves_t0 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_t0" --time-limit 2.0 --score \
  Voxel 2>&1 | tee "$OUT/t0_voxel_2h.log"

echo "[v124] $(date) T1: 30-MIN RUNG x6 (Phase-2 budget, never tested)"
BASELINE_LOG_DIR=$OUT/curves_t1 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_t1" --time-limit 0.5 --score \
  Cryptic Chesseract CIFARTile Sudoku AddNIST GameOfLife 2>&1 | tee "$OUT/t1_rung30min.log"

echo "[v124] $(date) T2: gate tie-break OFF, 3 rolls, Chesseract @2h"
for R in 1 2 3; do
  echo "[v124] $(date) T2 roll $R (V12_GATE_TIEBREAK=0)"
  BASELINE_LOG_DIR=$OUT/curves_t2_r$R V12_GATE_TIEBREAK=0 \
    $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir "$OUT/pred_t2_r$R" --time-limit 2.0 --score \
    Chesseract 2>&1 | tee "$OUT/t2_r${R}_off.log"
done

echo "[v124] $(date) T3: Chesseract @8h, tie-break OFF"
BASELINE_LOG_DIR=$OUT/curves_t3 V12_GATE_TIEBREAK=0 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_t3" --time-limit 8.0 --score \
  Chesseract 2>&1 | tee "$OUT/t3_chesseract_8h.log"

echo "[v124] $(date) BATTERY DONE"
