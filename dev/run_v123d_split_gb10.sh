#!/usr/bin/env bash
# v12.3d GB10: the half-split A/B, re-sized (7h). The 2h attempt was vacuous:
# GB10 calibrated MultNIST at ~15s/epoch -> 375 projected epochs, below the
# [600, 1200) half-split band. At 3.5h the projection is ~800:
#   old rule -> K=1 x ~800 epochs; new default -> K=2 x ~400.
# Same dataset, same budget, only V12_MIN_SEED_EPOCHS differs.
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v123d_split
mkdir -p "$OUT"

echo "[v123d] $(date) A: MultNIST @3.5h OLD split rule (expect K=1)"
BASELINE_LOG_DIR=$OUT/curves_old V12_MIN_SEED_EPOCHS=99999 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_old" --time-limit 3.5 --score \
  MultNIST 2>&1 | tee "$OUT/a_multnist_35h_old.log"

echo "[v123d] $(date) B: MultNIST @3.5h NEW half-split default (expect K=2)"
BASELINE_LOG_DIR=$OUT/curves_new \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_new" --time-limit 3.5 --score \
  MultNIST 2>&1 | tee "$OUT/b_multnist_35h_new.log"

echo "[v123d] $(date) SPLIT A/B DONE"
