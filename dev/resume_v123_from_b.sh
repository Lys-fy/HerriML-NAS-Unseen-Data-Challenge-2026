#!/usr/bin/env bash
# Resume the GB10 v12.3 chain from leg B (leg A done 2026-07-12: Cryptic@2h
# roll1 = 76.47, gate fired 0.7533 vs 0.6339). Sequential:
#   B  (8h): Cryptic @8h token pipeline (GB10 budget-scaling point)
#   C  (6h): no-trigger regression sweep @2h x {Chesseract, Gutenberg, Volga}
#   D  (7h): v12.3d MultNIST @3.5h half-split A/B
#   A2 (2h): Cryptic @2h roll 2 (gate stability)
# Launch:
#   nohup bash dev/resume_v123_from_b.sh > ~/local/v123_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v123b_token
mkdir -p "$OUT" "$HOME/local/v123d_split"

echo "[resume] $(date) B: Cryptic @8h token-integrated pipeline"
BASELINE_LOG_DIR=$OUT/curves_b \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_b" --time-limit 8.0 --score \
  Cryptic 2>&1 | tee "$OUT/b_cryptic_8h.log"

echo "[resume] $(date) C: no-trigger regression sweep @2h"
BASELINE_LOG_DIR=$OUT/curves_c \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_c" --time-limit 2.0 --score \
  Chesseract Gutenberg Volga 2>&1 | tee "$OUT/c_regression_2h.log"

echo "[resume] $(date) D: v12.3d MultNIST half-split A/B"
bash dev/run_v123d_split_gb10.sh 2>&1 | tee "$HOME/local/v123d_split/battery.log"

echo "[resume] $(date) A2: Cryptic @2h token pipeline (roll 2)"
BASELINE_LOG_DIR=$OUT/curves_a2 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_a2" --time-limit 2.0 --score \
  Cryptic 2>&1 | tee "$OUT/a2_cryptic_2h.log"

echo "[resume] $(date) ALL DONE"
