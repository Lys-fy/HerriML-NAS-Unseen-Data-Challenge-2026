#!/usr/bin/env bash
# v12.3b GB10 overnight->noon queue: token-pipeline confirmation, sized to
# hand the box back by ~12:00. Waits for run_v123_gb10.sh to finish first.
#   A  (2h): Cryptic @2h -- token gate must fire (1x6x768, ratio 128) and the
#            planner fill the window. Bar: excursion 77.3 single / 78.7 ens;
#            conv-pipeline plateau 72-73.8.
#   C  (6h): no-trigger regression sweep @2h x {Chesseract, Gutenberg, Volga}
#            -- decision logs must show NO token_gate stage, scores in band.
#   A2 (2h): Cryptic @2h second roll -- the token probe is unseeded; two
#            independent rolls bracket gate-trigger stability.
# B (Cryptic @8h) deferred: the H200 already holds the 8h token point (77.02).
# Launch:
#   cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && git pull && \
#   nohup bash dev/queue_v123b_gb10_noon.sh > ~/local/v123b_token/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v123b_token
mkdir -p "$OUT"

echo "[v123b-noon] $(date) waiting for run_v123_gb10.sh to finish"
while pgrep -f "run_v123_gb10.sh" > /dev/null; do sleep 120; done
echo "[v123b-noon] $(date) box free -- starting"

echo "[v123b-noon] $(date) A: Cryptic @2h token-integrated pipeline (roll 1)"
BASELINE_LOG_DIR=$OUT/curves_a \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_a" --time-limit 2.0 --score \
  Cryptic 2>&1 | tee "$OUT/a_cryptic_2h.log"

echo "[v123b-noon] $(date) C: no-trigger regression sweep @2h"
BASELINE_LOG_DIR=$OUT/curves_c \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_c" --time-limit 2.0 --score \
  Chesseract Gutenberg Volga 2>&1 | tee "$OUT/c_regression_2h.log"

echo "[v123b-noon] $(date) A2: Cryptic @2h token pipeline (roll 2)"
BASELINE_LOG_DIR=$OUT/curves_a2 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_a2" --time-limit 2.0 --score \
  Cryptic 2>&1 | tee "$OUT/a2_cryptic_2h.log"

echo "[v123b-noon] $(date) QUEUE DONE"
