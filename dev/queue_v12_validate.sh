#!/usr/bin/env bash
# v12 PACKAGE validation on Spark via the grader-mirror (dev/mock_harness.py):
#   P1  5-min/dataset crash+path coverage on 10 archetypes (encoded / image /
#       collapse / big-input paths) -- catches grader-visible failures cheaply.
#   P2  30-min on 3 representative sets (the Phase-2 smoke-budget tier).
#   P3  2h on 4 sets -> compare against the T2 prototype numbers + v10-real.
# All through the REAL submission entry points (DataProcessor/NAS/Trainer import
# from baselines/v12), with --score for adjusted scores.
# Detach:  setsid bash dev/queue_v12_validate.sh > ~/local/v12val.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export DATASETS_DIR="$HOME/local/nas-comp-datasets"
export BASELINE_LOG_DIR="$HOME/local/v12val/curves"
OUT="$HOME/local/v12val"; mkdir -p "$OUT" "$BASELINE_LOG_DIR"
SUB="baselines/v12"

echo "[v12val] $(date) waiting for GPU..."
while pgrep -f "v12_integration.py|harvest_ab.py|mock_harness.py" >/dev/null 2>&1; do sleep 60; done
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[v12val] $(date) P0: package smoke"
python dev/smoke_v12.py || { echo "[v12val] SMOKE FAILED -- abort"; exit 1; }

echo "[v12val] $(date) P1: 5-min crash/path coverage (10 sets)"
python dev/mock_harness.py --submission-dir "$SUB" --time-limit 0.083 --score \
    Sudoku Chesseract Language Cryptic GeoClassing CIFARTile AddNIST GameOfLife \
    Myofibre Windspeed 2>&1 | tee "$OUT/p1_5min.log"

echo "[v12val] $(date) P2: 30-min tier (3 sets)"
python dev/mock_harness.py --submission-dir "$SUB" --time-limit 0.5 --score \
    Chesseract Language CIFARTile 2>&1 | tee "$OUT/p2_30min.log"

echo "[v12val] $(date) P3: 2h tier (4 sets; compare to T2 prototype + v10-real)"
python dev/mock_harness.py --submission-dir "$SUB" --time-limit 2.0 --score \
    Chesseract Language CIFARTile Sudoku 2>&1 | tee "$OUT/p3_2h.log"

echo "[v12val] $(date) V12 VALIDATION DONE"
