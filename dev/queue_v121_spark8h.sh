#!/usr/bin/env bash
# Spark 8h tiers for v12.1(b): the Phase-3 dress rehearsal.
#   Phase A -- LAST YEAR'S FINALS TRIO (GameOfLife, Windspeed, Cryptic): the
#              closest known proxy for the 2026 hidden final datasets.
#   Phase B -- the three near-all-time sets (Gutenberg, MultNIST, GeoClassing)
#              where 8h + ensembles + the W96 capacity arm can flip external
#              records (GeoClassing extends our existing all-time lead).
# 6 x 8h ~ 48h (~2 days). Waits for the running 2h tier to finish first.
# First real exercise of the v12.1b capacity arm (watch for "capacity seed").
# Detach:  setsid bash dev/queue_v121_spark8h.sh > ~/local/v121_8h.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export DATASETS_DIR="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/v121_8h"; mkdir -p "$OUT"

echo "[v121-8h] $(date) waiting for the running harness / GPU..."
while pgrep -f "mock_harness.py|v12_integration.py" >/dev/null 2>&1; do sleep 120; done
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[v121-8h] $(date) SMOKE"
python dev/smoke_v12_1.py || { echo "[v121-8h] SMOKE FAILED -- abort"; exit 1; }

echo "[v121-8h] $(date) PHASE A: 2025-finals trio @8h"
export BASELINE_LOG_DIR="$OUT/curves_finals"
python -u dev/mock_harness.py --submission-dir baselines/v12_1 --time-limit 8.0 \
    --predictions-dir "$OUT/predictions_finals" --score \
    GameOfLife Windspeed Cryptic 2>&1 | tee "$OUT/finals_8h.log"

echo "[v121-8h] $(date) PHASE B: near-all-time trio @8h"
export BASELINE_LOG_DIR="$OUT/curves_wave2"
python -u dev/mock_harness.py --submission-dir baselines/v12_1 --time-limit 8.0 \
    --predictions-dir "$OUT/predictions_wave2" --score \
    Gutenberg MultNIST GeoClassing 2>&1 | tee "$OUT/wave2_8h.log"

echo "[v121-8h] $(date) V12.1 SPARK 8H DONE"
