#!/usr/bin/env bash
# v12.3c H200 battery (~19h, fits a 20h allocation):
#   1. Cryptic @2h  -- token family on H-class silicon (indep. gate roll,
#      ~8x GB10 epochs; excursion bar 77.3 single / 78.7 ens)
#   2. Cryptic @8h  -- does the token family scale with budget where conv
#      never did (72.0-73.8 plateau at every budget)?
#   3. CIFARTile @8h with a W128 capacity arm -- record chase: 89.18 was
#      K=3 x W48 on H200; W128 seed at 8h is the untested cell
#      (PC-DARTS all-time 92.28).
# Run inside the allocated shell:
#   conda activate nas && cd /bigwork/$USER/HerriML-NAS-Unseen-Data-Challenge-2026 && git pull && bash dev/run_v123c_h200.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-/bigwork/$USER/nas-comp-datasets}

echo "[v123c] $(date) 1/3 Cryptic @2h (token-integrated)"
BASELINE_LOG_DIR=$PWD/dev/logs/v123c_cryptic_2h \
python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v123c_cr2h --time-limit 2.0 --score \
    Cryptic 2>&1 | tee v123c_cryptic_2h.log

echo "[v123c] $(date) 2/3 Cryptic @8h (token-integrated)"
BASELINE_LOG_DIR=$PWD/dev/logs/v123c_cryptic_8h \
python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v123c_cr8h --time-limit 8.0 --score \
    Cryptic 2>&1 | tee v123c_cryptic_8h.log

echo "[v123c] $(date) 3/3 CIFARTile @8h, W128 capacity arm"
BASELINE_LOG_DIR=$PWD/dev/logs/v123c_caitie_w128 V12_CAP_ARM_WIDTH=128 \
python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v123c_ct8h --time-limit 8.0 --score \
    CIFARTile 2>&1 | tee v123c_cifartile_8h_w128.log

echo "[v123c] $(date) H200 BATTERY DONE"
