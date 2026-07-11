#!/usr/bin/env bash
# v12.3 ADOPTION GATE: the full 16-set @2h tier, same order/silicon as the
# v12.1 tier (A100-MIG 4g, Total +74.52) -- validates half-split default,
# gate tie-break and token family in one shot. ~33h, fits a 40h allocation.
# Run inside the allocated shell:
#   conda activate nas && cd /bigwork/$USER/HerriML-NAS-Unseen-Data-Challenge-2026 && git pull && bash dev/run_v123_tier16_2h.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-/bigwork/$USER/nas-comp-datasets}
export BASELINE_LOG_DIR=$PWD/dev/logs/v123_tier16
echo "[v123-tier16] $(date) start"
python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v123_tier16 --time-limit 2.0 --score \
    AddNIST CIFARTile Cauchy Chesseract Cryptic GameOfLife GeoClassing \
    Gutenberg Language MultNIST Myofibre Sedol Sloan Sudoku Voxel Windspeed \
    2>&1 | tee v123_tier16_2h.log
echo "[v123-tier16] $(date) DONE"
