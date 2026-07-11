#!/usr/bin/env bash
# v12.3 ADOPTION GATE: the full 16-set @2h tier, same order/silicon as the
# v12.1 tier (A100-MIG 4g, Total +74.52) -- validates half-split default,
# gate tie-break and token family in one shot. Split into two halves so it
# runs on TWO MIG slices in parallel (~17h each, fits a 20h allocation).
# Run inside each allocated shell:
#   conda activate nas && cd /bigwork/$USER/HerriML-NAS-Unseen-Data-Challenge-2026 && git pull && bash dev/run_v123_tier16_2h.sh 1
#   ... and on the second slice: ... bash dev/run_v123_tier16_2h.sh 2
set -u
HALF=${1:?usage: run_v123_tier16_2h.sh <1|2>}
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-/bigwork/$USER/nas-comp-datasets}
export BASELINE_LOG_DIR=$PWD/dev/logs/v123_tier16_h$HALF

if [ "$HALF" = "1" ]; then
  SETS="AddNIST CIFARTile Cauchy Chesseract Cryptic GameOfLife GeoClassing Gutenberg"
else
  SETS="Language MultNIST Myofibre Sedol Sloan Sudoku Voxel Windspeed"
fi

echo "[v123-tier16-h$HALF] $(date) start: $SETS"
python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v123_tier16_h$HALF --time-limit 2.0 --score \
    $SETS 2>&1 | tee v123_tier16_2h_h$HALF.log
echo "[v123-tier16-h$HALF] $(date) DONE"
