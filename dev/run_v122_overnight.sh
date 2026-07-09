#!/usr/bin/env bash
# v12.2 overnight test battery -- the three no-code-needed A/Bs, sequential.
# Run inside an salloc'd GPU shell with the conda env ACTIVE:
#   salloc ... --time=10:00:00
#   conda activate nas && cd <repo> && git pull && bash dev/run_v122_overnight.sh
# Est. ~9h total (T1 ~4.5h, T2 ~2h, T3 ~2h) -- fits a 10h allocation.
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-/bigwork/$USER/nas-comp-datasets}

echo "[v122] $(date) T1: INDEPENDENT-INIT A/B @2h (vs the shared-init v12.1 tier)"
export BASELINE_LOG_DIR=$PWD/dev/logs/v122_indep
V12_INDEP_INIT=1 python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v122_indep --time-limit 2.0 --score \
    Chesseract Sudoku GameOfLife 2>&1 | tee v122_indep_2h.log

echo "[v122] $(date) T2: SPLIT-THRESHOLD probe @2h (CIFARTile K=2 vs the tier's K=1)"
export BASELINE_LOG_DIR=$PWD/dev/logs/v122_split
V12_FULL_EPOCHS=250 python -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir dev/predictions_v122_split --time-limit 2.0 --score \
    CIFARTile 2>&1 | tee v122_split_2h.log

echo "[v122] $(date) T3: BDH mult-128 retry @2h (paper scale, OOM fix batch=16)"
mkdir -p /bigwork/$USER/bdh_scale/m128retry
python -u dev/bdh_excursion.py --dataset-dir "$DATASETS_DIR/Sudoku" \
    --out-dir /bigwork/$USER/bdh_scale/m128retry --model bidir --mlp-mult 128 \
    --batch 16 --hours 2.0 2>&1 | tee /bigwork/$USER/bdh_scale/Sudoku_m128retry.log

echo "[v122] $(date) OVERNIGHT BATTERY DONE"
