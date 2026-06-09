#!/bin/bash
# Tier-2a / P1 v9 — the dilated TCN-2d backbone, now SEARCHED (3 seeds).
# The fixed TCN-2d probe beat the searched conv backbone by +11 total over the
# 13 datasets (conv +48.2 -> TCN-2d-fixed +59.2, seed 42). v9 puts that exact
# dilated block inside the v8 pipeline (EGNAS depth/width search + macro-prune
# verify + train-to-budget + aug-probe) so depth and width are tuned per
# dataset instead of fixed at 6 blocks. Question: does searching the dilated
# backbone hold or beat the +59.2 fixed result, multi-seed?
# Micro is OFF (every block is identical dilated, nothing to op-search).
# EGNAS_DMAX=8: the dilated net keeps full spatial resolution (no downsampling),
# so cap depth to keep big inputs (GeoClassing 64x64) affordable.
#
# 13 datasets x 3 seeds x --time-limit 2.0. Paired vs v8 conv + TCN-2d-fixed.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/tier2a_p1_augsafety_v9
ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)
mkdir -p dev/logs/tier2a_p1_v9
run() {
    local seed="$1"; local tag="p1v9_s${seed}"
    local log_dir="dev/logs/tier2a_p1_v9/${tag}"
    local pred_dir="dev/predictions-tier2a_p1_v9-${tag}"
    local sweep_log="dev/logs/tier2a_p1_v9/${tag}.log"
    echo "================================================================"
    echo "  config=tier2a-p1-v9(dilated TCN-2d backbone, searched)  seed=$seed"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$log_dir" \
        P1_PROBE=1 VERIFY_MACRO=1 EGNAS_MICRO_OFF=1 EGNAS_DMAX=8 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" --time-limit 2.0 \
            --score "${ALL_DATASETS[@]}" > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'P1 probe|raw=|Total score|RESCUED|REJECTED|v9 backbone|EGNAS full done' "$sweep_log" | sed 's/^/    /' || echo "    (no scores)"
    echo
}
echo "Tier 2a / P1 v9 sweep starting $(date -Iseconds) on $(hostname)"
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "Tier 2a / P1 v9 sweep done $(date -Iseconds)"
