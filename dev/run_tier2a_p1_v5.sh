#!/bin/bash
# Tier-2a / P1 v5 sweep — three bundled changes vs v4:
#   1. Probe driver swapped A (multi-trial K=4) → C (multi-trial K=8).
#      v4 shadow data showed K=4 unanimity failed on Language s2024 false-ON
#      while both K=8 shadows were correct. K=8 is the lever, not multi-trial.
#   2. Mixup α=0.2 applied automatically whenever the probe decides aug-off.
#      Language excursion V3 showed +1.6 adj on the aug-off direction; this
#      generalises that lift to any probe-OFF dataset in the full pipeline.
#   3. OP_VOCAB extended with GroupConv3 + ResNeXt bottleneck block.
#      Literature (Geada et al. arXiv:2404.02189) says grouped convolutions
#      are the architectural lever for Language (ResNeXt-50 vanilla hits
#      93.97% vs PC-DARTS 90.12% — NAS doesn't beat ResNeXt because the
#      search spaces don't include grouped convs). EGNAS micro hill-climb
#      now A/B-tests both new ops against Conv3/Conv5/SepConv3 per layer.
#
# 13 datasets × 3 seeds, paired vs EG + v4. ~12-16 h wall.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2a_p1_augsafety_v5

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/tier2a_p1_v5

run() {
    local seed="$1"
    local tag="p1v5_s${seed}"
    local log_dir="dev/logs/tier2a_p1_v5/${tag}"
    local pred_dir="dev/predictions-tier2a_p1_v5-${tag}"
    local sweep_log="dev/logs/tier2a_p1_v5/${tag}.log"

    echo "================================================================"
    echo "  config=tier2a-p1-v5(driver C + Mixup-on-OFF + GroupConv3 + ResNeXt)  seed=$seed"
    echo "  datasets(${#ALL_DATASETS[@]}): ${ALL_DATASETS[*]}"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        P1_PROBE=1 \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${ALL_DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'P1 probe|raw=|Total score|EGNAS full done' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier 2a / P1 v5 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "Tier 2a / P1 v5 sweep done $(date -Iseconds)"
