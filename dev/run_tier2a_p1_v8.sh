#!/bin/bash
# Tier-2a / P1 v8 — the CONSOLIDATED config. v7 (EGNAS-full clean search +
# 4-op conv vocab + macro-prune-verify + C-only aug-probe; verdict +41.94,
# beat EGNAS-full +35.51) PLUS the two findings from the post-v7 sweeps:
#   1. TRAIN-TO-BUDGET (trainer default now PATIENCE=999 / MAX_EPOCHS=600):
#      early-stop wasted the non-transferable per-dataset budget and cut the
#      cosine LR anneal short. Validated 5 wins / 2 ties / 0 regress vs
#      patience-30 — and it RECOVERED MultNIST without the (dropped) micro-fix.
#   2. ADAPTIVE STEM (helpers build_cnn, gate in_channels>=8): high-channel
#      feature-plane inputs get a 1x1 channel-embed before the 3x3 spatial
#      conv. Bake-off: Voxel +0.5, 12ch wash, low-channel correctly excluded.
# Dropped vs explorations: micro-fix (budget absorbed MultNIST), patience-80
# (superseded by train-to-budget). LSTM/recurrence stays a documented arm.
#
# This is the "probe-driven self-configuring pipeline": data-processor
# (adaptive stem) + NAS (EGNAS+verify) + training (aug-probe + train-to-budget)
# each adapt to the unseen dataset, blind, from metadata. EGNAS is only the
# middle component.
#
# 13 datasets x 3 seeds x --time-limit 2.0. Paired vs v7 + EGNAS-full.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2a_p1_augsafety_v8

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/tier2a_p1_v8

run() {
    local seed="$1"
    local tag="p1v8_s${seed}"
    local log_dir="dev/logs/tier2a_p1_v8/${tag}"
    local pred_dir="dev/predictions-tier2a_p1_v8-${tag}"
    local sweep_log="dev/logs/tier2a_p1_v8/${tag}.log"
    echo "================================================================"
    echo "  config=tier2a-p1-v8(v7 + train-to-budget + adaptive-stem)  seed=$seed"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        P1_PROBE=1 \
        VERIFY_MACRO=1 \
        EGNAS_MICRO_OFF=1 \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${ALL_DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'P1 probe|raw=|Total score|RESCUED|REJECTED|stem|EGNAS full done' "$sweep_log" | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier 2a / P1 v8 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "Tier 2a / P1 v8 sweep done $(date -Iseconds)"
