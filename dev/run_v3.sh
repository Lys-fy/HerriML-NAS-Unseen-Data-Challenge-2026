#!/bin/bash
# Tier-2a / P1 v3 sweep — full EGNAS + the v3 aug-safety probe.
#
# v3 vs v2: the probe decision rule is asymmetric. v2 used a symmetric
# ±0.5 pp margin and coin-flipped on borderline datasets (the K=4 probe
# has a ~±6 pp noise floor; AddNIST Δscore swung +7.66 → −4.13 across
# seeds). A false-OFF (~−9, the −10-floor crater) costs ~3× a false-ON
# (~−3). So v3 stays aug-ON unless Δscore < −6 pp — only a Language-class
# signal (Δscore −15..−31) clears the band. Verified against all 6
# v2-sweep probe traces: every wrong call (AddNIST s2024, MultNIST s42)
# flips correct, Language stays correctly OFF on all 3 seeds.
#
# 13 datasets × 3 seeds, paired vs dev/logs/egnas_full/ (EGNAS-full) and
# vs dev/logs/v2/ (P1 v2). ~12-16 h wall.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/v3

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/v3

run() {
    local seed="$1"
    local tag="v3_s${seed}"
    local log_dir="dev/logs/v3/${tag}"
    local pred_dir="dev/predictions-v3-${tag}"
    local sweep_log="dev/logs/v3/${tag}.log"

    echo "================================================================"
    echo "  config=tier2a-p1-v3(egnas-full + asymmetric aug-probe)  seed=$seed"
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

echo "Tier 2a / P1 v3 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "Tier 2a / P1 v3 sweep done $(date -Iseconds)"
