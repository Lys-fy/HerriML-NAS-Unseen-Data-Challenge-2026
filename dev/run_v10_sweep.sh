#!/bin/bash
# v10 sweep: the comparability run (user-decided order: 2h FIRST, budget
# ladder after). 13 datasets x 3 seeds x --time-limit 2.0.
# Comparable to: v7 +41.94 (3-seed), v8 +48.2 (s42), v9 +52.4 (s42),
# fixed tcn2d +59.2 (s42), A1 routing estimate +61.2 (s42).
# Decision records: dev/logs/v10_sweep/v10_s<seed>/<codename>_decisions.json
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/v10
ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)
mkdir -p dev/logs/v10_sweep
run() {
    local seed="$1"; local tag="v10_s${seed}"
    local log_dir="dev/logs/v10_sweep/${tag}"
    local pred_dir="dev/predictions-v10-${tag}"
    local sweep_log="dev/logs/v10_sweep/${tag}.log"
    echo "================================================================"
    echo "  config=v10(measure-gate-train)  seed=$seed"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$log_dir" \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" --time-limit 2.0 \
            --score "${ALL_DATASETS[@]}" > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'v10 fingerprint|v10 backbone|v10 final|raw=|Total score' "$sweep_log" | sed 's/^/    /' | tail -40
    echo
}
echo "v10 sweep starting $(date -Iseconds) on $(hostname)"
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "v10 sweep done $(date -Iseconds)"
