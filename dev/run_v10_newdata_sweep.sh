#!/bin/bash
# v10 sweep on the four teammate-built additional datasets (The Count),
# 3 seeds x --time-limit 2.0, comparable to the main v10 sweep.
# Datasets (spark dirs under ~/local/nas-comp-datasets/, codenames):
#   Magnus    (mathvision)        3x64x64   5 classes
#   Cornelius (connect4)          3x64x64   11 classes
#   Pavel     (polygon_collision) 4x64x64   2 classes (temporal, frames-as-channels)
#   Dominik   (domino)            3x44x374  5 classes (localisation)
# Decision records: dev/logs/v10_newdata_sweep/v10_s<seed>/<codename>_decisions.json
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/v10
ALL_DATASETS=(Magnus Cornelius Pavel Dominik)
SEEDS=(42 1337 2024)
mkdir -p dev/logs/v10_newdata_sweep
run() {
    local seed="$1"; local tag="v10_s${seed}"
    local log_dir="dev/logs/v10_newdata_sweep/${tag}"
    local pred_dir="dev/predictions-v10nd-${tag}"
    local sweep_log="dev/logs/v10_newdata_sweep/${tag}.log"
    echo "================================================================"
    echo "  config=v10(measure-gate-train)  seed=$seed  (new datasets)"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    # V10_PERFECT_STOP_S: stop 5 min after a dataset's val accuracy hits 100%
    # (saves budget on easy datasets). Only set here, not in the main sweep.
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$log_dir" V10_PERFECT_STOP_S=300 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" --time-limit 2.0 \
            --score "${ALL_DATASETS[@]}" > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'v10 fingerprint|v10 backbone|v10 final|raw=|Total score' "$sweep_log" | sed 's/^/    /' | tail -40
    echo
}
echo "v10 newdata sweep starting $(date -Iseconds) on $(hostname)"
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "v10 newdata sweep done $(date -Iseconds)"
