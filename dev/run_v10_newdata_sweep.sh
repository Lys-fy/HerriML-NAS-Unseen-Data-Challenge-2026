#!/bin/bash
# v10 sweep on our five custom additional datasets (Quincy home-built + four
# from The Count), 3 seeds x --time-limit 2.0, comparable to the main v10 sweep.
# Datasets (spark dirs under ~/local/nas-comp-datasets/, codenames):
#   Quincy    (QR codes)          1x21x21   20 classes (encoded, corr1~0)
#   Magnus    (mathvision)        3x64x64   5 classes
#   Cornelius (connect4)          3x64x64   11 classes
#   Pavel     (polygon_collision) 4x64x64   2 classes (temporal, frames-as-channels)
#   Dominik   (domino)            3x44x374  5 classes (localisation)
# Decision records: dev/logs/v10_newdata_sweep/v10_s<seed>/<codename>_decisions.json
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SUB=baselines/v10
ALL_DATASETS=(Quincy Magnus Cornelius Pavel Dominik)
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
# ResNet-18 baseline pass (1 seed): the zero-point for the adjusted score on
# these custom datasets (their shipped benchmark is the 0.0 placeholder; the
# official datasets use a ResNet-18 number here). Same 2h budget as v10.
resnet_baseline() {
    local seed=42
    local log_dir="dev/logs/v10_newdata_sweep/resnet18_s${seed}"
    local pred_dir="dev/predictions-resnet18-newdata-s${seed}"
    local sweep_log="dev/logs/v10_newdata_sweep/resnet18_s${seed}.log"
    echo "================================================================"
    echo "  resnet18 baseline  seed=$seed  started: $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$log_dir" V10_PERFECT_STOP_S=300 \
        "$VENV" -u dev/mock_harness.py --submission-dir baselines/resnet18 \
            --predictions-dir "$pred_dir" --time-limit 2.0 \
            --score "${ALL_DATASETS[@]}" > "$sweep_log" 2>&1
    echo "  resnet18 finished: $(date -Iseconds)  exit=$?"
    grep -E 'resnet18:|raw=|Total score' "$sweep_log" | sed 's/^/    /' | tail -20
    echo
}

echo "newdata sweep starting $(date -Iseconds) on $(hostname)"
resnet_baseline
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "v10 newdata sweep done $(date -Iseconds)"
