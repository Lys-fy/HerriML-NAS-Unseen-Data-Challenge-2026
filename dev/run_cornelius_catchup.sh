#!/bin/bash
# Cornelius catch-up: its npy were only partially shipped (broken pipe left
# train_*/valid_* missing), so it FAILED on all 3 v10 seeds and the resnet
# baseline. After re-shipping Cornelius, run just Cornelius for v10 (3 seeds)
# and the resnet baseline (1 seed), appending into the existing newdata dirs.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
for seed in 42 1337 2024; do
    echo "v10 Cornelius seed=$seed $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="dev/logs/v10_newdata_sweep/v10_s${seed}" V10_PERFECT_STOP_S=300 \
        "$VENV" -u dev/mock_harness.py --submission-dir baselines/v10 \
            --predictions-dir "dev/predictions-v10nd-v10_s${seed}" --time-limit 2.0 \
            --score Cornelius >> "dev/logs/v10_newdata_sweep/v10_s${seed}.log" 2>&1
done
echo "resnet18 Cornelius seed=42 $(date -Iseconds)"
env BASELINE_SEED=42 BASELINE_LOG_DIR="dev/logs/v10_newdata_sweep/resnet18_s42" V10_PERFECT_STOP_S=300 \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/resnet18 \
        --predictions-dir "dev/predictions-resnet18-newdata-s42" --time-limit 2.0 \
        --score Cornelius >> "dev/logs/v10_newdata_sweep/resnet18_s42.log" 2>&1
echo "cornelius catchup done $(date -Iseconds)" >> dev/logs/v10_newdata_sweep/sweep.log
