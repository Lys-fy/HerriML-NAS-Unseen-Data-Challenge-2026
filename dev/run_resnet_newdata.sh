#!/bin/bash
# ResNet-18 baseline (redo) on the 5 custom datasets, 1 seed, 2h each.
# The zero-point for the adjusted score (customs ship benchmark=0.0).
# Standalone so it can be re-queued; the first attempt inside
# run_v10_newdata_sweep.sh crashed (resnet18/ was missing helpers.py).
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
ALL_DATASETS=(Quincy Magnus Cornelius Pavel Dominik)
seed=42
log_dir="dev/logs/v10_newdata_sweep/resnet18_s${seed}"
pred_dir="dev/predictions-resnet18-newdata-s${seed}"
sweep_log="dev/logs/v10_newdata_sweep/resnet18_s${seed}.log"
mkdir -p "$log_dir" "$pred_dir"
echo "resnet18 baseline (redo) seed=$seed started $(date -Iseconds)"
env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$log_dir" V10_PERFECT_STOP_S=300 \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/resnet18 \
        --predictions-dir "$pred_dir" --time-limit 2.0 \
        --score "${ALL_DATASETS[@]}" > "$sweep_log" 2>&1
echo "resnet18 baseline (redo) done $(date -Iseconds)  exit=$?"
grep -E 'resnet18:|raw=|Total score' "$sweep_log" | sed 's/^/    /' | tail -20
echo "resnet18 newdata done $(date -Iseconds)" >> dev/logs/v10_newdata_sweep/sweep.log
