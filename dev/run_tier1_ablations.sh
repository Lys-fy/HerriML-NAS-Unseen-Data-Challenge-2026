#!/bin/bash
# Tier 1 ablation runner — runs 3 single-feature-disabled configs against
# 3 representative datasets at --time-limit 2.0 each. Designed to be invoked
# under nohup on Spark for an overnight run.
#
# Configs (each isolates one Tier 1 component vs the full baseline):
#   1. search-off — TIER1_SEARCH_OFF=1  (skip NAS search, use all-Conv3)
#   2. ema-off    — TIER1_EMA_OFF=1     (no EMA shadow weights)
#   3. aug-off    — TIER1_AUG_OFF=1     (no augmentation regardless of metadata)
#
# Datasets (the canonical representatives from Tier 1 results):
#   - AddNIST   high-benchmark big winner (+11.86 adj over Tier 0)
#   - CIFARTile architectural-ceiling case (+0.66 adj)
#   - Language  metadata-policy regression (-5.72 adj)
#
# Outputs:
#   dev/logs/ablation-<config>/<codename>.csv     per-epoch CSV
#   dev/logs/ablation-<config>.log                full sweep log
#   dev/predictions-ablation-<config>/<codename>.npy  predictions
#
# Each config × dataset gets its own clock; total wall ~5–6 h on Spark GB10.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
DATASETS=(AddNIST CIFARTile Language)
SUB=baselines/tier1_aug_op_search

mkdir -p dev/logs

run_config() {
    local name="$1"
    local env_var="$2"
    local log_dir="dev/logs/ablation-${name}"
    local pred_dir="dev/predictions-ablation-${name}"
    local sweep_log="dev/logs/ablation-${name}.log"

    echo "================================================================"
    echo "  Ablation: $name   ($env_var=1)"
    echo "  Datasets: ${DATASETS[*]}"
    echo "  Started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env "$env_var=1" \
        BASELINE_LOG_DIR="$log_dir" \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${DATASETS[@]}" \
        > "$sweep_log" 2>&1
    local rc=$?

    echo "  Finished: $(date -Iseconds)   exit=$rc"
    echo "  Total score line:"
    grep "Total score" "$sweep_log" || echo "  (no total — sweep may have failed)"
    echo
}

echo "Ablation runner starting at $(date -Iseconds)"
echo "Spark host: $(hostname)"
echo "Python: $($VENV --version)"
echo

run_config "search-off" "TIER1_SEARCH_OFF"
run_config "ema-off"    "TIER1_EMA_OFF"
run_config "aug-off"    "TIER1_AUG_OFF"

echo "================================================================"
echo "  ALL ABLATIONS COMPLETE — $(date -Iseconds)"
echo "================================================================"
echo
echo "Per-config scoring summary:"
for name in search-off ema-off aug-off; do
    echo "--- $name ---"
    awk '/========== Scoring/,EOF' "dev/logs/ablation-${name}.log" | head -20
    echo
done
