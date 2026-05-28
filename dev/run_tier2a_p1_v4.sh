#!/bin/bash
# Tier-2a / P1 v4 — 3-probe shadow comparison sweep.
#
# Each dataset runs THREE probes; the multi-trial K=4 (A) drives the
# main training (chosen for robustness via 3-trial unanimity); single-
# trial K=8 (B) and multi-trial K=8 (C) run as shadows and log their
# decisions for post-sweep agreement analysis. Cost: ~132 s of probe
# per dataset, <1% of the 2 h budget.
#
# Built after the v3 verdict (Δ = −10 vs EGNAS-full) showed the K=4
# noise floor is ~±8 pp, wider than estimated. v4 doesn't pick a single
# fix — it tests multi-trial, longer K, and both, in one sweep.
#
# 13 datasets × 3 seeds × 2.0h budget ≈ 12-16 h wall.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2a_p1_augsafety_v4

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/tier2a_p1_v4

run() {
    local seed="$1"
    local tag="p1v4_s${seed}"
    local log_dir="dev/logs/tier2a_p1_v4/${tag}"
    local pred_dir="dev/predictions-tier2a_p1_v4-${tag}"
    local sweep_log="dev/logs/tier2a_p1_v4/${tag}.log"

    echo "================================================================"
    echo "  config=tier2a-p1-v4(3-probe shadow comp)  seed=$seed"
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

echo "Tier 2a / P1 v4 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "Tier 2a / P1 v4 sweep done $(date -Iseconds)"
