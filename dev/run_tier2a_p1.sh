#!/bin/bash
# Tier-2a / P1 aug-safety probe sweep — Tier-2b full EGNAS + the P1 probe
# in DataProcessor. 13 datasets × 3 seeds, paired vs the existing
# Tier-2b-full sweep (same datasets / seeds / budget). The ONLY delta in
# the pipeline is the probe's aug on/off decision flowing through
# DataProcessor — clean attribution of P1's marginal value.
#
# Targets: Language (aug-cost −6.82 ± 1.88, the seed-stable regression
# under EGNAS-full) and Myofibre (−0.96 medical regression). Everything
# else should be unchanged within σ — a correctly-calibrated probe is
# transparent where it picks the same answer as the default.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2a_p1_augsafety

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/tier2a_p1

run() {
    local seed="$1"
    local tag="p1_s${seed}"
    local log_dir="dev/logs/tier2a_p1/${tag}"
    local pred_dir="dev/predictions-tier2a_p1-${tag}"
    local sweep_log="dev/logs/tier2a_p1/${tag}.log"

    echo "================================================================"
    echo "  config=tier2a-p1(egnas-full + aug-probe)  seed=$seed"
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

echo "Tier 2a / P1 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "Tier 2a / P1 sweep done $(date -Iseconds)"
