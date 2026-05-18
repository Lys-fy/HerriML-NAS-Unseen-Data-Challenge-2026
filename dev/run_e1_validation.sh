#!/bin/bash
# E1 validation sweep — does the budget-filling ensemble collapse the
# single-run variance (thesis Q4)?
#
# Arm: e1 (baselines/tier1e1_ensemble) × 5 datasets × 3 seeds = 15 runs,
#   --time-limit 2.0 (same budget as the multi-seed sweep).
#
# The Tier-1 BASELINE is NOT re-run: the multi-seed sweep already measured
# Tier 1 at these exact 5 datasets × seeds 42/1337/2024 at --time-limit 2.0
# (dev/logs/multiseed/full_s*.log). E1's member 0 IS that Tier-1 model, so
# the comparison is paired and same-conditions; re-running it would just
# burn ~half the Spark budget for no extra signal.
#
# Datasets: the high-variance + representative set —
#   AddNIST   (σ_adj 5.45 — the variance bomb)
#   MultNIST  (σ_adj 3.00 — the other high-variance one)
#   CIFARTile (the architectural ceiling — stable, sanity that E1 ≠ free lunch)
#   Language  (the seed-stable aug-regression — must not get worse)
#   GeoClassing (rock-stable +7.7 win — must stay stable, no regression)
#
# Success: E1 total σ across the 3 seeds shrinks substantially vs. Tier 1
#   (target: AddNIST σ_adj 5.45 -> <2) AND mean >= the Tier 1 mean, with no
#   new -10 and no regression on the stable datasets.
#
# E1 trains extra members into slack, so e1 runs use MORE of the 2 h budget
# than Tier 1 did — expect up to ~2 h/dataset. ~25-30 h wall. nohup-friendly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier1e1_ensemble

DATASETS=(AddNIST CIFARTile GeoClassing Language MultNIST)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/e1val

run() {
    local seed="$1"
    local tag="e1_s${seed}"
    local log_dir="dev/logs/e1val/${tag}"
    local pred_dir="dev/predictions-e1val-${tag}"
    local sweep_log="dev/logs/e1val/${tag}.log"

    echo "================================================================"
    echo "  config=e1  seed=$seed"
    echo "  datasets(${#DATASETS[@]}): ${DATASETS[*]}"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'raw=|Total score|E1: ensemble of|E1 predict: averaged' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "E1 validation sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "================================================================"
echo "  E1 VALIDATION COMPLETE — $(date -Iseconds)"
echo "================================================================"
for seed in "${SEEDS[@]}"; do
    echo "--- e1_s${seed} ---"
    grep -E 'raw=|Total score' "dev/logs/e1val/e1_s${seed}.log" 2>/dev/null \
        | sed 's/^/  /' || echo "  (missing)"
done
