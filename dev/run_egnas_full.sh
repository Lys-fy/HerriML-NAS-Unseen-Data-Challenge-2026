#!/bin/bash
# Tier-2b FULL EGNAS-light (macro + micro) — full 11-dataset × 3-seed sweep.
#
# User decision 2026-05-18: skip the cheap 3-ds gate, run the complete
# method on everything for a proper understanding of actual performance.
# (A ~15-min correctness smoke is still run first — that only guards
# against burning ~30 h on never-executed code, it is NOT the strategic
# gate being skipped.)
#
# Arm: egnas_full × 11 datasets × 3 seeds = 33 runs, --time-limit
#   2.0. Paired vs the Tier-1 multi-seed baseline dev/logs/multiseed/
#   full_s*.log (same datasets/seeds/budget). Clean "what does MICRO add"
#   attribution remains on the 3-ds egnas_macro (egnas_macro is frozen
#   as the macro-only comparator).
#
# ~28-32 h wall. nohup-friendly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/egnas_full

ALL_DATASETS=(AddNIST CIFARTile Chesseract GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/egnas_full

run() {
    local seed="$1"
    local tag="t2bfull_s${seed}"
    local log_dir="dev/logs/egnas_full/${tag}"
    local pred_dir="dev/predictions-egnas_full-${tag}"
    local sweep_log="dev/logs/egnas_full/${tag}.log"

    echo "================================================================"
    echo "  config=tier2b-full(egnas macro+micro)  seed=$seed"
    echo "  datasets(${#ALL_DATASETS[@]}): ${ALL_DATASETS[*]}"
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
            "${ALL_DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'raw=|Total score|EGNAS full done' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier-2b FULL EGNAS sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "================================================================"
echo "  TIER-2B FULL SWEEP COMPLETE — $(date -Iseconds)"
echo "================================================================"
for seed in "${SEEDS[@]}"; do
    echo "--- t2bfull_s${seed} ---"
    grep -E 'raw=|Total score' "dev/logs/egnas_full/t2bfull_s${seed}.log" \
        2>/dev/null | sed 's/^/  /' || echo "  (missing)"
done
