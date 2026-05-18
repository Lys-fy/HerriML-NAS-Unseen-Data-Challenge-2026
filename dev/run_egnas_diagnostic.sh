#!/bin/bash
# Tier-2b EGNAS-light (macro) — CIFARTile-focused diagnostic.
#
# The make-or-break question: does multi-seed macro architecture search
# (depth-grow + width-prune) beat Tier-1's CIFARTile ceiling (-0.5 ± 0.5,
# the canonical +46.7pp NAS-reward dataset where training-side levers are
# exhausted)? Plus AddNIST (high-variance sanity) and GeoClassing (stable
# win — must-not-regress).
#
# Arm: tier2b_egnas_macro × {CIFARTile AddNIST GeoClassing} × 3 seeds
#   = 9 runs, --time-limit 2.0. Tier-1 baseline is NOT re-run — paired
#   against dev/logs/multiseed/full_s*.log (same datasets/seeds/budget).
#
# Cheap gate BEFORE any full 11-dataset sweep. ~6-9 h wall. nohup-friendly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2b_egnas_macro

DATASETS=(CIFARTile AddNIST GeoClassing)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/egnasdiag

run() {
    local seed="$1"
    local tag="egnas_s${seed}"
    local log_dir="dev/logs/egnasdiag/${tag}"
    local pred_dir="dev/predictions-egnasdiag-${tag}"
    local sweep_log="dev/logs/egnasdiag/${tag}.log"

    echo "================================================================"
    echo "  config=egnas-macro  seed=$seed"
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
    grep -E 'raw=|Total score|EGNAS macro done|best D=' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "EGNAS-macro diagnostic starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "================================================================"
echo "  EGNAS DIAGNOSTIC COMPLETE — $(date -Iseconds)"
echo "================================================================"
for seed in "${SEEDS[@]}"; do
    echo "--- egnas_s${seed} ---"
    grep -E 'raw=|Total score' "dev/logs/egnasdiag/egnas_s${seed}.log" \
        2>/dev/null | sed 's/^/  /' || echo "  (missing)"
done
