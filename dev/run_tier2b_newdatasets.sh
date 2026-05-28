#!/bin/bash
# Tier-2b FULL EGNAS — 3-seed run on the 2 newly-resolved 2025 datasets
# (Windspeed + Cryptic), composes with the main 11-ds tier2bfull sweep
# (dev/logs/tier2bfull/t2bfull_s*.log) for a full 13-ds × 3-seed picture.
#
# Note: no Tier-1 paired baseline exists for these 2 (not in multiseed) —
# this run gives absolute EGNAS-full numbers, no paired Tier-1 delta.
#
# Cryptic ships placeholder benchmark (raw/10 shipped scoring, like Sudoku).
# Windspeed has a real benchmark 13.492 → standard shipped scoring is honest.
#
# ~2 h/dataset budget × 2 datasets × 3 seeds ≈ ~12 h. nohup-friendly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2b_egnas_full

DATASETS=(Windspeed Cryptic)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/tier2bfull

run() {
    local seed="$1"
    local tag="t2bfull_new_s${seed}"
    local log_dir="dev/logs/tier2bfull/${tag}"
    local pred_dir="dev/predictions-tier2bfull-${tag}"
    local sweep_log="dev/logs/tier2bfull/${tag}.log"

    echo "================================================================"
    echo "  config=tier2b-full-new (Windspeed + Cryptic)  seed=$seed"
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
    grep -E 'raw=|Total score|EGNAS full done' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier-2b new-datasets sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "================================================================"
echo "  TIER-2B NEWDATASETS SWEEP COMPLETE — $(date -Iseconds)"
echo "================================================================"
for seed in "${SEEDS[@]}"; do
    echo "--- t2bfull_new_s${seed} ---"
    grep -E 'raw=|Total score' "dev/logs/tier2bfull/t2bfull_new_s${seed}.log" \
        2>/dev/null | sed 's/^/  /' || echo "  (missing)"
done
