#!/bin/bash
# Multi-seed variance + pattern-confirmation sweep (full-coverage version).
#
# Purpose (White et al. 2023 survey Sec. 9.2 — NAS has high variance,
# single-seed deltas untrustworthy): full per-dataset × per-seed variance
# profile + confirm the Tier-1 patterns / aug double-edge are seed-stable,
# BEFORE the probe thesis leans on single-seed numbers.
#
# Matrix:
#   full config   × all 11 datasets        × 3 seeds = 33 runs
#   aug-off config × 7 aug-firing datasets  × 3 seeds = 21 runs
#   (aug-off is a no-op on Sudoku/GameOfLife/Chesseract/Voxel — aug policy
#    is 'none' there regardless — so aug-off ≡ full for those 4; not re-run.)
#   = 54 runs, --time-limit 2.0 (matches the Tier 0 / Tier 1 sweeps so the
#   single-vs-multi-seed comparison is apples-to-apples; patience early-stop
#   self-limits, so most datasets finish well under 2 h anyway).
#
# ~24-28 h wall on Spark GB10 (early-stop-dominated). nohup-friendly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/aug_op_search

ALL_DATASETS=(AddNIST CIFARTile Chesseract GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel)
# Datasets where the metadata-only aug policy actually fires (1/3-channel,
# big enough). aug-off only differs from full on these.
AUG_DATASETS=(AddNIST MultNIST CIFARTile GeoClassing Gutenberg Language Myofibre)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/multiseed

run() {
    local cfg="$1"; local seed="$2"; local extra_env="$3"; shift 3
    local datasets=("$@")
    local tag="${cfg}_s${seed}"
    local log_dir="dev/logs/multiseed/${tag}"
    local pred_dir="dev/predictions-multiseed-${tag}"
    local sweep_log="dev/logs/multiseed/${tag}.log"

    echo "================================================================"
    echo "  config=$cfg  seed=$seed  env=[$extra_env]"
    echo "  datasets(${#datasets[@]}): ${datasets[*]}"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env BASELINE_SEED="$seed" $extra_env \
        BASELINE_LOG_DIR="$log_dir" \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${datasets[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'raw=|Total score' "$sweep_log" | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Multi-seed full-coverage sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "full"   "$seed" ""                "${ALL_DATASETS[@]}"
    run "augoff" "$seed" "TIER1_AUG_OFF=1" "${AUG_DATASETS[@]}"
done

echo "================================================================"
echo "  ALL MULTI-SEED RUNS COMPLETE — $(date -Iseconds)"
echo "  (aug-off ≡ full for Sudoku/GameOfLife/Chesseract/Voxel)"
echo "================================================================"
echo
for seed in "${SEEDS[@]}"; do
  for cfg in full augoff; do
    echo "--- ${cfg}_s${seed} ---"
    grep -E 'raw=|Total score' "dev/logs/multiseed/${cfg}_s${seed}.log" 2>/dev/null | sed 's/^/  /' || echo "  (missing)"
  done
done
