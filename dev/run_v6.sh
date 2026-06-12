#!/bin/bash
# Tier-2a / P1 v6 sweep — six bundled changes vs v5:
#   1. Drop A and B probes; only C (multi-trial K=8). Saves ~60s/dataset.
#   2. OP_VOCAB expanded to 9 swappable ops: Conv3, Conv5, Conv7, SepConv3,
#      GroupConv3, ResNeXt, DilatedConv3, SEConv3, SpatialSelfAttn2D.
#   3. MAX_SHORT_EPOCHS bumped 6 → 12 (longer K reduces ranker noise).
#   4. Verification-of-accepts: each accepted micro swap re-trained at 2× K;
#      if val doesn't hold up, swap reverted. Catches the v5 GameOfLife-class
#      false positives caused by noisy short-train rankings.
#   5. Clock-extend mode: NAS doesn't deduct from trainer budget. Trainer
#      always sees the full 2h regardless of how long NAS spent. Each dataset
#      now takes (2h + nas_duration) wall-clock — exploration mode.
#   6. Zero-shot proxies (NASWOT, Synflow, SNIP, ZenScore) computed per
#      candidate and logged in the pool JSON as SHADOW signals — no decision
#      impact, but data for post-sweep Spearman analysis. If any proxy
#      correlates with test acc, v7 can use it as a fast pre-filter.
#
# 13 datasets × 3 seeds, paired vs EG + v4 + v5. ~70 h wall-clock with
# early-stop. Finishes within 3 days.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/v6

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/v6

run() {
    local seed="$1"
    local tag="v6_s${seed}"
    local log_dir="dev/logs/v6/${tag}"
    local pred_dir="dev/predictions-v6-${tag}"
    local sweep_log="dev/logs/v6/${tag}.log"

    echo "================================================================"
    echo "  config=tier2a-p1-v6(C-only + Mixup-OFF + 9 ops + verify + clock-extend + zero-shot)  seed=$seed"
    echo "  datasets(${#ALL_DATASETS[@]}): ${ALL_DATASETS[*]}"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        P1_PROBE=1 \
        EXTEND_CLOCK_FOR_NAS=1 \
        VERIFY_MICRO_ACCEPTS=1 \
        COMPUTE_ZEROSHOT=1 \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${ALL_DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'P1 probe|raw=|Total score|EGNAS v6 full done|\[clock-extend\]' "$sweep_log" \
        | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier 2a / P1 v6 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for seed in "${SEEDS[@]}"; do
    run "$seed"
done

echo "Tier 2a / P1 v6 sweep done $(date -Iseconds)"
