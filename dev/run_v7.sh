#!/bin/bash
# Tier-2a / P1 v7 — the SIMPLIFY pivot after the v6 3-seed verdict
# (v6 was −12.1 vs EGNAS-full, σ ×2.98 — the rich vocab + aggressive search
#  were liabilities). v7 = EGNAS-full's clean, proven search + ONLY the two
# things that work, all metadata/probe-blind (no dataset gates — Unseen Data
# Challenge legal):
#   1. OP_VOCAB cut to 4 CONV ops (Conv3/Conv5/Conv7/SepConv3). Exotic ops
#      (attention/ResNeXt/...) are OUT — they poisoned Sudoku (−20pp). They are
#      tested out-of-band via baselines/tcn_probe + run_lstm_language_probe.sh.
#   2. Macro-prune VERIFICATION (the headline fix): depth verify-of-rejects
#      (rescue deep nets wrongly struck at short-K — GameOfLife) + width
#      verify-of-accepts (reject narrow nets that look fine at short-K but
#      crater at full budget — Sudoku). Knobs: VERIFY_K_MULT, MAX_MACRO_VERIFY,
#      MAX_VERIFY_EPOCHS — CALIBRATE on GPU before trusting the sweep.
#   3. The C-only Language aug-probe (aug-off + Mixup) — the ONE durable P1 win
#      (+4.23 vs EGNAS-full). Runs blind from metadata, fires per dataset.
# Dropped vs v6: clock-extend (submission-illegal), zero-shot (weak τ),
# verify-of-accepts-micro (the real bug was macro, not micro).
#
# 13 datasets × 3 seeds × --time-limit 2.0. Paired vs EGNAS-full + v6.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/v7

ALL_DATASETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
SEEDS=(42 1337 2024)

mkdir -p dev/logs/v7

run() {
    local seed="$1"
    local tag="v7_s${seed}"
    local log_dir="dev/logs/v7/${tag}"
    local pred_dir="dev/predictions-v7-${tag}"
    local sweep_log="dev/logs/v7/${tag}.log"
    echo "================================================================"
    echo "  config=tier2a-p1-v7(EGNAS-full + 4-op vocab + macro-prune-verify + Language probe)  seed=$seed"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"
    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        P1_PROBE=1 \
        VERIFY_MACRO=1 \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit 2.0 \
            --score \
            "${ALL_DATASETS[@]}" \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'P1 probe|raw=|Total score|RESCUED|REJECTED|EGNAS full done' "$sweep_log" | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Tier 2a / P1 v7 sweep starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
for seed in "${SEEDS[@]}"; do run "$seed"; done
echo "Tier 2a / P1 v7 sweep done $(date -Iseconds)"
