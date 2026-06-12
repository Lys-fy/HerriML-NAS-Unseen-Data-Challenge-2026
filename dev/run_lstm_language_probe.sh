#!/bin/bash
# LSTM Language probe — the RESEARCH GATE for the recurrent op family.
#
# Question it answers: does recurrence (or attention) beat plain conv on
# Language-like data when handed to it ARCHITECTURE-WIDE and trained to the
# full 2h budget? This is the out-of-band test the deep analysis demands —
# the live NAS search cannot be trusted to judge exotic ops (they win the
# noisy K=12 short-train and crater at full training, e.g. Sudoku −20pp).
#
# It is ONLY a research gate, NOT a deployment: the Unseen Data Challenge
# forbids dataset-specific options, so even if a recurrent arm wins here, a
# blind metadata/probe rule (validated on all 13 ds) would be needed to
# actually use it. No point designing that rule if recurrence loses on a
# plate — hence this gate first.
#
# Multi-seed because Language is high-variance (v6 best_val swung 75.8/79.3/
# 83.1 across seeds — a single seed could mislead). 4 arms × 3 seeds × ~2h
# ≈ 24h. All arms are 5-block @ W=24 (== v6's shipped Language arch), so the
# only variable is the op family. Built on the FIXED_ARCH hook in nas.py,
# which skips search and trains the named arch directly.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tcn_probe
TIME_LIMIT="${TIME_LIMIT:-2.0}"
WIDTH="${WIDTH:-24}"
SEEDS=(42 1337 2024)
LOG_ROOT=dev/logs/lstm_lang_probe
mkdir -p "$LOG_ROOT"

# arm : comma-separated 5-block op list
declare -A ARMS=(
    [conv]="Conv3,Conv3,Conv7,Conv3,Conv3"
    [lstm_heavy]="RowBiLSTM,RowBiLSTM,Conv3,RowBiLSTM,Conv3"
    [crossaxis]="CrossAxisBiLSTM,Conv3,CrossAxisBiLSTM,Conv3,Conv3"
    [attn_heavy]="SpatialSelfAttn2D,SpatialSelfAttn2D,Conv3,SpatialSelfAttn2D,Conv3"
)
ORDER=(conv lstm_heavy crossaxis attn_heavy)

echo "LSTM Language probe — start $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)  seeds=${SEEDS[*]}  width=$WIDTH  time-limit=$TIME_LIMIT"
echo

for arm in "${ORDER[@]}"; do
    ops="${ARMS[$arm]}"
    for seed in "${SEEDS[@]}"; do
        log_dir="$LOG_ROOT/${arm}_s${seed}"
        pred_dir="dev/predictions-lstm_lang-${arm}_s${seed}"
        arm_log="$LOG_ROOT/${arm}_s${seed}.log"
        mkdir -p "$log_dir" "$pred_dir"
        echo "──────── arm=$arm seed=$seed  ops=[$ops] ────────"
        echo "  started: $(date -Iseconds)"
        env BASELINE_SEED="$seed" \
            BASELINE_LOG_DIR="$log_dir" \
            P1_PROBE=1 \
            FIXED_ARCH="$ops" \
            FIXED_WIDTH="$WIDTH" \
            EXTEND_CLOCK_FOR_NAS=0 \
            VERIFY_MICRO_ACCEPTS=0 \
            COMPUTE_ZEROSHOT=0 \
            "$VENV" -u dev/mock_harness.py \
                --submission-dir "$SUB" \
                --predictions-dir "$pred_dir" \
                --time-limit "$TIME_LIMIT" \
                --score \
                Language \
            > "$arm_log" 2>&1
        echo "  finished: $(date -Iseconds)  exit=$?"
        grep -E 'FIXED_ARCH|raw=.*adj=' "$arm_log" | sed 's/^/    /' || echo "    (no score)"
        echo
    done
done

echo "── LSTM Language probe summary (raw best_val proxy + scored raw) ──"
for arm in "${ORDER[@]}"; do
    for seed in "${SEEDS[@]}"; do
        printf '%-12s s%-5s ' "$arm" "$seed"
        grep -E 'raw=.*adj=' "$LOG_ROOT/${arm}_s${seed}.log" 2>/dev/null | tail -1 || echo "(no score)"
    done
done
echo "LSTM Language probe done $(date -Iseconds)"
