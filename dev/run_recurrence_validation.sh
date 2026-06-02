#!/bin/bash
# Two jobs in one pass on the free GPU:
#
# PHASE 0 — v7 GPU smoke: confirm the simplified v7 core (EGNAS-full + 4-op
#   vocab + macro-prune-verify + aug-probe) runs end-to-end on GPU (only
#   CPU-smoked so far). Short 0.2h budget on Chesseract.
#
# PHASE 1 — RECURRENCE A/B VALIDATION: the guard against "the A/B always picks
#   LSTM". The Language probe showed row-BiLSTM beats conv THERE. To trust a
#   blind conv-vs-recurrent A/B we must confirm conv WINS on conv-optimal
#   datasets. Same fixed 5-block arch, only the op family differs, both trained
#   to convergence (patience-stop) — the honest full-budget comparison.
#     conv = Conv3,Conv3,Conv7,Conv3,Conv3
#     lstm = RowBiLSTM,RowBiLSTM,Conv3,RowBiLSTM,Conv3
#   Datasets: Sudoku (1x9x9), AddNIST (3x28x28), CIFARTile (3x64x64). 2 seeds.
#   Expected: conv ≥ lstm on all three (→ the A/B would correctly pick conv).
#   If lstm wins somewhere unexpected, that's a finding worth knowing BEFORE
#   deploying the A/B.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
LOG=dev/logs/recur_val
mkdir -p "$LOG"

echo "=== recurrence validation + v7 smoke — start $(date -Iseconds) on $(hostname) ==="

# ---- PHASE 0: v7 GPU smoke ----
echo; echo "── PHASE 0: v7 GPU smoke (Chesseract, 0.2h) ──"
env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/v7smoke" P1_PROBE=1 VERIFY_MACRO=1 \
    "$VENV" -u dev/mock_harness.py \
        --submission-dir baselines/tier2a_p1_augsafety_v7 \
        --predictions-dir dev/predictions-v7smoke \
        --time-limit 0.2 --score Chesseract \
    > "$LOG/v7smoke.log" 2>&1
echo "  v7 smoke exit=$?  result:"
grep -E 'raw=.*adj=|RESCUED|REJECTED|EGNAS full done|Traceback|Error' "$LOG/v7smoke.log" | sed 's/^/    /' | head

# ---- PHASE 1: recurrence A/B validation ----
declare -A ARMS=(
    [conv]="Conv3,Conv3,Conv7,Conv3,Conv3"
    [lstm]="RowBiLSTM,RowBiLSTM,Conv3,RowBiLSTM,Conv3"
)
DATASETS=(Sudoku AddNIST CIFARTile)
SEEDS=(42 1337)
WIDTH=48

for ds in "${DATASETS[@]}"; do
  for arm in conv lstm; do
    for seed in "${SEEDS[@]}"; do
        tag="${ds}_${arm}_s${seed}"
        echo; echo "── A/B: $tag  ops=[${ARMS[$arm]}] ──"
        env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
            FIXED_ARCH="${ARMS[$arm]}" FIXED_WIDTH="$WIDTH" \
            EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
            "$VENV" -u dev/mock_harness.py \
                --submission-dir baselines/fixed_arch_probe \
                --predictions-dir "dev/predictions-recur-$tag" \
                --time-limit 2.0 --score "$ds" \
            > "$LOG/$tag.log" 2>&1
        printf '    %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1 || echo "(no score)"
    done
  done
done

echo; echo "=== A/B SUMMARY (conv vs lstm, raw acc) ==="
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    c=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_conv_s${seed}.log" 2>/dev/null | head -1 | cut -d= -f2)
    l=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_lstm_s${seed}.log" 2>/dev/null | head -1 | cut -d= -f2)
    printf '  %-10s s%-5s conv=%-8s lstm=%-8s\n' "$ds" "$seed" "${c:-?}" "${l:-?}"
  done
done
echo "=== done $(date -Iseconds) ==="
