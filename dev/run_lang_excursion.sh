#!/bin/bash
# Language-only excursion — test 6 variations × 3 seeds × ONLY Language.
#
# Hypothesis we're testing: aug-off alone is structurally insufficient
# to cross Language benchmark (best ever observed: Tier-1 aug-off 83.47 %
# raw, 1.7 pp below benchmark 85.20 %). Try variations of training +
# architecture knobs to see if anything actually beats it.
#
# All variations: TIER1_AUG_OFF=1 forces no spatial aug (the multi-seed
# direction; probe is bypassed entirely). Architecture is fixed (no
# EGNAS) via tier2a_language_excursion/nas.py. The variations differ
# only in training-side knobs + model width.
#
# Variations:
#   V0_baseline       : aug-off, default Tier-1 hyperparameters
#   V1_patience60     : + LANG_PATIENCE=60   (longer patience to escape plateau)
#   V2_lr_low_warmup  : + LANG_BASE_LR=0.01 LANG_WARMUP=10
#   V3_mixup          : + LANG_MIXUP_ALPHA=0.2
#   V4_larger_model   : + LANG_WIDTH=72  (vs default 48)
#   V5_combo          : + LANG_PATIENCE=60 LANG_BASE_LR=0.01 LANG_WARMUP=10 LANG_MIXUP_ALPHA=0.2
#
# Per-run budget: 0.5 h. Language under aug-off historically converges
# in <10 min; 30 min is plenty for any longer-patience variation.
# Total: 6 variations × 3 seeds × ≤30 min = ≤9 h on Spark.

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

VENV=.venv/bin/python
SUB=baselines/tier2a_language_excursion
SEEDS=(42 1337 2024)
TIME_LIMIT=0.5

declare -A VARIATIONS=(
    [V0_baseline]=""
    [V1_patience60]="LANG_PATIENCE=60"
    [V2_lr_low_warmup]="LANG_BASE_LR=0.01 LANG_WARMUP=10"
    [V3_mixup]="LANG_MIXUP_ALPHA=0.2"
    [V4_larger_model]="LANG_WIDTH=72"
    [V5_combo]="LANG_PATIENCE=60 LANG_BASE_LR=0.01 LANG_WARMUP=10 LANG_MIXUP_ALPHA=0.2"
)

# Stable iteration order
VAR_ORDER=(V0_baseline V1_patience60 V2_lr_low_warmup V3_mixup V4_larger_model V5_combo)

mkdir -p dev/logs/lang_excursion

run() {
    local var="$1"
    local seed="$2"
    local extra="${VARIATIONS[$var]}"
    local tag="${var}_s${seed}"
    local log_dir="dev/logs/lang_excursion/${tag}"
    local pred_dir="dev/predictions-lang-${tag}"
    local sweep_log="dev/logs/lang_excursion/${tag}.log"

    echo "================================================================"
    echo "  variation=$var  seed=$seed   extra-env=[$extra]"
    echo "  started:  $(date -Iseconds)"
    echo "================================================================"
    mkdir -p "$log_dir" "$pred_dir"

    env BASELINE_SEED="$seed" \
        BASELINE_LOG_DIR="$log_dir" \
        TIER1_AUG_OFF=1 \
        $extra \
        "$VENV" -u dev/mock_harness.py \
            --submission-dir "$SUB" \
            --predictions-dir "$pred_dir" \
            --time-limit "$TIME_LIMIT" \
            --score \
            Language \
        > "$sweep_log" 2>&1
    echo "  finished: $(date -Iseconds)  exit=$?"
    grep -E 'aug policy|fixed default arch|raw=|Total score|best_val' "$sweep_log" \
        | tail -8 | sed 's/^/    /' || echo "    (no scores)"
    echo
}

echo "Language excursion starting $(date -Iseconds) on $(hostname)"
echo "$($VENV --version)"
echo

for var in "${VAR_ORDER[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run "$var" "$seed"
    done
done

echo "Language excursion done $(date -Iseconds)"

# Compact summary.
echo
echo "========================================"
echo "Per-variation Language raw acc + adj:"
echo "========================================"
for var in "${VAR_ORDER[@]}"; do
    echo "=== $var ==="
    for seed in "${SEEDS[@]}"; do
        sweep_log="dev/logs/lang_excursion/${var}_s${seed}.log"
        line=$(grep -E 'LaMelo:.*raw=' "$sweep_log" 2>/dev/null | tail -1)
        echo "  s${seed}: ${line:-(no scoring line)}"
    done
done
