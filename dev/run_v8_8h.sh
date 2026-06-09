#!/bin/bash
# v8 Phase-3 mimic: the same consolidated v8 config at --time-limit 8.0 (a
# generous Phase-3 budget) on 3 datasets likely in the Phase-2/3 rotation that
# also have headroom for more budget:
#   GameOfLife (deep net loves budget) + Sudoku (+2.7 from budget already,
#   likely 2024 rotation) + Cryptic (2025 leaderboard set).
# Question: does train-to-budget keep scaling at 8h, or plateau/overfit?
# 1 seed first (24h); .done markers so we can extend to more seeds.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tier2a_p1_augsafety_v8
LOG=dev/logs/v8_8h; mkdir -p "$LOG"
for seed in 42; do
  for ds in GameOfLife Sudoku Cryptic; do
    tag="${ds}_s${seed}"; [ -f "$LOG/$tag.done" ] && continue
    echo; echo "── v8 8h $tag ──  $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
        VERIFY_MACRO=1 EGNAS_MICRO_OFF=1 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-v8-8h-$tag" \
            --time-limit 8.0 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/$tag.log"|head -1
  done
done
echo "=== v8 8h done $(date -Iseconds) ==="
