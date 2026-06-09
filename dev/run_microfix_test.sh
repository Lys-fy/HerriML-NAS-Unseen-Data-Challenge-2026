#!/bin/bash
# EXTENDED micro-incumbent BUGFIX sweep: full v7 pipeline WITH the micro-fix on
# ALL 13 datasets, seeds 42->1337->2024 (gets ~1 full seed overnight). Default
# patience (isolates the micro-fix from the separate patience-80 change).
# Compare per-dataset to the v7 sweep (WITHOUT fix) in dev/logs/tier2a_p1_v7/.
# Win: micro re-activates (micro-changed > 0 where ops help), MultNIST recovers
# toward EGNAS-full ~94.5, net no regression. MultNIST first (target lands early).
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tier2a_p1_augsafety_v7
LOG=dev/logs/microfix_test; mkdir -p "$LOG"
DATASETS=(MultNIST AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language Myofibre Sudoku Voxel Windspeed)
for seed in 42 1337 2024; do
  for ds in "${DATASETS[@]}"; do
    tag="${ds}_s${seed}"
    [ -f "$LOG/$tag.done" ] && continue   # resume-safe
    echo; echo "── microfix $tag ──  $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 VERIFY_MACRO=1 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-microfix-$tag" \
            --time-limit 2.0 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1
    grep -hoE 'micro-changed [0-9]+/[0-9]+' "$LOG/$tag.log" | head -1 | sed 's/^/      /'
  done
done
echo "=== microfix sweep done $(date -Iseconds) ==="
