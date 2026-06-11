#!/bin/bash
# v10 smoke part 2: the FULL encoded path needs a budget above the 35-min
# collapse threshold (part 1 proved the collapse path works, but at 0.3h
# every dataset collapses, so the gate/depth/random stages never ran).
# Language at 0.8h (48 min): gate + family aug probe + depth + random arm.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
LOG=dev/logs/v10_smoke; mkdir -p "$LOG"
echo "-- v10 fullpath smoke Language (tl=0.8) --  $(date -Iseconds)"
env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/Language_full" \
    .venv/bin/python -u dev/mock_harness.py --submission-dir baselines/v10 \
        --predictions-dir dev/predictions-v10smoke-LanguageFull \
        --time-limit 0.8 --score Language > "$LOG/Language_full.log" 2>&1
echo "  exit=$?"
grep -hE 'v10 |raw=.*adj=|Traceback|FALLBACK' "$LOG/Language_full.log" | head -16
echo "=== V10 FULLPATH SMOKE DONE $(date -Iseconds) ==="
