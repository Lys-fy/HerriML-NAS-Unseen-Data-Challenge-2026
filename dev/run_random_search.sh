#!/bin/bash
# Random-search arm (random-Bonsai style) on the 3 datasets where every fixed
# backbone is stuck and conv beats TCN-2d: Chesseract, Sudoku, Gutenberg.
# Question: does a budget-limited random search over a WIDE op vocab beat our
# best (conv) on these weird/odd-shaped inputs, the way offline random-Bonsai
# beats directed NAS on Chesseract (68.8% vs our ~59%)?
# RAND_SEARCH=1 samples K=12 random archs (random depth/width/ops), short-trains
# each, keeps the best, trains it to budget. 3 seeds (random search is high
# variance, so seeds matter). Compare vs v8 conv: Chesseract +0.30, Sudoku
# +9.45, Gutenberg +1.03.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tcn_probe
LOG=dev/logs/random_search; mkdir -p "$LOG"
for ds in Chesseract Sudoku Gutenberg; do
  for seed in 42 1337 2024; do
    tag="${ds}_s${seed}"; [ -f "$LOG/$tag.done" ] && continue
    echo; echo "-- random $tag --  $(date -Iseconds)"
    env BASELINE_SEED="$seed" BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
        RAND_SEARCH=1 RAND_K=12 RAND_BUDGET_FRAC=0.45 \
        TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-rand-$tag" \
            --time-limit 2.0 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hE 'raw=.*adj=' "$LOG/$tag.log" | tail -1 || echo "(no score)"
  done
done
echo; echo "=== RANDOM SEARCH SUMMARY (raw% | adj, vs v8 conv) ==="
echo "  (v8 conv: Chesseract +0.30, Sudoku +9.45, Gutenberg +1.03)"
for ds in Chesseract Sudoku Gutenberg; do for s in 42 1337 2024; do
  r=$(grep -hE 'raw=.*adj=' "$LOG/${ds}_s${s}.log" 2>/dev/null|tail -1|grep -oE 'raw=[0-9.]+%.*adj= *[+-]?[0-9.]+')
  printf '  %-11s s%-5s %s\n' "$ds" "$s" "${r:-...}"
done; done
echo "=== random search done $(date -Iseconds) ==="
