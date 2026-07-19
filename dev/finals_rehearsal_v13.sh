#!/usr/bin/env bash
# v13 finals dress rehearsal at the Phase-3 operating point: the 2025-finals trio
# (GameOfLife / Windspeed / Cryptic) at ~8h each = 24h total, exactly the confirmed
# Phase-3 budget. Runs the SHIPPED submission (submissions/v2) so it also confirms
# the cleaned code runs an 8h job clean. Projection vs 2025 winner:
#   2025 winner (approx): GameOfLife ~100, Windspeed ~17.3, Cryptic 76.15
#   prior v12.1 rehearsal: 99.99 / 18.93 / 72.84  (~+11.2 adj vs winner +12.2)
#   v13 now has Cryptic ~78 (token family) -> expect a clear improvement.
# Launch (spark):
#   nohup bash dev/finals_rehearsal_v13.sh > ~/local/finals_v13/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
export BASELINE_SEED=42
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/finals_v13
mkdir -p "$OUT"

run() {  # <dataset>
  echo "[finals] $(date) $1 @8h"
  BASELINE_LOG_DIR=$OUT/dec_$1 \
    $PY -u dev/mock_harness.py --submission-dir submissions/v2 \
    --predictions-dir "$OUT/pred_$1" --time-limit 8.0 --score "$1" \
    2>&1 | tee "$OUT/$1.log"
}

run GameOfLife
run Windspeed
run Cryptic

echo "[finals] $(date) ===== DRESS REHEARSAL DONE ====="
grep -hE "raw=|Total score" "$OUT"/GameOfLife.log "$OUT"/Windspeed.log "$OUT"/Cryptic.log 2>/dev/null
