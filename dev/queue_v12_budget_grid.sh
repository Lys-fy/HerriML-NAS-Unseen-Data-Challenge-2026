#!/usr/bin/env bash
# v12 FULL BUDGET GRID on Spark: all 16 datasets x {2h, 4h, 6h, 8h} through the
# grader-mirror harness. Assumes the old validate queue's P3 (2h on Chesseract/
# Language/CIFARTile/Sudoku) already ran -- the 2h tier here covers the REMAINING
# 12. Each tier gets its OWN predictions dir (clean per-tier scoring) and its own
# BASELINE_LOG_DIR (no curve overwrites across tiers). python -u for live logs.
# Gap datasets (vs 2025 winners / never-run-v12) lead each tier.
# Detach:  setsid bash dev/queue_v12_budget_grid.sh > ~/local/v12grid.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export DATASETS_DIR="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/v12grid"; mkdir -p "$OUT"
SUB="baselines/v12"

ALL16="Cryptic MultNIST Gutenberg Chesseract Sudoku Language CIFARTile AddNIST GeoClassing GameOfLife Myofibre Windspeed Sedol Sloan Cauchy Voxel"
REMAIN12="Cryptic MultNIST Gutenberg AddNIST GeoClassing GameOfLife Myofibre Windspeed Sedol Sloan Cauchy Voxel"

run_tier() {  # run_tier <hours> <name> <datasets...>
    local hours="$1" name="$2"; shift 2
    export BASELINE_LOG_DIR="$OUT/curves_${name}"
    mkdir -p "$BASELINE_LOG_DIR"
    echo "[grid] $(date) TIER ${name} (${hours}h/dataset): $*"
    # shellcheck disable=SC2086
    python -u dev/mock_harness.py --submission-dir "$SUB" --time-limit "$hours" \
        --predictions-dir "$OUT/predictions_${name}" --score $* 2>&1 | tee "$OUT/${name}.log"
}

echo "[grid] $(date) waiting for the running P3 harness / GPU..."
while pgrep -f "mock_harness.py|v12_integration.py|harvest_ab.py" >/dev/null 2>&1; do sleep 120; done
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

run_tier 2.0 2h $REMAIN12
run_tier 4.0 4h $ALL16
run_tier 6.0 6h $ALL16
run_tier 8.0 8h $ALL16
echo "[grid] $(date) V12 BUDGET GRID DONE"
