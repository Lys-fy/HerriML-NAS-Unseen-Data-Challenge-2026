#!/usr/bin/env bash
# v12 long-budget tier (Phase-3-scale): 6h and 8h per-dataset runs through the
# grader-mirror harness. Chains after queue_v12_validate.sh. Reference points:
# prototype T4 @6h = Chesseract 67.25 / CIFARTile 82.36; v10-real = 57.66 / 73.65.
# Detach:  setsid bash dev/queue_v12_longbudget.sh > ~/local/v12long.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export DATASETS_DIR="$HOME/local/nas-comp-datasets"
export BASELINE_LOG_DIR="$HOME/local/v12val/curves_long"
OUT="$HOME/local/v12val"; mkdir -p "$OUT" "$BASELINE_LOG_DIR"

echo "[v12long] $(date) waiting for the validation queue / GPU..."
while pgrep -f "queue_v12_validate|mock_harness.py|v12_integration" >/dev/null 2>&1; do sleep 120; done
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[v12long] $(date) P4a: 6h tier (Chesseract, CIFARTile)"
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit 6.0 --score \
    Chesseract CIFARTile 2>&1 | tee "$OUT/p4_6h.log"

echo "[v12long] $(date) P4b: 8h tier (Chesseract, CIFARTile, Language)"
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit 8.0 --score \
    Chesseract CIFARTile Language 2>&1 | tee "$OUT/p5_8h.log"

echo "[v12long] $(date) V12 LONG-BUDGET DONE"
