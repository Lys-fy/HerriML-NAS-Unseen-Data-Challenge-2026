#!/usr/bin/env bash
# After v11.2 finishes, full-budget-validate each ADOPTED winner with v10's real
# recipe (2h train-to-budget). Abstained datasets are skipped (v10 stays). Self-cd.
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
V112="$HOME/local/v112"; OUT="$HOME/local/v112_validated"; mkdir -p "$OUT"
DATA="$HOME/local/nas-comp-datasets"
HOURS=${HOURS:-2.0}

echo "[val] $(date) waiting for v11.2 to finish ..."
until grep -q "v11.2 done" "$V112/queue.log" 2>/dev/null; do sleep 180; done
echo "[val] $(date) v11.2 done -> validating adopted winners at ${HOURS}h"

for js in "$V112"/*.json; do
    [ -f "$js" ] || continue
    info=$(python -c "import json;r=json.load(open('$js'));print(r['decision']['adopt_over_v10'],r['dataset'])" 2>/dev/null)
    adopt=$(echo "$info" | cut -d' ' -f1); ds=$(echo "$info" | cut -d' ' -f2)
    code=$(basename "$js" .json)
    if [ "$adopt" != "True" ]; then
        echo "[val] $code ($ds): abstained -> skip (v10 stays)"; continue
    fi
    echo "[val] === $code ($ds)  $(date '+%H:%M') ==="
    python dev/validate_winner.py --dataset-dir "$DATA/$ds" --winner-json "$js" \
        --hours "$HOURS" --out-dir "$OUT" 2>&1 | tee "$OUT/${code}.log"
done
echo "[val] ALL DONE -> $OUT  $(date)"
