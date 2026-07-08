#!/usr/bin/env bash
# v11.2 (fixed: inline fingerprint + op shape-prefilter) on the BLIND custom
# hold-outs + controls, then full-budget-validate EVERY winner (not just adopted
# -- the iso-budget experiment needs each search winner's true test acc, and the
# in-search v10 reference is known-weak). Generalisation evidence at GB10 scale,
# in parallel with the cluster's full matrix. Self-cd + detachable with setsid.
#
#   setsid bash dev/queue_v112_customs.sh > ~/local/v112_customs/queue.log 2>&1 &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate

OUT="$HOME/local/v112_customs"; VAL="$HOME/local/v112_customs_validated"
DATA="$HOME/local/nas-comp-datasets"
mkdir -p "$OUT" "$VAL"
HOURS=${HOURS:-2.0}
# customs first (the new hold-out evidence), then the Gutenberg re-run (confirms
# the shape fix) and Sudoku (control: encoded, search should abstain).
SETS=(Sedol Sloan Cauchy Gutenberg Sudoku)

echo "[cust] $(date) v11.2-fixed search -> ${SETS[*]}  out=$OUT"
OUT="$OUT" bash dev/run_v112.sh "${SETS[@]}"
echo "[cust] $(date) search done -> validating EVERY winner at ${HOURS}h"

for js in "$OUT"/*.json; do
    [ -f "$js" ] || continue
    info=$(python -c "import json;r=json.load(open('$js'));print(r['decision']['winner'],r['decision']['adopt_over_v10'],r['dataset'])" 2>/dev/null)
    winner=$(echo "$info" | cut -d' ' -f1); adopt=$(echo "$info" | cut -d' ' -f2); ds=$(echo "$info" | cut -d' ' -f3)
    code=$(basename "$js" .json)
    echo "[cust] === validate $code ($ds) winner=$winner adopt=$adopt  $(date '+%H:%M') ==="
    python dev/validate_winner.py --dataset-dir "$DATA/$ds" --winner-json "$js" \
        --hours "$HOURS" --out-dir "$VAL" 2>&1 | tee "$VAL/${code}.log"
done
echo "[cust] ALL DONE -> search=$OUT validated=$VAL  $(date)"
