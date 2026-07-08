#!/usr/bin/env bash
# Real-search tests: the FAITHFUL EGNAS replica (dev/egnas_faithful.py) and the
# v11.3 EVOLUTIONARY search (dev/evo_search.py), across all datasets, each
# discovered arch then full-budget-validated with v10's real recipe vs v10's
# known numbers. Phase 1 = all searches (fast -> every discovered architecture
# early); Phase 2 = all validations (slow). Priority-ordered. Self-cd, detach:
#   setsid bash dev/queue_search_tests.sh > ~/local/search_tests.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
DATA="$HOME/local/nas-comp-datasets"
HOURS=${HOURS:-1.5}                 # full-budget validation hours/dataset
SEARCH_HOURS=${SEARCH_HOURS:-1.5}   # search wall-clock safety cap
SEED=${SEED:-42}                    # search seed; dirs are seed-suffixed (s42 = bare)
SUF=""; [ "$SEED" != "42" ] && SUF="_s${SEED}"
V11_ONLY=${V11_ONLY:-0}             # 1 = run only v11.1 (skip egnas/evo + validation)

# decision-relevant datasets first (board/encoded levers), ceilinged ones last
SETS=(Chesseract Sedol CIFARTile Sudoku Cryptic Language Gutenberg Sloan Cauchy \
      AddNIST MultNIST GeoClassing GameOfLife Voxel Windspeed Myofibre)
[ $# -gt 0 ] && SETS=("$@")

EG="$HOME/local/egnas_faithful${SUF}"; EV="$HOME/local/evo${SUF}"
EGV="${EG}_validated"; EVV="${EV}_validated"
V11="$HOME/local/v11_1${SUF}"
mkdir -p "$EG" "$EV" "$EGV" "$EVV" "$V11"
echo "[tests] SEED=$SEED  dirs: $V11 $EG $EV"

echo "[tests] $(date) datasets: ${SETS[*]}  search<=${SEARCH_HOURS}h validate=${HOURS}h"
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[tests] ===== PHASE 1: searches ====="
for ds in "${SETS[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds (no metadata)"; continue; }
    echo "=== v11.1 $ds s$SEED  $(date '+%H:%M:%S') ==="
    python dev/v11_1_run.py --dataset-dir "$DATA/$ds" --out-dir "$V11" \
        --hours "$HOURS" --seed "$SEED" 2>&1 | tee "$V11/${ds}.log"
    [ "$V11_ONLY" = 1 ] && continue
    echo "=== EGNAS-faithful $ds s$SEED  $(date '+%H:%M:%S') ==="
    python dev/egnas_faithful.py --dataset-dir "$DATA/$ds" --out-dir "$EG" \
        --search-hours "$SEARCH_HOURS" --seed "$SEED" 2>&1 | tee "$EG/${ds}.log"
    echo "=== EVO $ds s$SEED  $(date '+%H:%M:%S') ==="
    python dev/evo_search.py --dataset-dir "$DATA/$ds" --out-dir "$EV" \
        --search-hours "$SEARCH_HOURS" --seed "$SEED" 2>&1 | tee "$EV/${ds}.log"
done
echo "[tests] $(date) ALL SEARCHES DONE"
if [ "$V11_ONLY" = 1 ]; then echo "[tests] V11_ONLY -> done (no validation)"; exit 0; fi

echo "[tests] ===== PHASE 2: full-budget validation ====="
for pair in "$EG:$EGV:egnas" "$EV:$EVV:evo"; do
    src="${pair%%:*}"; rest="${pair#*:}"; dst="${rest%%:*}"; tagm="${rest##*:}"
    for js in "$src"/*.json; do
        [ -f "$js" ] || continue
        ds=$(python -c "import json;print(json.load(open('$js'))['dataset'])" 2>/dev/null)
        code=$(basename "$js" .json)
        [ -f "$DATA/$ds/metadata" ] || { echo "SKIP validate $code (no $ds)"; continue; }
        echo "=== validate[$tagm] $code ($ds)  $(date '+%H:%M') ==="
        python dev/validate_winner.py --dataset-dir "$DATA/$ds" --winner-json "$js" \
            --hours "$HOURS" --out-dir "$dst" 2>&1 | tee "$dst/${code}.log"
    done
done
echo "[tests] ALL DONE -> egnas=$EG/$EGV  evo=$EV/$EVV  $(date)"
