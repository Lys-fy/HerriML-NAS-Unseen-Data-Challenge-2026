#!/usr/bin/env bash
# In-depth test queue for the BDH-harvested build (Kosowski 2025):
#   Phase A -- fixed-arch A/B for C1 (LinearAttn2D), C2 (SparseGated), C4
#              (weight-shared depth) vs a default all-Conv3 backbone, on
#              representative datasets. The queue's OWN 'default' arm is the
#              primary baseline (same budget, same recipe); v10 real is a
#              secondary reference. This isolates each op's marginal value
#              BEFORE any search.
#   Phase B -- BDH bidirectional (B1 O(T) attention + B2 all-layer readout) on
#              the viable short-T + now-reachable medium-T sets (viability-gated).
# Runs a smoke test first and ABORTS the whole queue if it fails.
# Self-cd, detach:
#   setsid bash dev/queue_harvest.sh > ~/local/harvest.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export BDH_COMPILE=${BDH_COMPILE:-0}       # eager is predictable; opt-in compile
DATA="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/harvest"; mkdir -p "$OUT"
HOURS=${HOURS:-1.5}                        # per A/B run
HOURS_BDH=${HOURS_BDH:-2.0}                # per BDH-bidir run

# representative A/B sets: encoded-board, encoded, sequence, sequence, image
AB_SETS=(${AB_SETS:-Chesseract Sudoku Language Cryptic GeoClassing})
# BDH-bidir: short-T viable + medium-T now-reachable via B1
BDH_SETS=(${BDH_SETS:-Sudoku Chesseract Sedol Language Cryptic AddNIST MultNIST})

python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[harvest] $(date) SMOKE TEST"
if ! python dev/smoke_harvest.py; then
    echo "[harvest] SMOKE FAILED -- aborting queue (fix the ops before running)"; exit 1
fi

echo "[harvest] $(date) PHASE A: C1/C2/C4 A/B (${HOURS}h each) on: ${AB_SETS[*]}"
# variant table: "tag|extra-flags"
VARIANTS=("default|" \
          "C1|--inject LinearAttn2D" \
          "C2|--inject SparseGated" \
          "C4|--repeat 3")
for ds in "${AB_SETS[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds (no metadata)"; continue; }
    code=$(python -c "import json;print(json.load(open('$DATA/$ds/metadata')).get('codename','$ds'))" 2>/dev/null)
    for v in "${VARIANTS[@]}"; do
        tag="${v%%|*}"; flags="${v#*|}"
        done_json="$OUT/${code}_ab_${tag}.json"
        [ -f "$done_json" ] && { echo "SKIP $ds/$tag ($code done)"; continue; }
        echo "=== A/B $ds  tag=$tag  $(date '+%H:%M:%S') ==="
        # shellcheck disable=SC2086
        python dev/harvest_ab.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
            --hours "$HOURS" --tag "$tag" $flags 2>&1 | tee "$OUT/${ds}_ab_${tag}.log"
    done
done

echo "[harvest] $(date) PHASE B: BDH-bidir (B1+B2, ${HOURS_BDH}h each) on: ${BDH_SETS[*]}"
for ds in "${BDH_SETS[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds (no metadata)"; continue; }
    code=$(python -c "import json;print(json.load(open('$DATA/$ds/metadata')).get('codename','$ds'))" 2>/dev/null)
    [ -f "$OUT/${code}_bdh_bidir.json" ] && { echo "SKIP $ds ($code bidir done)"; continue; }
    echo "=== BDH-bidir $ds  $(date '+%H:%M:%S') ==="
    python dev/bdh_excursion.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --model bidir --hours "$HOURS_BDH" 2>&1 | tee "$OUT/${ds}_bdh_bidir.log"
done

echo "[harvest] $(date) QUEUE DONE"
