#!/usr/bin/env bash
# BDH (Dragon Hatchling) excursion across the datasets, REASONING/SEQUENCE ones
# first, then -- when it finishes -- AUTO-RESUMES the paused 3-seed v11 sweep
# (no manual restart). Self-cd, detach:
#   setsid bash dev/queue_bdh.sh > ~/local/bdh.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export BDH_COMPILE=${BDH_COMPILE:-1}   # compiled is faster (fused); drop_last keeps shapes static
DATA="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/bdh"; mkdir -p "$OUT"
HOURS=${HOURS:-1.5}

# reasoning / sequence first (BDH's domain), then the rest; CIFARTile last (longest seq)
SETS=(Sudoku Chesseract Sedol Cryptic Language \
      AddNIST MultNIST GeoClassing GameOfLife Gutenberg Myofibre Voxel Windspeed CIFARTile)
[ $# -gt 0 ] && SETS=("$@")

echo "[bdh] $(date) excursion (${HOURS}h each): ${SETS[*]}"
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
for ds in "${SETS[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds (no metadata)"; continue; }
    code=$(python -c "import json;print(json.load(open('$DATA/$ds/metadata')).get('codename','$ds'))" 2>/dev/null)
    [ -f "$OUT/${code}_bdh.json" ] && { echo "SKIP $ds ($code already done)"; continue; }
    echo "=== BDH $ds  $(date '+%H:%M:%S') ==="
    python dev/bdh_excursion.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --hours "$HOURS" 2>&1 | tee "$OUT/${ds}.log"
done
echo "[bdh] $(date) EXCURSION DONE -> auto-resuming the 3-seed v11 sweep"
exec bash dev/queue_search_seeds.sh
