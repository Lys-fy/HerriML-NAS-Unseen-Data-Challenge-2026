#!/usr/bin/env bash
# v11.2 overnight: EGNAS-style heterogeneous micro-macro hill-climb + verify,
# one dataset at a time. Self-cd + self-activate. Override knobs via env.
#   SE=12 VE=40 FE=120 NST=4 bash dev/run_v112.sh [dataset names...]
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
OUT=${OUT:-"$HOME/local/v112"}; mkdir -p "$OUT"
DATA="$HOME/local/nas-comp-datasets"
SE=${SE:-12}; VE=${VE:-40}; FE=${FE:-120}; NST=${NST:-4}; SEED=${SEED:-42}

SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(Chesseract Gutenberg CIFARTile)
echo "v11.2: ${SETS[*]}  screen=$SE verify=$VE final=$FE n_stages=$NST  out=$OUT"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
for ds in "${SETS[@]}"; do
    if [ ! -f "$DATA/$ds/metadata" ]; then echo "SKIP $ds (no metadata)"; continue; fi
    echo "=== $ds  $(date '+%H:%M:%S') ==="
    python dev/v112_search.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --screen-epochs "$SE" --verify-epochs "$VE" --final-epochs "$FE" \
        --n-stages "$NST" --seed "$SEED" 2>&1 | tee "$OUT/${ds}.log"
done
echo "DONE -> $OUT  $(date)"
