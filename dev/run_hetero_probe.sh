#!/usr/bin/env bash
# v11.2 overnight probe: heterogeneous multi-fidelity NAS on the probe datasets.
# Self-cd + self-activate. Override the knobs via env (N / R0 / R1 / K).
#   N=16 R0=12 R1=120 K=3 bash dev/run_hetero_probe.sh [dataset names...]
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
OUT="$HOME/local/hetero_probe"; mkdir -p "$OUT"
DATA="$HOME/local/nas-comp-datasets"
N=${N:-16}; R0=${R0:-12}; R1=${R1:-120}; K=${K:-3}; SEED=${SEED:-42}

SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(CIFARTile Chesseract Gutenberg)
echo "hetero probe: ${SETS[*]}  N=$N rung0=$R0 rung1=$R1 k=$K  out=$OUT"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

for ds in "${SETS[@]}"; do
    if [ ! -f "$DATA/$ds/metadata" ]; then
        echo "SKIP $ds (no metadata at $DATA/$ds)"; continue
    fi
    echo "=== $ds  $(date '+%H:%M:%S') ==="
    python dev/hetero_search_probe.py \
        --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --n-candidates "$N" --rung0-epochs "$R0" --rung1-epochs "$R1" \
        --shortlist "$K" --seed "$SEED" \
        2>&1 | tee "$OUT/${ds}.log"
done
echo "DONE -> $OUT  (bring the *.json + *_curves.npz + *.log back)  $(date)"
