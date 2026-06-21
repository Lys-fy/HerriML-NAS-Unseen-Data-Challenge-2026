#!/usr/bin/env bash
# Run the in-depth per-stage macro search over all datasets (v11 development).
# Serial on one GPU by default; see the SLURM-array note at the bottom to
# parallelise one-dataset-per-GPU.
#
# Usage:
#   VENV=/project/NHWP25179/$USER/nas-venv \
#   DATA_ROOT=/bigwork/$USER/nas-comp-datasets \
#   OUT_ROOT=/bigwork/$USER/macro_search_out \
#   ARM_EPOCHS=150 \
#   bash dev/run_macro_search.sh [dataset names...]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-/project/NHWP25179/$USER/nas-venv}
DATA_ROOT=${DATA_ROOT:-/bigwork/$USER/nas-comp-datasets}
OUT_ROOT=${OUT_ROOT:-/bigwork/$USER/macro_search_out}
ARM_EPOCHS=${ARM_EPOCHS:-150}            # fixed epochs/arm, trained with cosine LR
SEED=${SEED:-42}

# default dataset set: the 13 official + the two ENCODED custom probes.
# (override by passing names on the command line.)
DEFAULT_SETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing \
              Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed \
              Sedol Sloan Cauchy)
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=("${DEFAULT_SETS[@]}")

# activate env: a loaded PyTorch module already provides torch; else the venv.
python -c "import torch" 2>/dev/null || source "$VENV/bin/activate"

mkdir -p "$OUT_ROOT"
echo "macro search: ${#SETS[@]} datasets, cosine($ARM_EPOCHS epochs/arm), out=$OUT_ROOT"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

for ds in "${SETS[@]}"; do
    dir="$DATA_ROOT/$ds"
    if [ ! -f "$dir/metadata" ]; then
        echo "SKIP $ds (no metadata at $dir)"; continue
    fi
    echo "=== $ds  $(date '+%H:%M:%S') ==="
    python "$REPO/dev/macro_search_indepth.py" \
        --dataset-dir "$dir" --out-dir "$OUT_ROOT" \
        --arm-epochs "$ARM_EPOCHS" --seed "$SEED" \
        2>&1 | tee "$OUT_ROOT/${ds}.log"
done
echo "DONE -> $OUT_ROOT  (bring the *.json + *_curves.npz + *.log back for analysis)"

# ---------------------------------------------------------------------------
# SLURM array (one dataset per GPU) -- save as a .sbatch and submit instead:
#
#   #!/bin/bash
#   #SBATCH --job-name=macro-search
#   #SBATCH --partition=<a100-partition>
#   #SBATCH --gres=gpu:1
#   #SBATCH --time=04:00:00
#   #SBATCH --array=0-14            # 15 datasets (0..14)
#   #SBATCH --output=macro_%a.out
#   SETS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing \
#         Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed Sedol Cauchy)
#   ds=${SETS[$SLURM_ARRAY_TASK_ID]}
#   source /project/NHWP25179/$USER/nas-venv/bin/activate   # or: module load PyTorch
#   python dev/macro_search_indepth.py --dataset-dir $DATA_ROOT/$ds \
#       --out-dir $OUT_ROOT --arm-epochs 150
# ---------------------------------------------------------------------------
