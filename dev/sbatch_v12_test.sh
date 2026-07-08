#!/usr/bin/env bash
#SBATCH --job-name=v12-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1            # select the GPU type via --constraint or partition, see below
#SBATCH --time=12:00:00
#SBATCH --output=v12test_%j.log
#
# v12 package test on the LUIS cluster (V100 / A100). Submit variants:
#   sbatch --partition=gpu --gres=gpu:v100:1 dev/sbatch_v12_test.sh
#   sbatch --partition=gpu --gres=gpu:a100:1 dev/sbatch_v12_test.sh
# (check `sinfo -o "%P %G"` for the exact partition/gres names on your account)
#
# Env overrides:
#   REPO      repo checkout (default: submit dir)
#   VENV      venv path (default matches dev/setup_cluster_env.sh)
#   DATASETS_DIR  dataset root containing <Name>/{train_x.npy,...,metadata}
#   TL        per-dataset time limit in HOURS (default 2.0)
#   SETS      dataset names (default: the 4-set comparison tier)
set -u
REPO=${REPO:-$SLURM_SUBMIT_DIR}
VENV=${VENV:-/project/NHWP25179/$USER/nas-venv}
export DATASETS_DIR=${DATASETS_DIR:-/project/NHWP25179/$USER/nas-comp-datasets}
TL=${TL:-2.0}
SETS=${SETS:-"Chesseract Language CIFARTile Sudoku"}

cd "$REPO"
source "$VENV/bin/activate"
export BASELINE_LOG_DIR="$REPO/dev/logs/v12val_${SLURM_JOB_ID}"
mkdir -p "$BASELINE_LOG_DIR"

echo "== node $(hostname), GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF

echo "== v12 smoke"
python dev/smoke_v12.py || exit 1

echo "== mock harness: v12 @ ${TL}h/dataset on: $SETS"
# shellcheck disable=SC2086
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit "$TL" --score $SETS
echo "== DONE (curves in $BASELINE_LOG_DIR, predictions in dev/predictions/)"
