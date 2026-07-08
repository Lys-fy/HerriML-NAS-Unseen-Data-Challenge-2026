# LUIS cluster: conda env + interactive v12 runs

Conda variant of the env setup (see `dev/setup_cluster_env.sh` for the venv
variant). CRITICAL on LUIS: conda writes package caches into the quota'd HOME
by default and dies with "Disk quota exceeded" -- point EVERYTHING at /project
first.

## 1. One-time: create the env (login node)

```bash
module load Miniforge3            # or: module spider conda / Miniconda3
# keep conda off the HOME quota:
export CONDA_PKGS_DIRS=/project/NHWP25179/$USER/.conda-pkgs
mkdir -p "$CONDA_PKGS_DIRS"

conda create -y -p /project/NHWP25179/$USER/conda/nas python=3.12
conda activate /project/NHWP25179/$USER/conda/nas

# torch separately, CUDA-matched (cu128 = the validated Spark build; covers
# V100 through B300 on torch 2.11):
pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r requirements.txt

python -c "import torch, numpy, sklearn; print(torch.__version__, numpy.__version__, sklearn.__version__)"
# torch.cuda.is_available() is False on the LOGIN node -- verify inside the job.
```

## 2. Every session: screen + salloc + run

```bash
screen -S nas                      # detachable session (Ctrl-a d / screen -r nas)

# pick a GPU (all partitions AllowAccounts=ALL; gpu MaxTime 2d, ai* 8d8h):
salloc --partition=gpu     --gres=gpu:v100:1 --cpus-per-task=8 --mem=48G --time=16:00:00
salloc --partition=gpu     --gres=gpu:a100:1 --cpus-per-task=8 --mem=48G --time=16:00:00
salloc --partition=ainlp   --gres=gpu:a100:1 --cpus-per-task=8 --mem=48G --time=30:00:00
salloc --partition=ai.h100 --gres=gpu:h100:1 --cpus-per-task=8 --mem=48G --time=30:00:00
salloc --partition=ai.b300 --gres=gpu:b300:1 --cpus-per-task=8 --mem=64G --time=30:00:00

# if salloc leaves you on the login node, hop onto the allocation:
srun --pty bash

# then, on the node:
module load Miniforge3
conda activate /project/NHWP25179/$USER/conda/nas
cd <repo>                          # git pull first for the latest v12
export DATASETS_DIR=/project/NHWP25179/$USER/nas-comp-datasets
export BASELINE_LOG_DIR=$PWD/dev/logs/v12val_$(hostname)   # optional: per-epoch curves
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python dev/smoke_v12.py            # ~1 min; MUST pass before long runs

# the actual tests (grader-mirror harness, --score prints adjusted scores):
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit 2.0 --score Chesseract Language CIFARTile Sudoku
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit 6.0 --score Chesseract CIFARTile
python dev/mock_harness.py --submission-dir baselines/v12 --time-limit 8.0 --score Chesseract CIFARTile Language
```

Reference numbers to compare against: v10-real Chesseract 57.66 / CIFARTile
73.65 / Language 90.03 / Sudoku 97.70; v12 prototype @6h Chesseract 67.25 /
CIFARTile 82.36 (GB10). Watch the "v12 plan:" line in the output -- it shows
the measured s/epoch and the K x schedule the throughput-keyed planner chose,
which is where the V100 / A100 / H100 / B300 differences should appear.
