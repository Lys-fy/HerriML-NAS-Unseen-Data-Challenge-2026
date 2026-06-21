#!/usr/bin/env bash
# Set up the env for the in-depth per-stage macro search on the LUIS/HPC cluster.
#
# IMPORTANT: do NOT use conda here -- it writes cache into the quota'd HOME and
# fails with "Disk quota exceeded". Use a venv on /bigwork, ideally on top of a
# prebuilt PyTorch module. Mirrors the validated Spark env (torch 2.11, py3.12).
set -e

# Put the venv where there is FILE-COUNT headroom: /bigwork is near its 1M-file
# limit, /project has a 10M quota. A torch venv is ~40k small files. Override
# VENV/PIP_CACHE_DIR to taste (e.g. a personal subdir under /project).
VENV=${VENV:-/project/NHWP25179/$USER/nas-venv}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/project/NHWP25179/$USER/.pip-cache}

# --- 0. free HOME if it is full (conda/pip junk lives there) ------------------
rm -rf ~/.cache/pip 2>/dev/null || true

# --- 1. prefer a prebuilt PyTorch module (zero install, zero quota) -----------
# Inspect first:  module spider PyTorch   |   module avail
if module load PyTorch 2>/dev/null; then
    echo "loaded PyTorch module: $(python -c 'import torch; print(torch.__version__)')"
    python -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    pip install --no-cache-dir scikit-learn      # numpy usually already in module
else
    # --- 2. fallback: venv + pip, everything on /bigwork ----------------------
    module load Python/3.12 2>/dev/null || module load Python 2>/dev/null || true
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    # match the A100 CUDA (check nvidia-smi); cu128 = our dev env
    pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    pip install --no-cache-dir -r requirements.txt
fi

# --- 3. verify ----------------------------------------------------------------
python -c "import torch, numpy, sklearn; print('torch', torch.__version__, '| cuda-build', torch.version.cuda)"
echo "venv ready at $VENV  (activate with: source $VENV/bin/activate)"
echo "NOTE: torch.cuda.is_available() is False on the login node -- check it inside a SLURM GPU job."
