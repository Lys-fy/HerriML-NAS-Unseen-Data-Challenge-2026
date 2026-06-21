#!/usr/bin/env bash
# Detached v11 chain on spark (GB10), run after the extension sweep frees the GPU:
#   1. wait for the macro sweep (13 macro predictions)
#   2. ResNet-18 baselines for the 3 new datasets -> set their metadata benchmark
#   3. v10 on the 3 new datasets (3 seeds, perfect-acc early stop)
#   4. in-depth per-stage macro search over all 16 datasets
# Self-cd + self-activate (detached wrappers must set up their own env).
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
OUT="$HOME/local/macro_search_out"; mkdir -p "$OUT"

echo "[chain] $(date) waiting for macro sweep (13 macro predictions) ..."
until [ "$(ls dev/predictions-macro_probe/*.npy 2>/dev/null | wc -l)" -ge 13 ]; do sleep 300; done
echo "[chain] $(date) sweep done"

echo "[chain] $(date) resnet18 baselines: Sedol Sloan Cauchy ..."
python dev/mock_harness.py --submission-dir baselines/resnet18 \
    --predictions-dir dev/predictions-resnet18 --time-limit 2.0 \
    --score Sedol Sloan Cauchy > "$OUT/resnet.log" 2>&1
python dev/set_benchmark_from_resnet.py >> "$OUT/resnet.log" 2>&1
echo "[chain] $(date) benchmarks set"

echo "[chain] $(date) v10 on new datasets (3 seeds, perfect-stop) ..."
for seed in 42 1337 2024; do
    BASELINE_SEED=$seed V10_PERFECT_STOP_S=300 python dev/mock_harness.py \
        --submission-dir baselines/v10 --predictions-dir "dev/predictions-v10new-s$seed" \
        --time-limit 2.0 --score Sedol Sloan Cauchy >> "$OUT/v10new.log" 2>&1
    echo "[chain] $(date) v10 seed $seed done"
done

echo "[chain] $(date) in-depth macro search (16 datasets) ..."
VENV="$PWD/.venv" DATA_ROOT="$HOME/local/nas-comp-datasets" \
    OUT_ROOT="$OUT" ARM_SECONDS=300 SEED=42 \
    bash dev/run_macro_search.sh > "$OUT/sweep.log" 2>&1
echo "[chain] $(date) ALL DONE"
