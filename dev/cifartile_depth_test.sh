#!/usr/bin/env bash
# Quick test: does DEPTH close the CIFARTile gap? v10 runs our backbone at
# depth ~6 (73.7% test); PC-DARTS uses depth ~20 (92.3%). Train our backbone
# (uniform DilatedConv3) at depths 6 / 12 / 20 with the cosine recipe and
# compare, to isolate depth from op/cell. Self-cd + self-activate (detached).
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
OUT="$HOME/local/cifartile_depth"; mkdir -p "$OUT"
for d in 6 12 20; do
    echo "=== depth $d  $(date '+%H:%M:%S') ==="
    python dev/macro_search_indepth.py \
        --dataset-dir "$HOME/local/nas-comp-datasets/CIFARTile" \
        --out-dir "$OUT/d$d" --depth "$d" --stages 1 --candidates DilatedConv3 \
        --arm-epochs 120 --arm-cap-seconds 3000 --seed 42
done
echo "DONE -> $OUT  $(date)"
