#!/bin/bash
# Queued after the random-arm validation. Three experiments, seed 42, 2h each:
#  (1) architecture gauntlet: attn / recur / modern fixed archs x 4 datasets
#  (2) ensemble probe: conv+tcn2d+tcn1d logit-average x 2 datasets
#  (3) verified width-search: v10 with the systematic WIDTH_SEARCH stage
#      (default-incumbent W=48 vs {32,64}, short-train + verify before adopt --
#      the protected version of "search width again"). x 4 encoded datasets.
# ~36h total; GPU stays busy. Self-contained (self-cd).
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python
SEED=42
MARK=dev/logs/extensions.marker
echo "extensions sweep starting $(date -Iseconds)" >> "$MARK"

ARCH_DS=(Chesseract Sudoku Language CIFARTile)
for arch in attn recur modern; do
    ld="dev/logs/arch_probe/$arch"; pd="dev/predictions-arch_probe-$arch"
    mkdir -p "$ld" "$pd"
    env BASELINE_SEED=$SEED BASELINE_LOG_DIR="$ld" PROBE_ARCH=$arch \
        "$VENV" -u dev/mock_harness.py --submission-dir baselines/arch_probe \
            --predictions-dir "$pd" --time-limit 2.0 --score "${ARCH_DS[@]}" > "$ld/run.log" 2>&1
    echo "arch_probe[$arch] done $(date -Iseconds)" >> "$MARK"
done

ENS_DS=(Chesseract Sudoku)
mkdir -p dev/logs/ensemble_probe dev/predictions-ensemble_probe
env BASELINE_SEED=$SEED BASELINE_LOG_DIR=dev/logs/ensemble_probe \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/ensemble_probe \
        --predictions-dir dev/predictions-ensemble_probe --time-limit 2.0 --score "${ENS_DS[@]}" > dev/logs/ensemble_probe/run.log 2>&1
echo "ensemble_probe done $(date -Iseconds)" >> "$MARK"

WIDTH_DS=(Chesseract Sudoku Gutenberg Language)
mkdir -p dev/logs/width_search dev/predictions-width_search
env BASELINE_SEED=$SEED BASELINE_LOG_DIR=dev/logs/width_search WIDTH_SEARCH=1 \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/v10 \
        --predictions-dir dev/predictions-width_search --time-limit 2.0 --score "${WIDTH_DS[@]}" > dev/logs/width_search/run.log 2>&1
echo "width_search done $(date -Iseconds)" >> "$MARK"

# (4) macro-ONLY search on ALL 13 datasets: baselines/macro_probe always trains
#     the per-stage macro arch (uniform DilatedConv3 incumbent vs per-stage swaps
#     to {SpatialSelfAttn2D, SEConv3, Conv5}, verified). Isolates the macro
#     search's own standalone performance -- NO family-default fallback.
MACRO_DS=(AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed)
mkdir -p dev/logs/macro_probe dev/predictions-macro_probe
env BASELINE_SEED=$SEED BASELINE_LOG_DIR=dev/logs/macro_probe \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/macro_probe \
        --predictions-dir dev/predictions-macro_probe --time-limit 2.0 --score "${MACRO_DS[@]}" > dev/logs/macro_probe/run.log 2>&1
echo "macro_probe (all 13) done $(date -Iseconds)" >> "$MARK"
echo "extensions sweep done $(date -Iseconds)" >> "$MARK"
