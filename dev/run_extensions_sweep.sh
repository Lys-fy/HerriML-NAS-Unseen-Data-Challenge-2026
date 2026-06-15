#!/bin/bash
# Queued after the random-arm validation. Three experiments, seed 42, 2h each:
#  (1) architecture gauntlet: attn / recur / modern fixed archs x 4 datasets
#  (2) ensemble probe: conv+tcn2d+tcn1d logit-average x 2 datasets
#  (3) micro-search expansion: v10 with a bigger random arm (RAND_K=12,
#      RAND_FRAC=0.20, wider widths) -- now safe to revisit since the arm
#      verifies before adopting. x 4 encoded datasets.
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

MICRO_DS=(Chesseract Sudoku Gutenberg Language)
mkdir -p dev/logs/micro_expand dev/predictions-micro_expand
env BASELINE_SEED=$SEED BASELINE_LOG_DIR=dev/logs/micro_expand RAND_K=12 RAND_FRAC=0.20 RAND_WIDTHS=16,24,32,48,64,96 \
    "$VENV" -u dev/mock_harness.py --submission-dir baselines/v10 \
        --predictions-dir dev/predictions-micro_expand --time-limit 2.0 --score "${MICRO_DS[@]}" > dev/logs/micro_expand/run.log 2>&1
echo "micro_expand done $(date -Iseconds)" >> "$MARK"
echo "extensions sweep done $(date -Iseconds)" >> "$MARK"
