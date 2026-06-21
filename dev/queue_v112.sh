#!/usr/bin/env bash
# Wait for the CIFARTile depth test to finish, then run v11.2 on the freed GPU
# (clean timings, no contention). Self-cd. Detach with setsid.
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
echo "[queue] $(date) waiting for depth test (d20/Caitie.json) ..."
until [ -f "$HOME/local/cifartile_depth/d20/Caitie.json" ]; do sleep 180; done
echo "[queue] $(date) depth test done -> launching v11.2"
bash dev/run_v112.sh Chesseract Gutenberg CIFARTile
echo "[queue] $(date) v11.2 done"
