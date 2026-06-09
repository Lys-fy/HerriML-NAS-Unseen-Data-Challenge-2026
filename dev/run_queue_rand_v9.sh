#!/bin/bash
# Sequential queue: random-search arm first (3 stuck datasets x 3 seeds, ~18h),
# then the v9 dilated-backbone 3-seed sweep (~3 days). Each sub-script blocks
# until its own loop finishes, so plain sequential calls are enough.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
echo "=== QUEUE START $(date -Iseconds) ==="
echo "--- [1/2] random search ---"
bash dev/run_random_search.sh
echo "--- [2/2] v9 dilated-backbone 3-seed sweep ---"
bash dev/run_tier2a_p1_v9.sh
echo "=== QUEUE DONE $(date -Iseconds) ==="
