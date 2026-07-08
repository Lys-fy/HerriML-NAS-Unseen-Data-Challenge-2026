#!/usr/bin/env bash
# v12 extra tests (post-screen, pre-build), per the Screening Report plan:
#   T1  TrivialAug flip-ablation (fast): TA without flip_h on the 3 biggest TA
#       movers -> compare to the existing TrivialAug arms (flip in) + default.
#   T2  THE DECISIVE INTEGRATION TEST: full v11.1 pipeline + TrivialAug +
#       K=3-seed val-gated selection (ensemble +/- TTA) at 2h TOTAL on the
#       9-set -> head-to-head vs v10-real / v11.1-s42 (matched budget).
#   T3  C1/C4 in-pipeline op tests (--plain = pure v11.1 path, TA off):
#       C1 via the existing verified macro search (MACRO_OPS=LinearAttn2D),
#       C4 via V11_SHARED_REPEAT=3, on Language + Sudoku.
#   T4  rung-threshold matrix cells: K3 @ 0.5h x 3 sets; K3 @ 6h x 2 sets;
#       K1-plain @ 6h x CIFARTile. (K3@2h = T2; K1@2h = existing v11.1-s42.)
# Waits for the GPU (Chesseract long4h-v2 rerun may still be going). Detach:
#   setsid bash dev/queue_v12_tests.sh > ~/local/v12tests.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
DATA="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/v12tests"; mkdir -p "$OUT"
HARVEST="$HOME/local/harvest"
SCREEN=(Chesseract Sudoku Language GeoClassing CIFARTile AddNIST GameOfLife Myofibre Windspeed)

code_of() { python -c "import json;print(json.load(open('$DATA/$1/metadata')).get('codename','$1'))" 2>/dev/null; }

echo "[v12t] $(date) waiting for GPU (harvest_ab long4h rerun etc.)..."
while pgrep -f "harvest_ab.py|bdh_excursion.py|soup_ab.py|v12_integration.py" >/dev/null 2>&1; do sleep 60; done
echo "[v12t] $(date) GPU free"
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[v12t] SMOKE: tiny integration run (Sudoku, 0.15h, k=2)"
if ! env TRAIN_TRIVIAL_AUG=1 python dev/v12_integration.py --dataset-dir "$DATA/Sudoku" \
    --out-dir "$OUT/smoke" --hours 0.15 --k 2 --tag smoke 2>&1 | tee "$OUT/smoke.log"; then
    echo "[v12t] SMOKE FAILED -- abort"; exit 1
fi
grep -q "DONE" "$OUT/smoke.log" || { echo "[v12t] SMOKE produced no DONE -- abort"; exit 1; }

echo "[v12t] $(date) T1: TrivialAug flip-ablation (1.5h each)"
for ds in Sudoku GameOfLife Chesseract; do
    code=$(code_of "$ds")
    [ -f "$HARVEST/${code}_ab_TrivAugNoFlip.json" ] && { echo "SKIP $ds"; continue; }
    echo "=== TA-noflip $ds  $(date '+%H:%M:%S') ==="
    env TRAIN_TRIVIAL_AUG=1 TRAIN_TA_NO_FLIP=1 python dev/harvest_ab.py \
        --dataset-dir "$DATA/$ds" --out-dir "$HARVEST" --hours 1.5 \
        --tag TrivAugNoFlip 2>&1 | tee "$HARVEST/${ds}_ab_TrivAugNoFlip.log"
done

echo "[v12t] $(date) T2: DECISIVE integration test -- v11.1 + TA + K3 val-gated selection @ 2h total"
for ds in "${SCREEN[@]}"; do
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_v12.json" ] && { echo "SKIP $ds"; continue; }
    echo "=== v12 $ds  $(date '+%H:%M:%S') ==="
    env TRAIN_TRIVIAL_AUG=1 python dev/v12_integration.py --dataset-dir "$DATA/$ds" \
        --out-dir "$OUT" --hours 2.0 --k 3 --tag v12 2>&1 | tee "$OUT/${ds}_v12.log"
done

echo "[v12t] $(date) T3: C1/C4 in-pipeline op tests (--plain, TA off, 2h)"
for ds in Language Sudoku; do
    code=$(code_of "$ds")
    if [ ! -f "$OUT/${code}_c1pipe.json" ]; then
        echo "=== C1-inpipe $ds  $(date '+%H:%M:%S') ==="
        env MACRO_SEARCH=1 MACRO_OPS="LinearAttn2D" python dev/v12_integration.py \
            --dataset-dir "$DATA/$ds" --out-dir "$OUT" --hours 2.0 --plain \
            --tag c1pipe 2>&1 | tee "$OUT/${ds}_c1pipe.log"
    fi
    if [ ! -f "$OUT/${code}_c4pipe.json" ]; then
        echo "=== C4-inpipe $ds  $(date '+%H:%M:%S') ==="
        env V11_SHARED_REPEAT=3 python dev/v12_integration.py \
            --dataset-dir "$DATA/$ds" --out-dir "$OUT" --hours 2.0 --plain \
            --tag c4pipe 2>&1 | tee "$OUT/${ds}_c4pipe.log"
    fi
done

echo "[v12t] $(date) T4: rung-threshold matrix cells"
for ds in Chesseract Language CIFARTile; do
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_v12h05.json" ] && { echo "SKIP $ds h05"; continue; }
    echo "=== v12@0.5h $ds  $(date '+%H:%M:%S') ==="
    env TRAIN_TRIVIAL_AUG=1 python dev/v12_integration.py --dataset-dir "$DATA/$ds" \
        --out-dir "$OUT" --hours 0.5 --k 3 --tag v12h05 2>&1 | tee "$OUT/${ds}_v12h05.log"
done
for ds in Chesseract CIFARTile; do
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_v12h6.json" ] && { echo "SKIP $ds h6"; continue; }
    echo "=== v12@6h $ds  $(date '+%H:%M:%S') ==="
    env TRAIN_TRIVIAL_AUG=1 TRAIN_MAX_EPOCHS=2400 python dev/v12_integration.py \
        --dataset-dir "$DATA/$ds" --out-dir "$OUT" --hours 6.0 --k 3 \
        --tag v12h6 2>&1 | tee "$OUT/${ds}_v12h6.log"
done
code=$(code_of CIFARTile)
if [ ! -f "$OUT/${code}_plainh6.json" ]; then
    echo "=== plain@6h CIFARTile  $(date '+%H:%M:%S') ==="
    env TRAIN_MAX_EPOCHS=2400 python dev/v12_integration.py --dataset-dir "$DATA/CIFARTile" \
        --out-dir "$OUT" --hours 6.0 --plain --tag plainh6 2>&1 | tee "$OUT/CIFARTile_plainh6.log"
fi

echo "[v12t] $(date) V12 TEST QUEUE DONE"
