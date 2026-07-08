#!/usr/bin/env bash
# Recipe / architecture-improvement screen from the research scout's top-payoff
# list, run on ONE consistent, archetype-spanning 9-dataset set so every result
# is comparable. Winners graduate to a full-16 3-seed x multi-interval verify.
#   R0  default all-Conv3 baseline (skip-if-done; harvest Phase A already made 4)
#   R1  RepConv3 (RepVGG/RepViT structural-reparameterization op) vs default
#   R2  Model Soups (uniform+greedy) vs logit-ensemble vs single, each +/- flip-TTA
#   R3  recipe toggles LAST: SAM, self-distillation, TrivialAugment (env-gated)
# WAITS for the harvest queue to free the GPU first. Detach:
#   setsid bash dev/queue_recipes.sh > ~/local/recipes.log 2>&1 < /dev/null &
set -u
cd "$HOME/projects/HerriML-NAS-Unseen-Data-Challenge-2026" || exit 1
source .venv/bin/activate
export BDH_COMPILE=${BDH_COMPILE:-0}
DATA="$HOME/local/nas-comp-datasets"
OUT="$HOME/local/harvest"; mkdir -p "$OUT"
HOURS=${HOURS:-1.5}
SOUP_H=${SOUP_H:-0.5}; SOUP_K=${SOUP_K:-3}
# canonical archetype-spanning screen: board / reasoning / sequence / satellite /
# hard-natural / arithmetic / deterministic-CA / scientific(fibre) / scientific(wind)
SCREEN=(${SCREEN:-Chesseract Sudoku Language GeoClassing CIFARTile AddNIST GameOfLife Myofibre Windspeed})

code_of() { python -c "import json;print(json.load(open('$DATA/$1/metadata')).get('codename','$1'))" 2>/dev/null; }

echo "[recipes] $(date) waiting for the harvest queue / GPU to free..."
while pgrep -f "queue_harvest|harvest_ab.py|bdh_excursion.py|v112_search|validate_winner" >/dev/null 2>&1; do
    sleep 60
done
echo "[recipes] $(date) GPU free -- starting.  screen=${SCREEN[*]}"
python -c "import torch;print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "[recipes] SMOKE (RepConv3 fold + recipe toggles)"
if ! python dev/smoke_harvest.py; then echo "[recipes] SMOKE (harvest) FAILED -- abort"; exit 1; fi
if ! python dev/smoke_recipes.py; then echo "[recipes] SMOKE (recipes) FAILED -- abort"; exit 1; fi

echo "[recipes] $(date) R0: default baselines (${HOURS}h; skip-if-done)"
for ds in "${SCREEN[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds"; continue; }
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_ab_default.json" ] && { echo "SKIP $ds/default ($code done)"; continue; }
    echo "=== default $ds  $(date '+%H:%M:%S') ==="
    python dev/harvest_ab.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --hours "$HOURS" --tag default 2>&1 | tee "$OUT/${ds}_ab_default.log"
done

echo "[recipes] $(date) R0b: CIFARTile DEPTH ladder, BOTH families (the ~19pp open question: v10 73.7 vs PC-DARTS d20 92.3)"
# conv d6 baseline comes from R0; tcn2d gets its own d6 baseline arm.
# variant table: "tag|family|blocks"
DEPTH_LADDER=("deep12|cnn|12" "deep20|cnn|20" \
              "tcn2d_d6|tcn2d|6" "tcn2d_d12|tcn2d|12" "tcn2d_d20|tcn2d|20")
for v in "${DEPTH_LADDER[@]}"; do
    IFS="|" read -r tag fam d <<< "$v"
    code=$(code_of CIFARTile)
    [ -f "$OUT/${code}_ab_${tag}.json" ] && { echo "SKIP ${tag} ($code done)"; continue; }
    echo "=== CIFARTile ${tag} (${fam} ${d}L)  $(date '+%H:%M:%S') ==="
    python dev/harvest_ab.py --dataset-dir "$DATA/CIFARTile" --out-dir "$OUT" \
        --hours "$HOURS" --family "$fam" --blocks "$d" --tag "$tag" 2>&1 | tee "$OUT/CIFARTile_ab_${tag}.log"
done

echo "[recipes] $(date) R1: RepConv3 A/B (${HOURS}h)"
for ds in "${SCREEN[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds"; continue; }
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_ab_RepConv3.json" ] && { echo "SKIP $ds/RepConv3 ($code done)"; continue; }
    echo "=== RepConv3 $ds  $(date '+%H:%M:%S') ==="
    python dev/harvest_ab.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --hours "$HOURS" --tag RepConv3 --inject RepConv3 2>&1 | tee "$OUT/${ds}_ab_RepConv3.log"
done

echo "[recipes] $(date) R2: Model-Soups/ensemble/TTA (K=${SOUP_K} x ${SOUP_H}h)"
for ds in "${SCREEN[@]}"; do
    [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds"; continue; }
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_soup.json" ] && { echo "SKIP $ds/soup ($code done)"; continue; }
    echo "=== SOUP $ds  $(date '+%H:%M:%S') ==="
    python dev/soup_ab.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --hours-per-seed "$SOUP_H" --k "$SOUP_K" 2>&1 | tee "$OUT/${ds}_soup.log"
done

echo "[recipes] $(date) R3: recipe toggles LAST -- SAM / self-distill / TrivialAug / AdamW (${HOURS}h)"
declare -A R3_ENV=( [SAM]="TRAIN_SAM=1" [SelfDistill]="TRAIN_SELF_DISTILL=1" [TrivialAug]="TRAIN_TRIVIAL_AUG=1" [AdamW]="TRAIN_ADAMW=1" )
for tag in SAM SelfDistill TrivialAug AdamW; do
    for ds in "${SCREEN[@]}"; do
        [ -f "$DATA/$ds/metadata" ] || { echo "SKIP $ds"; continue; }
        code=$(code_of "$ds")
        [ -f "$OUT/${code}_ab_${tag}.json" ] && { echo "SKIP $ds/$tag ($code done)"; continue; }
        echo "=== $tag $ds  $(date '+%H:%M:%S') ==="
        # shellcheck disable=SC2086
        env ${R3_ENV[$tag]} python dev/harvest_ab.py --dataset-dir "$DATA/$ds" \
            --out-dir "$OUT" --hours "$HOURS" --tag "$tag" 2>&1 | tee "$OUT/${ds}_ab_${tag}.log"
    done
done

echo "[recipes] $(date) R4: budget-scaling probe -- default arch at 4h (does train-to-budget saturate? informs the Phase-3 ~8h/dataset surplus policy)"
for ds in Chesseract CIFARTile; do
    code=$(code_of "$ds")
    [ -f "$OUT/${code}_ab_long4h.json" ] && { echo "SKIP $ds/long4h ($code done)"; continue; }
    echo "=== long4h $ds  $(date '+%H:%M:%S') ==="
    python dev/harvest_ab.py --dataset-dir "$DATA/$ds" --out-dir "$OUT" \
        --hours 4.0 --tag long4h 2>&1 | tee "$OUT/${ds}_ab_long4h.log"
done
echo "[recipes] $(date) RECIPE QUEUE DONE"
