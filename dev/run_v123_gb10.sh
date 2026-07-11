#!/usr/bin/env bash
# v12.3 GB10 battery -- the three open tests, sequential (~17h total):
#   T1 half-split confirmation (4h): MultNIST @2h lands in the new K=2 band
#      (~9.2s/epoch -> ~700 proj epochs). Arm a = old rule (K=1, via
#      V12_MIN_SEED_EPOCHS=99999), arm b = new default (K=2 x ~350). Same seed
#      -> matched pair; grid + A100 rows as extra reference.
#   T2 gate tie-break A/B (8h): Chesseract @2h, 2x OFF (V12_GATE_TIEBREAK=0)
#      vs 2x ON (new default: sub-noise family race -> continue top-2 arms at
#      2x slice, re-measure, adopt only if decisive). NOTE: the gate stage is
#      unseeded, so these are 4 independent rolls -- read the gate_tiebreak
#      log lines (margin vs noise, family picked) alongside the scores.
#   T3 Cryptic token-specialist excursion (5h): dev/cryptic_specialist.py,
#      3 arms (tok256 / tok384 / mixer) + surplus 2-seed ensemble.
# Run on spark:
#   cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026 && nohup bash dev/run_v123_gb10.sh > ~/local/v123_gb10/battery.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
export DATASETS_DIR=${DATASETS_DIR:-$HOME/local/nas-comp-datasets}
PY=${PY:-.venv/bin/python}
OUT=$HOME/local/v123_gb10
mkdir -p "$OUT"

echo "[v123] $(date) T3: Cryptic token-specialist (5h)"
$PY -u dev/cryptic_specialist.py \
  --dataset-dir "$DATASETS_DIR/Cryptic" \
  --out-dir "$OUT/cryptic_spec" --hours 5 2>&1 | tee "$OUT/t3_cryptic_spec.log"

echo "[v123] $(date) T1a: MultNIST @2h OLD split rule (expect K=1)"
BASELINE_LOG_DIR=$OUT/curves_t1a V12_MIN_SEED_EPOCHS=99999 \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_t1a" --time-limit 2.0 --score \
  MultNIST 2>&1 | tee "$OUT/t1a_multnist_old.log"

echo "[v123] $(date) T1b: MultNIST @2h NEW half-split default (expect K=2)"
BASELINE_LOG_DIR=$OUT/curves_t1b \
  $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
  --predictions-dir "$OUT/pred_t1b" --time-limit 2.0 --score \
  MultNIST 2>&1 | tee "$OUT/t1b_multnist_half.log"

for R in 1 2; do
  echo "[v123] $(date) T2 rep=$R tie-break OFF: Chesseract @2h"
  BASELINE_LOG_DIR=$OUT/curves_t2_r${R}_off V12_GATE_TIEBREAK=0 \
    $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir "$OUT/pred_t2_r${R}_off" --time-limit 2.0 --score \
    Chesseract 2>&1 | tee "$OUT/t2_r${R}_off.log"

  echo "[v123] $(date) T2 rep=$R tie-break ON: Chesseract @2h"
  BASELINE_LOG_DIR=$OUT/curves_t2_r${R}_on \
    $PY -u dev/mock_harness.py --submission-dir baselines/v12_1 \
    --predictions-dir "$OUT/pred_t2_r${R}_on" --time-limit 2.0 --score \
    Chesseract 2>&1 | tee "$OUT/t2_r${R}_on.log"
done

echo "[v123] $(date) BATTERY DONE"
