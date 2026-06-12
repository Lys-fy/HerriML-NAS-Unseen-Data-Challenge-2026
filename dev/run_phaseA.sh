#!/bin/bash
# Phase A: decision-harness tests (see v10 - Design + Idea Ledger note).
# Tests the DECISION machinery only, no full trainings. Auto-queued after
# v9 seed 42. Total ~7h GPU.
#
#  A1 backbone-gate arms: conv / tcn1d / tcn2d short-train (~7 min each,
#     matched budget, aug off) on all 13 datasets -> curves for the gate
#     analysis (abstain-to-default thresholds, LOO, regret).
#  A4 TSE-in-search: 4 depth-sensitive datasets, conv depth search with
#     TSE_LOG lines (per-candidate epoch losses) -> offline TSE-vs-valacc
#     ranking comparison against known-best depths.
#  A3 aug pre-filter: zero-cost proxies (naswot+snip) on aug vs clean
#     minibatches vs the probe's logged decisions. Seconds per dataset.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/tcn_probe
LOG=dev/logs/phaseA; mkdir -p "$LOG/gate" "$LOG/tse"

echo "=== PHASE A start $(date -Iseconds) ==="
echo "--- A1: backbone-gate arms (13 ds x 3 arms x 0.12h) ---"
for ds in AddNIST CIFARTile Chesseract Cryptic GameOfLife GeoClassing Gutenberg Language MultNIST Myofibre Sudoku Voxel Windspeed; do
  for arm in conv tcn1d tcn2d; do
    tag="${ds}_${arm}"; [ -f "$LOG/gate/$tag.done" ] && continue
    case "$arm" in
      conv)  EXTRA="EGNAS_OFF=1" ;;
      *)     EXTRA="PROBE_ARCH=$arm" ;;
    esac
    echo "-- A1 $tag --  $(date -Iseconds)"
    env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/gate/$tag" \
        TIER1_AUG_OFF=1 TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        $EXTRA \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-phaseA-$tag" \
            --time-limit 0.12 --score "$ds" > "$LOG/gate/$tag.log" 2>&1
    touch "$LOG/gate/$tag.done"
  done
done

echo "--- A4: TSE-in-search (4 ds x 0.5h conv depth search) ---"
for ds in GameOfLife Sudoku AddNIST Language; do
  tag="tse_${ds}"; [ -f "$LOG/tse/$tag.done" ] && continue
  echo "-- A4 $tag --  $(date -Iseconds)"
  env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/tse/$tag" P1_PROBE=0 \
      VERIFY_MACRO=0 EGNAS_MICRO_OFF=1 \
      EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
      "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
          --predictions-dir "dev/predictions-phaseA-$tag" \
          --time-limit 0.5 --score "$ds" > "$LOG/tse/$tag.log" 2>&1
  touch "$LOG/tse/$tag.done"
done

echo "--- A3: aug pre-filter (proxies vs probe decisions) ---"
"$VENV" -u dev/test_aug_prefilter.py > "$LOG/aug_prefilter.log" 2>&1 || true
tail -16 "$LOG/aug_prefilter.log"

echo "--- A5: regime retests (old verdicts re-checked on the NEW backbone) ---"
# A5a: adaptive stem x dilated backbone (stem was validated on conv only).
mkdir -p "$LOG/retest"
for ds in Voxel Chesseract; do
  for st in conv3 channel_embed; do
    tag="stem_${ds}_${st}"; [ -f "$LOG/retest/$tag.done" ] && continue
    echo "-- A5a $tag --  $(date -Iseconds)"
    env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/retest/$tag" P1_PROBE=1 \
        PROBE_ARCH=tcn2d PROBE_TCN_STEM="$st" \
        TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-phaseA-$tag" \
            --time-limit 0.5 --score "$ds" > "$LOG/retest/$tag.log" 2>&1
    touch "$LOG/retest/$tag.done"
    printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/retest/$tag.log"|head -1
  done
done
# A5b: does the aug-probe's Language decision (aug OFF, decided with CONV
# arms) still hold for the sequence backbone? tcn1d converges in ~7 min,
# so 0.5h is plenty.
for aug in off on; do
  tag="aug_Language_tcn1d_${aug}"; [ -f "$LOG/retest/$tag.done" ] && continue
  if [ "$aug" = off ]; then AUGENV="TIER1_AUG_OFF=1"; else AUGENV="TIER1_AUG_OFF=0"; fi
  echo "-- A5b $tag --  $(date -Iseconds)"
  env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/retest/$tag" \
      PROBE_ARCH=tcn1d $AUGENV \
      TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
      EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
      "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
          --predictions-dir "dev/predictions-phaseA-$tag" \
          --time-limit 0.5 --score Language > "$LOG/retest/$tag.log" 2>&1
  touch "$LOG/retest/$tag.done"
  printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/retest/$tag.log"|head -1
done

echo "=== PHASE A done $(date -Iseconds) ==="
