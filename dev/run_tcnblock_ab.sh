#!/bin/bash
# TCN block A/B (low-priority ledger item, queued before v10): does the
# canonical TCN residual block (Bai et al. 2018 via darts TCNModel: weight
# norm + in-block dropout, no BN) or a GELU variant beat our BN+ReLU block?
# Fixed tcn2d arch (D=6 W=48), 3 datasets where the dilated backbone is the
# deployed winner, half budget (1h), seed 42.
# Predeclared bar: a variant must beat ours (bn) beyond noise (>=0.5 raw pp)
# on >=2 of 3 datasets, else we keep ours.
set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026
VENV=.venv/bin/python; SUB=baselines/fixed_arch_probe
LOG=dev/logs/tcnblock_ab; mkdir -p "$LOG"
for ds in CIFARTile Myofibre Language; do
  for blk in bn canonical gelu; do
    tag="${ds}_${blk}"; [ -f "$LOG/$tag.done" ] && continue
    echo; echo "-- block A/B $tag --  $(date -Iseconds)"
    env BASELINE_SEED=42 BASELINE_LOG_DIR="$LOG/$tag" P1_PROBE=1 \
        PROBE_ARCH=tcn2d PROBE_TCN_BLOCK="$blk" \
        TRAIN_PATIENCE=999 TRAIN_MAX_EPOCHS=600 \
        EXTEND_CLOCK_FOR_NAS=0 VERIFY_MICRO_ACCEPTS=0 COMPUTE_ZEROSHOT=0 \
        "$VENV" -u dev/mock_harness.py --submission-dir "$SUB" \
            --predictions-dir "dev/predictions-tcnblock-$tag" \
            --time-limit 1.0 --score "$ds" > "$LOG/$tag.log" 2>&1
    touch "$LOG/$tag.done"
    printf '   %s ' "$tag"; grep -hoE 'raw=[0-9.]+%' "$LOG/$tag.log"|head -1
  done
done
echo; echo "=== TCN BLOCK A/B SUMMARY (raw %, bn | canonical | gelu) ==="
for ds in CIFARTile Myofibre Language; do
  a=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_bn.log" 2>/dev/null|head -1|cut -d= -f2)
  b=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_canonical.log" 2>/dev/null|head -1|cut -d= -f2)
  c=$(grep -hoE 'raw=[0-9.]+' "$LOG/${ds}_gelu.log" 2>/dev/null|head -1|cut -d= -f2)
  printf '  %-10s bn=%-8s canonical=%-8s gelu=%-8s\n' "$ds" "${a:-...}" "${b:-...}" "${c:-...}"
done
echo "=== TCN block A/B done $(date -Iseconds) ==="
