#!/bin/bash
# Unattended gated launcher for the Tier-2b FULL EGNAS sweep.
#
# User wants the ~30 h sweep running overnight (checks tomorrow), but the
# full-EGNAS code (micro phase) has only been syntax-checked, never run.
# So: wait for the already-running ~12-min correctness smoke, check it
# passed, and ONLY THEN launch the full 11-ds × 3-seed sweep. If the smoke
# failed, do NOT launch — leave a clear message instead of burning 30 h on
# broken code. Everything nohup'd; survives disconnect.
#
# Status to check tomorrow:
#   tail -n 30 dev/logs/egnasfull-orchestrator.log
#   (then the usual sweep live-status on dev/logs/egnas_full-all.log)

set -uo pipefail
cd ~/projects/HerriML-NAS-Unseen-Data-Challenge-2026

SMOKE_LOG=dev/logs/egnasfullsmoke.log
SMOKE_PRED=dev/predictions-egnasfullsmoke

echo "[orch $(date -Iseconds)] waiting for smoke to finish..."
while pgrep -f "predictions-egnasfullsmoke" >/dev/null 2>&1; do sleep 20; done
echo "[orch $(date -Iseconds)] smoke process exited — evaluating gate"

fail=0
if grep -qE "Traceback|Exception while running|EGNAS macro FAILED" "$SMOKE_LOG"; then
    echo "[orch] FAIL: traceback/exception in smoke log"; fail=1
fi
if ! grep -q "Total score" "$SMOKE_LOG"; then
    echo "[orch] FAIL: smoke did not reach scoring"; fail=1
fi
if [ ! -f "$SMOKE_PRED/Chester.npy" ] || [ ! -f "$SMOKE_PRED/Sokoto.npy" ]; then
    echo "[orch] FAIL: smoke predictions missing"; fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "[orch $(date -Iseconds)] ===== SMOKE FAILED — FULL SWEEP NOT LAUNCHED ====="
    echo "[orch] inspect $SMOKE_LOG ; relevant lines:"
    grep -nE "Traceback|Exception while running|FAILED|micro FAILED" "$SMOKE_LOG" | tail -25
    exit 1
fi

echo "[orch $(date -Iseconds)] ===== SMOKE PASSED ====="
grep -E "EGNAS full done|micro L|raw=.*adj=|Total score" "$SMOKE_LOG" | tail -25
echo "[orch $(date -Iseconds)] launching FULL 11-ds × 3-seed sweep (foreground of this nohup'd orchestrator)"
exec bash dev/run_egnas_full.sh
