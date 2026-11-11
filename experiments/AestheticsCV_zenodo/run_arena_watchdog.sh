#!/usr/bin/env bash
# Watchdog: run the resilient feature extractor, restart it past any stall, until done.
# Guarantees a SINGLE python process per pass (prevents GPU-OOM cascades from overlap).
set -u
cd "C:/Users/yanyl/Downloads/AestheticsCV"
PY="C:/Users/yanyl/miniconda3/envs/aesthetics/python.exe"
OUT="features_arena.csv"
STALL_LIMIT=120      # seconds of no new rows -> assume hung
POLL=20

wait_no_python () {   # block until no python.exe is running (max ~40s)
  for _ in $(seq 1 20); do
    if ! tasklist 2>/dev/null | grep -qi python.exe; then return 0; fi
    taskkill //IM python.exe //F >/dev/null 2>&1
    sleep 2
  done
}

for pass in $(seq 1 80); do
  wait_no_python
  echo "===== PASS $pass ($(date +%H:%M:%S)) ====="
  "$PY" pipeline/run_resilient.py --manifest stimuli_manifest_arena.csv \
        --out "$OUT" --reference control_embeddings.npz > "arena_pass.log" 2>&1 &
  PID=$!
  last=$( (wc -l < "$OUT") 2>/dev/null || echo 0)
  stall=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL"
    now=$( (wc -l < "$OUT") 2>/dev/null || echo 0)
    if [ "$now" -gt "$last" ]; then last="$now"; stall=0; else stall=$((stall+POLL)); fi
    if [ "$stall" -ge "$STALL_LIMIT" ]; then
      echo "  !! STALL ($stall s, rows=$now) -> killing"
      taskkill //IM python.exe //F >/dev/null 2>&1
      kill -9 "$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null
  if grep -q "nothing to do" arena_pass.log; then
    echo "ALL DONE at pass $pass, rows=$(wc -l < "$OUT")"
    break
  fi
  echo "  pass $pass ended, rows=$(wc -l < "$OUT")"
done
wait_no_python
echo "WATCHDOG FINISHED ($(date +%H:%M:%S)) rows=$(wc -l < "$OUT")"
