#!/usr/bin/env bash
# Topic 11 (PART 26): expert-label fine-tuning with active patch selection, scored on ALL
# 109 expert-labelled LOFAR images by 5-fold cross-validation. Builds on the headline
# hybrid checkpoints (runs/lofar/hybrid_b08_nocw_seed*/best.pt); no existing code changes.
#
#   ./scripts/run_lofar_al.sh            # seeds 0 1 2: 195 fine-tuning runs, ~3-9 h on AC
#   ./scripts/run_lofar_al.sh 0          # seed 0 only, ~1-3 h
#
# Resumable: rerun the same command and finished pieces are skipped.
# Progress:  grep "ETA" runs/lofar/logs/al_runs.log | tail -1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$HOME/torch-env/bin/python"

on_ac() { [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" = "1" ]; }
if ! on_ac; then
    echo "REFUSING TO START: the laptop is on battery (GPU capped at 210 of 2100 MHz)." >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: another process is using the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
fi

SEEDS=(0 1 2)
[ $# -ge 1 ] && SEEDS=("$@")
mkdir -p runs/lofar/logs

# PyTorch occasionally aborts at start-up with "getCount is non-monotonic" (a clock
# glitch, PART 25.5); everything below retries, and the run resumes where it stopped.
for attempt in 1 2 3; do
    echo "-- self-test, attempt $attempt ($(date +%H:%M))"
    if "$PY" experiments/active_learning/al_lofar.py --selftest; then break; fi
    if [ "$attempt" = 3 ]; then echo "SELF-TEST FAILED -- not starting." >&2; exit 1; fi
    sleep 10
done

for attempt in 1 2 3 4 5; do
    echo; echo "-- run, attempt $attempt ($(date +%H:%M)), seeds ${SEEDS[*]}"
    if "$PY" experiments/active_learning/al_lofar.py --seeds "${SEEDS[@]}"; then break; fi
    if [ "$attempt" = 5 ]; then echo "giving up after 5 attempts" >&2; exit 1; fi
    if ! on_ac; then echo "stopped, and the laptop is on battery: plug in and rerun this script" >&2; exit 1; fi
    echo "-- stopped unexpectedly; resuming in 20 s (finished pieces are kept)"
    sleep 20
done

echo; echo "All done ($(date +%H:%M)). Results:"
"$PY" analysis/active_learning/summarise_al.py
