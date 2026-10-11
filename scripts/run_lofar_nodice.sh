#!/usr/bin/env bash
# Dice-off on LOFAR (PARTS 18.4, 19.2, 20.2): is the CE + Dice loss what puts the
# hybrid ~0.11 above tf_unet?
#
# Same configuration as the headline hybrid (PART 13.11: base 8, fixed norm, no class
# weight, 28,000 steps, seeds 0-2, identical train/val splits); the ONLY difference is
# --dice_weight 0, i.e. plain cross-entropy as in tf_unet. The flag already exists in
# experiments/lofar_hybrid.py, so no code changes.
# Baseline: max F1 0.6603 +/- 0.0040 (runs/lofar/hybrid_b08_nocw_seed*).
#
#   ./scripts/run_lofar_nodice.sh            # seeds 0 1 2 (~1.5-2 h on AC)
#   ./scripts/run_lofar_nodice.sh 0          # one seed (~30 min)
#
# Resumable: rerun the same command; finished seeds are skipped and an interrupted
# seed continues from its progress.json.
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
for seed in "${SEEDS[@]}"; do
    out="runs/lofar/nodice_b08_nocw_seed${seed}"
    if [ -f "$out/eval_test/metrics.json" ]; then echo "-- skip seed $seed (done)"; continue; fi
    # PyTorch occasionally aborts with "getCount is non-monotonic" (PART 25.5): retry,
    # and the run resumes from its progress.json.
    for attempt in 1 2 3; do
        echo; echo "-- Dice off, seed $seed, attempt $attempt -> $out   ($(date +%H:%M))"
        if "$PY" experiments/lofar_hybrid.py --base 8 --norm fixed --no_class_weight \
               --dice_weight 0 --seed "$seed" --output_dir "$out"; then break; fi
        if [ "$attempt" = 3 ]; then echo "seed $seed failed 3 times -- stopping" >&2; exit 1; fi
        if ! on_ac; then echo "stopped, and the laptop is on battery: plug in and rerun this script" >&2; exit 1; fi
        echo "-- stopped unexpectedly; resuming in 20 s"
        sleep 20
    done
done

echo; echo "All done ($(date +%H:%M)). Results:"
"$PY" analysis/summarise_nodice.py
