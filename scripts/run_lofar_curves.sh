#!/usr/bin/env bash
# Learning curves on LOFAR (PART 41): repeat the two headline runs with per-epoch logging.
#
# Same settings as the reported runs, plus --log_curves (logging only, training unchanged):
#   Kestrel : runs/lofar/hybrid_b08_nocw_seed<S>  (base 8, fixed norm, no class weight,
#             CE + Dice, lr 1e-3 cosine, 40 x 700 steps, batch 1)
#   tf_unet : runs/lofar/lr1e-4_seed<S>           (Adam 1e-4, 150 x 175 steps, batch 4)
# Each writes <run>/curves.csv. These are NEW runs: their final scores are reported as
# measured, not swapped for the original ones.
#
#   ./scripts/run_lofar_curves.sh        # about 1.4 h (Kestrel) + 4 h (tf_unet) on AC
#
# Resumable: rerun the same command; finished runs are skipped, an interrupted run
# continues from its progress.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TORCH="$HOME/torch-env/bin/python"
TF="$HOME/tf-env/bin/python"

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

run() {   # run <out_dir> <command...>   with 3 attempts (PyTorch getCount abort, PART 25.5)
    local out="$1"; shift
    if [ -f "$out/eval_test/metrics.json" ]; then echo "-- skip $out (done)"; return 0; fi
    for attempt in 1 2 3; do
        echo; echo "-- $out, attempt $attempt   ($(date '+%Y-%m-%d %H:%M'))"
        if "$@" --output_dir "$out"; then echo "   finished $(date +%H:%M)"; return 0; fi
        if [ "$attempt" = 3 ]; then echo "$out failed 3 times -- stopping" >&2; exit 1; fi
        if ! on_ac; then echo "stopped, and the laptop is on battery: plug in and rerun" >&2; exit 1; fi
        echo "-- stopped unexpectedly; resuming in 20 s"; sleep 20
    done
}

for s in 0 1 2; do
    run "runs/lofar/curves_hybrid_b08_nocw_seed$s" "$TORCH" experiments/lofar_hybrid.py \
        --base 8 --norm fixed --no_class_weight --seed "$s" --log_curves
done

export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
for s in 0 1 2; do
    run "runs/lofar/curves_tfunet_lr1e-4_seed$s" "$TF" experiments/lofar_tfunet_baseline.py \
        --norm fixed --lr 1e-4 --epochs 150 --seed "$s" --log_curves
done
echo; echo "ALL DONE $(date '+%Y-%m-%d %H:%M')"
