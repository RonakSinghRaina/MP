#!/usr/bin/env bash
# Edge-aware training on LOFAR (PART 22.4).
#
# Same configuration as the headline hybrid (PART 13.11: base 8, fixed norm,
# no class weight, 28,000 steps, seeds 0-2, identical train/val splits); the
# ONLY difference is how the 1-pixel rim of each AOFlagger mask is treated.
# Baseline to beat: max F1 0.6603 +/- 0.0040 (runs/lofar/hybrid_b08_nocw_seed*).
#
#   ./scripts/run_lofar_edge.sh              # svls and ignore, seeds 0 1 2 (~2.5 h)
#   ./scripts/run_lofar_edge.sh svls         # one mode only, seeds 0 1 2 (~1.25 h)
#   ./scripts/run_lofar_edge.sh ignore 0     # one mode, one seed (~25 min)
#
# Resumable: rerun the same command and finished runs are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$HOME/torch-env/bin/python"

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
    echo "REFUSING TO START: the laptop is on battery (GPU capped at 210 of 2100 MHz)." >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "REFUSING TO START: another process is using the GPU:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
fi

MODES=(svls ignore)
[ $# -ge 1 ] && MODES=("$1")
SEEDS=(0 1 2)
[ $# -ge 2 ] && SEEDS=("${@:2}")

mkdir -p runs/lofar/logs
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="runs/lofar/edge_${mode}_b08_nocw_seed${seed}"
    if [ -f "$out/eval_test/metrics.json" ]; then echo "-- skip $mode seed $seed (done)"; continue; fi
    echo; echo "-- edge mode $mode, seed $seed -> $out   ($(date +%H:%M))"
    "$PY" experiments/lofar_hybrid.py --base 8 --norm fixed --no_class_weight \
        --edge_mode "$mode" --seed "$seed" --output_dir "$out"
  done
done

echo; echo "All done ($(date +%H:%M)). Results:"
"$PY" analysis/summarise_edge.py
