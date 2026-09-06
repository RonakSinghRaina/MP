#!/usr/bin/env bash
# Follow-up to the PART 12.8 finding: confirm the lr 1e-4 result with seeds 1
# and 2, then test the one remaining difference against Mesarcik et al. --
# their per-image normalisation.
#
#   ./run_lofar_lr1e-4.sh
#
# Order matters: the two seeds come FIRST because they are what the paper
# needs. If you stop it early you still have the result that counts.
#
# ~1.3 h per run on AC. Seeds only: ~2.6 h. Everything: ~3.9 h.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
  echo "REFUSING TO START: on battery. The GPU throttles ~15x (PART 11.10c)."
  exit 1
fi

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "REFUSING TO START: something is already using the GPU."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  echo "A second run would OOM. Wait for it, or kill it first."
  exit 1
fi

export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
PY=~/tf-env/bin/python

# --- priority: the two missing seeds ---------------------------------------
for SEED in 1 2; do
  echo "=== lr 1e-4, seed $SEED  ($(date '+%H:%M')) ==="
  $PY experiments/lofar_tfunet_baseline.py \
      --norm fixed --lr 1e-4 --epochs 150 --seed "$SEED" \
      --output_dir "runs/lofar/lr1e-4_seed${SEED}"
done

# --- secondary: the last remaining difference vs the paper -----------------
echo "=== lr 1e-4, per-image normalisation, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_tfunet_baseline.py \
    --norm per_image --lr 1e-4 --epochs 150 --seed 0 \
    --output_dir lofar_runs/lr1e-4_perimage_seed0

echo
echo "All finished. Run ./status.sh for the summary."
