#!/usr/bin/env bash
# Test the last untested difference against Mesarcik et al.: patch-based training.
# See PART 12.12 of RFI-project-context.md.
#
# The paper uses 32x32 patches. tf_unet with layers=3 and valid padding REJECTS
# 32 -- it shrinks the input by 40 px, so 32 produces no output at all. The
# smallest that builds is 48; 64 is the smallest sensible one. So this is the
# nearest workable equivalent, not a reproduction, and the paper must say so.
#
# Budget is matched by OUTPUT PIXELS, not by steps. A 64x64 patch at batch 32
# yields 18,432 output pixels per step against 891,136 for a full 512x512 image
# at batch 4 -- 48x less. Matching step counts would undertrain the patch model
# by that factor and prove nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" != "1" ]; then
  echo "REFUSING TO START: on battery."; exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "REFUSING TO START: GPU is busy."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
  exit 1
fi

export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
PY=~/tf-env/bin/python

# fixed-range norm, matching every other run
echo "=== 64x64 patches, fixed norm, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_tfunet_baseline.py \
    --patch_size 64 --batch_size 32 --norm fixed --lr 1e-4 \
    --epochs 150 --iters_per_epoch 8460 --chunk 10 --seed 0 \
    --output_dir lofar_runs/patch64_fixed_seed0

# per-image norm ON PATCHES -- this is the actual PART 12.11 hypothesis:
# per-image normalisation may only work when the "image" is a small patch.
echo "=== 64x64 patches, PER-IMAGE norm, seed 0  ($(date '+%H:%M')) ==="
$PY experiments/lofar_tfunet_baseline.py \
    --patch_size 64 --batch_size 32 --norm per_image --lr 1e-4 \
    --epochs 150 --iters_per_epoch 8460 --chunk 10 --seed 0 \
    --output_dir lofar_runs/patch64_perimage_seed0

echo; echo "Done. ./status.sh for the summary."
