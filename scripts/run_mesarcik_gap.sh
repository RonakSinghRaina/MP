#!/usr/bin/env bash
# PART 30: why does Mesarcik et al.'s own U-Net score ~0.62 here vs 0.5876 published?
# Run A: their exact pipeline (June preprocessing, all 7500 images) + their 25k tf.data
#        shuffle buffer + float32 inputs, test scored every 5 epochs.
# Run B: same but full shuffle -> isolates the buffer; with A, also isolates float16 caching
#        against the earlier mesarcik_unet_paper_code_june_seed0 (0.6258).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" = "1" ] || { echo "on battery" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "GPU busy" >&2; exit 1; fi
export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY="$HOME/tf-env/bin/python"
for buf in 25000 0; do
  "$PY" experiments/mesarcik_repro/train_mesarcik_unet.py --setup paper_code --clip june \
      --shuffle_buffer "$buf" --input_dtype float32 --eval_every 5 --seed 0 \
      2>&1 | grep --line-buffered -vE "^WARNING|E0000|I0000|W0000|oneDNN|cuda_|computation placer|rebuild TensorFlow"
done
