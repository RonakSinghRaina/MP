#!/usr/bin/env bash
# PART 30 Run C: Mesarcik et al.'s U-Net, June pipeline + their 25k shuffle buffer + float32
# inputs (= Run A), but under Keras 2 (tf_keras 2.21 via TF_USE_LEGACY_KERAS=1, env ~/tfk2-env,
# which reuses tf-env's TensorFlow 2.21). Isolates Keras 3 vs Keras 2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" = "1" ] || { echo "on battery" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then echo "GPU busy" >&2; exit 1; fi
export LD_LIBRARY_PATH="$(ls -d ~/tf-env/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_USE_LEGACY_KERAS=1
PY="$HOME/tfk2-env/bin/python"
"$PY" -c "import tensorflow as tf; print('  keras backend     :', tf.keras.__name__)" 2>/dev/null
"$PY" experiments/mesarcik_repro/train_mesarcik_unet.py --setup paper_code --clip june \
    --shuffle_buffer 25000 --input_dtype float32 --eval_every 5 --seed 0 \
    --output_dir runs/lofar/mesarcik_unet_paper_code_june_buf25000_f32_keras2_seed0 \
    2>&1 | grep --line-buffered -vE "^WARNING|E0000|I0000|W0000|oneDNN|cuda_|computation placer|rebuild TensorFlow"
