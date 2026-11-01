#!/usr/bin/env bash
# All HERA runs for the report (PART 38): every model trained on AOFlagger labels (threshold 10)
# and scored on the simulator's labels, 135 leak-free test images, as in Mesarcik et al.'s Table 1.
#
#   ./scripts/run_hera.sh            seeds 0 1 2, about 7 h on AC
#   ./scripts/run_hera.sh 0          one seed, about 2 h 20 min
#
# It checks the laptop first (plug in, GPU working and free), then carries on in the background,
# so the terminal can be closed. Watch it with:   tail -f runs/hera/logs/run_hera.log
# See the table so far with:   ~/torch-env/bin/python analysis/hera_aoflagger/summarise_hera.py
#
# Resumable: run the same command again; finished runs are skipped and an interrupted run
# continues from where it stopped.
#
# For each seed, cheapest first, so every model has a result early:
#   Mesarcik U-Net  their settings  about  9 min   runs/hera/mesarcik_unet_clean_seed<S>
#   HybridRFINet    their settings  about 21 min   runs/hera/hybrid_mesarcik_clean_bce_seed<S>
#   HybridRFINet    our settings    about 27 min   runs/hera/hybrid_b08_nocw_seed<S>
#   tf_unet         our settings    about 78 min   runs/hera/tfunet_lr1e-4_seed<S>
# tf_unet cannot take 32x32 patches, so "closest to their settings" is the last-epoch model of
# the same run (scored at the end). The U-Net keeps their code's loss and HybridRFINet uses plain
# BCE, as on LOFAR (PART 31), so AUROC and AUPRC are not comparable between those two.
#
# HERA_SMOKE=1 runs a tiny version on the CPU into HERA_RUNS, to test this script.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOKE="${HERA_SMOKE:-}"
RUNS="${HERA_RUNS:-runs/hera}"
LOG="$RUNS/logs/run_hera.log"
LOCK="$RUNS/.run_hera.lock"
TORCH="$HOME/torch-env/bin/python"
TF="$HOME/tf-env/bin/python"
AOF="$HOME/aof-env/bin/python"
TF_LIBS="$(ls -d "$HOME"/tf-env/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"

on_ac() { [ "$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)" = "1" ]; }

check_machine() {
    local bad=0
    if ! on_ac; then
        echo "NOT STARTED: the laptop is on battery. Plug it in (the GPU is 15x slower on battery)."
        bad=1
    fi
    if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "NOT STARTED: the NVIDIA driver is not working. Restart the laptop, then run this again."
        return 1
    fi
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
        echo "NOT STARTED: another program is using the GPU, and only one run fits:"
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
        bad=1
    fi
    if ! "$TORCH" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo "NOT STARTED: PyTorch cannot see the GPU."
        bad=1
    fi
    if ! LD_LIBRARY_PATH="$TF_LIBS${LD_LIBRARY_PATH:-}" TF_CPP_MIN_LOG_LEVEL=3 "$TF" -c \
        'import tensorflow as tf,sys; sys.exit(0 if tf.config.list_physical_devices("GPU") else 1)' 2>/dev/null; then
        echo "NOT STARTED: TensorFlow cannot see the GPU."
        bad=1
    fi
    return $bad
}

# ---- first call: check in the foreground, then continue in the background ----------------
if [ "${RUN_HERA_CHILD:-}" != "1" ]; then
    mkdir -p "$RUNS/logs"
    if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then
        echo "NOT STARTED: run_hera.sh is already running. Watch it with: tail -f $LOG"
        exit 1
    fi
    if [ ! -x "$AOF" ] && [ ! -d data/hera/HERA_arrays ]; then
        echo "NOT STARTED: ~/aof-env is missing and data/hera/HERA_arrays does not exist."
        exit 1
    fi
    if [ -z "$SMOKE" ]; then
        check_machine || exit 1
    fi
    export RUN_HERA_CHILD=1
    if [ -z "$SMOKE" ] && command -v systemd-inhibit >/dev/null; then
        setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who="RFI training" \
            --why="HERA runs" --mode=block "$ROOT/scripts/run_hera.sh" "$@" >> "$LOG" 2>&1 < /dev/null &
    else
        setsid nohup "$ROOT/scripts/run_hera.sh" "$@" >> "$LOG" 2>&1 < /dev/null &
    fi
    echo "Started in the background. You can close this terminal, but keep the laptop plugged in."
    echo "Watch it with:  tail -f $LOG"
    exit 0
fi

# ---- background part ----------------------------------------------------------------------
exec 9>"$LOCK"
if ! flock -n 9; then echo "already running"; exit 1; fi
echo; echo "=============== run_hera.sh started $(date '+%F %H:%M') ==============="

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
UNET_EXTRA=(); HYB_M_EXTRA=(); HYB_EXTRA=(); TF_EXTRA=()
if [ -n "$SMOKE" ]; then
    export CUDA_VISIBLE_DEVICES=-1
    UNET_EXTRA=(--epochs 1 --limit_train 8)
    HYB_M_EXTRA=(--epochs 1 --limit_train 8 --eval_every 0)
    HYB_EXTRA=(--total_epochs 2 --iters_per_epoch 2 --epochs_per_chunk 1 --n_val_images 2)
    TF_EXTRA=(--iters_per_epoch 1 --batch_size 1)
fi

# run <folder name> <command...>: skip if finished, retry after a crash (resumes where it stopped)
run() {
    local name="$1"; shift
    local out="$RUNS/$name" mlog="$RUNS/logs/$name.log" attempt
    if [ -f "$out/eval_test/metrics.json" ]; then echo "-- $name already finished, skipping"; return 0; fi
    for attempt in 1 2 3; do
        echo "-- $name, attempt $attempt  ($(date '+%F %H:%M'))"
        if "$@" --output_dir "$out" >> "$mlog" 2>&1 && [ -f "$out/eval_test/metrics.json" ]; then
            echo "   finished  ($(date '+%H:%M')): $(grep -E 'max F1|max_f1' "$mlog" | tail -1 | cut -c1-70)"
            return 0
        fi
        tail -3 "$mlog"
        if ! on_ac; then echo "STOPPED: the laptop is on battery. Plug in and run the same command again."; exit 1; fi
        echo "   stopped unexpectedly; resuming in 20 s"
        sleep 20
    done
    echo "STOPPED: $name failed 3 times. See $mlog"
    exit 1
}
tf_env() { LD_LIBRARY_PATH="$TF_LIBS${LD_LIBRARY_PATH:-}" "$@"; }

if [ ! -f data/hera/HERA_arrays/test_masks_aof.npy ]; then
    echo "-- making the AOFlagger labels"
    "$AOF" analysis/hera_aoflagger/make_hera_arrays.py || { echo "STOPPED: AOFlagger step failed"; exit 1; }
fi

for s in "${SEEDS[@]}"; do
    run "mesarcik_unet_clean_seed$s" tf_env "$TF" experiments/mesarcik_repro/train_mesarcik_unet.py \
        --dataset hera --setup clean --seed "$s" "${UNET_EXTRA[@]}"
    run "hybrid_mesarcik_clean_bce_seed$s" "$TORCH" experiments/mesarcik_repro/train_hybrid_mesarcik_settings.py \
        --dataset hera --setup clean --loss bce --eval_every 10 --seed "$s" "${HYB_M_EXTRA[@]}"
    run "hybrid_b08_nocw_seed$s" "$TORCH" experiments/lofar_hybrid.py \
        --dataset hera --base 8 --norm fixed --no_class_weight --seed "$s" "${HYB_EXTRA[@]}"
    if [ -n "$SMOKE" ]; then
        run "tfunet_lr1e-4_seed$s" tf_env "$TF" experiments/lofar_tfunet_baseline.py \
            --dataset hera --norm fixed --lr 1e-4 --epochs 2 --chunk 2 --seed "$s" "${TF_EXTRA[@]}"
    else
        run "tfunet_lr1e-4_seed$s" tf_env "$TF" experiments/lofar_tfunet_baseline.py \
            --dataset hera --norm fixed --lr 1e-4 --epochs 150 --seed "$s"
    fi
done

echo "-- scoring tf_unet's last-epoch models  ($(date '+%H:%M'))"
if ! tf_env "$TF" analysis/last_vs_best/tfunet_last_auc.py --dataset hera --seeds "${SEEDS[@]}" \
        --run_root "$RUNS" >> "$RUNS/logs/tfunet_last_auc.log" 2>&1; then
    echo "STOPPED: tf_unet last-epoch scoring failed. See $RUNS/logs/tfunet_last_auc.log"; exit 1
fi

echo; echo "=============== ALL DONE $(date '+%F %H:%M') ==============="
"$TORCH" analysis/hera_aoflagger/summarise_hera.py "$RUNS"
