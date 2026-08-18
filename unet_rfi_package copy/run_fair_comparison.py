"""
FAIR COMPARISON RUNNER
======================
Trains the authors' UNMODIFIED tf_unet (Akeret et al. 2017) on the SAME
276x600 dataset, with settings matched to the hybrid run, so that the only
remaining difference is the ARCHITECTURE.

WHY THIS SCRIPT EXISTS
-----------------------
The earlier baseline result (ROC 0.55, F1 0.34) is NOT comparable to the
hybrid result (ROC 0.9995, F1 0.9808), because THREE things differed at
once: architecture, dataset (1024x1024 vs 276x600), and batch size (1 vs 8).
With three variables changed simultaneously it is impossible to say which
caused the gap. This script holds dataset and training settings fixed so
the comparison actually means something.

WHAT IS MATCHED TO THE HYBRID
------------------------------
  dataset          276x600, identical train/val/test split
  batch size       8
  patch size       0  (full images, no cropping)
  optimizer        adam
  learning rate    0.001
  epochs           22   (the hybrid was stopped at epoch 22)
  val images       150  (ALL of them -- see note below)
  model selection  validation only; test set never touched during training

THE training_iters DETAIL (easy to get wrong)
----------------------------------------------
In tf_unet, one "epoch" is NOT a full pass over the dataset -- it is exactly
`training_iters` gradient steps (default 64). The hybrid's epoch IS a full
pass: 700 images / batch 8 = 87 steps.

Left at the default, the baseline would perform 22 x 64 = 1408 steps versus
the hybrid's 22 x 87 = 1914 steps -- only 74% as much training, which would
unfairly handicap the baseline and inflate any apparent hybrid advantage.
This script sets training_iters = 87 so both models see the same amount of
data. If you change the batch size, this recomputes automatically.

THE n_eval_images DETAIL (this one already burned us once)
-----------------------------------------------------------
The first baseline run used only 10 validation images per checkpoint. With
per-image RFI fraction ranging 0-60%, a 10-image sample is extremely noisy,
and taking the MAX over 20 such noisy checkpoints produced an inflated
"best F1 = 0.8138" that collapsed to 0.34 on the real test set -- the
classic winner's curse. This script uses all 150 validation images so
checkpoint selection is actually reliable.

WHAT IS *NOT* MATCHED (be honest about this in the report)
------------------------------------------------------------
The loss functions differ. The hybrid uses weighted cross-entropy + Dice;
tf_unet supports weighted cross-entropy only. Dice is part of what makes the
hybrid approach work on sparse targets and cannot be transplanted into the
authors' code without modifying it -- which would break the "unmodified
authors' code" guarantee. So the comparison is
"authors' method vs hybrid method", not "architecture alone, all else equal".
State it that way.
"""
import os
import sys
import glob
import subprocess
import argparse


def main():
    p = argparse.ArgumentParser(description="Run the authors' tf_unet on the 276x600 dataset, matched to the hybrid")
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset 276x600")))
    p.add_argument("--output_dir", default=os.path.normpath(os.path.join(_here, "..", "unet_run_faircompare")))
    p.add_argument("--batch_size", type=int, default=8, help="Matched to the hybrid run")
    p.add_argument("--epochs", type=int, default=22, help="Matched: the hybrid was stopped at epoch 22")
    p.add_argument("--optimizer", default="adam", choices=["adam", "momentum"])
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--layers", type=int, default=3, help="paper's recommended depth")
    p.add_argument("--features_root", type=int, default=64, help="paper's recommended width")
    p.add_argument("--epochs_per_chunk", type=int, default=2)
    p.add_argument("--dry_run", action="store_true", help="Print the command without running it")
    args = p.parse_args()

    train_img = os.path.join(args.dataset_dir, "train", "images")
    if not os.path.isdir(train_img):
        print(f"ERROR: {train_img} not found. Check --dataset_dir.")
        return 1
    n_train = len(glob.glob(os.path.join(train_img, "*.npy")))

    val_img = os.path.join(args.dataset_dir, "val", "images")
    if not os.path.isdir(val_img):
        print(f"ERROR: {val_img} not found.")
        print("Without a val/ split the baseline would select checkpoints using the")
        print("TEST set, which invalidates the comparison. Create one first.")
        return 1
    n_val = len(glob.glob(os.path.join(val_img, "*.npy")))

    # One tf_unet "epoch" == one full pass over the training set
    training_iters = max(1, n_train // args.batch_size)

    script = os.path.join(_here, "train_unet_rfi_gpu.py")
    if not os.path.exists(script):
        print(f"ERROR: {script} not found.")
        print("This runner must sit in the same folder as train_unet_rfi_gpu.py")
        print("(i.e. inside your 'unet_rfi_package copy' folder).")
        return 1

    cmd = [
        sys.executable, script,
        "--dataset_dir", args.dataset_dir,
        "--output_dir", args.output_dir,
        "--patch_size", "0",
        "--batch_size", str(args.batch_size),
        "--optimizer", args.optimizer,
        "--learning_rate", str(args.learning_rate),
        "--layers", str(args.layers),
        "--features_root", str(args.features_root),
        "--training_iters", str(training_iters),
        "--total_epochs", str(args.epochs),
        "--epochs_per_chunk", str(args.epochs_per_chunk),
        "--n_eval_images", str(n_val),
    ]

    print("=" * 72)
    print("FAIR COMPARISON: authors' tf_unet on the 276x600 dataset")
    print("=" * 72)
    print(f"  train images      : {n_train}")
    print(f"  val images        : {n_val}  (all used for checkpoint selection)")
    print(f"  batch size        : {args.batch_size}          [matched to hybrid]")
    print(f"  training_iters    : {training_iters}         [= {n_train}/{args.batch_size}, one full pass per epoch]")
    print(f"  epochs            : {args.epochs}          [matched to hybrid]")
    print(f"  total grad steps  : {training_iters * args.epochs}")
    print(f"  optimizer         : {args.optimizer} @ lr={args.learning_rate}   [matched to hybrid]")
    print(f"  architecture      : layers={args.layers}, features_root={args.features_root}  [paper's recommendation]")
    print(f"  images, full size : yes (patch_size=0)")
    print("=" * 72)
    print()

    if args.dry_run:
        print("DRY RUN -- command that would be executed:\n")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
