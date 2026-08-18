"""
Evaluates a trained checkpoint (default: best_checkpoint/) on the test set,
reporting ROC AUC / PR AUC / F1 exactly the way the paper does, plus a
visual prediction panel for your report.

Reuses RFINpyDataProvider and the evaluate() function from
train_unet_rfi_gpu.py directly (same file, same tested code) rather than
duplicating that logic.

IMPORTANT METHODOLOGY NOTE
----------------------------
If your dataset only has train/ and test/ folders (no val/), then
train_unet_rfi_gpu.py's training loop already used the test set as its
validation set -- meaning the checkpoint you're about to evaluate here was
*selected* because it scored well on this exact test set. Running this
script will largely just confirm that, not give you an independent measure
of generalization. This script prints that caveat before showing results
so the number isn't accidentally overclaimed in a report.
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_unet_rfi_gpu import (  # noqa: E402  (reusing already-tested code)
    tf1, unet, util, RFINpyDataProvider, evaluate, print_device_info,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set")
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset")))
    parser.add_argument("--checkpoint_dir", default=None,
                         help="Default: <output_dir>/best_checkpoint")
    parser.add_argument("--output_dir", default=os.path.normpath(os.path.join(_here, "..", "unet_run_gpu")))
    parser.add_argument("--split", default="test", help="Which subfolder to evaluate on")
    parser.add_argument("--layers", type=int, default=3, help="MUST match the trained model's --layers")
    parser.add_argument("--features_root", type=int, default=64, help="MUST match the trained model's --features_root")
    parser.add_argument("--patch_size", type=int, default=512,
                         help="MUST match what the model was trained/evaluated with for a fair comparison "
                              "to your training_log.csv numbers. Set 0 to evaluate on full-size images instead "
                              "(different from training numbers, but sometimes wanted for a final report).")
    parser.add_argument("--n_images", type=int, default=0, help="0 = evaluate on the entire split")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                         help="Images per forward pass. Keep at 1 unless you've confirmed higher fits VRAM.")
    parser.add_argument("--n_preview_images", type=int, default=4,
                         help="How many example predictions to save as a visual panel")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "best_checkpoint")
    if not os.path.exists(os.path.join(checkpoint_dir, "checkpoint")):
        print(f"ERROR: no checkpoint found at {checkpoint_dir}")
        print("Pass --checkpoint_dir explicitly if your checkpoint lives somewhere else, e.g.:")
        print(f'  python3 {os.path.basename(__file__)} --checkpoint_dir "{args.output_dir}/checkpoints"')
        return

    split_images = os.path.join(args.dataset_dir, args.split, "images")
    split_masks = os.path.join(args.dataset_dir, args.split, "masks")
    if not os.path.isdir(split_images):
        print(f"ERROR: could not find {split_images}")
        return

    print("=" * 70)
    print("METHODOLOGY NOTE: if your dataset has no separate val/ folder,")
    print("training already used this same test set to pick the best")
    print("checkpoint. This evaluation largely confirms that selection --")
    print("it is not an independent generalization estimate in that case.")
    print("=" * 70 + "\n")

    print_device_info()

    n_total = len(os.listdir(split_images))
    n_images = args.n_images if args.n_images > 0 else n_total
    print(f"Evaluating on {n_images} of {n_total} images in '{args.split}/' "
          f"using checkpoint: {checkpoint_dir}")
    print(f"Architecture: layers={args.layers}, features_root={args.features_root} "
          f"(must match training -- mismatches will fail to restore or give garbage results)\n")

    provider = RFINpyDataProvider(
        split_images, split_masks,
        patch_size=(args.patch_size if args.patch_size > 0 else None),
        shuffle_data=False,
    )

    net = unet.Unet(
        channels=1, n_class=2,
        layers=args.layers, features_root=args.features_root,
        cost="cross_entropy", cost_kwargs=dict(regularizer=0.001),
    )
    model_path = os.path.join(checkpoint_dir, "model.ckpt")

    metrics = evaluate(net, model_path, provider, n_images, args.eval_batch_size)

    print("\n" + "=" * 40)
    print(f" Results on {args.split}/ ({n_images} images)")
    print("=" * 40)
    print(f"  ROC AUC : {metrics['roc_auc']:.4f}   (paper's simulated-data result: ~0.96)")
    print(f"  PR  AUC : {metrics['pr_auc']:.4f}   (paper's simulated-data result: ~0.92)")
    print(f"  Max F1  : {metrics['max_f1']:.4f}   (paper's simulated-data result: ~0.85)")
    if metrics["collapsed"]:
        print("  WARNING: this checkpoint is in the dead-network state (see earlier discussion). "
              "These numbers are meaningless -- use a different checkpoint.")
    print("=" * 40)

    eval_dir = os.path.join(args.output_dir, f"eval_{args.split}")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "n_images": n_images, "split": args.split,
                    "checkpoint_dir": checkpoint_dir}, f, indent=2)
    print(f"\nSaved: {eval_dir}/metrics.json")

    # Visual prediction panel for a handful of images -- useful for a report
    if args.n_preview_images > 0:
        n_prev = min(args.n_preview_images, n_images)
        x_prev, y_prev = provider(n_prev)
        pred_prev = net.predict(model_path, x_prev)
        panel = util.combine_img_prediction(x_prev, y_prev, pred_prev)
        panel_path = os.path.join(eval_dir, "prediction_panel.jpg")
        util.save_image(panel, panel_path)
        print(f"Saved: {panel_path}  (input / ground truth / prediction, {n_prev} examples)")


if __name__ == "__main__":
    main()
