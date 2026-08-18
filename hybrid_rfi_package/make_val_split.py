"""
Creates a proper validation split by moving a portion of the TRAINING set
into a new val/ folder.

WHY THIS EXISTS
----------------
Your dataset folder has train/ and test/ but no val/. That means training
had to fall back to using the TEST set to pick the best checkpoint --
which is exactly the leakage you said you wanted to avoid.

WHY CARVE FROM TRAIN INSTEAD OF REGENERATING
----------------------------------------------
Regenerating the full 1000-image dataset takes ~20 minutes and 5 GB. Moving
150 images from train/ to val/ takes seconds, touches the test set ZERO
times, and gives exactly the same guarantee: the model never sees test data
during training or model selection.

Cost: your training set shrinks from 700 to 550 images. That is the correct
trade -- a slightly smaller training set is far less damaging to your
project than an invalid test score.

The split is deterministic (fixed seed), so it is reproducible and you can
state exactly how it was made in your report.
"""
import os
import json
import shutil
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Carve a val split out of train/ (test/ untouched)")
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset")))
    parser.add_argument("--n_val", type=int, default=150, help="How many images to move from train to val")
    parser.add_argument("--seed", type=int, default=1234, help="Fixed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Show what would happen without moving anything")
    args = parser.parse_args()

    train_img = os.path.join(args.dataset_dir, "train", "images")
    train_msk = os.path.join(args.dataset_dir, "train", "masks")
    val_img = os.path.join(args.dataset_dir, "val", "images")
    val_msk = os.path.join(args.dataset_dir, "val", "masks")

    if not os.path.isdir(train_img):
        print(f"ERROR: {train_img} not found. Check --dataset_dir.")
        return

    if os.path.isdir(val_img) and len(os.listdir(val_img)) > 0:
        print(f"val/ already exists with {len(os.listdir(val_img))} images -- nothing to do.")
        print("Delete it first if you want to redo the split.")
        return

    imgs = sorted(f for f in os.listdir(train_img) if f.endswith(".npy"))
    if len(imgs) <= args.n_val:
        print(f"ERROR: only {len(imgs)} training images; cannot move {args.n_val} to val.")
        return

    rng = np.random.RandomState(args.seed)
    pick = set(rng.choice(len(imgs), size=args.n_val, replace=False).tolist())
    to_move = [imgs[i] for i in sorted(pick)]

    print(f"Dataset: {args.dataset_dir}")
    print(f"  train currently: {len(imgs)} images")
    print(f"  moving to val  : {args.n_val} images (seed={args.seed}, deterministic)")
    print(f"  train after    : {len(imgs) - args.n_val} images")
    print(f"  test           : UNTOUCHED")
    if args.dry_run:
        print("\n--dry_run set; nothing moved. First 5 that would move:")
        for f in to_move[:5]:
            print("   ", f)
        return

    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_msk, exist_ok=True)

    # metadata.jsonl: split records to match
    train_meta_path = os.path.join(args.dataset_dir, "train", "metadata.jsonl")
    meta_by_idx = {}
    if os.path.exists(train_meta_path):
        for line in open(train_meta_path):
            r = json.loads(line)
            meta_by_idx[int(r["global_idx"])] = r

    moved_meta, kept_meta = [], []
    moved = 0
    for fname in imgs:
        idx = int(fname.replace("spectrogram_", "").replace(".npy", ""))
        rec = meta_by_idx.get(idx)
        if fname in to_move:
            mname = f"mask_{idx:04d}.npy"
            shutil.move(os.path.join(train_img, fname), os.path.join(val_img, fname))
            shutil.move(os.path.join(train_msk, mname), os.path.join(val_msk, mname))
            # strength maps too, if the newer generator produced them
            s_src = os.path.join(args.dataset_dir, "train", "strength", f"strength_{idx:04d}.npy")
            if os.path.exists(s_src):
                s_dst_dir = os.path.join(args.dataset_dir, "val", "strength")
                os.makedirs(s_dst_dir, exist_ok=True)
                shutil.move(s_src, os.path.join(s_dst_dir, f"strength_{idx:04d}.npy"))
            if rec:
                rec["split"] = "val"
                moved_meta.append(rec)
            moved += 1
        elif rec:
            kept_meta.append(rec)

    if meta_by_idx:
        with open(train_meta_path, "w") as f:
            for r in kept_meta:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.dataset_dir, "val", "metadata.jsonl"), "w") as f:
            for r in moved_meta:
                f.write(json.dumps(r) + "\n")

    print(f"\nDone. Moved {moved} image/mask pairs to val/.")
    print("Training will now use val/ for checkpoint selection, and the test set")
    print("stays completely unseen until your single final evaluation.")


if __name__ == "__main__":
    main()
