"""
Remove the train/test leakage from the HERA dataset.

WHY THIS EXISTS
---------------
A SHA-256 check over the raw pixels of HERA_npy found:

    unique train images : 382 / 420   (38 duplicates inside train)
    unique test images  : 135 / 140   ( 5 duplicates inside test)
    train/test overlap  :  31         (22% of the test set)

31 test images are byte-identical copies of training images. Any model trained on
HERA and scored on that test set has already memorised nearly a quarter of it.
This is the dataset's own published split, not something introduced here.

WHAT IT DOES
------------
1. Hashes every image AND every mask.
2. Reports duplicates within each split and across splits.
3. Checks whether identical images carry identical masks -- if the same picture
   appears with two different answers, that is contradictory labelling and worth
   knowing about separately from leakage.
4. Writes a CLEAN copy to a NEW folder. The original HERA_npy is never modified.

WHICH COPY GETS DROPPED
-----------------------
For an image present in both splits, the default drops it from TRAIN and keeps it
in TEST (--drop_from train). Reason: the test set is the scarce resource and its
job is to be unseen; dropping from train achieves that while leaving the test set
at its full unique size. Training data is comparatively plentiful.

Use --drop_from test for the opposite convention. Either produces disjoint
splits; only the sizes differ.

USAGE
    source ~/torch-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"
    python3 hera_transfer_test/dedupe_hera.py

    # look first, change nothing
    python3 hera_transfer_test/dedupe_hera.py --dry_run
"""
import os, sys, glob, json, shutil, hashlib, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    return hashlib.sha256(np.load(path).tobytes()).hexdigest()


def index(split_dir):
    imgs = sorted(glob.glob(os.path.join(split_dir, "images", "*.npy")))
    msks = sorted(glob.glob(os.path.join(split_dir, "masks", "*.npy")))
    assert len(imgs) == len(msks) and imgs, "bad split: " + split_dir
    return [dict(img=i, msk=m, ih=sha(i), mh=sha(m)) for i, m in zip(imgs, msks)]


def first_occurrence(recs):
    """Keep the first record for each image hash; return (kept, dropped)."""
    seen, kept, dropped = set(), [], []
    for r in recs:
        (kept if r["ih"] not in seen else dropped).append(r)
        seen.add(r["ih"])
    return kept, dropped


def main():
    p = argparse.ArgumentParser(description="Deduplicate the HERA train/test split")
    p.add_argument("--in_dir", default=os.path.join(_HERE, "HERA_npy"))
    p.add_argument("--out_dir", default=os.path.join(_HERE, "HERA_npy_clean"))
    p.add_argument("--drop_from", choices=["train", "test"], default="train",
                   help="which side loses the cross-split duplicates (default train, "
                        "which keeps the test set at full size)")
    p.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    a = p.parse_args()

    print("=" * 74)
    print(" HERA DEDUPLICATION")
    print("=" * 74)
    print("  reading {} ...".format(a.in_dir), flush=True)
    tr = index(os.path.join(a.in_dir, "train"))
    te = index(os.path.join(a.in_dir, "test"))
    print("  train {} images | test {} images".format(len(tr), len(te)))

    # ---- 1. duplicates within each split
    tr_keep, tr_dup = first_occurrence(tr)
    te_keep, te_dup = first_occurrence(te)
    print("\n--- duplicates INSIDE each split ---")
    print("  train: {} unique, {} dropped".format(len(tr_keep), len(tr_dup)))
    print("  test : {} unique, {} dropped".format(len(te_keep), len(te_dup)))

    # ---- 2. contradictory labels: same image, different mask?
    by_img = {}
    for r in tr + te:
        by_img.setdefault(r["ih"], set()).add(r["mh"])
    contra = [h for h, ms in by_img.items() if len(ms) > 1]
    print("\n--- label consistency ---")
    if contra:
        print("  !! {} images appear with MORE THAN ONE mask.".format(len(contra)))
        print("     Identical input, different answer -- contradictory labelling.")
        print("     Report this separately; deduplication does not fix it.")
    else:
        print("  OK: every duplicated image carries an identical mask.")

    # ---- 3. cross-split overlap
    tr_h = {r["ih"] for r in tr_keep}
    te_h = {r["ih"] for r in te_keep}
    overlap = tr_h & te_h
    print("\n--- cross-split overlap ---")
    print("  {} images appear in BOTH train and test".format(len(overlap)))
    print("  ({:.1f}% of the unique test set)".format(100 * len(overlap) / max(len(te_keep), 1)))

    if a.drop_from == "train":
        tr_final = [r for r in tr_keep if r["ih"] not in overlap]
        te_final = te_keep
    else:
        tr_final = tr_keep
        te_final = [r for r in te_keep if r["ih"] not in overlap]
    print("  dropping them from: {}".format(a.drop_from.upper()))

    print("\n" + "=" * 74)
    print("  FINAL CLEAN SPLIT")
    print("    train : {:>4}  (was {})".format(len(tr_final), len(tr)))
    print("    test  : {:>4}  (was {})".format(len(te_final), len(te)))
    fh_tr = {r["ih"] for r in tr_final}
    fh_te = {r["ih"] for r in te_final}
    print("    overlap after cleaning: {}   <-- must be 0".format(len(fh_tr & fh_te)))
    print("    all train unique: {} | all test unique: {}".format(
        len(fh_tr) == len(tr_final), len(fh_te) == len(te_final)))
    print("=" * 74)

    if a.dry_run:
        print("\n--dry_run: nothing written.")
        return
    if fh_tr & fh_te:
        raise SystemExit("REFUSING TO WRITE: overlap is not zero.")

    shutil.rmtree(a.out_dir, ignore_errors=True)
    for split, recs in (("train", tr_final), ("test", te_final)):
        for k in ("images", "masks"):
            os.makedirs(os.path.join(a.out_dir, split, k), exist_ok=True)
        for i, r in enumerate(recs):
            shutil.copyfile(r["img"], os.path.join(a.out_dir, split, "images",
                                                   "spectrogram_{:04d}.npy".format(i)))
            shutil.copyfile(r["msk"], os.path.join(a.out_dir, split, "masks",
                                                   "mask_{:04d}.npy".format(i)))
        print("  wrote {}: {} pairs".format(split, len(recs)))

    json.dump(dict(source=os.path.abspath(a.in_dir), drop_from=a.drop_from,
                   original=dict(train=len(tr), test=len(te)),
                   dropped_within=dict(train=len(tr_dup), test=len(te_dup)),
                   cross_split_overlap=len(overlap),
                   contradictory_label_images=len(contra),
                   final=dict(train=len(tr_final), test=len(te_final)),
                   overlap_after=0),
              open(os.path.join(a.out_dir, "dedupe_report.json"), "w"), indent=2)
    print("\nSaved: {}/dedupe_report.json".format(a.out_dir))
    print("\nNow run BOTH fine-tuning arms against the clean folder:")
    print("  python3 hera_transfer_test/finetune_on_hera.py --init pretrained \\")
    print("      --data_dir hera_transfer_test/HERA_npy_clean --fresh")
    print("  python3 hera_transfer_test/finetune_on_hera.py --init scratch \\")
    print("      --data_dir hera_transfer_test/HERA_npy_clean --fresh")


if __name__ == "__main__":
    main()
