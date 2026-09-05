"""One-time conversion of LOFAR_Full_RFI_dataset.pkl into memory-mapped .npy files.

Run this ONCE (it needs ~10 GB RAM for a few minutes, so close other apps).
After that, never open the pickle again -- use lofar_data.py, which memory-maps
these files and uses almost no RAM.

Each array is freed from memory as soon as it is written, so peak usage falls
steadily through the run instead of holding everything at once.
"""
import gc
import os
import pickle
import sys

import numpy as np

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
OUT = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_npy"

NAMES = ["train_images", "train_masks", "test_images", "test_masks"]


def main():
    os.makedirs(OUT, exist_ok=True)

    if all(os.path.exists(os.path.join(OUT, n + ".npy")) for n in NAMES):
        print(f"All four .npy files already exist in {OUT} -- nothing to do.")
        return

    print(f"Loading {PKL} (this needs ~10 GB RAM and takes a few minutes) ...")
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    print("Loaded. Writing .npy files and freeing memory as we go.\n")

    for i, name in enumerate(NAMES):
        arr = data[i]
        path = os.path.join(OUT, name + ".npy")
        print(f"  {name:<14} shape={str(arr.shape):<22} dtype={arr.dtype} "
              f"-> {arr.nbytes / 2**30:.2f} GiB")
        np.save(path, arr)
        # drop our reference and the list's, so the memory is actually released
        data[i] = None
        del arr
        gc.collect()

    del data
    gc.collect()

    print("\nVerifying (memory-mapped, so this uses no real RAM) ...")
    for name in NAMES:
        a = np.load(os.path.join(OUT, name + ".npy"), mmap_mode="r")
        print(f"  {name:<14} {str(a.shape):<22} {a.dtype}")
        del a

    print(f"\nDone. Files are in {OUT}")
    print("From now on use:  from lofar_data import load_lofar")


if __name__ == "__main__":
    sys.exit(main())
