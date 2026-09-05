"""Low-RAM access to the LOFAR dataset.

The 9.3 GB pickle is converted once by lofar_analysis/convert_pickle_to_memmap.py
into LOFAR_npy/*.npy. This module memory-maps those files: indexing works exactly
like a normal numpy array, but only the slice you touch is read from disk, so RAM
stays at a few MB instead of 10 GB.

    from lofar_data import load_lofar, preprocess

    d = load_lofar()
    d.train_images.shape        # (7500, 512, 512, 1)   -- nothing loaded yet
    img = d.train_images[1]     # ~1 MB read from disk
    img = preprocess(img)       # paper's clip -> log -> min-max, per image

Axis order in these arrays is rows = TIME, columns = FREQUENCY (measured; this is
the opposite of our synthetic datasets). See PART 11 of RFI-project-context.md.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NPY = ROOT / "LOFAR_npy"
IDX = ROOT / "lofar_analysis"


@dataclass
class Lofar:
    train_images: np.ndarray  # (7500, 512, 512, 1) float32, RAW
    train_masks: np.ndarray   # (7500, 512, 512, 1) bool, AOFlagger labels
    test_images: np.ndarray   # (109, 512, 512, 1) float32, RAW
    test_masks: np.ndarray    # (109, 512, 512, 1) bool, HUMAN expert labels
    clean_train_idx: np.ndarray   # 7356 training indices safe to train on
    leak_train_idx: np.ndarray    # 109 training indices duplicated in the test set

    def summary(self) -> str:
        mb = sum(a.nbytes for a in (self.train_images, self.train_masks,
                                    self.test_images, self.test_masks)) / 2**30
        return (
            f"train_images {self.train_images.shape} {self.train_images.dtype}  (RAW)\n"
            f"train_masks  {self.train_masks.shape} {self.train_masks.dtype}  (AOFlagger)\n"
            f"test_images  {self.test_images.shape} {self.test_images.dtype}  (RAW)\n"
            f"test_masks   {self.test_masks.shape} {self.test_masks.dtype}  (HUMAN)\n"
            f"clean_train_idx: {len(self.clean_train_idx)} usable training images\n"
            f"leak_train_idx:  {len(self.leak_train_idx)} training images that are "
            f"byte-identical to test images\n"
            f"total on disk {mb:.2f} GiB, but memory-mapped -- RAM use is ~0"
        )


def load_lofar(mmap: bool = True) -> Lofar:
    """Open the dataset. With mmap=True (default) almost no RAM is used."""
    if not NPY.exists():
        raise FileNotFoundError(
            f"{NPY} not found. Run lofar_analysis/convert_pickle_to_memmap.py once first."
        )
    mode = "r" if mmap else None
    return Lofar(
        train_images=np.load(NPY / "train_images.npy", mmap_mode=mode),
        train_masks=np.load(NPY / "train_masks.npy", mmap_mode=mode),
        test_images=np.load(NPY / "test_images.npy", mmap_mode=mode),
        test_masks=np.load(NPY / "test_masks.npy", mmap_mode=mode),
        clean_train_idx=np.load(IDX / "lofar_clean_train_idx.npy"),
        leak_train_idx=np.load(IDX / "lofar_leak_train_idx.npy"),
    )


def preprocess(img: np.ndarray) -> np.ndarray:
    """Mesarcik et al. 2022 §4.2: clip to [|mu-sigma|, mu+4sigma], log, min-max to [0,1].

    Applied PER IMAGE -- the per-image clip bounds vary by ~490x across this
    dataset, so a single global scale does not work here (PART 11.8).
    Degenerate images (all zero) come back as all zeros instead of NaN.
    """
    x = np.asarray(img, dtype=np.float64)
    mu, sd = float(x.mean()), float(x.std())
    lo, hi = abs(mu - sd), mu + 4.0 * sd
    lo = max(lo, 1e-6)
    if not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.log(np.clip(x, lo, hi))
    rng = x.max() - x.min()
    if rng <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x.min()) / rng).astype(np.float32)


def batches(images, masks, indices, batch_size=8, preprocess_fn=preprocess):
    """Yield (X, Y) batches, reading only one batch from disk at a time.

    indices: which image indices to draw from (use clean_train_idx for training).
    """
    indices = np.asarray(indices)
    for s in range(0, len(indices), batch_size):
        sel = np.sort(indices[s : s + batch_size])
        X = np.stack([preprocess_fn(images[i, :, :, 0]) for i in sel])[..., None]
        Y = np.stack([masks[i, :, :, 0] for i in sel])[..., None]
        yield X, Y


if __name__ == "__main__":
    d = load_lofar()
    print(d.summary())
