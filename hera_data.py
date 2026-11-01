"""Low-RAM access to the leak-free HERA arrays, with the same fields as lofar_data.load_lofar().

The arrays are built once by analysis/hera_aoflagger/make_hera_arrays.py (PART 37, 38):
    train_masks  AOFlagger at threshold 10, what models train on (Mesarcik et al. 2022, Table 1)
    test_masks   the simulator's masks, what models are scored on
There are no expert labels: HERA is simulated, so the simulator knows every RFI pixel.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NPY = ROOT / "data" / "hera" / "HERA_arrays"


@dataclass
class Hera:
    train_images: np.ndarray     # (351, 512, 512, 1) float32, RAW
    train_masks: np.ndarray      # (351, 512, 512, 1) bool, AOFlagger labels
    train_masks_sim: np.ndarray  # (351, 512, 512, 1) bool, simulator labels; never trained on
    test_images: np.ndarray      # (135, 512, 512, 1) float32, RAW
    test_masks: np.ndarray       # (135, 512, 512, 1) bool, simulator labels
    test_masks_aof: np.ndarray   # (135, 512, 512, 1) bool, AOFlagger labels
    clean_train_idx: np.ndarray  # all 351: duplicates were already removed
    leak_train_idx: np.ndarray   # empty, for the same reason


def load_hera() -> Hera:
    if not NPY.exists():
        raise FileNotFoundError(
            f"{NPY} not found. Run: ~/aof-env/bin/python analysis/hera_aoflagger/make_hera_arrays.py")
    load = lambda name: np.load(NPY / f"{name}.npy", mmap_mode="r")
    train_images = load("train_images")
    return Hera(train_images=train_images, train_masks=load("train_masks"),
                train_masks_sim=load("train_masks_sim"), test_images=load("test_images"),
                test_masks=load("test_masks"), test_masks_aof=load("test_masks_aof"),
                clean_train_idx=np.arange(train_images.shape[0]),
                leak_train_idx=np.array([], dtype=np.int64))
