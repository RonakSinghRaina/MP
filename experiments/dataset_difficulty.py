"""
Characterise how hard this benchmark actually is.

WHY THIS EXISTS
---------------
A reported F1 of 0.98 is only impressive if the task is hard. Nothing in this
project measured that. This script measures it from the generator's own
strength maps.

The key structural fact it checks: `dataset_generator_v3_strength.py` builds the
label as

    mask = (hard injected footprint) OR (blurred RFI power > 0.5 * local sigma)

so the ground truth is a deterministic, noise-free threshold on a smooth latent
field. That is very different from real telescope data, where the "true" flag is
genuinely ambiguous near the noise floor. This script quantifies the
consequence: what fraction of the labelled RFI sits comfortably above the noise
(and is therefore findable by thresholding alone), and how many clean-labelled
pixels carry any RFI power at all (i.e. how much label ambiguity exists).

Report these numbers in the paper. They set the context for any headline score.
"""
import os
import glob
import json
import argparse

import numpy as np

BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, np.inf)]


def main():
    p = argparse.ArgumentParser(description="Measure intrinsic difficulty of the synthetic benchmark")
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--dataset_dir", default=os.path.normpath(os.path.join(_here, "..", "Synthetic Dataset 276x600")))
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    sdir = os.path.join(args.dataset_dir, args.split, "strength")
    mdir = os.path.join(args.dataset_dir, args.split, "masks")
    if not os.path.isdir(sdir):
        raise SystemExit(
            f"ERROR: {sdir} not found.\n"
            "Strength maps are written only by dataset_generator_v3_strength.py. "
            "Regenerate the dataset with that generator."
        )

    st = sorted(glob.glob(os.path.join(sdir, "*.npy")))
    mk = sorted(glob.glob(os.path.join(mdir, "*.npy")))
    if len(st) != len(mk) or not st:
        raise SystemExit(f"ERROR: {len(st)} strength maps vs {len(mk)} masks in {args.dataset_dir}/{args.split}")

    counts = np.zeros(len(BUCKETS), dtype=np.int64)
    n_rfi = n_tot = 0
    label_ambiguity = 0          # clean-labelled pixels that still carry >0.5 sigma of RFI
    per_img_frac = []

    for sp, mp in zip(st, mk):
        s = np.load(sp).astype(np.float32)
        m = np.load(mp).astype(bool)
        n_tot += m.size
        n_rfi += int(m.sum())
        per_img_frac.append(float(m.mean()))
        v = s[m]
        for i, (lo, hi) in enumerate(BUCKETS):
            counts[i] += int(((v >= lo) & (v < hi)).sum())
        label_ambiguity += int((s[~m] > 0.5).sum())

    n_clean = n_tot - n_rfi
    print(f"split              : {args.split}  ({len(st)} images)")
    print(f"total pixels       : {n_tot:,}")
    print(f"RFI-labelled       : {n_rfi:,}  ({100 * n_rfi / n_tot:.4f}%)")
    print(f"label ambiguity    : {label_ambiguity:,} clean-labelled pixels carry >0.5 sigma of RFI "
          f"({100 * label_ambiguity / max(n_clean, 1):.4f}% of clean)")

    print("\nRFI-labelled pixels by injected strength (units of local noise sigma):")
    cum = 0
    for (lo, hi), c in zip(BUCKETS, counts):
        cum += c
        lab = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {lab:>9} sigma : {c:>11,}  ({100 * c / max(n_rfi, 1):5.2f}%)   cumulative {100 * cum / max(n_rfi, 1):6.2f}%")

    ge1 = counts[2:].sum() / max(n_rfi, 1)
    ge2 = counts[3:].sum() / max(n_rfi, 1)
    print(f"\n>= 1 sigma : {100 * ge1:.2f}% of labelled RFI")
    print(f">= 2 sigma : {100 * ge2:.2f}% of labelled RFI")
    print(f"\nINTERPRETATION: a detector that flagged every pixel at or above 1 sigma")
    print(f"and nothing else would already reach recall {100 * ge1:.2f}% before any")
    print(f"learning. Quote this next to any headline F1.")

    pf = np.array(per_img_frac)
    print(f"\nper-image RFI fraction: min {100 * pf.min():.2f}%  median {100 * np.median(pf):.2f}%  "
          f"mean {100 * pf.mean():.2f}%  max {100 * pf.max():.2f}%")
    print(f"images containing no RFI at all: {(pf == 0).sum()} / {len(pf)}  "
          f"(these are trivially solved and inflate per-image averages)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(split=args.split, n_images=len(st), n_total_px=int(n_tot),
                       n_rfi_px=int(n_rfi), rfi_fraction=float(n_rfi / n_tot),
                       buckets=[f"{a:g}-{b:g}" for a, b in BUCKETS],
                       bucket_counts=counts.tolist(),
                       frac_ge_1sigma=float(ge1), frac_ge_2sigma=float(ge2),
                       label_ambiguity_px=int(label_ambiguity),
                       n_empty_images=int((pf == 0).sum())),
                  open(args.out, "w"), indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
