"""
Verification for the v4 bandpass dataset.
=========================================

Run this before trusting anything the dataset produces. It checks structure,
looks for leakage, measures the numbers that have previously broken training
runs in this project, and renders previews.

    python3 verify_dataset_v4.py --dataset_dir "../Synthetic Dataset 1024x265"

    # also prove the dataset regenerates bit-exactly from its seed:
    python3 verify_dataset_v4.py --dataset_dir "../Synthetic Dataset 1024x265" \
        --compare_dir /path/to/second/generation

Exits non-zero if any check fails, so it can gate a pipeline.

WHAT IT CHECKS AND WHY EACH ONE EARNED ITS PLACE

  structure      Shape/dtype drift silently breaks loaders. Also re-derives the
                 mask from the saved strength map -- if those two disagree, the
                 labels on disk are not the labels the generator computed.

  leakage        The published HERA benchmark shipped with 31 byte-identical
                 images shared between train and test and nobody noticed. Hash
                 everything and assume our own data is guilty until checked.

  dynamic range  Per-image min-max normalisation broke the tf_unet baseline
                 because the per-image maximum spanned 8.2x. This reports the
                 spread for this dataset, and the fixed-range LO/HI that fixed it.

  class weights  Measured from the train split, never pasted in.

  difficulty     What F1 a single global brightness threshold reaches with no
                 model at all. v3 scored 0.7421, which is why it was judged too
                 easy.

  bandpass cost  How many injected RFI pixels the bandpass pushed below the
                 label threshold, and where in the band they were lost.

Everything measured for later use (LO, HI, class weights) comes from TRAIN only.
The test split is read for structural checks and reported statistics, never used
to choose a constant.
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import copy


SPLITS = ("train", "val", "test")

EXPECTED = {
    # subdir      prefix          dtype        ndim
    "images":   ("spectrogram", np.float32, 2),
    "masks":    ("mask",        np.uint8,   2),
    "strength": ("strength",    np.float16, 2),
    "bandpass": ("bandpass",    np.float32, 1),
}


class Report:
    """Collects pass/fail lines so the summary can be printed in one place."""

    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok, message):
        if not ok:
            self.failures.append(message)
        return ok

    def warn(self, message):
        self.warnings.append(message)


def sha256_array(a):
    """Hash the raw pixel bytes, not the .npy container.

    Hashing the file would let two identical images differ because of header
    padding, and would let a re-save with different metadata look like new data.
    """
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pass 1 — structure, hashes, per-image statistics
# ---------------------------------------------------------------------------

def scan(dataset_dir, p, rep):
    """One pass over every array. Returns per-split accumulated statistics."""
    data = {}
    floor_keep = {}

    for split in SPLITS:
        split_dir = os.path.join(dataset_dir, split)
        if not rep.check(os.path.isdir(split_dir), f"missing split directory: {split_dir}"):
            continue

        files = {}
        for sub, (prefix, _, _) in EXPECTED.items():
            d = os.path.join(split_dir, sub)
            if not rep.check(os.path.isdir(d), f"missing directory: {d}"):
                files[sub] = []
                continue
            files[sub] = sorted(os.listdir(d))

        n = len(files["images"])
        for sub in EXPECTED:
            rep.check(len(files[sub]) == n,
                      f"{split}: {sub} has {len(files[sub])} files, images has {n}")

        stats = {
            "n": n, "hashes": [], "ids": [], "shapes": set(), "dtypes": set(),
            "img_min": [], "img_max": [], "p_lo": [], "p_hi": [],
            "rfi_frac": [], "mask_mismatch": 0, "bp_min": [], "bp_max": [],
        }
        keep_imgs, keep_masks = [], []

        # Which images to retain for the difficulty floor. Fixed stride rather
        # than a random subsample so the verifier is itself deterministic.
        stride = max(1, n // p.floor_max_images) if n else 1

        for i, fn in enumerate(files["images"]):
            idx = fn.split("_")[-1].split(".")[0]
            stats["ids"].append(idx)

            arrays = {}
            for sub, (prefix, dt, ndim) in EXPECTED.items():
                path = os.path.join(split_dir, sub, f"{prefix}_{idx}.npy")
                if not rep.check(os.path.exists(path), f"missing file: {path}"):
                    continue
                a = np.load(path)
                arrays[sub] = a
                rep.check(a.dtype == dt,
                          f"{split}/{sub}/{prefix}_{idx}: dtype {a.dtype}, expected {dt}")
                rep.check(a.ndim == ndim,
                          f"{split}/{sub}/{prefix}_{idx}: ndim {a.ndim}, expected {ndim}")

            if len(arrays) != len(EXPECTED):
                continue

            img, mask, strength, bp = (arrays["images"], arrays["masks"],
                                       arrays["strength"], arrays["bandpass"])

            stats["shapes"].add(img.shape)
            stats["dtypes"].add(str(img.dtype))
            rep.check(mask.shape == img.shape,
                      f"{split}/{idx}: mask shape {mask.shape} != image {img.shape}")
            rep.check(strength.shape == img.shape,
                      f"{split}/{idx}: strength shape {strength.shape} != image {img.shape}")
            rep.check(bp.shape == (img.shape[0],),
                      f"{split}/{idx}: bandpass shape {bp.shape} != ({img.shape[0]},)")
            rep.check(np.isfinite(img).all(), f"{split}/{idx}: image contains NaN or inf")
            rep.check(bool((bp > 0).all()), f"{split}/{idx}: bandpass has non-positive gain")
            rep.check(set(np.unique(mask)).issubset({0, 1}),
                      f"{split}/{idx}: mask is not binary")

            # The mask on disk must be exactly the strength map thresholded.
            stats["mask_mismatch"] += int(
                (mask != (strength > p.mask_sigma_threshold).astype(np.uint8)).sum())

            stats["hashes"].append(sha256_array(img))
            stats["img_min"].append(float(img.min()))
            stats["img_max"].append(float(img.max()))
            lo, hi = np.percentile(img, [p.lo_percentile, p.hi_percentile])
            stats["p_lo"].append(float(lo)); stats["p_hi"].append(float(hi))
            stats["rfi_frac"].append(float(mask.mean()))
            stats["bp_min"].append(float(bp.min())); stats["bp_max"].append(float(bp.max()))

            if i % stride == 0 and len(keep_imgs) < p.floor_max_images:
                keep_imgs.append(img.astype(np.float32))
                keep_masks.append(mask.astype(bool))

        data[split] = stats
        floor_keep[split] = (keep_imgs, keep_masks)

    return data, floor_keep


# ---------------------------------------------------------------------------
# Difficulty floor
# ---------------------------------------------------------------------------

def difficulty_floor(images, masks, n_thresholds):
    """
    Best F1 achievable by ONE global brightness threshold over the whole split.

    This is an oracle: the threshold is chosen with knowledge of the labels, so
    it is an upper bound on what a no-model baseline could do. That is
    deliberate -- it is a floor on task difficulty, not a proposed method.
    """
    if not images:
        return float("nan"), float("nan")
    scores = np.concatenate([im.ravel() for im in images])
    labels = np.concatenate([m.ravel() for m in masks])
    pos = int(labels.sum())
    if pos == 0:
        return float("nan"), float("nan")

    lo, hi = np.percentile(scores, [1.0, 99.9])
    best_f1, best_t = 0.0, lo
    for t in np.linspace(lo, hi, n_thresholds):
        pred = scores > t
        tp = int(np.count_nonzero(pred & labels))
        if tp == 0:
            continue
        prec = tp / int(np.count_nonzero(pred))
        rec = tp / pos
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

def render_previews(dataset_dir, out_dir, data, p):
    split = p.preview_split
    ids = data[split]["ids"][:p.n_preview_images]

    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 3.1 * len(ids)))
    if len(ids) == 1:
        axes = axes[np.newaxis, :]
    for row, idx in enumerate(ids):
        img = np.load(os.path.join(dataset_dir, split, "images", f"spectrogram_{idx}.npy"))
        mask = np.load(os.path.join(dataset_dir, split, "masks", f"mask_{idx}.npy"))
        vmin, vmax = np.percentile(img, [p.lo_percentile, p.hi_percentile])
        axes[row, 0].imshow(img, aspect="auto", cmap="hot", origin="upper",
                            vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{split}/{idx} — spectrogram", fontsize=9, fontweight="bold")
        axes[row, 0].set_ylabel("Freq channel", fontsize=8)
        axes[row, 1].imshow(mask, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")
        axes[row, 1].set_title(f"mask — {100*mask.mean():.1f}% RFI", fontsize=9, fontweight="bold")
        for c in (0, 1):
            axes[row, c].tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("Time sample", fontsize=8)
    axes[-1, 1].set_xlabel("Time sample", fontsize=8)
    plt.tight_layout()
    img_path = os.path.join(out_dir, "previews_images_masks.png")
    plt.savefig(img_path, dpi=p.preview_dpi, bbox_inches="tight")
    plt.close(fig)

    # Bandpass curves overlaid, so the per-image variation is visible directly.
    curves = []
    for idx in data[split]["ids"][:p.n_bandpass_curves]:
        curves.append(np.load(os.path.join(dataset_dir, split, "bandpass",
                                           f"bandpass_{idx}.npy")))
    C = np.stack(curves)
    n_freq = C.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    for c in C:
        ax[0].plot(c, lw=0.9, alpha=0.75)
    ax[0].set_title(f"{len(C)} per-image bandpass draws", fontweight="bold")
    ax[0].set_xlabel("Frequency channel"); ax[0].set_ylabel("Gain B(f)"); ax[0].grid(alpha=0.3)

    edge = max(4, int(p.preview_edge_frac * n_freq))
    for c in C:
        ax[1].plot(c, lw=0.9, alpha=0.75)
    ax[1].set_xlim(0, edge * 2)
    ax[1].set_title("low-frequency edge, zoomed", fontweight="bold")
    ax[1].set_xlabel("Frequency channel"); ax[1].grid(alpha=0.3)

    ax[2].plot(C.mean(axis=0), lw=1.4, color="black", label="mean")
    ax[2].fill_between(np.arange(n_freq), C.min(axis=0), C.max(axis=0),
                       alpha=0.25, color="#1f77b4", label="min–max envelope")
    ax[2].set_title("spread across images", fontweight="bold")
    ax[2].set_xlabel("Frequency channel"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    bp_path = os.path.join(out_dir, "bandpass_curves.png")
    plt.savefig(bp_path, dpi=p.preview_dpi, bbox_inches="tight")
    plt.close(fig)
    return img_path, bp_path


# ---------------------------------------------------------------------------
# Bit-exactness
# ---------------------------------------------------------------------------

def compare_trees(a_dir, b_dir, rep, ignore_ext):
    """SHA-256 every file in both trees and report differences."""
    def listing(root):
        out = {}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                out[os.path.relpath(full, root)] = full
        return out

    A, B = listing(a_dir), listing(b_dir)
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    shared = sorted(set(A) & set(B))

    compared, identical, differing, skipped = 0, 0, [], []
    for rel in shared:
        if os.path.splitext(rel)[1].lower() in ignore_ext:
            skipped.append(rel)
            continue
        compared += 1
        if sha256_file(A[rel]) == sha256_file(B[rel]):
            identical += 1
        else:
            differing.append(rel)

    rep.check(not only_a, f"{len(only_a)} files only in {a_dir}")
    rep.check(not only_b, f"{len(only_b)} files only in {b_dir}")
    rep.check(not differing, f"{len(differing)} files differ between the two generations")
    return {"compared": compared, "identical": identical, "differing": differing,
            "skipped": skipped, "only_a": only_a, "only_b": only_b}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Verify the v4 bandpass dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--compare_dir", default=None,
                    help="a second generation from the same seed; every file is "
                         "SHA-256 compared against it to prove bit-exactness")
    ap.add_argument("--out_dir", default=None,
                    help="where reports and PNGs go (default: <script dir>/verification)")

    g = ap.add_argument_group("reference values from the previous dataset")
    g.add_argument("--reference_rfi_fraction", type=float, default=0.1467,
                   help="v3 mean RFI pixel fraction")
    g.add_argument("--reference_floor_f1", type=float, default=0.7421,
                   help="v3 global-threshold F1")
    g.add_argument("--reference_max_spread", type=float, default=8.2,
                   help="v3 per-image maximum spread as documented")

    g = ap.add_argument_group("measurement settings")
    g.add_argument("--lo_percentile", type=float, default=0.5,
                   help="low percentile for the fixed normalisation range. Percentiles "
                        "rather than min/max because min/max are set by a single pixel.")
    g.add_argument("--hi_percentile", type=float, default=99.5)
    g.add_argument("--calibration_split", default="train",
                   help="split used to measure LO/HI and class weights; never test")
    g.add_argument("--floor_split", default="train",
                   help="split used for the difficulty floor")
    g.add_argument("--floor_max_images", type=int, default=200,
                   help="images retained for the floor sweep; a fixed stride is used "
                        "so the verifier stays deterministic")
    g.add_argument("--floor_thresholds", type=int, default=600)
    g.add_argument("--mask_sigma_threshold", type=float, default=0.5,
                   help="must match the generator, or the mask/strength check will fail")

    g = ap.add_argument_group("previews")
    g.add_argument("--preview_split", default="train")
    g.add_argument("--n_preview_images", type=int, default=6)
    g.add_argument("--n_bandpass_curves", type=int, default=12)
    g.add_argument("--preview_edge_frac", type=float, default=0.08)
    g.add_argument("--preview_dpi", type=int, default=130)
    g.add_argument("--ignore_ext", nargs="*", default=[],
                   help="extensions excluded from the bit-exactness comparison. Empty "
                        "by default: matplotlib's PNG output was checked and is "
                        "byte-identical between runs, so nothing needs excusing.")

    p = ap.parse_args()
    if p.out_dir is None:
        p.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification")
    os.makedirs(p.out_dir, exist_ok=True)

    rep = Report()
    results = {}

    print("=" * 78)
    print(f"VERIFYING  {p.dataset_dir}")
    print("=" * 78)

    data, floor_keep = scan(p.dataset_dir, p, rep)

    # --- 1. structure --------------------------------------------------------
    print("\n1. STRUCTURE")
    print("-" * 78)
    all_shapes = set()
    for split in SPLITS:
        s = data.get(split)
        if not s:
            continue
        all_shapes |= s["shapes"]
        shape = next(iter(s["shapes"])) if len(s["shapes"]) == 1 else s["shapes"]
        print(f"  {split:5s}  {s['n']:4d} images   shape {shape}   "
              f"dtype {'/'.join(sorted(s['dtypes']))}")
        rep.check(len(s["shapes"]) == 1, f"{split}: inconsistent image shapes {s['shapes']}")
        rep.check(s["mask_mismatch"] == 0,
                  f"{split}: {s['mask_mismatch']} pixels where mask != (strength > "
                  f"{p.mask_sigma_threshold})")
    rep.check(len(all_shapes) == 1, f"shapes differ across splits: {all_shapes}")
    print(f"  mask == (strength > {p.mask_sigma_threshold}) everywhere: "
          f"{'yes' if all(data[s]['mask_mismatch'] == 0 for s in data) else 'NO'}")
    results["shape"] = str(next(iter(all_shapes))) if len(all_shapes) == 1 else "INCONSISTENT"
    results["counts"] = {s: data[s]["n"] for s in data}

    # --- 2. metadata ---------------------------------------------------------
    print("\n2. METADATA")
    print("-" * 78)
    for split in SPLITS:
        path = os.path.join(p.dataset_dir, split, "metadata.jsonl")
        if not rep.check(os.path.exists(path), f"missing {path}"):
            continue
        rows = [json.loads(l) for l in open(path)]
        rep.check(len(rows) == data[split]["n"],
                  f"{split}: metadata has {len(rows)} rows, {data[split]['n']} images")
        recorded = np.array([r["rfi_fraction"] for r in rows])
        measured = np.array(data[split]["rfi_frac"])
        drift = float(np.abs(recorded - measured).max()) if len(rows) == len(measured) else float("nan")
        rep.check(not (drift > 1e-6), f"{split}: metadata rfi_fraction disagrees with the "
                                      f"masks on disk (max {drift:.2e})")
        print(f"  {split:5s}  {len(rows):4d} rows   rfi_fraction matches masks "
              f"(max drift {drift:.1e})")

    # --- 3. leakage ----------------------------------------------------------
    print("\n3. LEAKAGE  (SHA-256 over raw pixels)")
    print("-" * 78)
    sets = {}
    for split in SPLITS:
        if split not in data:
            continue
        h = data[split]["hashes"]
        uniq = set(h)
        dupes = len(h) - len(uniq)
        sets[split] = uniq
        rep.check(dupes == 0, f"{split}: {dupes} duplicate images within the split")
        print(f"  {split:5s}  {len(h):4d} images, {len(uniq):4d} unique, {dupes} duplicates")
    overlaps = {}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and a in sets and b in sets:
                n = len(sets[a] & sets[b])
                overlaps[f"{a}^{b}"] = n
                rep.check(n == 0, f"{a} and {b} share {n} byte-identical images")
                print(f"  {a} ∩ {b}: {n} shared")
    results["leakage"] = overlaps

    # --- 4. RFI fraction -----------------------------------------------------
    print("\n4. RFI PIXEL FRACTION")
    print("-" * 78)
    rfi_by_split = {}
    for split in SPLITS:
        if split not in data:
            continue
        f = np.array(data[split]["rfi_frac"])
        rfi_by_split[split] = float(f.mean())
        empty = int((f == 0).sum())
        print(f"  {split:5s}  mean {100*f.mean():6.2f}%   median {100*np.median(f):6.2f}%   "
              f"min {100*f.min():5.2f}%   max {100*f.max():6.2f}%   ({empty} images with no RFI)")
    overall = float(np.mean([x for s in data for x in data[s]["rfi_frac"]]))
    print(f"  overall {100*overall:.2f}%   vs v3's {100*p.reference_rfi_fraction:.2f}%   "
          f"(difference {100*(overall - p.reference_rfi_fraction):+.2f} points)")
    results["rfi_fraction"] = {"by_split": rfi_by_split, "overall": overall}

    # --- 5. dynamic range ----------------------------------------------------
    print("\n5. DYNAMIC RANGE")
    print("-" * 78)
    for split in SPLITS:
        if split not in data:
            continue
        mn = np.array(data[split]["img_min"]); mx = np.array(data[split]["img_max"])
        lo = np.array(data[split]["p_lo"]);    hi = np.array(data[split]["p_hi"])
        print(f"  {split:5s}  min [{mn.min():8.2f}, {mn.max():8.2f}]   "
              f"max [{mx.min():8.2f}, {mx.max():8.2f}]   "
              f"max-spread {mx.max()/mx.min():5.2f}x")
        print(f"         {p.lo_percentile}th pct mean {lo.mean():8.3f} (sd {lo.std():6.3f})   "
              f"{p.hi_percentile}th pct mean {hi.mean():8.3f} (sd {hi.std():6.3f})")
    all_max = np.array([x for s in data for x in data[s]["img_max"]])
    spread = float(all_max.max() / all_max.min())
    print(f"\n  PER-IMAGE MAXIMUM SPREAD ACROSS THE WHOLE DATASET: {spread:.2f}x")
    print(f"  (v3's documented figure was {p.reference_max_spread}x — the number that")
    print("   made per-image min-max normalisation break the tf_unet baseline)")
    results["max_spread"] = spread

    # --- 6. normalisation constants and class weights ------------------------
    cs = p.calibration_split
    print(f"\n6. FIXED-RANGE NORMALISATION AND CLASS WEIGHTS  ({cs} split only)")
    print("-" * 78)
    rep.check(cs != "test", "calibration split must not be test")
    LO = float(np.mean(data[cs]["p_lo"]))
    HI = float(np.mean(data[cs]["p_hi"]))
    mean_rfi = float(np.mean(data[cs]["rfi_frac"]))
    w = [0.5 / (1 - mean_rfi), 0.5 / mean_rfi]
    print(f"  LO = {LO:.4f}   HI = {HI:.4f}")
    print(f"    use as: np.clip((data - LO) / (HI - LO), 0, 1)")
    print(f"  measured RFI fraction on {cs}: {100*mean_rfi:.4f}%")
    print(f"  class weights [clean, rfi] = [{w[0]:.4f}, {w[1]:.4f}]")
    print("    NOTE: class weighting drives the tf_unet baseline into a dead state")
    print("    (its output ReLU zeroes the clean logit's gradient). These weights")
    print("    are safe for the hybrid, which emits raw logits.")
    results["normalisation"] = {"LO": LO, "HI": HI, "split": cs}
    results["class_weights"] = {"mean_rfi": mean_rfi, "weights": w}

    # --- 7. difficulty floor -------------------------------------------------
    print(f"\n7. DIFFICULTY FLOOR  ({p.floor_split} split, "
          f"{len(floor_keep[p.floor_split][0])} images)")
    print("-" * 78)
    imgs, msks = floor_keep[p.floor_split]
    f1, thr = difficulty_floor(imgs, msks, p.floor_thresholds)
    print(f"  best F1 from ONE global brightness threshold: {f1:.4f}  (at {thr:.3f})")
    print(f"  v3 reference: {p.reference_floor_f1:.4f}")
    delta = f1 - p.reference_floor_f1
    print(f"  difference: {delta:+.4f}  "
          f"({'HARDER than v3' if delta < 0 else 'EASIER than v3'})")
    results["difficulty_floor"] = {"f1": f1, "threshold": thr,
                                   "reference": p.reference_floor_f1, "delta": delta}

    # --- 8. bandpass ---------------------------------------------------------
    print("\n8. BANDPASS")
    print("-" * 78)
    for split in SPLITS:
        if split not in data:
            continue
        bmn = np.array(data[split]["bp_min"]); bmx = np.array(data[split]["bp_max"])
        print(f"  {split:5s}  gain min [{bmn.min():.5f}, {bmn.max():.5f}]   "
              f"max [{bmx.min():.4f}, {bmx.max():.4f}]")

    excl_path = os.path.join(p.dataset_dir, "exclusion_by_channel.json")
    if os.path.exists(excl_path):
        ex = json.load(open(excl_path))
        inj = np.array(ex["total"]["injected"]); exc = np.array(ex["total"]["excluded"])
        n_freq = ex["n_freq"]
        tot_i, tot_e = int(inj.sum()), int(exc.sum())
        print(f"\n  injected RFI pixels : {tot_i:,}")
        print(f"  excluded by bandpass: {tot_e:,}  ({100*tot_e/max(tot_i,1):.3f}%)")
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(inj > 0, exc / np.maximum(inj, 1), 0.0)
        for frac in (0.9, 0.5, 0.1):
            n_bad = int((rate > frac).sum())
            print(f"    channels losing >{100*frac:4.0f}% of injected RFI: "
                  f"{n_bad:4d} / {n_freq}  ({100*n_bad/n_freq:5.2f}% of the band)")
        results["bandpass_cost"] = {
            "injected": tot_i, "excluded": tot_e,
            "excluded_frac": tot_e / max(tot_i, 1),
            "channels_losing_over_half": int((rate > 0.5).sum()), "n_freq": n_freq}
    else:
        rep.warn(f"no exclusion_by_channel.json in {p.dataset_dir}; "
                 "skipping the bandpass cost report")

    # --- 9. previews ---------------------------------------------------------
    print("\n9. PREVIEWS")
    print("-" * 78)
    a, b = render_previews(p.dataset_dir, p.out_dir, data, p)
    print(f"  {a}")
    print(f"  {b}")

    # --- 10. bit-exactness ---------------------------------------------------
    if p.compare_dir:
        print("\n10. BIT-EXACT REGENERATION")
        print("-" * 78)
        cmp = compare_trees(p.dataset_dir, p.compare_dir, rep, set(p.ignore_ext))
        print(f"  files compared : {cmp['compared']}")
        print(f"  identical      : {cmp['identical']}")
        print(f"  differing      : {len(cmp['differing'])}")
        if cmp["skipped"]:
            print(f"  skipped ({'/'.join(p.ignore_ext)}): {len(cmp['skipped'])}")
        for f in cmp["differing"][:10]:
            print(f"    DIFFERS: {f}")
        results["bit_exact"] = {"compared": cmp["compared"],
                                "identical": cmp["identical"],
                                "differing": len(cmp["differing"])}

    # --- summary -------------------------------------------------------------
    print("\n" + "=" * 78)
    if rep.failures:
        print(f"FAILED — {len(rep.failures)} problem(s)")
        for f in rep.failures[:40]:
            print(f"  - {f}")
        if len(rep.failures) > 40:
            print(f"  ... and {len(rep.failures) - 40} more")
    else:
        print("ALL CHECKS PASSED")
    for wmsg in rep.warnings:
        print(f"  warning: {wmsg}")
    print("=" * 78)

    results["passed"] = not rep.failures
    results["failures"] = rep.failures
    results["warnings"] = rep.warnings
    with open(os.path.join(p.out_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMachine-readable report: {os.path.join(p.out_dir, 'report.json')}")

    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
