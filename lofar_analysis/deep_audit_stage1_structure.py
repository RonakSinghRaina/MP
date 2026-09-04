"""Deep structural analysis of LOFAR_Full_RFI_dataset.pkl.
Chunked throughout so we never make a full-size temporary copy.
Writes results to JSON incrementally so a crash loses nothing.
"""
import pickle, json, os, sys, gc
import numpy as np

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
OUT = "/tmp/claude-1000/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b/scratchpad/lofar_report.json"
R = {}

def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=float)

def log(*a):
    print(*a, flush=True)

log("loading pickle ...")
with open(PKL, "rb") as f:
    data = pickle.load(f)
log("loaded.")

# ---------------------------------------------------------------- structure
R["structure"] = {
    "container_type": str(type(data)),
    "n_elements": len(data),
    "elements": [
        {
            "index": i,
            "type": str(type(a)),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "nbytes_GiB": round(a.nbytes / 2**30, 3),
            "c_contiguous": bool(a.flags["C_CONTIGUOUS"]),
        }
        for i, a in enumerate(data)
    ],
}
save()
log(json.dumps(R["structure"], indent=2))

TRX, TRY, TEX, TEY = data[0], data[1], data[2], data[3]

# ------------------------------------------------------- chunked value stats
def value_stats(arr, name, chunk=200, sample_for_pct=400):
    """Streaming min/max/mean/std + NaN/Inf + percentiles from a subsample."""
    n = arr.shape[0]
    mn, mx = np.inf, -np.inf
    s = 0.0
    ss = 0.0
    cnt = 0
    n_nan = 0
    n_inf = 0
    n_neg = 0
    n_zero = 0
    for i in range(0, n, chunk):
        blk = arr[i : i + chunk].astype(np.float64, copy=False)
        finite = np.isfinite(blk)
        n_nan += int(np.isnan(blk).sum())
        n_inf += int(np.isinf(blk).sum())
        v = blk[finite]
        if v.size:
            mn = min(mn, float(v.min()))
            mx = max(mx, float(v.max()))
            s += float(v.sum())
            ss += float((v.astype(np.float64) ** 2).sum())
            cnt += v.size
            n_neg += int((v < 0).sum())
            n_zero += int((v == 0).sum())
        del blk, v, finite
    mean = s / cnt
    var = ss / cnt - mean**2
    # percentiles from a random subsample of images
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample_for_pct, n), replace=False)
    sub = arr[np.sort(idx)].astype(np.float64).ravel()
    sub = sub[np.isfinite(sub)]
    pcts = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    pv = np.percentile(sub, pcts)
    out = {
        "n_values": cnt,
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": float(np.sqrt(max(var, 0))),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_negative": n_neg,
        "n_exactly_zero": n_zero,
        "frac_negative": n_neg / cnt,
        "percentiles": {str(p): float(v) for p, v in zip(pcts, pv)},
        "pct_sample_images": int(len(idx)),
    }
    del sub
    gc.collect()
    R.setdefault("value_stats", {})[name] = out
    save()
    log(f"value_stats[{name}]:", json.dumps(out, indent=2))
    return out

value_stats(TRX, "train_spectrograms")
value_stats(TEX, "test_spectrograms")

# ----------------------------------------------------------- mask statistics
def mask_stats(arr, name, chunk=200):
    n = arr.shape[0]
    per_image = np.zeros(n, dtype=np.float64)
    total_true = 0
    for i in range(0, n, chunk):
        blk = arr[i : i + chunk]
        f = blk.reshape(blk.shape[0], -1).mean(axis=1, dtype=np.float64)
        per_image[i : i + blk.shape[0]] = f
        total_true += int(blk.sum())
        del blk, f
    out = {
        "n_images": int(n),
        "global_rfi_fraction": float(per_image.mean()),
        "total_flagged_pixels": total_true,
        "per_image_rfi_fraction": {
            "min": float(per_image.min()),
            "p1": float(np.percentile(per_image, 1)),
            "p25": float(np.percentile(per_image, 25)),
            "median": float(np.median(per_image)),
            "p75": float(np.percentile(per_image, 75)),
            "p99": float(np.percentile(per_image, 99)),
            "max": float(per_image.max()),
            "mean": float(per_image.mean()),
            "std": float(per_image.std()),
        },
        "n_images_completely_clean": int((per_image == 0).sum()),
        "n_images_under_0.1pct": int((per_image < 0.001).sum()),
        "n_images_over_10pct": int((per_image > 0.10).sum()),
        "n_images_over_50pct": int((per_image > 0.50).sum()),
        "imbalance_ratio_clean_to_rfi": float((1 - per_image.mean()) / max(per_image.mean(), 1e-12)),
    }
    R.setdefault("mask_stats", {})[name] = out
    save()
    log(f"mask_stats[{name}]:", json.dumps(out, indent=2))
    np.save(OUT.replace(".json", f"_{name}_perimage.npy"), per_image)
    return per_image

pi_train = mask_stats(TRY, "train_masks_AOFlagger")
pi_test = mask_stats(TEY, "test_masks_human_expert")

log("STAGE 1 COMPLETE")
