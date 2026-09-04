"""Stage 4: leak indices, AOFlagger-vs-human agreement, contrast in preprocessed space."""
import pickle, json, hashlib
import numpy as np

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
BASE = "/tmp/claude-1000/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b/scratchpad"
OUT = f"{BASE}/lofar_report4.json"
R = {}
def save():
    with open(OUT, "w") as f: json.dump(R, f, indent=2, default=float)
def log(*a): print(*a, flush=True)

with open(PKL, "rb") as f: data = pickle.load(f)
TRX, TRY, TEX, TEY = data[0], data[1], data[2], data[3]
log("loaded")

# ------------------------------------------------- exact train<->test mapping
def h(a): return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
te_map = {h(TEX[i]): i for i in range(TEX.shape[0])}
pairs = []
for i in range(TRX.shape[0]):
    k = h(TRX[i])
    if k in te_map:
        pairs.append((i, te_map[k]))
log(f"{len(pairs)} train->test matches")
R["leak_pairs"] = {"n": len(pairs), "train_indices": [p[0] for p in pairs],
                   "test_indices_covered": sorted({p[1] for p in pairs})}
save()

# --------------------------- AOFlagger (train label) vs HUMAN (test label) on same image
TP = FP = FN = TN = 0
per_img = []
for tr_i, te_i in pairs:
    a = TRY[tr_i, :, :, 0]      # AOFlagger
    hm = TEY[te_i, :, :, 0]     # human
    tp = int((a & hm).sum()); fp = int((a & ~hm).sum())
    fn = int((~a & hm).sum()); tn = int((~a & ~hm).sum())
    TP += tp; FP += fp; FN += fn; TN += tn
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    per_img.append([p, r, 2 * p * r / max(p + r, 1e-12), a.mean(), hm.mean()])
per_img = np.array(per_img)
prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
R["aoflagger_vs_human"] = {
    "n_image_pairs": len(pairs),
    "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    "pixelwise_precision": prec, "pixelwise_recall": rec,
    "pixelwise_F1": 2 * prec * rec / max(prec + rec, 1e-12),
    "iou": TP / max(TP + FP + FN, 1),
    "paper_reported_AOFlagger_F1_table2": 0.5698,
    "mean_per_image_F1": float(np.nanmean(per_img[:, 2])),
    "median_per_image_F1": float(np.nanmedian(per_img[:, 2])),
    "mean_AOFlagger_rfi_frac_on_these": float(per_img[:, 3].mean()),
    "mean_human_rfi_frac_on_these": float(per_img[:, 4].mean()),
    "aoflagger_over_under_flag_ratio": float(per_img[:, 3].mean() / max(per_img[:, 4].mean(), 1e-12)),
}
save(); log("aoflagger_vs_human:", json.dumps(R["aoflagger_vs_human"], indent=2))

# ------------------------------- contrast AFTER the paper's preprocessing (z within image)
def prep(img):
    mu, sd = float(img.mean()), float(img.std())
    lo, hi = abs(mu - sd), mu + 4 * sd
    if not np.isfinite(hi) or hi <= lo: return None
    c = np.clip(img, max(lo, 1e-6), hi)
    l = np.log(c)
    rng = l.max() - l.min()
    if rng <= 0: return None
    return (l - l.min()) / rng

rows = []
for i in range(TEX.shape[0]):
    s = prep(TEX[i, :, :, 0].astype(np.float64))
    if s is None: continue
    m = TEY[i, :, :, 0]
    if m.sum() < 20: continue
    f, u = s[m], s[~m]
    rows.append([np.median(f), np.median(u), f.mean(), u.mean(), u.std(),
                 (np.median(f) - np.median(u)) / max(u.std(), 1e-9),
                 (f < np.percentile(u, 75)).mean(),
                 (f < np.median(u)).mean(),
                 (f > np.percentile(u, 99)).mean()])
rows = np.array(rows)
R["prep_space_contrast_test"] = {
    "n_images": int(rows.shape[0]),
    "median_flagged": float(rows[:, 0].mean()), "median_clean": float(rows[:, 1].mean()),
    "separation_in_clean_sigmas_mean": float(rows[:, 5].mean()),
    "separation_in_clean_sigmas_median": float(np.median(rows[:, 5])),
    "frac_flagged_below_clean_p75_mean": float(rows[:, 6].mean()),
    "frac_flagged_below_clean_median_mean": float(rows[:, 7].mean()),
    "frac_flagged_above_clean_p99_mean": float(rows[:, 8].mean()),
    "note": "our synthetic v4 label is by construction >0.5 sigma; compare that to separation_in_clean_sigmas",
}
save(); log("prep_space_contrast_test:", json.dumps(R["prep_space_contrast_test"], indent=2))

# ---------------------------------- how many flagged px are 'invisible' per image (test)
inv = rows[:, 7]
R["invisibility"] = {
    "per_image_frac_flagged_below_clean_median": {
        "min": float(inv.min()), "p25": float(np.percentile(inv, 25)),
        "median": float(np.median(inv)), "p75": float(np.percentile(inv, 75)),
        "max": float(inv.max())},
    "n_images_where_over_half_of_RFI_is_below_clean_median": int((inv > 0.5).sum()),
}
save(); log("invisibility:", json.dumps(R["invisibility"], indent=2))

# ------------------------------------------- clean train index list (usable subset)
pi = np.load(f"{BASE}/lofar_report_train_masks_AOFlagger_perimage.npy")
leak = set(p[0] for p in pairs)
bad_all = set(np.where(pi > 0.5)[0].tolist())
# dead images (all zero)
dead = []
for i in range(TRX.shape[0]):
    if i in bad_all or i in leak:
        continue
keep = [i for i in range(TRX.shape[0]) if i not in leak and i not in bad_all]
R["recommended_train_subset"] = {
    "n_original": int(TRX.shape[0]),
    "n_removed_leak": len(leak),
    "n_removed_over50pct_flagged": len(bad_all),
    "n_removed_total": len(leak | bad_all),
    "n_keep": len(keep),
}
np.save(f"{BASE}/lofar_clean_train_idx.npy", np.array(sorted(keep), dtype=np.int32))
np.save(f"{BASE}/lofar_leak_train_idx.npy", np.array(sorted(leak), dtype=np.int32))
save(); log("recommended_train_subset:", json.dumps(R["recommended_train_subset"], indent=2))
log("STAGE 4 COMPLETE")
