"""Stage 5: trivial baselines, threshold floor, normalization strategy comparison."""
import pickle, json
import numpy as np

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
BASE = "/tmp/claude-1000/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b/scratchpad"
OUT = f"{BASE}/lofar_report5.json"
R = {}
def save():
    with open(OUT, "w") as f: json.dump(R, f, indent=2, default=float)
def log(*a): print(*a, flush=True)

with open(PKL, "rb") as f: data = pickle.load(f)
TRX, TRY, TEX, TEY = data[0], data[1], data[2], data[3]
log("loaded")

def prep(img):
    mu, sd = float(img.mean()), float(img.std())
    lo, hi = abs(mu - sd), mu + 4 * sd
    c = np.clip(img, max(lo, 1e-6), hi)
    l = np.log(c)
    r = l.max() - l.min()
    return (l - l.min()) / r if r > 0 else np.zeros_like(l)

# stack the whole (small) test set in prep space
X = np.stack([prep(TEX[i, :, :, 0].astype(np.float64)) for i in range(TEX.shape[0])])
Y = TEY[:, :, :, 0]
log("test prep stack", X.shape, "pos frac", Y.mean())

def f1_at(scores, y, th):
    p = scores > th
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-12), pr, rc

# ---- baseline 1: global constant threshold on per-image-normalised amplitude
ths = np.percentile(X, np.linspace(80, 99.9, 60))
best = max(((f1_at(X, Y, t)[0], t) for t in ths))
f1, pr, rc = f1_at(X, Y, best[1])
R["baseline_global_threshold_perimage_norm"] = {
    "best_F1": f1, "precision": pr, "recall": rc, "threshold": float(best[1]),
    "note": "oracle threshold picked on the test set itself -> optimistic upper bound"}

# ---- baseline 2: per-image sigma threshold (classic sigma clipping)
res = {}
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    P = np.zeros_like(Y)
    for i in range(X.shape[0]):
        im = X[i]; P[i] = im > (im.mean() + k * im.std())
    tp = int((P & Y).sum()); fp = int((P & ~Y).sum()); fn = int((~P & Y).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    res[f"{k}sigma"] = {"F1": 2 * pr * rc / max(pr + rc, 1e-12), "precision": pr, "recall": rc}
R["baseline_per_image_sigma_clip"] = res

# ---- baseline 3: per-COLUMN (frequency) sigma clip -- exploits the freq axis
res2 = {}
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    P = np.zeros_like(Y)
    for i in range(X.shape[0]):
        im = X[i]
        med = np.median(im, axis=0, keepdims=True)          # per-column baseline
        mad = np.median(np.abs(im - med), axis=0, keepdims=True) * 1.4826 + 1e-9
        P[i] = (im - med) / mad > k
    tp = int((P & Y).sum()); fp = int((P & ~Y).sum()); fn = int((~P & Y).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    res2[f"{k}sigma"] = {"F1": 2 * pr * rc / max(pr + rc, 1e-12), "precision": pr, "recall": rc}
R["baseline_per_column_MAD_clip"] = res2

# ---- trivial baselines
pos = float(Y.mean())
R["trivial_baselines"] = {
    "predict_all_RFI": {"precision": pos, "recall": 1.0, "F1": 2 * pos / (pos + 1)},
    "predict_no_RFI": {"precision": 0.0, "recall": 0.0, "F1": 0.0,
                       "pixel_accuracy": 1 - pos},
    "accuracy_of_all_zero_predictor": 1 - pos,
}
save(); log(json.dumps({k: R[k] for k in R}, indent=2))

# ------------------------- normalization strategy: per-image vs one global scale
gmin, gmax = np.inf, -np.inf
s = ss = c = 0.0
for i in range(0, 2000, 100):
    b = TRX[i:i+100].astype(np.float64)
    gmin = min(gmin, float(b.min())); gmax = max(gmax, float(b.max()))
    s += float(b.sum()); ss += float((b**2).sum()); c += b.size
    del b
gmean = s / c; gstd = (ss / c - gmean**2) ** .5
# per-image clip bounds spread
los, his = [], []
for i in range(0, TEX.shape[0]):
    im = TEX[i, :, :, 0].astype(np.float64)
    mu, sd = im.mean(), im.std()
    los.append(abs(mu - sd)); his.append(mu + 4 * sd)
los, his = np.array(los), np.array(his)
R["normalisation"] = {
    "global_min_first2000": gmin, "global_max_first2000": gmax,
    "global_mean": gmean, "global_std": gstd,
    "global_dynamic_range": gmax / max(gmin, 1e-9) if gmin > 0 else float("inf"),
    "per_image_clip_lo": {"min": float(los.min()), "median": float(np.median(los)), "max": float(los.max())},
    "per_image_clip_hi": {"min": float(his.min()), "median": float(np.median(his)), "max": float(his.max())},
    "per_image_hi_spread_ratio": float(his.max() / his.min()),
    "verdict": "per-image clip bounds vary by the ratio above; a single global scale would crush most images",
}
save(); log("normalisation:", json.dumps(R["normalisation"], indent=2))
log("STAGE 5 COMPLETE")
