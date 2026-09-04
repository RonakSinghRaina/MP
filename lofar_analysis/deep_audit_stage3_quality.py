"""Stage 3: per-image periodicity, leakage, pathological images, preprocessing, figures."""
import pickle, json, gc, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
BASE = "/tmp/claude-1000/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b/scratchpad"
OUT = f"{BASE}/lofar_report3.json"
R = {}
def save():
    with open(OUT, "w") as f: json.dump(R, f, indent=2, default=float)
def log(*a): print(*a, flush=True)

log("loading ...");
with open(PKL, "rb") as f: data = pickle.load(f)
TRX, TRY, TEX, TEY = data[0], data[1], data[2], data[3]
log("loaded")

# ================================ 1. PER-IMAGE AUTOCORR (subband hunt, crop-safe)
def per_image_ac(x, name, n=200, axis="cols"):
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False))
    acc = np.zeros(120)
    c = 0
    for i in idx:
        img = x[i, :, :, 0].astype(np.float64)
        img = np.clip(img, 0, np.percentile(img, 99))
        prof = img.mean(axis=0) if axis == "cols" else img.mean(axis=1)
        d = prof - np.convolve(prof, np.ones(41)/41, mode="same")
        d = d[20:-20]
        ac = np.correlate(d, d, mode="full")[len(d)-1:]
        if ac[0] > 0:
            acc += ac[:120] / ac[0]; c += 1
    acc /= c
    top = sorted(np.argsort(acc[2:100])[::-1][:8] + 2)
    out = {"n_images": c, "top_lags": [{"lag": int(l), "ac": float(acc[l])} for l in top],
           "ac_at_selected": {str(l): float(acc[l]) for l in [6,7,10,12,13,14,16,20,24,28,32,48,64]}}
    R.setdefault("per_image_autocorr", {})[f"{name}_{axis}"] = out
    save(); log(f"per_image_ac[{name}_{axis}]:", json.dumps(out["ac_at_selected"]))
    log("   top:", [(d_["lag"], round(d_["ac"],3)) for d_ in out["top_lags"]])
    return acc

per_image_ac(TRX, "train", n=200, axis="cols")
per_image_ac(TRX, "train", n=200, axis="rows")
per_image_ac(TEX, "test", n=109, axis="cols")
per_image_ac(TEX, "test", n=109, axis="rows")

# ============================== 2. MASK AXIS CONCENTRATION (which axis is fixed?)
def mask_concentration(m, name, limit=1000, chunk=50):
    """For each image: what fraction of its flagged px live in its top-k rows vs top-k cols?"""
    n = min(m.shape[0], limit)
    fr_rows, fr_cols = [], []
    for i in range(0, n, chunk):
        b = m[i:i+chunk, :, :, 0]
        rs = b.sum(axis=2).astype(np.float64)   # (B,512) per-row counts
        cs = b.sum(axis=1).astype(np.float64)   # (B,512) per-col counts
        tot = b.reshape(b.shape[0], -1).sum(axis=1).astype(np.float64)
        ok = tot > 50
        if ok.any():
            k = 16
            rs_s = np.sort(rs[ok], axis=1)[:, ::-1][:, :k].sum(axis=1)
            cs_s = np.sort(cs[ok], axis=1)[:, ::-1][:, :k].sum(axis=1)
            fr_rows.append(rs_s / tot[ok]); fr_cols.append(cs_s / tot[ok])
        del b, rs, cs
    fr_rows = np.concatenate(fr_rows); fr_cols = np.concatenate(fr_cols)
    out = {"n_images": int(fr_rows.size),
           "frac_in_top16_rows": {"mean": float(fr_rows.mean()), "median": float(np.median(fr_rows))},
           "frac_in_top16_cols": {"mean": float(fr_cols.mean()), "median": float(np.median(fr_cols))},
           "verdict_axis_with_more_concentration": "cols" if fr_cols.mean() > fr_rows.mean() else "rows"}
    R.setdefault("mask_concentration", {})[name] = out
    save(); log(f"mask_concentration[{name}]:", json.dumps(out, indent=2))
    return out

mask_concentration(TRY, "train", limit=1000)
mask_concentration(TEY, "test", limit=109)

# ================================================ 3. PATHOLOGICAL TRAIN IMAGES
pi = np.load(f"{BASE}/lofar_report_train_masks_AOFlagger_perimage.npy")
bad = np.where(pi > 0.5)[0]
allflag = np.where(pi >= 0.999)[0]
R["pathological"] = {
    "n_over_50pct_flagged": int(bad.size),
    "indices_over_50pct": bad.tolist()[:60],
    "n_100pct_flagged": int(allflag.size),
    "indices_100pct": allflag.tolist()[:60],
    "n_over_20pct": int((pi > 0.2).sum()),
    "n_over_5pct": int((pi > 0.05).sum()),
}
save(); log("pathological:", json.dumps(R["pathological"], indent=2))

# ============================================= 4. ZEROS in train spectrograms
def zero_structure(x, limit=1500, chunk=50):
    n = min(x.shape[0], limit)
    imgs_with_zero, whole_row, whole_col, zfrac = 0, 0, 0, []
    for i in range(0, n, chunk):
        b = x[i:i+chunk, :, :, 0]
        z = (b == 0)
        per = z.reshape(z.shape[0], -1).mean(axis=1)
        zfrac.append(per)
        imgs_with_zero += int((per > 0).sum())
        whole_row += int((z.all(axis=2)).sum())
        whole_col += int((z.all(axis=1)).sum())
        del b, z
    zfrac = np.concatenate(zfrac)
    out = {"n_images_checked": int(zfrac.size), "n_images_with_any_zero": imgs_with_zero,
           "frac_images_with_any_zero": float((zfrac > 0).mean()),
           "mean_zero_frac_when_present": float(zfrac[zfrac > 0].mean()) if (zfrac>0).any() else 0.0,
           "max_zero_frac": float(zfrac.max()),
           "n_fully_zero_rows": whole_row, "n_fully_zero_cols": whole_col,
           "n_images_fully_zero": int((zfrac >= 0.999).sum())}
    R["zeros_train"] = out; save(); log("zeros_train:", json.dumps(out, indent=2))
zero_structure(TRX)

# ==================================================== 5. LEAKAGE / DUPLICATES
def hashes(x, limit=None, chunk=100):
    n = x.shape[0] if limit is None else min(limit, x.shape[0])
    hs = []
    for i in range(0, n, chunk):
        b = x[i:i+chunk]
        for j in range(b.shape[0]):
            hs.append(hashlib.md5(np.ascontiguousarray(b[j]).tobytes()).hexdigest())
        del b
    return hs
h_te = hashes(TEX)
h_tr = hashes(TRX)
set_te = set(h_te)
overlap = [h for h in h_tr if h in set_te]
R["leakage"] = {
    "n_train": len(h_tr), "n_test": len(h_te),
    "n_unique_train": len(set(h_tr)), "n_unique_test": len(set_te),
    "n_duplicate_train_images": len(h_tr) - len(set(h_tr)),
    "n_train_images_identical_to_a_test_image": len(overlap),
}
save(); log("leakage:", json.dumps(R["leakage"], indent=2))

# ============================== 6. PAPER PREPROCESSING EFFECT (clip->log->minmax)
def prep_effect(x, name, n=300):
    rng = np.random.default_rng(2)
    idx = np.sort(rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False))
    raw_dr, out_stats = [], []
    for i in idx:
        img = x[i, :, :, 0].astype(np.float64)
        mu, sd = img.mean(), img.std()
        lo, hi = abs(mu - sd), mu + 4 * sd
        raw_dr.append(img.max() / max(img.min() + 1e-9, 1e-9))
        c = np.clip(img, lo, hi)
        l = np.log(c)
        s = (l - l.min()) / max(l.max() - l.min(), 1e-12)
        out_stats.append([s.mean(), s.std(), np.percentile(s, 1), np.percentile(s, 99)])
    o = np.array(out_stats)
    out = {"n": len(idx),
           "raw_dynamic_range_median": float(np.median(raw_dr)),
           "after_prep_mean": float(o[:,0].mean()), "after_prep_std_within_image": float(o[:,1].mean()),
           "after_prep_p1": float(o[:,2].mean()), "after_prep_p99": float(o[:,3].mean()),
           "across_image_spread_of_mean": float(o[:,0].std())}
    R.setdefault("preprocessing", {})[name] = out; save()
    log(f"preprocessing[{name}]:", json.dumps(out, indent=2))
prep_effect(TRX, "train"); prep_effect(TEX, "test")

# ================================================================ 7. FIGURES
def prep(img):
    mu, sd = img.mean(), img.std()
    c = np.clip(img, abs(mu - sd), mu + 4 * sd)
    l = np.log(c)
    return (l - l.min()) / max(l.max() - l.min(), 1e-12)

fig, ax = plt.subplots(4, 5, figsize=(20, 16))
picks_tr = [0, 3, 11, int(np.argsort(pi)[len(pi)//2]), int(bad[0]) if bad.size else 7]
for c, i in enumerate(picks_tr):
    ax[0, c].imshow(prep(TRX[i,:,:,0]), aspect="auto", cmap="viridis")
    ax[0, c].set_title(f"TRAIN raw->prep #{i}\nrfi={pi[i]*100:.2f}%", fontsize=9)
    ax[1, c].imshow(TRY[i,:,:,0], aspect="auto", cmap="viridis")
    ax[1, c].set_title(f"TRAIN mask (AOFlagger) #{i}", fontsize=9)
pi_te = np.load(f"{BASE}/lofar_report_test_masks_human_expert_perimage.npy")
picks_te = [0, 5, 20, int(np.argmax(pi_te)), int(np.argmin(pi_te))]
for c, i in enumerate(picks_te):
    ax[2, c].imshow(prep(TEX[i,:,:,0]), aspect="auto", cmap="viridis")
    ax[2, c].set_title(f"TEST raw->prep #{i}\nrfi={pi_te[i]*100:.2f}%", fontsize=9)
    ax[3, c].imshow(TEY[i,:,:,0], aspect="auto", cmap="viridis")
    ax[3, c].set_title(f"TEST mask (HUMAN) #{i}", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{BASE}/fig_overview.png", dpi=65); plt.close()

# profile figure
r2 = json.load(open(f"{BASE}/lofar_report2.json"))
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
ax[0,0].plot(r2["profiles"]["train"]["profile_over_rows_axis1"]); ax[0,0].set_title("TRAIN data: mean vs ROW index (axis1)")
ax[0,1].plot(r2["profiles"]["train"]["profile_over_cols_axis2"], color="tab:orange"); ax[0,1].set_title("TRAIN data: mean vs COL index (axis2)")
ax[1,0].plot(r2["profiles"]["train"]["maskprofile_over_rows_axis1"]); ax[1,0].set_title("TRAIN mask: RFI frac vs ROW index (axis1)")
ax[1,1].plot(r2["profiles"]["train"]["maskprofile_over_cols_axis2"], color="tab:orange"); ax[1,1].set_title("TRAIN mask: RFI frac vs COL index (axis2)")
for a in ax.ravel(): a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{BASE}/fig_profiles.png", dpi=75); plt.close()
log("STAGE 3 COMPLETE")
