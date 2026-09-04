"""Stage 2: axis order, RFI morphology, contrast, leakage, preprocessing effect."""
import pickle, json, gc
import numpy as np

PKL = "/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl"
BASE = "/tmp/claude-1000/-home-ronaksingh-Downloads-Antigravity-Antigravity-x64/dbe9e263-0f9a-4d0b-9c18-6fc448e05e3b/scratchpad"
OUT = f"{BASE}/lofar_report2.json"
R = {}

def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=float)

def log(*a):
    print(*a, flush=True)

log("loading ...")
with open(PKL, "rb") as f:
    data = pickle.load(f)
TRX, TRY, TEX, TEY = data[0], data[1], data[2], data[3]
log("loaded")

# ============================================================ 1. AXIS PROFILES
# Mean over N and over the *other* spatial axis -> a 512-long profile per axis.
def axis_profiles(x, m, name, chunk=100, limit=2000):
    n = min(x.shape[0], limit)
    prof_ax1 = np.zeros(512, dtype=np.float64)  # collapse axis2 -> profile over rows
    prof_ax2 = np.zeros(512, dtype=np.float64)  # collapse axis1 -> profile over cols
    mprof_ax1 = np.zeros(512, dtype=np.float64)
    mprof_ax2 = np.zeros(512, dtype=np.float64)
    c = 0
    for i in range(0, n, chunk):
        b = x[i : i + chunk, :, :, 0].astype(np.float64)
        mb = m[i : i + chunk, :, :, 0]
        # use median-ish robust: clip extreme to avoid one bright RFI dominating
        bc = np.clip(b, 0, np.percentile(b, 99.5))
        prof_ax1 += bc.mean(axis=2).sum(axis=0)
        prof_ax2 += bc.mean(axis=1).sum(axis=0)
        mprof_ax1 += mb.mean(axis=2).sum(axis=0)
        mprof_ax2 += mb.mean(axis=1).sum(axis=0)
        c += b.shape[0]
        del b, mb, bc
    out = {
        "n_images_used": c,
        "profile_over_rows_axis1": (prof_ax1 / c).tolist(),
        "profile_over_cols_axis2": (prof_ax2 / c).tolist(),
        "maskprofile_over_rows_axis1": (mprof_ax1 / c).tolist(),
        "maskprofile_over_cols_axis2": (mprof_ax2 / c).tolist(),
    }
    R.setdefault("profiles", {})[name] = out
    save()
    # summary numbers
    for k in ["profile_over_rows_axis1", "profile_over_cols_axis2",
              "maskprofile_over_rows_axis1", "maskprofile_over_cols_axis2"]:
        v = np.array(out[k])
        log(f"{name}.{k}: mean={v.mean():.4g} std={v.std():.4g} "
            f"cv={v.std()/max(v.mean(),1e-12):.4f} min={v.min():.4g} max={v.max():.4g}")
    return out

p_tr = axis_profiles(TRX, TRY, "train", limit=2000)
p_te = axis_profiles(TEX, TEY, "test", limit=109)

# ============================================ 2. PERIODICITY (subband hunting)
def periodicity(profile, name):
    v = np.array(profile, dtype=np.float64)
    v = v - v.mean()
    # detrend with a wide moving average to kill the bandpass envelope
    k = 51
    kern = np.ones(k) / k
    trend = np.convolve(v, kern, mode="same")
    d = v - trend
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    ac /= ac[0]
    # find peaks in lag 2..60
    lags = np.arange(len(ac))
    win = (lags >= 2) & (lags <= 80)
    cand = ac.copy()
    cand[~win] = -np.inf
    top = np.argsort(cand)[::-1][:6]
    out = {"top_lags": [{"lag": int(l), "autocorr": float(ac[l])} for l in sorted(top)]}
    R.setdefault("periodicity", {})[name] = out
    save()
    log(f"periodicity[{name}]: " + ", ".join(f"lag{d_['lag']}={d_['autocorr']:.3f}" for d_ in out["top_lags"]))
    return out

periodicity(p_tr["profile_over_rows_axis1"], "train_rows_axis1")
periodicity(p_tr["profile_over_cols_axis2"], "train_cols_axis2")
periodicity(p_te["profile_over_rows_axis1"], "test_rows_axis1")
periodicity(p_te["profile_over_cols_axis2"], "test_cols_axis2")

# ========================================== 3. RFI MORPHOLOGY (run lengths)
def run_lengths(m, name, chunk=50, limit=1000):
    """Mean run length of True along axis1 (down rows) and axis2 (across cols)."""
    n = min(m.shape[0], limit)
    stats = {"axis1_down_rows": [], "axis2_across_cols": []}
    tot = {"axis1_down_rows": [0, 0], "axis2_across_cols": [0, 0]}  # [sum_len, n_runs]
    for i in range(0, n, chunk):
        b = m[i : i + chunk, :, :, 0]
        # runs along axis 1 (vertical, down the rows)
        pad = np.zeros((b.shape[0], 1, b.shape[2]), dtype=bool)
        bb = np.concatenate([pad, b, pad], axis=1)
        starts = (~bb[:, :-1] & bb[:, 1:]).sum()
        tot["axis1_down_rows"][0] += int(b.sum())
        tot["axis1_down_rows"][1] += int(starts)
        # runs along axis 2 (horizontal, across the cols)
        pad2 = np.zeros((b.shape[0], b.shape[1], 1), dtype=bool)
        bb2 = np.concatenate([pad2, b, pad2], axis=2)
        starts2 = (~bb2[:, :, :-1] & bb2[:, :, 1:]).sum()
        tot["axis2_across_cols"][0] += int(b.sum())
        tot["axis2_across_cols"][1] += int(starts2)
        del b, bb, bb2, pad, pad2
    out = {}
    for k, (s, nr) in tot.items():
        out[k] = {"total_true_px": s, "n_runs": nr,
                  "mean_run_length": (s / nr) if nr else 0.0}
    out["elongation_ratio_vertical_over_horizontal"] = (
        out["axis1_down_rows"]["mean_run_length"]
        / max(out["axis2_across_cols"]["mean_run_length"], 1e-12)
    )
    R.setdefault("morphology", {})[name] = out
    save()
    log(f"morphology[{name}]: " + json.dumps(out, indent=2))
    return out

run_lengths(TRY, "train_masks", limit=1000)
run_lengths(TEY, "test_masks", limit=109)

# ================================ 4. CONTRAST: flagged vs unflagged pixel values
def contrast(x, m, name, chunk=50, limit=500):
    n = min(x.shape[0], limit)
    fl, un = [], []
    rng = np.random.default_rng(1)
    for i in range(0, n, chunk):
        b = x[i : i + chunk, :, :, 0].astype(np.float64)
        mb = m[i : i + chunk, :, :, 0]
        f = b[mb]
        u = b[~mb]
        if f.size:
            fl.append(rng.choice(f, size=min(f.size, 200000), replace=False))
        if u.size:
            un.append(rng.choice(u, size=min(u.size, 200000), replace=False))
        del b, mb, f, u
    fl = np.concatenate(fl)
    un = np.concatenate(un)
    pcts = [1, 5, 25, 50, 75, 95, 99]
    out = {
        "n_flagged_sampled": int(fl.size),
        "n_unflagged_sampled": int(un.size),
        "flagged": {"mean": float(fl.mean()), "median": float(np.median(fl)),
                    "percentiles": {str(p): float(v) for p, v in zip(pcts, np.percentile(fl, pcts))}},
        "unflagged": {"mean": float(un.mean()), "median": float(np.median(un)),
                      "percentiles": {str(p): float(v) for p, v in zip(pcts, np.percentile(un, pcts))}},
        "median_ratio_flagged_over_unflagged": float(np.median(fl) / max(np.median(un), 1e-12)),
        "frac_flagged_below_unflagged_median": float((fl < np.median(un)).mean()),
        "frac_flagged_below_unflagged_p75": float((fl < np.percentile(un, 75)).mean()),
        "frac_flagged_above_unflagged_p99": float((fl > np.percentile(un, 99)).mean()),
    }
    R.setdefault("contrast", {})[name] = out
    save()
    log(f"contrast[{name}]: " + json.dumps(out, indent=2))
    del fl, un
    gc.collect()
    return out

contrast(TRX, TRY, "train", limit=500)
contrast(TEX, TEY, "test", limit=109)

log("STAGE 2 COMPLETE")
