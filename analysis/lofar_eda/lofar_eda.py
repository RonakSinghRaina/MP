"""Basic data analysis of the LOFAR dataset (Mesarcik et al. 2022).

Everything is measured from data/lofar/LOFAR_npy (memory-mapped, so RAM stays low).
Writes lofar_eda.json and four figures next to this file. Run:
    ~/torch-env/bin/python analysis/lofar_eda/lofar_eda.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lofar_data import load_lofar, preprocess  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, ACCENT, NAVY = "#1E1E1E", "#9AA3B2", "#C0392B", "#22324F"
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def split(d, seed=0, val_frac=0.1):
    # same split as experiments/lofar_hybrid.py (seed 0)
    rng = np.random.default_rng(seed)
    idx = d.clean_train_idx.copy()
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    return idx[n_val:], idx[:n_val]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    d = load_lofar()
    out = {}
    n_tr, H, W, _ = d.train_images.shape
    n_te = d.test_images.shape[0]
    tr_idx, va_idx = split(d)
    out["shapes"] = {"train": list(d.train_images.shape), "test": list(d.test_images.shape)}
    out["splits"] = {"released_train": n_tr, "test": n_te, "leaked": int(len(d.leak_train_idx)),
                     "clean": int(len(d.clean_train_idx)), "train": int(len(tr_idx)),
                     "val": int(len(va_idx))}

    # ---- pass over every training image -----------------------------------
    rfi_frac_tr = np.zeros(n_tr)
    zero_frac_tr = np.zeros(n_tr)
    col_tr = np.zeros(W)
    row_tr = np.zeros(H)
    in_train = np.zeros(n_tr, bool)
    in_train[tr_idx] = True
    hashes = []
    mins, maxs, sums, sq = [], [], 0.0, 0.0
    rng = np.random.default_rng(0)
    sample_raw = []
    for i in range(n_tr):
        x = np.asarray(d.train_images[i, :, :, 0])
        m = np.asarray(d.train_masks[i, :, :, 0])
        rfi_frac_tr[i] = m.mean()
        zero_frac_tr[i] = (x == 0).mean()
        if in_train[i]:            # profiles over the 6621 training images only
            col_tr += m.sum(0)
            row_tr += m.sum(1)
        hashes.append(hashlib.md5(x.tobytes()).hexdigest())
        mins.append(float(x.min())); maxs.append(float(x.max()))
        sums += float(x.sum(dtype=np.float64)); sq += float((x.astype(np.float64) ** 2).sum())
        sample_raw.append(rng.choice(x.ravel(), 200))
        if i % 1000 == 0:
            print("train", i, flush=True)
    npx = n_tr * H * W
    sample_raw = np.concatenate(sample_raw)
    out["raw_train"] = {
        "min": min(mins), "max": max(maxs), "mean": sums / npx,
        "std": float(np.sqrt(sq / npx - (sums / npx) ** 2)),
        "median_sampled": float(np.median(sample_raw)),
        "p1_sampled": float(np.percentile(sample_raw, 1)),
        "p99_sampled": float(np.percentile(sample_raw, 99)),
        "images_with_any_zero": int((zero_frac_tr > 0).sum()),
        "images_all_zero": int((zero_frac_tr == 1).sum()),
    }
    out["duplicates_in_train"] = {"unique_images": len(set(hashes)), "of": n_tr}

    # ---- test images ------------------------------------------------------
    rfi_frac_te = np.zeros(n_te)
    col_te = np.zeros(W)
    row_te = np.zeros(H)
    te_hash = []
    for i in range(n_te):
        x = np.asarray(d.test_images[i, :, :, 0])
        m = np.asarray(d.test_masks[i, :, :, 0])
        rfi_frac_te[i] = m.mean()
        col_te += m.sum(0); row_te += m.sum(1)
        te_hash.append(hashlib.md5(x.tobytes()).hexdigest())
    pos = {h: k for k, h in enumerate(hashes)}
    pair = [pos.get(h, -1) for h in te_hash]
    out["test_in_train"] = int(sum(p >= 0 for p in pair))

    def frac_stats(fr):
        return {"pooled_pct": None, "median_pct": float(np.median(fr) * 100),
                "mean_pct": float(fr.mean() * 100), "max_pct": float(fr.max() * 100),
                "images_0pct": int((fr == 0).sum()), "images_over_10pct": int((fr > 0.10).sum()),
                "images_100pct": int((fr == 1).sum())}
    s_all = frac_stats(rfi_frac_tr); s_all["pooled_pct"] = float(rfi_frac_tr.mean() * 100)
    s_tr = frac_stats(rfi_frac_tr[tr_idx]); s_tr["pooled_pct"] = float(rfi_frac_tr[tr_idx].mean() * 100)
    s_va = frac_stats(rfi_frac_tr[va_idx]); s_va["pooled_pct"] = float(rfi_frac_tr[va_idx].mean() * 100)
    s_te = frac_stats(rfi_frac_te); s_te["pooled_pct"] = float(rfi_frac_te.mean() * 100)
    out["rfi_fraction"] = {"released_train_7500": s_all, "train": s_tr, "val": s_va, "test": s_te}

    # ---- where the RFI sits: frequency channels vs time -------------------
    def concentration(col):
        c = np.sort(col)[::-1]
        return {"top16_channels_pct": float(c[:16].sum() / c.sum() * 100),
                "top_channels": [int(k) for k in np.argsort(col)[::-1][:5]]}
    out["by_frequency"] = {"train": concentration(col_tr), "test": concentration(col_te)}
    out["by_frequency"]["train"]["cv_columns"] = float(col_tr.std() / col_tr.mean())
    out["by_frequency"]["train"]["cv_rows"] = float(row_tr.std() / row_tr.mean())
    out["by_frequency"]["test"]["cv_columns"] = float(col_te.std() / col_te.mean())
    out["by_frequency"]["test"]["cv_rows"] = float(row_te.std() / row_te.mean())

    # ---- brightness of RFI vs clean (test, after the paper's preprocessing) --
    # brightness in units of the image's own median clean pixel
    rfi_v, cln_v, ratio = [], [], []
    for i in range(n_te):
        raw = np.asarray(d.test_images[i, :, :, 0]).astype(np.float64)
        m = np.asarray(d.test_masks[i, :, :, 0]).astype(bool)
        med = np.median(raw[~m])
        rfi_v.append(raw[m] / med); cln_v.append(rng.choice(raw[~m], 5000) / med)
        ratio.append(np.median(raw[m]) / med)
    rfi_v = np.concatenate(rfi_v); cln_v = np.concatenate(cln_v)
    raw_rfi = np.concatenate([np.asarray(d.test_images[i, :, :, 0])[np.asarray(d.test_masks[i, :, :, 0])]
                              for i in range(n_te)])
    raw_cln_med = float(np.median(np.concatenate(
        [rng.choice(np.asarray(d.test_images[i, :, :, 0])[~np.asarray(d.test_masks[i, :, :, 0])], 5000)
         for i in range(n_te)])))
    out["brightness_test"] = {
        "median_rfi_over_median_clean_per_image": float(np.median(ratio)),
        "rfi_below_clean_median_pct": float((raw_rfi < raw_cln_med).mean() * 100),
        "rfi_below_own_image_clean_median_pct": float((rfi_v < 1).mean() * 100),
        "rfi_above_2x_pct": float((rfi_v > 2).mean() * 100),
        "rfi_above_10x_pct": float((rfi_v > 10).mean() * 100),
        "clean_above_2x_pct": float((cln_v > 2).mean() * 100),
    }

    # ---- AOFlagger vs the human expert on the same 109 images -------------
    tp = fp = fn = 0
    for k in range(n_te):
        a = np.asarray(d.train_masks[pair[k], :, :, 0]).astype(bool)
        h = np.asarray(d.test_masks[k, :, :, 0]).astype(bool)
        tp += int((a & h).sum()); fp += int((a & ~h).sum()); fn += int((~a & h).sum())
    p, r, f = f1(tp, fp, fn)
    out["aoflagger_vs_expert"] = {"precision": p, "recall": r, "f1": f,
                                  "aoflagger_rfi_pixels": tp + fp, "expert_rfi_pixels": tp + fn}

    with open(os.path.join(HERE, "lofar_eda.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

    # ================= figures ==============================================
    # 1. per-image RFI fraction
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bins = np.logspace(-5, 0, 41)
    for fr, c, lab in [(rfi_frac_tr[tr_idx], NAVY, "training, 6621 images (AOFlagger labels)"),
                       (rfi_frac_te, ACCENT, "test, 109 images (expert labels)")]:
        ax.hist(fr, bins=bins, weights=np.full(len(fr), 100 / len(fr)), histtype="step",
                lw=1.8, color=c, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of pixels flagged as RFI in one image")
    ax.set_ylabel("% of images")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    fig.savefig(os.path.join(HERE, "fig_rfi_fraction_per_image.png")); plt.close(fig)

    # 2. RFI by frequency channel and by time
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=False)
    axs[0].plot(col_tr / (len(tr_idx) * H) * 100, color=NAVY, lw=1.2, label="training")
    axs[0].plot(col_te / (n_te * H) * 100, color=ACCENT, lw=1.2, label="test")
    axs[0].set_xlabel("frequency channel (column)"); axs[0].set_ylabel("% of pixels flagged")
    axs[0].legend(frameon=False, fontsize=8.5)
    axs[1].plot(row_tr / (len(tr_idx) * W) * 100, color=NAVY, lw=1.2)
    axs[1].plot(row_te / (n_te * W) * 100, color=ACCENT, lw=1.2)
    axs[1].set_xlabel("time sample (row)")
    for a in axs:
        a.grid(color="#E5E5E5"); a.set_ylim(bottom=0)
    fig.savefig(os.path.join(HERE, "fig_rfi_by_frequency_time.png")); plt.close(fig)

    # 3. brightness of RFI vs clean pixels (test, preprocessed)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    b = np.logspace(-1.5, 3, 61)
    for v, c, lab in [(cln_v, MUTED, "clean pixels"), (rfi_v, ACCENT, "RFI pixels (expert labels)")]:
        ax.hist(np.clip(v, b[0], b[-1]), bins=b, weights=np.full(len(v), 100 / len(v)),
                histtype="step", lw=1.8, color=c, label=lab, zorder=3)
    ax.axvline(1, color="#555555", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("pixel brightness / median clean pixel of the same image")
    ax.set_ylabel("% of pixels")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    fig.savefig(os.path.join(HERE, "fig_brightness_rfi_vs_clean.png")); plt.close(fig)

    # 4. one test image: data, AOFlagger, expert
    k = int(np.argsort(rfi_frac_te)[len(rfi_frac_te) // 2])
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axs[0].imshow(preprocess(np.asarray(d.test_images[k, :, :, 0])), cmap="viridis", aspect="auto")
    axs[1].imshow(np.asarray(d.train_masks[pair[k], :, :, 0]), cmap="gray_r", aspect="auto")
    axs[2].imshow(np.asarray(d.test_masks[k, :, :, 0]), cmap="gray_r", aspect="auto")
    for a, t in zip(axs, ["spectrogram", "AOFlagger mask", "expert mask"]):
        a.set_title(t, fontsize=10); a.set_xlabel("frequency channel")
    axs[0].set_ylabel("time sample")
    for a in axs[1:]:
        a.set_yticks([])
    fig.savefig(os.path.join(HERE, "fig_aoflagger_vs_expert.png")); plt.close(fig)
    print("test image shown in fig 4:", k)


if __name__ == "__main__":
    main()
