"""Report figures drawn by us (no copied figures), PART 41.

    ~/torch-env/bin/python report/Images/Figures/make_data_figures.py [name ...]

  lofar_label_pairs  Fig 3.2: LOFAR test images, AOFlagger mask beside the human mask
  rfi_types          Fig 2.1: a simulated spectrogram with the four kinds of RFI
  unet_ronneberger   Fig 2.2: the U-Net of Ronneberger et al. (2015), redrawn
  tfunet_schematic   Fig 3.3: tf_unet as used here (3 levels, 32 filters, valid padding)
  curves             learning curves and loss curves from runs/lofar/curves_*/curves.csv
  hera_vs_lofar      the HERA-vs-LOFAR label figure, from analysis/hera_vs_lofar/

Every number drawn comes from the data, a run file, or the authors' code; nothing is typed
in from memory. The figure functions print what they used.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)
    print("saved", name + ".pdf")


# ---------------------------------------------------------------------------------------
def lofar_twins(d):
    """Index of each LOFAR test image's byte-identical twin in the training set.
    The twin's training mask is AOFlagger's mask of the test image (PART 11.6)."""
    fp = {}
    for j in np.asarray(d.leak_train_idx):
        fp.setdefault(float(d.train_images[int(j), :, :, 0].sum(dtype=np.float64)), []).append(int(j))
    twin = []
    for i in range(d.test_images.shape[0]):
        hits = [j for j in fp.get(float(d.test_images[i, :, :, 0].sum(dtype=np.float64)), [])
                if np.array_equal(d.train_images[j], d.test_images[i])]
        assert len(hits) == 1, (i, hits)
        twin.append(hits[0])
    assert len(set(twin)) == len(twin)
    return np.array(twin)


def lofar_label_pairs():
    from lofar_data import load_lofar
    d = load_lofar()
    twin = lofar_twins(d)
    tp = fp = fn = 0
    rows = []
    for i, j in enumerate(twin):
        a = np.asarray(d.train_masks[j, :, :, 0]).astype(bool)
        h = np.asarray(d.test_masks[i, :, :, 0]).astype(bool)
        t, f, n = int((a & h).sum()), int((a & ~h).sum()), int((~a & h).sum())
        tp += t; fp += f; fn += n
        rows.append((i, 2 * t / max(2 * t + f + n, 1)))
    pooled = 2 * tp / (2 * tp + fp + fn)
    assert abs(pooled - 0.5698) < 5e-5, pooled            # Mesarcik et al. Table 2
    order = sorted(rows, key=lambda r: r[1])
    pick = [order[int(round(q * (len(order) - 1)))] for q in (0.2, 0.4, 0.6, 0.8)]
    print("pooled AOFlagger vs human F1 {:.7f}; shown (test index, per-image F1):".format(pooled),
          [(i, round(f, 3)) for i, f in pick])

    fig, ax = plt.subplots(len(pick), 3, figsize=(6.3, 2.15 * len(pick)))
    heads = ["Spectrogram", "AOFlagger mask\n(training labels)", "Human expert mask\n(test labels)"]
    for r, (i, f1) in enumerate(pick):
        # display only: log of the raw power, colour range = this image's 1st..99.5th percentile
        img = np.log10(np.maximum(np.asarray(d.test_images[i, :, :, 0], dtype=np.float64), 1e-3))
        lo, hi = np.percentile(img, [1, 99.5])
        a = np.asarray(d.train_masks[twin[i], :, :, 0]).astype(bool)
        h = np.asarray(d.test_masks[i, :, :, 0]).astype(bool)
        ax[r, 0].imshow(img, cmap="viridis", vmin=lo, vmax=hi, interpolation="none", aspect="auto")
        ax[r, 1].imshow(a, cmap="Greys", vmin=0, vmax=1, interpolation="none", aspect="auto")
        ax[r, 2].imshow(h, cmap="Greys", vmin=0, vmax=1, interpolation="none", aspect="auto")
        ax[r, 0].set_ylabel("Test image {}\nagreement {:.2f}".format(i + 1, f1), fontsize=9)
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values():
                s.set_color("#888888")
            if r == 0:
                ax[r, c].set_title(heads[c], fontsize=9)
    for c in range(3):
        ax[-1, c].set_xlabel("frequency channel $\\rightarrow$", fontsize=8)
    fig.text(0.005, 0.5, "$\\leftarrow$ time", rotation=90, va="center", fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "lofar_label_pairs")


# ---------------------------------------------------------------------------------------
def rfi_types():
    """Fig 2.1: a SIMULATED spectrogram (made here, not real data) showing the four kinds of
    RFI that Chen et al. (2023) describe in FAST pulsar data."""
    rng = np.random.default_rng(7)
    nf, nt = 400, 91
    freq = np.linspace(1.0, 1.5, nf)                       # GHz
    bandpass = 1.0 - 0.6 * ((freq - 1.25) / 0.25) ** 8     # gain falls at the band edges
    img = 50.0 * bandpass[:, None] + rng.normal(0.0, 4.0, (nf, nt))
    for lo, hi in ((1.17, 1.20), (1.215, 1.235), (1.255, 1.29)):     # constant broadband
        m = (freq >= lo) & (freq <= hi)
        img[m, :] += 55.0 + rng.normal(0.0, 8.0, (int(m.sum()), nt))
    img[:, 20] += 65.0                                     # instantaneous wideband
    m = (freq >= 1.14) & (freq <= 1.32)                    # broadband, a few minutes
    img[np.ix_(m, np.arange(43, 49))] += 75.0
    img[int(np.argmin(np.abs(freq - 1.35))), :] += 85.0    # constant narrowband

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, nt - 1, freq[0], freq[-1]], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("power (arbitrary units)")
    ax.set_xlabel("time step")
    ax.set_ylabel("frequency (GHz)")
    style = dict(fontsize=8.5, color="#1E1E1E", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=0.6),
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.3))
    ax.annotate("instantaneous\nwideband RFI", xy=(20, 1.43), xytext=(9, 1.465), **style)
    ax.annotate("constant narrowband RFI", xy=(70, 1.35), xytext=(70, 1.43), **style)
    ax.annotate("broadband RFI lasting\na few minutes", xy=(46, 1.30), xytext=(40, 1.395), **style)
    ax.annotate("constant broadband RFI", xy=(80, 1.23), xytext=(72, 1.08), **style)
    save(fig, "rfi_types_simulated")


# ---------------------------------------------------------------------------------------
def unet_ronneberger():
    """Fig 2.2: the U-Net of Ronneberger et al. (2015), redrawn. Every channel count and
    size is from their Fig. 1 (572x572 input, valid 3x3 convolutions, 388x388 output).
    As in their figure, a box where a max-pool or up-conv arrow arrives or leaves carries
    no channel label (its count is the same as the box it comes from)."""
    from matplotlib.patches import Rectangle
    C_MAP, C_EDGE = "#F2C38B", "#7A4B1E"
    COL = dict(conv="#22324F", copy="#9AA3B2", pool="#C0392B", up="#1E8449", one="#8E44AD")
    H = [2.4, 1.6, 1.05, 0.7, 0.5]                            # box height per level
    Y = [0.0]
    for l in range(1, 5):                                     # room for size labels + arrow
        Y.append(Y[-1] - (H[l - 1] + H[l]) / 2 - 1.15)
    W = lambda ch: 0.07 + 0.034 * np.log2(max(ch, 1))         # box width from channels
    GAP = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 6.8))

    def box(x, lvl, ch, size, copied=False, label=True, size_label=True):
        w, h = W(ch), H[lvl]
        ax.add_patch(Rectangle((x, Y[lvl] - h / 2), w, h, fc="white" if copied else C_MAP,
                               ec=C_EDGE, lw=0.7, hatch="////" if copied else None))
        if label:
            ax.text(x + w / 2, Y[lvl] + h / 2 + 0.05, str(ch), ha="center", va="bottom", fontsize=7)
        if size_label:
            ax.text(x + w / 2, Y[lvl] - h / 2 - 0.05, "{}$\\times${}".format(size, size), ha="center",
                    va="top", fontsize=6, rotation=90, color="#444444")
        return x, x + w

    def arrow(x0, y0, x1, y1, kind):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=COL[kind], lw=1.4, mutation_scale=8,
                                    shrinkA=0, shrinkB=0))

    enc = [[(1, 572), (64, 570), (64, 568)], [(64, 284), (128, 282), (128, 280)],
           [(128, 140), (256, 138), (256, 136)], [(256, 68), (512, 66), (512, 64)],
           [(512, 32), (1024, 30), (1024, 28)]]
    dec = {3: [(512, 56), (512, 54), (512, 52)], 2: [(256, 104), (256, 102), (256, 100)],
           1: [(128, 200), (128, 198), (128, 196)], 0: [(64, 392), (64, 390), (64, 388)]}
    SIZE_DROP = 0.8                 # room the rotated size label takes under a box
    last = {}                       # (level, side) -> (x_left, x_right) of the last box
    x = 0.6
    for lvl, maps in enumerate(enc):
        if lvl > 0:                 # max pool: straight down from the last box above
            a0, a1 = last[(lvl - 1, "e")]
            xc = (a0 + a1) / 2
            x = xc - W(maps[0][0]) / 2
            arrow(xc, Y[lvl - 1] - H[lvl - 1] / 2 - SIZE_DROP, xc, Y[lvl] + H[lvl] / 2 + 0.04, "pool")
        prev = None
        for k, (ch, sz) in enumerate(maps):
            first_pooled = lvl > 0 and k == 0
            starts_up = lvl == 4 and k == len(maps) - 1
            l, r = box(x, lvl, ch, sz, label=not (first_pooled or starts_up))
            if prev is not None:
                arrow(prev, Y[lvl], l, Y[lvl], "conv")
            prev, x = r, r + GAP
        last[(lvl, "e")] = (l, r)
    for lvl in (3, 2, 1, 0):
        b0, b1 = last[(lvl + 1, "e" if lvl == 3 else "d")]
        xc = (b0 + b1) / 2
        ch, sz = dec[lvl][0]
        xb = xc - W(ch) / 2                                   # up-conv result, above the box below
        arrow(xc, Y[lvl + 1] + H[lvl + 1] / 2 + 0.04, xc, Y[lvl] - H[lvl] / 2 - 0.04, "up")
        wl, _ = box(xb - W(ch), lvl, ch, sz, copied=True, label=False)          # copied map
        _, r = box(xb, lvl, ch, sz, label=False, size_label=False)              # up-conv map
        ax.text(xb, Y[lvl] + H[lvl] / 2 + 0.05, str(2 * ch), ha="center", va="bottom", fontsize=7)
        e0, e1 = last[(lvl, "e")]
        arrow(e1 + 0.05, Y[lvl], wl - 0.05, Y[lvl], "copy")
        prev, x = r, r + GAP
        for k, (ch2, sz2) in enumerate(dec[lvl][1:]):
            starts_up = lvl > 0 and k == len(dec[lvl]) - 2
            l, r = box(x, lvl, ch2, sz2, label=not starts_up)
            arrow(prev, Y[lvl], l, Y[lvl], "conv")
            prev, x = r, r + GAP
        last[(lvl, "d")] = (l, r)
    l, r = box(x, 0, 2, 388)                                  # output: 2 channels
    arrow(prev, Y[0], l, Y[0], "one")
    ax.text(0.5, Y[0], "input\nimage", ha="right", va="center", fontsize=8)
    ax.text(r + 0.12, Y[0], "output\nmap", ha="left", va="center", fontsize=8)

    key = [("conv", "3$\\times$3 convolution, then ReLU"), ("copy", "copy and crop"),
           ("pool", "max pool 2$\\times$2"), ("up", "up-convolution 2$\\times$2"),
           ("one", "1$\\times$1 convolution")]
    kx, ky = last[(2, "d")][1] + 0.9, Y[2] + 0.2
    for i, (k, t) in enumerate(key):
        yy = ky - 0.45 * i
        arrow(kx, yy, kx + 0.5, yy, k)
        ax.text(kx + 0.65, yy, t, va="center", fontsize=7.5)
    yy = ky - 0.45 * 5
    ax.add_patch(Rectangle((kx + 0.12, yy - 0.16), 0.13, 0.32, fc=C_MAP, ec=C_EDGE, lw=0.7))
    ax.text(kx + 0.65, yy, "feature map", va="center", fontsize=7.5)
    yy = ky - 0.45 * 6
    ax.add_patch(Rectangle((kx + 0.12, yy - 0.16), 0.13, 0.32, fc="white", ec=C_EDGE, lw=0.7, hatch="////"))
    ax.text(kx + 0.65, yy, "copied feature map", va="center", fontsize=7.5)
    ax.set_xlim(-0.5, r + 1.1)
    ax.set_ylim(Y[4] - H[4] / 2 - 0.75, Y[0] + H[0] / 2 + 0.35)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "unet_ronneberger_redrawn")


# ---------------------------------------------------------------------------------------
def tfunet_schematic():
    """Fig 3.3: tf_unet as trained here. Sizes measured by building the network in
    TensorFlow on a 512x512 input (PART 41): 508, 254, 250, 125, 121, 242, 238, 476, 472."""
    from matplotlib.patches import FancyBboxPatch
    NAVY, MID, GREY, INK = "#0F2043", "#2E4A7D", "#5A6170", "#1E1E1E"
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def block(x, y, w, h, title, ops, fill=MID):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec="none"))
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, ops, ha="center", va="center", color="white", fontsize=7.2)

    def arrow(x0, y0, x1, y1, dashed=False, color=INK):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9,
                                    linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))

    w, h = 3.3, 0.8
    E1, E2, BN = (0.1, 4.35), (0.75, 2.95), (3.35, 1.45)
    D2, D1 = (5.95, 2.95), (6.6, 4.35)
    block(*E1, w, h, "Encoder 1 · 32 × 508 × 508", "2 × (3×3 conv → ReLU)")
    block(*E2, w, h, "Encoder 2 · 64 × 250 × 250", "2 × (3×3 conv → ReLU)")
    block(*BN, w, h, "Bottleneck · 128 × 121 × 121", "2 × (3×3 conv → ReLU)", fill=NAVY)
    block(*D2, w, h, "Decoder 2 · 64 × 238 × 238", "join skip → 2 × (3×3 conv → ReLU)")
    block(*D1, w, h, "Decoder 1 · 32 × 472 × 472", "join skip → 2 × (3×3 conv → ReLU)")

    ax.text(0.1, 5.65, "▼ Input: spectrogram, 1 × 512 × 512 (time × frequency)",
            fontsize=7.6, color=GREY, va="center")
    ax.text(9.9, 5.65, "Output: RFI probability, 472 × 472 ▲",
            fontsize=7.6, color=GREY, va="center", ha="right")
    arrow(D1[0] + w - 0.4, D1[1] + h, D1[0] + w - 0.4, 5.5)
    ax.text(D1[0] + w - 0.45, (D1[1] + h + 5.5) / 2, "1×1 conv → ReLU → softmax ",
            fontsize=7, color=GREY, va="center", ha="right")
    # down: max pool
    arrow(E1[0] + 0.4, E1[1], E1[0] + 0.4, E2[1] + h)
    ax.text(E1[0] + 0.45, (E1[1] + E2[1] + h) / 2, " max-pool: half size (508 → 254)",
            fontsize=7, color=GREY, va="center")
    arrow(E2[0] + 0.4, E2[1], E2[0] + 0.4, BN[1] + 0.4)
    arrow(E2[0] + 0.4, BN[1] + 0.4, BN[0], BN[1] + 0.4)
    ax.text(E2[0] + 0.45, (E2[1] + BN[1] + 0.4) / 2 + 0.05, " max-pool: half size (250 → 125)",
            fontsize=7, color=GREY, va="center")
    # up: up-convolution
    xr2 = D2[0] + w - 0.4
    arrow(BN[0] + w, BN[1] + 0.4, xr2, BN[1] + 0.4)
    arrow(xr2, BN[1] + 0.4, xr2, D2[1])
    ax.text(xr2 - 0.05, (D2[1] + BN[1] + 0.4) / 2 + 0.05, "up-conv: double size (121 → 242) ",
            fontsize=7, color=GREY, va="center", ha="right")
    xr1 = D1[0] + w - 0.4
    arrow(xr1, D2[1] + h, xr1, D1[1])
    ax.text(xr1 - 0.05, (D1[1] + D2[1] + h) / 2, "up-conv: double size (238 → 476) ",
            fontsize=7, color=GREY, va="center", ha="right")
    # skip connections
    arrow(E1[0] + w, E1[1] + h / 2, D1[0], D1[1] + h / 2, dashed=True, color=GREY)
    ax.text((E1[0] + w + D1[0]) / 2, E1[1] + h / 2 + 0.1, "skip: copy, cut to\n476 × 476, join",
            fontsize=7, color=GREY, ha="center", va="bottom")
    arrow(E2[0] + w, E2[1] + h / 2, D2[0], D2[1] + h / 2, dashed=True, color=GREY)
    ax.text((E2[0] + w + D2[0]) / 2, E2[1] + h / 2 + 0.1, "skip: copy, cut to\n242 × 242, join",
            fontsize=7, color=GREY, ha="center", va="bottom")
    # notes
    for i, t in enumerate(["No padding: every 3×3 conv trims 1 pixel from each side, so a 512 × 512 input",
                           "gives a 472 × 472 output, and the copies sent along the skips are cut to fit.",
                           "Numbers: channels × height × width.  465,986 parameters in total."]):
        ax.text(0.1, 0.85 - 0.3 * i, t, fontsize=7.6, color=INK, va="center")
    save(fig, "tfunet_schematic")


# ---------------------------------------------------------------------------------------
def hera_rim_example():
    """HERA: exact label, AOFlagger label, Kestrel and tf_unet (seed 0, each at the threshold
    of its reported max F1, analysis/hera_vs_lofar/error_split.json) on one 96x96 region.
    The region is the one with the most Kestrel false alarms that AOFlagger also made
    (searched over all test images, stride 32, inside tf_unet's centre 472x472)."""
    import json
    from matplotlib.colors import ListedColormap
    from hera_data import load_hera
    d = load_hera()
    E = json.load(open(os.path.join(ROOT, "analysis", "hera_vs_lofar", "error_split.json")))
    thK = [r for r in E["hera"]["kestrel"] if r["seed"] == 0][0]["threshold"]
    thU = [r for r in E["hera"]["tf_unet"] if r["seed"] == 0][0]["threshold"]
    PK = np.load(os.path.join(ROOT, "runs/hera/hybrid_b08_nocw_seed0/eval_test/test_pred_full512.npy"),
                 mmap_mode="r")
    PU = np.load(os.path.join(ROOT, "runs/hera/tfunet_lr1e-4_seed0/eval_test/test_pred_crop472.npy"),
                 mmap_mode="r")
    C0, S = 20, 96
    best = (-1, 0, 0, 0)
    for i in range(d.test_images.shape[0]):
        T = np.asarray(d.test_masks[i, C0:C0 + 472, C0:C0 + 472, 0]).astype(bool)
        A = np.asarray(d.test_masks_aof[i, C0:C0 + 472, C0:C0 + 472, 0]).astype(bool)
        K = np.asarray(PK[i, C0:C0 + 472, C0:C0 + 472]) > thK
        copied = K & ~T & A
        for r in range(0, 472 - S + 1, 32):
            for c in range(0, 472 - S + 1, 32):
                n = int(copied[r:r + S, c:c + S].sum())
                if n > best[0]:
                    best = (n, i, r, c)
    n, i, r, c = best
    T = np.asarray(d.test_masks[i, C0:C0 + 472, C0:C0 + 472, 0]).astype(bool)[r:r + S, c:c + S]
    A = np.asarray(d.test_masks_aof[i, C0:C0 + 472, C0:C0 + 472, 0]).astype(bool)[r:r + S, c:c + S]
    K = (np.asarray(PK[i, C0:C0 + 472, C0:C0 + 472]) > thK)[r:r + S, c:c + S]
    U = (np.asarray(PU[i]) > thU)[r:r + S, c:c + S]
    print("HERA test image {} (0-based), centre rows {}..{}, cols {}..{}; Kestrel copied false alarms "
          "there: {}; thresholds Kestrel {:.4f}, tf_unet {:.4f}".format(i, r, r + S, c, c + S, n, thK, thU))
    # 0 = clean, 1 = correct RFI, 2 = false alarm (flagged, not RFI), 3 = missed RFI
    cmap = ListedColormap(["#FFFFFF", "#1E1E1E", "#D62728", "#7FB3E6"])
    code = lambda m: np.where(m & T, 1, np.where(m & ~T, 2, np.where(~m & T, 3, 0)))
    panels = [("Exact label\n(simulator)", np.where(T, 1, 0)), ("AOFlagger label\n(used for training)", code(A)),
              ("Kestrel", code(K)), ("tf_unet", code(U))]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.25))
    for a, (title, img) in zip(ax, panels):
        a.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        a.set_title(title, fontsize=8.5)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#888888")
        fa = int((img == 2).sum())
        if title not in ("Exact label\n(simulator)",):
            a.set_xlabel("{} false alarm{}".format(fa, "" if fa == 1 else "s"), fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#1E1E1E", label="RFI, correctly flagged"),
                        Patch(fc="#D62728", label="false alarm (flagged, but not RFI)"),
                        Patch(fc="#7FB3E6", label="missed RFI")],
               loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save(fig, "hera_rim_example")


# ---------------------------------------------------------------------------------------
def _load_curves(pattern, seeds=(0, 1, 2)):
    """Read <run>/curves.csv for each seed that exists. Returns {column: (epochs, values[seed])}."""
    out = {}
    runs = []
    for s in seeds:
        f = os.path.join(ROOT, pattern.format(s), "curves.csv")
        if os.path.exists(f):
            runs.append(np.genfromtxt(f, delimiter=",", names=True))
    if not runs:
        return None, 0
    n_ep = min(int(r["epoch"].max()) for r in runs)
    for col in runs[0].dtype.names:
        if col == "epoch":
            continue
        ep = runs[0]["epoch"][:n_ep]
        vals = np.stack([r[col][:n_ep] for r in runs])
        keep = ~np.isnan(vals).any(axis=0)
        out[col] = (ep[keep], vals[:, keep])
    return out, len(runs)


def _band(ax, ep, vals, color, label, ls="-"):
    m = vals.mean(axis=0)
    ax.plot(ep, m, color=color, lw=1.4, ls=ls, label=label)
    if vals.shape[0] > 1:
        sd = vals.std(axis=0, ddof=1)
        ax.fill_between(ep, m - sd, m + sd, color=color, alpha=0.18, lw=0)


def curves():
    """Learning curve (validation max F1 per epoch) and loss curves, LOFAR, from the
    repeat runs with --log_curves (scripts/run_lofar_curves.sh). Mean and spread over the
    seeds that have finished; the number of seeds is printed and must be 3 for the report."""
    K, nK = _load_curves("runs/lofar/curves_hybrid_b08_nocw_seed{}")
    U, nU = _load_curves("runs/lofar/curves_tfunet_lr1e-4_seed{}")
    print("seeds found: Kestrel {}, tf_unet {}".format(nK, nU))
    CK, CU = "#C0392B", "#22324F"

    # 1. learning curve: validation max F1
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for D, c, lab in ((K, CK, "Kestrel"), (U, CU, "tf_unet")):
        if D:
            _band(ax, *D["val_max_f1"], c, lab)
    ax.set_xlabel("epoch (700 training images each)")
    ax.set_ylabel("validation max $F_1$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "learning_curves")

    # 2. loss curves: Kestrel's two loss terms, tf_unet's cross-entropy
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    if K:
        _band(ax[0], K["train_dice_loss"][0], 1 - K["train_dice_loss"][1], CK, "training")
        _band(ax[0], K["val_dice_loss"][0], 1 - K["val_dice_loss"][1], CK, "validation", ls="--")
        ax[0].set_title("Kestrel: Dice score", fontsize=9)
        _band(ax[1], *K["train_ce"], CK, "training")
        _band(ax[1], *K["val_ce"], CK, "validation", ls="--")
        ax[1].set_title("Kestrel: cross-entropy part", fontsize=9)
    if U:
        _band(ax[2], *U["train_loss"], CU, "training (with\nweight penalty)")
        _band(ax[2], *U["val_ce"], CU, "validation", ls="--")
        ax[2].set_title("tf_unet: cross-entropy", fontsize=9)
    for a in ax:
        a.set_xlabel("epoch", fontsize=8.5)
        a.tick_params(labelsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "loss_curves")


if __name__ == "__main__":
    names = sys.argv[1:] or ["lofar_label_pairs"]
    for nm in names:
        globals()[nm]()
