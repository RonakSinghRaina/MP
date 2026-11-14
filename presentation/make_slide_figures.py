"""Figures for the final presentation (presentation/figures/*.png, 300 dpi).

Everything shown is measured from the saved data and predictions:
  intro_example.png   LOFAR test image 1: spectrogram and the human expert's RFI mask
  visual_results.png  LOFAR test images 14 and 49 (the 60th and 80th percentile of the
                      AOFlagger-vs-expert agreement, the same rule as Fig. 3.2 of the report):
                      spectrogram, expert mask, AOFlagger mask, tf_unet and Kestrel masks.
                      Seed-0 models, each at the threshold of its reported max F1; centre
                      472x472 pixels (tf_unet's output area); per-image F1 against the expert.
  problem_input.png, problem_output.png
                      LOFAR test image 1 and Kestrel's RFI probability for it (seed 0).
  kestrel_schematic.png, unet_schematic.png, label_pairs.png
                      the report's own figures, rasterised for the slides.

    ~/torch-env/bin/python presentation/make_slide_figures.py
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402
from sklearn.metrics import precision_recall_curve   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "report", "Images", "Figures"))
from lofar_data import load_lofar                    # noqa: E402
from make_data_figures import lofar_twins            # noqa: E402

MAROON = "#920815"
GREY = "#595959"
C0, C1 = 20, 492                                     # tf_unet's 472x472 output window
plt.rcParams.update({"font.family": "Carlito", "font.size": 13, "axes.edgecolor": "#8C8C8C"})


def show_spec(ax, img):
    """Display only: log of the raw power, colour range = this image's 1st..99.5th percentile."""
    v = np.log10(np.maximum(np.asarray(img, dtype=np.float64), 1e-3))
    lo, hi = np.percentile(v, [1, 99.5])
    ax.imshow(v, cmap="viridis", vmin=lo, vmax=hi, interpolation="none", aspect="auto")


def show_mask(ax, m):
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1, interpolation="none", aspect="auto")


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])


def f1(p, t):
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def max_f1_threshold(P, T):
    pc, rc, th = precision_recall_curve(T.ravel(), P.ravel())
    f = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f[:-1]))
    return float(th[k]), float(f[k])


def intro_example(d):
    i = 0                                           # "test image 1"
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.25))
    show_spec(ax[0], d.test_images[i, :, :, 0])
    show_mask(ax[1], np.asarray(d.test_masks[i, :, :, 0]).astype(bool))
    ax[0].set_title("Spectrogram", fontsize=14)
    ax[1].set_title("RFI marked by a human expert", fontsize=14)
    for a in ax:
        bare(a)
        a.set_xlabel("frequency channel →", fontsize=12, color=GREY)
    ax[0].set_ylabel("time →", fontsize=12, color=GREY, rotation=-90, labelpad=14)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.1, top=0.9, wspace=0.06)
    fig.savefig(os.path.join(OUT, "intro_example.png"), dpi=300)
    plt.close(fig)


def visual_results(d):
    twin = lofar_twins(d)
    T = np.asarray(d.test_masks[..., 0]).astype(bool)[:, C0:C1, C0:C1]
    A = np.stack([np.asarray(d.train_masks[j, :, :, 0]) for j in twin]).astype(bool)[:, C0:C1, C0:C1]
    K = np.asarray(np.load(os.path.join(ROOT, "runs/lofar/hybrid_b08_nocw_seed0/eval_test/test_pred_full512.npy"),
                           mmap_mode="r")[:, C0:C1, C0:C1], dtype=np.float32)
    U = np.load(os.path.join(ROOT, "runs/lofar/lr1e-4_seed0/eval_test/test_pred_crop472.npy"))
    tk, fk = max_f1_threshold(K, T)
    tu, fu = max_f1_threshold(U, T)
    assert abs(fu - 0.5640) < 1e-4, fu              # tf_unet seed 0 reported max F1
    print("seed 0 max F1 on the centre 472: Kestrel {:.4f} (thr {:.3f}), tf_unet {:.4f} (thr {:.3f})".format(
        fk, tk, fu, tu))
    # the same rule as Fig. 3.2: rank the test images by AOFlagger-vs-expert agreement
    agree = sorted(range(len(T)), key=lambda i: f1(
        np.asarray(d.train_masks[twin[i], :, :, 0]).astype(bool), np.asarray(d.test_masks[i, :, :, 0]).astype(bool)))
    pick = [agree[int(round(q * (len(agree) - 1)))] for q in (0.6, 0.8)]
    assert [p + 1 for p in pick] == [14, 49], pick
    heads = ["Spectrogram", "Human expert\n(the truth)", "AOFlagger\n(training labels)", "tf_unet", "Kestrel (ours)"]
    fig, ax = plt.subplots(2, 5, figsize=(11.6, 5.2))
    for r, i in enumerate(pick):
        show_spec(ax[r, 0], d.test_images[i, C0:C1, C0:C1, 0])
        masks = [T[i], A[i], U[i] > tu, K[i] > tk]
        for c, m in enumerate(masks, start=1):
            show_mask(ax[r, c], m)
            if c >= 2:
                s = f1(m, T[i])
                ax[r, c].set_xlabel("F1 = {:.2f}".format(s), fontsize=14,
                                    color=MAROON if c == 4 else "#262626",
                                    fontweight="bold" if c == 4 else "normal", labelpad=4)
        ax[r, 0].set_ylabel("Test image {}".format(i + 1), fontsize=13, color=GREY)
        print("test image {}: AOFlagger {:.3f}, tf_unet {:.3f}, Kestrel {:.3f}".format(
            i + 1, f1(A[i], T[i]), f1(U[i] > tu, T[i]), f1(K[i] > tk, T[i])))
    for r in range(2):
        for c in range(5):
            bare(ax[r, c])
            if c == 4:
                for s in ax[r, c].spines.values():
                    s.set_color(MAROON); s.set_linewidth(2.2)
    for c in range(5):
        ax[0, c].set_title(heads[c], fontsize=15, color=MAROON if c == 4 else "#262626",
                           fontweight="bold" if c == 4 else "normal")
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.07, top=0.88, wspace=0.07, hspace=0.3)
    fig.savefig(os.path.join(OUT, "visual_results.png"), dpi=300)
    plt.close(fig)


def problem_thumbs(d):
    """Problem-statement slide: test image 1 in, Kestrel's RFI probability (seed 0) out."""
    i = 0
    P = np.load(os.path.join(ROOT, "runs/lofar/hybrid_b08_nocw_seed0/eval_test/test_pred_full512.npy"),
                mmap_mode="r")[i]
    for name, draw in (("problem_input", lambda a: show_spec(a, d.test_images[i, :, :, 0])),
                       ("problem_output", lambda a: a.imshow(np.asarray(P, dtype=np.float32), cmap="Greys",
                                                             vmin=0, vmax=1, interpolation="none",
                                                             aspect="auto"))):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        draw(ax)
        bare(ax)
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        plt.close(fig)


def raster(pdf, name, dpi=300, trim_top=0):
    base = os.path.join(OUT, name)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf, base], check=True)
    if trim_top:
        im = Image.open(base + ".png")
        im.crop((0, trim_top, im.width, im.height)).save(base + ".png")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_lofar()
    intro_example(d)
    visual_results(d)
    problem_thumbs(d)
    rep = os.path.join(ROOT, "report", "Images")
    raster(os.path.join(rep, "hybrid_schematic.pdf"), "kestrel_schematic")
    raster(os.path.join(rep, "basic_unet_schematic.pdf"), "unet_schematic")
    raster(os.path.join(rep, "Figures", "lofar_label_pairs.pdf"), "label_pairs")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
