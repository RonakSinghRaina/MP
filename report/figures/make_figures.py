"""Figures for the report. Every number here is copied from
RFI-project-context.md with its PART noted, so a figure can be traced to a
measurement. Run:  ~/torch-env/bin/python report/figures/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, ACCENT, NAVY = "#1E1E1E", "#9AA3B2", "#C0392B", "#22324F"
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)


# --- tf_unet: four controlled runs on the 276x600 synthetic set (PART 1) ---
def tfunet_controlled_runs():
    labels = ["1\nper-image\nweights on", "2\nper-image\nweights on\n(collapsed)",
              "3\nper-image\nweights off", "4\nfixed range\nweights off"]
    f1 = [0.3879, 0.3064, 0.7191, 0.9317]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(labels, f1, color=[MUTED, MUTED, NAVY, ACCENT], width=0.6, zorder=3)
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("max $F_1$ (test set)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    save(fig, "tfunet_controlled_runs")


# --- tf_unet on LOFAR against published methods (PART 12.10, 13.12) --------
def tfunet_lofar_ladder():
    names = ["$\\sigma$-clip", "tf_unet\nlr $10^{-3}$", "tf_unet\nlr $10^{-4}$",
             "AOFlagger", "Mesarcik\nU-Net", "RFI-Net"]
    v = [0.4103, 0.4901, 0.5482, 0.5698, 0.5876, 0.5979]
    e = [0, 0.0495, 0.0139, 0, 0.0031, 0]
    col = [MUTED, NAVY, NAVY, MUTED, MUTED, MUTED]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bars = ax.bar(names, v, yerr=e, color=col, width=0.6, zorder=3,
                  error_kw=dict(capsize=4, lw=1, ecolor="#555555"))
    for b, val, err in zip(bars, v, e):
        ax.text(b.get_x() + b.get_width() / 2, val + err + 0.012, f"{val:.4f}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel("max $F_1$ (109 test images)")
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    save(fig, "tfunet_lofar_ladder")


if __name__ == "__main__":
    tfunet_controlled_runs()
    tfunet_lofar_ladder()
    print("figures written to", HERE)
