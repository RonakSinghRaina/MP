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


if __name__ == "__main__":
    tfunet_controlled_runs()
    print("figures written to", HERE)
