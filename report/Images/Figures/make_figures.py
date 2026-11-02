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


# --- final LOFAR comparison incl. the hybrid (PART 13.12) --------------------
def hybrid_lofar_ladder(include_later=False):
    # RFDL: van Zyl & Grobler 2024, Table 9 (best F1, AOFlagger labels).
    # Swin-UNETR: Ouyang et al. 2024, Table 2 -- the largest of three sizes (408.1 M parameters).
    # include_later=False leaves these two out (the report currently omits them).
    names = ["$\\sigma$-clip", "tf_unet", "AOFlagger", "Mesarcik\nU-Net",
             "RFI-Net", "RFDL", "Swin-\nUNETR\n(408 M)", "Kestrel"]
    v = [0.4103, 0.5482, 0.5698, 0.5876, 0.5979, 0.639, 0.6401, 0.6603]
    e = [0, 0.0139, 0, 0.0031, 0, 0.007, 0, 0.0040]
    shown = ["0.4103", "0.5482", "0.5698", "0.5876", "0.5979", "0.639", "0.6401", "0.6603"]
    col = [MUTED, NAVY, MUTED, MUTED, MUTED, MUTED, MUTED, ACCENT]
    if not include_later:
        keep = [i for i, n in enumerate(names) if n not in ("RFDL", "Swin-\nUNETR\n(408 M)")]
        names, v, e, shown, col = ([x[i] for i in keep] for x in (names, v, e, shown, col))
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    bars = ax.bar(names, v, color=col, width=0.6, zorder=3)
    k = [i for i, x in enumerate(e) if x > 0]          # error bars only where a spread exists
    ax.errorbar(k, [v[i] for i in k], yerr=[e[i] for i in k], fmt="none",
                capsize=4, lw=1, ecolor="#555555", zorder=4)
    ax.tick_params(axis="x", labelsize=8)
    for b, val, err, txt in zip(bars, v, e, shown):
        ax.text(b.get_x() + b.get_width() / 2, val + err + 0.012, txt,
                ha="center", fontsize=8, color=INK)
    ax.set_ylabel("max $F_1$ (109 test images)")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    save(fig, "hybrid_lofar_ladder")


# --- every method of Table 5.3: max F1, AUROC, AUPRC, under both settings ------------
def lofar_same_settings():
    # Ours: runs/lofar/{lr1e-4_seed*, hybrid_b08_nocw_seed*, mesarcik_unet_clean_seed0,
    # hybrid_mesarcik_clean_bce_seed0}/eval_test/metrics.json; tf_unet closest = last-epoch
    # checkpoint, analysis/last_vs_best/tfunet_last_auc.json (PART 36.1).
    # Published: Mesarcik et al. 2022, Table 2. Sigma-clip: analysis/lofar_analysis/sigma_clip_auc.json.
    # name, (max F1, sd), (AUROC, sd), (AUPRC, sd); None = no such score
    groups = [
        ("Our settings", [
            ("tf_unet", (0.5482, 0.0139), (0.9470, 0.0074), (0.5562, 0.0152)),
            ("Kestrel", (0.6603, 0.0040), (0.9378, 0.0111), (0.6583, 0.0088)),
        ]),
        ("Mesarcik et al.'s settings", [
            ("tf_unet\n(closest)", (0.5551, 0.0155), (0.9482, 0.0052), (0.5608, 0.0167)),
            ("Mesarcik\nU-Net", (0.6166, 0), (0.8213, 0), (0.5283, 0)),
            ("Kestrel", (0.6258, 0), (0.9719, 0), (0.6401, 0)),
        ]),
        ("Published by Mesarcik et al.", [
            ("AOFlagger", (0.5698, 0), (0.7883, 0), (0.5716, 0)),
            ("Mesarcik\nU-Net", (0.5876, 0.0031), (0.8017, 0.0058), (0.5920, 0.0031)),
            ("RFI-Net", (0.5979, 0.0012), (0.8109, 0.0037), (0.5991, 0.0038)),
        ]),
        ("Threshold", [
            ("$\\sigma$-clip", (0.4103, 0), (0.7029, 0), (0.3165, 0)),
        ]),
    ]
    metrics = [("max $F_1$", NAVY), ("AUROC", MUTED), ("AUPRC", ACCENT)]
    w, step, gap = 0.27, 1.12, 0.5           # bar width; model spacing; extra space between groups
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    x, ticks, labels = 0.0, [], []
    for gi, (title, models) in enumerate(groups):
        x0 = x
        for name, *scores in models:
            for j, sc in enumerate(scores):
                if sc is None:
                    continue
                val, err = sc
                xb = x + (j - 1) * w
                ax.bar(xb, val, w * 0.92, color=metrics[j][1], zorder=3)
                if err:
                    ax.errorbar(xb, val, yerr=err, fmt="none", capsize=1.5, lw=0.7,
                                ecolor="#333333", zorder=4)
                ax.text(xb, val + err + 0.012, f"{val:.4f}", rotation=90, ha="center",
                        va="bottom", fontsize=6.6, color=INK)
            ticks.append(x); labels.append(name)
            x += step
        x1 = x - step
        ax.text((x0 + x1) / 2, 1.215, title, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK)
        ax.plot([x0 - 0.45, x1 + 0.45], [1.205, 1.205], color="#888888", lw=0.8,
                clip_on=False)
        if gi < len(groups) - 1:
            ax.axvline(x1 + (step + gap) / 2, color="#CCCCCC", lw=0.8, zorder=1)
        x += gap
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.6)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("score (109 LOFAR test images)", fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.6, x - gap - step + 0.6)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in metrics]
    ax.legend(handles, [m for m, _ in metrics], frameon=False, fontsize=8.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=1.2,
              columnspacing=1.6)
    save(fig, "lofar_same_settings")


# --- their U-Net: published vs their code run by us (PART 30) ------------------
def mesarcik_rerun():
    # runs/lofar/mesarcik_unet_clean_seed0/eval_test/metrics.json: max F1 0.6166 (test images
    # removed from training). Published: Mesarcik et al. 2022, Table 2, 0.5876 +/- 0.0031.
    names = ["Mesarcik et al. U-Net\n(published)", "Mesarcik et al. U-Net\n(their code, run by us)"]
    v, e = [0.5876, 0.6166], [0.0031, 0]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = ax.bar(names, v, color=[MUTED, NAVY], width=0.5, zorder=3)
    ax.errorbar([0], [v[0]], yerr=[e[0]], fmt="none", capsize=4, lw=1, ecolor="#555555", zorder=4)
    ax.tick_params(axis="x", labelsize=8.5)
    for b, val, err in zip(bars, v, e):
        ax.text(b.get_x() + b.get_width() / 2, val + err + 0.012, f"{val:.4f}",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("max $F_1$ (109 test images)")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    save(fig, "mesarcik_rerun")


if __name__ == "__main__":
    tfunet_controlled_runs()
    tfunet_lofar_ladder()
    hybrid_lofar_ladder()
    lofar_same_settings()
    mesarcik_rerun()
    print("figures written to", HERE)
