"""Summarise the LOFAR architecture ablation (PART 17.3).

Reads runs/lofar/abl_*/eval_test/metrics.json and prints the decomposition of
the +0.1110 gain over tf_unet, with per-seed spread where seeds exist.

Reference points, all measured in this project and recorded in
RFI-project-context.md:
    tf_unet @ lr 1e-4          0.5482 +/- 0.0139   (PART 12.10)
    hybrid b8, no class weight 0.6603 +/- 0.0040   (PART 13.11)
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TFUNET = 0.5482
TFUNET_SD = 0.0139
HYBRID = 0.6603
HYBRID_SD = 0.0040

ORDER = ["hybrid_full", "no_strip", "no_eca", "no_res", "plain_unet", "no_groupnorm"]


def main():
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "lofar", "abl_*", "eval_test", "metrics.json"))):
        m = json.load(open(p))
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        mo = re.match(r"abl_(.+)_seed(\d+)$", name)
        if not mo:
            continue
        runs[mo.group(1)].append(m)

    if not runs:
        print("No ablation runs found. Run ./scripts/run_lofar_ablation.sh first.")
        return

    print("=" * 78)
    print("  LOFAR ARCHITECTURE ABLATION — base 8, fixed norm, no class weight")
    print("=" * 78)
    print(f"{'variant':<15}{'n':>3}{'max F1':>10}{'sd':>8}{'params':>10}"
          f"{'vs full':>10}{'prec':>8}{'rec':>8}")

    full = np.array([m["max_f1"] for m in runs.get("hybrid_full", [])])
    ref = full.mean() if full.size else np.nan

    for v in ORDER + [k for k in runs if k not in ORDER]:
        if v not in runs:
            continue
        f = np.array([m["max_f1"] for m in runs[v]])
        sd = f.std(ddof=1) if f.size > 1 else float("nan")
        par = runs[v][0].get("parameters", 0)
        pr = np.mean([m.get("pooled_precision", np.nan) for m in runs[v]])
        rc = np.mean([m.get("pooled_recall", np.nan) for m in runs[v]])
        delta = f.mean() - ref if np.isfinite(ref) else np.nan
        sd_s = f"{sd:>8.4f}" if f.size > 1 else f"{'-':>8}"
        print(f"{v:<15}{f.size:>3}{f.mean():>10.4f}{sd_s}"
              f"{par:>10,}{delta:>+10.4f}{pr:>8.4f}{rc:>8.4f}")

    print("-" * 78)
    print(f"{'tf_unet lr1e-4':<15}{3:>3}{TFUNET:>10.4f}{TFUNET_SD:>8.4f}"
          f"{'~500,000':>10}{TFUNET - ref:>+10.4f}   (PART 12.10, quoted)")
    print(f"{'HybridRFINet':<15}{3:>3}{HYBRID:>10.4f}{HYBRID_SD:>8.4f}"
          f"{593842:>10,}{HYBRID - ref:>+10.4f}   (PART 13.11, quoted)")

    # ---- the decomposition the paper needs ---------------------------------
    print()
    print("=" * 78)
    print("  DECOMPOSITION of the +{:.4f} gain over tf_unet".format(HYBRID - TFUNET))
    print("=" * 78)

    def get(v):
        return np.mean([m["max_f1"] for m in runs[v]]) if v in runs else None

    steps = [
        ("normalisation (GroupNorm)", "no_groupnorm", "plain_unet"),
        ("residual blocks", "plain_unet", "no_strip"),
        ("strip convolutions", "no_strip", "no_eca"),
    ]
    ng = get("no_groupnorm")
    if ng is not None:
        print(f"  tf_unet -> no_groupnorm      : {ng - TFUNET:+.4f}   "
              f"(what is left: padding, output activation, framework)")
    for label, a, b in steps:
        va, vb = get(a), get(b)
        if va is not None and vb is not None:
            print(f"  {label:<28}: {vb - va:+.4f}   ({a} -> {b})")

    # single-component effects, the cleanest read
    print()
    if full.size:
        for v, label in [("no_strip", "strip convolutions"),
                         ("no_eca", "ECA"),
                         ("no_res", "residual blocks")]:
            x = get(v)
            if x is not None:
                d = ref - x
                sig = ""
                if full.size > 1 and len(runs[v]) > 1:
                    pooled = np.sqrt((full.std(ddof=1) ** 2 + np.std([m["max_f1"] for m in runs[v]], ddof=1) ** 2) / 2)
                    sig = f"   ({abs(d) / pooled:.1f} pooled sd)" if pooled > 0 else ""
                elif np.isfinite(HYBRID_SD):
                    sig = f"   ({abs(d) / HYBRID_SD:.1f} seed sd of the reference)"
                print(f"  removing {label:<22}: {-d:+.4f}{sig}")

    print()
    print("  Reminder (PART 17.3): a null result here is publishable. If")
    print("  plain_unet ~ hybrid_full, the components do nothing on real data")
    print("  either, and the win is normalisation + padding.")


if __name__ == "__main__":
    main()
