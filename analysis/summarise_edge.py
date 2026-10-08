"""Compare edge-aware training (PART 22.4) with the headline hybrid, seed by seed.

Baseline: runs/lofar/hybrid_b08_nocw_seed{0,1,2} (max F1 0.6603 +/- 0.0040).
Same seeds give the same train/val split, so differences are paired.

    ~/torch-env/bin/python analysis/summarise_edge.py
"""
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "runs", "lofar")


def load(pattern):
    out = {}
    for p in glob.glob(os.path.join(RUNS, pattern, "eval_test", "metrics.json")):
        m = json.load(open(p))
        out[int(m["seed"])] = m
    return out


def main():
    base = load("hybrid_b08_nocw_seed*")
    print("=" * 78)
    print("  EDGE-AWARE TRAINING vs the headline hybrid (base 8, no class weight)")
    print("=" * 78)
    rows = [("baseline (none)", base)] + [(f"edge: {m}", load(f"edge_{m}_b08_nocw_seed*"))
                                          for m in ("svls", "ignore")]
    keys = [("max_f1", "max F1"), ("pooled_f1", "pooled F1"),
            ("pooled_precision", "precision"), ("pooled_recall", "recall"), ("roc_auc", "ROC")]
    print(f"{'setup':<18}{'n':>3}" + "".join(f"{lab:>18}" for _, lab in keys))
    for name, runs in rows:
        if not runs:
            print(f"{name:<18}  -- no finished runs yet"); continue
        line = f"{name:<18}{len(runs):>3}"
        for k, _ in keys:
            v = np.array([runs[s][k] for s in sorted(runs)])
            line += f"{v.mean():>11.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
        print(line)

    for name, runs in rows[1:]:
        common = sorted(set(runs) & set(base))
        if not common:
            continue
        print(f"\n{name} -- paired by seed (max F1):")
        d = []
        for s in common:
            dv = runs[s]["max_f1"] - base[s]["max_f1"]; d.append(dv)
            print(f"   seed {s}: {base[s]['max_f1']:.4f} -> {runs[s]['max_f1']:.4f}   ({dv:+.4f})")
        d = np.array(d)
        if len(d) >= 3:
            t, p = stats.ttest_rel([runs[s]["max_f1"] for s in common],
                                   [base[s]["max_f1"] for s in common])
            print(f"   mean change {d.mean():+.4f}, paired t = {t:.2f}, p = {p:.3f}"
                  f"  ({'significant' if p < 0.05 else 'NOT significant'} at 5%)")
        else:
            print(f"   mean change {d.mean():+.4f} (need 3 seeds for a test)")


if __name__ == "__main__":
    main()
