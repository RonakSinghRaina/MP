"""Dice-off (PARTS 18.4, 19.2, 20.2): the headline hybrid trained with plain cross-entropy
(--dice_weight 0) against the same model with CE + Dice, seed by seed.

Baseline: runs/lofar/hybrid_b08_nocw_seed{0,1,2} (max F1 0.6603 +/- 0.0040).
Same seeds give the same train/val split, so differences are paired. Also shows how
much of the hybrid's lead over tf_unet (runs/lofar/lr1e-4_seed*, 0.5482 +/- 0.0139)
the Dice term accounts for.

    ~/torch-env/bin/python analysis/summarise_nodice.py
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "runs", "lofar")
# everything that must match the headline runs; only dice_weight may differ
SAME = ("model", "base", "depth", "dropout", "parameters", "norm", "fixed_range", "class_weights",
        "learning_rate", "batch_size", "epochs", "gradient_steps", "n_train", "n_val", "test_set")


def load(pattern):
    out = {}
    for p in glob.glob(os.path.join(RUNS, pattern, "eval_test", "metrics.json")):
        m = json.load(open(p))
        out[int(m["seed"])] = m
    return out


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="nodice_b08_nocw_seed*", help="run folders to compare (glob)")
    a = ap.parse_args()
    base = load("hybrid_b08_nocw_seed*")
    nod = load(a.runs)
    tfu = load("lr1e-4_seed*")
    print("=" * 78)
    print("  DICE OFF vs the headline hybrid (base 8, no class weight)")
    print("=" * 78)
    if not nod:
        print("  no finished Dice-off runs yet"); return

    bad = []
    for s, m in sorted(nod.items()):
        b = base.get(s)
        if b is None:
            bad.append(f"seed {s}: no headline run to pair with"); continue
        bad += [f"seed {s}: {k} {b.get(k)!r} -> {m.get(k)!r}" for k in SAME if m.get(k) != b.get(k)]
        if m.get("dice_weight") != 0.0:
            bad.append(f"seed {s}: dice_weight is {m.get('dice_weight')!r}, not 0")
        if m.get("edge_mode", "none") != "none" or m.get("variant", "hybrid") != "hybrid":
            bad.append(f"seed {s}: edge_mode {m.get('edge_mode')!r} / variant {m.get('variant')!r}")
    print("  config check: " + ("identical to the headline except dice_weight 1.0 -> 0.0" if not bad
                                else "MISMATCH -- do not compare:\n     " + "\n     ".join(bad)))

    keys = [("max_f1", "max F1"), ("pooled_f1", "pooled F1"), ("pooled_precision", "precision"),
            ("pooled_recall", "recall"), ("roc_auc", "ROC")]
    print(f"\n{'setup':<20}{'n':>3}" + "".join(f"{lab:>18}" for _, lab in keys))
    for name, runs in (("CE + Dice (headline)", base), ("CE only (Dice off)", nod), ("tf_unet (lr 1e-4)", tfu)):
        if runs:
            print(f"{name:<20}{len(runs):>3}" + "".join(f"{ms([runs[s][k] for s in sorted(runs)]):>18}"
                                                     for k, _ in keys))

    common = sorted(set(nod) & set(base))
    for k, lab in (("max_f1", "max F1"), ("pooled_f1", "pooled F1")):
        print(f"\n  Dice off - Dice on, paired by seed ({lab}):")
        for s in common:
            print(f"     seed {s}: {base[s][k]:.4f} -> {nod[s][k]:.4f}   ({nod[s][k] - base[s][k]:+.4f})")
        d = np.array([nod[s][k] - base[s][k] for s in common])
        if len(d) >= 3:
            t, p = stats.ttest_rel([nod[s][k] for s in common], [base[s][k] for s in common])
            print(f"     mean change {d.mean():+.4f}, paired t = {t:.2f}, p = {p:.3f}"
                  f"  ({'significant' if p < 0.05 else 'NOT significant'} at 5%)")
        else:
            print(f"     mean change {d.mean():+.4f} (need 3 seeds for a test)")

    if tfu and len(common) == len(base):
        lead = np.mean([base[s]["max_f1"] for s in base]) - np.mean([tfu[s]["max_f1"] for s in tfu])
        drop = np.mean([base[s]["max_f1"] for s in common]) - np.mean([nod[s]["max_f1"] for s in common])
        print(f"\n  lead of the hybrid over tf_unet (max F1): {lead:+.4f}")
        print(f"  lost when Dice is switched off:            {drop:+.4f}  = {100 * drop / lead:.0f}% of that lead")
        print("  (one change at a time, from the hybrid side; the other differences -- output ReLU,"
              "\n   padding, depth/width, batch, learning rate, framework -- are not isolated by this)")


if __name__ == "__main__":
    main()
