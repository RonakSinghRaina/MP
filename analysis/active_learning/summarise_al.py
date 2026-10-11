"""Summarise topic 11 (experiments/active_learning/al_lofar.py): F1 on ALL 109
expert-labelled LOFAR images, pooled over the cross-validation folds, mean +/- sd over
the headline seeds. Only (seed, arm) pairs with every fold finished are used.

  max F1   best single threshold on the pooled 109 (benchmark convention)
  val F1   each fold at its own validation-chosen threshold (no test information)
  <=600    the same model stopped within the pilot's 600 steps ("train longer" check)

    ~/torch-env/bin/python analysis/active_learning/summarise_al.py [--out runs/lofar/al]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "active_learning"))
from al_lofar import arm_name, max_f1_from_hist  # noqa: E402  (same metric code as the runs)

CHAINS = ("unc", "rfi", "rand", "img")
LABEL = {"unc": "uncertainty patches", "rfi": "most-RFI patches", "rand": "random patches",
         "img": "random whole images"}


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def pool_arm(out, seed, arm, folds):
    acc = {"hp": 0, "hn": 0, "hps": 0, "hns": 0, "tp": 0, "fp": 0, "fn": 0,
           "tps": 0, "fps": 0, "fns": 0}
    rows = []
    for k in folds:
        base = os.path.join(out, f"seed{seed}", f"fold{k}", arm)
        if not (os.path.exists(base + ".json") and os.path.exists(base + ".npz")):
            return None
        r = json.load(open(base + ".json"))
        z = np.load(base + ".npz")
        acc["hp"] = acc["hp"] + z["hp"]; acc["hn"] = acc["hn"] + z["hn"]
        acc["hps"] = acc["hps"] + z["hp_short"]; acc["hns"] = acc["hns"] + z["hn_short"]
        for key, part in (("", "best"), ("s", "short")):
            acc["tp" + key] += r[part]["test_tp"]; acc["fp" + key] += r[part]["test_fp"]
            acc["fn" + key] += r[part]["test_fn"]
        rows.append(r)
    n_test = sum(r["n_test"] for r in rows)
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    return {"max_f1": max_f1_from_hist(acc["hp"], acc["hn"])[0],
            "max_f1_short": max_f1_from_hist(acc["hps"], acc["hns"])[0],
            "val_f1": f1(tp, fp, fn), "val_f1_short": f1(acc["tps"], acc["fps"], acc["fns"]),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "n_test": n_test,
            "patches_per_fold": float(np.mean([r["labelled_patches"] for r in rows])),
            "pool_fraction": float(np.mean([r["labelled_fraction_of_pool"] for r in rows])),
            "rfi_px_per_fold": float(np.mean([r["labelled_rfi_pixels"] for r in rows])),
            "best_steps": [r["best"]["step"] for r in rows],
            "hit_cap": sum(r["hit_step_cap"] for r in rows),
            "n_runs": len(rows)}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f} (n=1)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "lofar", "al"))
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="default: all folds in config.json (use fewer only to test this script)")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    folds = a.folds if a.folds is not None else list(range(cfg["n_folds"]))
    all_folds = len(folds) == cfg["n_folds"]
    budgets = cfg["budgets"]
    arms = ["base", "full"] + [arm_name(c, b) for c in CHAINS for b in budgets]
    seeds = sorted(int(x[4:]) for x in os.listdir(a.out) if x.startswith("seed") and x[4:].isdigit())
    R = {arm: {} for arm in arms}
    for arm in arms:
        for s in seeds:
            r = pool_arm(a.out, s, arm, folds)
            if r:
                R[arm][s] = r

    print("=" * 100)
    print(f"  TOPIC 11 -- expert-label fine-tuning, {len(folds)}-fold cross-validation, "
          f"{'ALL 109 images' if all_folds else 'PARTIAL: folds ' + str(folds)}")
    print("=" * 100)
    if not R["base"]:
        print("  no finished base arm yet"); return
    if all_folds:
        print("  check -- base, pooled over the folds, must reproduce the headline max F1:")
        for s, r in sorted(R["base"].items()):
            m = json.load(open(os.path.join(ROOT, "runs", "lofar", f"hybrid_b08_nocw_seed{s}",
                                            "eval_test", "metrics.json")))
            print(f"     seed {s}: {r['max_f1']:.4f}  vs headline {m['max_f1']:.4f}  "
                  f"(diff {r['max_f1'] - m['max_f1']:+.5f}; histogram bins, exact within 5e-4)")
    n_val = cfg["n_val"]
    print(f"\n  labelling cost = patches labelled in the pool (shown as image-equivalents, 64 patches"
          f" = 1 image) + {n_val} validation images in every arm\n")
    head = f"  {'arm':<8}{'labelled / fold':>22}{'seeds':>6}{'max F1':>22}{'val F1':>22}" \
           f"{'prec / rec':>14}{'max F1 <=600':>16}{'best steps (median)':>21}"
    print(head)
    order = ["base", "full"] + [arm_name(c, b) for b in budgets for c in CHAINS]
    for arm in order:
        rs = R[arm]
        if not rs:
            print(f"  {arm:<8}   -- not finished"); continue
        v = list(rs.values())
        lab = f"{v[0]['patches_per_fold'] / 64:6.1f} img ({100 * v[0]['pool_fraction']:4.1f}%)"
        steps = [st for r in v for st in r["best_steps"]]
        cap = sum(r["hit_cap"] for r in v)
        print(f"  {arm:<8}{lab:>22}{len(v):>6}{ms([r['max_f1'] for r in v]):>22}"
              f"{ms([r['val_f1'] for r in v]):>22}"
              f"{np.mean([r['precision'] for r in v]):>7.3f}/{np.mean([r['recall'] for r in v]):.3f}"
              f"{ms([r['max_f1_short'] for r in v]):>16}"
              f"{int(np.median(steps)) if steps else 0:>12}"
              f"{('  (' + str(cap) + ' hit the step cap)') if cap else '':<9}")

    def paired(x, y, label):
        common = sorted(set(R[x]) & set(R[y]))
        if len(common) < 1:
            return
        d = np.array([R[x][s]["max_f1"] - R[y][s]["max_f1"] for s in common])
        extra = ""
        if len(common) >= 3:
            t, p = stats.ttest_rel([R[x][s]["max_f1"] for s in common], [R[y][s]["max_f1"] for s in common])
            extra = f", paired t = {t:.2f}, p = {p:.3f}"
        print(f"     {label:<44} {d.mean():+.4f}  (per seed: {', '.join(f'{v:+.4f}' for v in d)}){extra}")

    print("\n  gain over no fine-tuning (max F1, paired by seed):")
    for arm in order[1:]:
        if R[arm]:
            paired(arm, "base", f"{arm} - base")
    print("\n  which patches to label? (same budget, paired by seed):")
    for b in budgets:
        pct = int(round(100 * b))
        paired(arm_name("unc", b), arm_name("rand", b), f"{pct:>2}%: uncertainty patches - random patches")
        paired(arm_name("rfi", b), arm_name("rand", b), f"{pct:>2}%: most-RFI patches - random patches")
        paired(arm_name("rand", b), arm_name("img", b), f"{pct:>2}%: random patches - random whole images")
    print("\n  train longer? (same runs: best state within 600 steps vs the full schedule):")
    for arm in order[1:]:
        common = sorted(R[arm])
        if common:
            d = [R[arm][s]["max_f1"] - R[arm][s]["max_f1_short"] for s in common]
            print(f"     {arm:<8} longer - <=600: {np.mean(d):+.4f}")

    best_al = max((arm for arm in order[2:] if R[arm]), default=None,
                  key=lambda x: np.mean([r["max_f1"] for r in R[x].values()]))
    print("\n  FINAL (max F1 on all 109 images, cross-validated; expert-label protocol -- compare"
          " with RFDL 0.669 / 0.675, NOT with the AOFlagger-only 0.660):")
    print(f"     no fine-tuning       {ms([r['max_f1'] for r in R['base'].values()])}")
    if R["full"]:
        print(f"     all pool labels      {ms([r['max_f1'] for r in R['full'].values()])}")
    if best_al:
        print(f"     best budgeted arm    {best_al}: {ms([r['max_f1'] for r in R[best_al].values()])}")
    out = {arm: {str(s): {k: v for k, v in r.items()} for s, r in rs.items()} for arm, rs in R.items()}
    path = os.path.join(a.out, "summary.json")
    json.dump({"folds": folds, "all_folds": all_folds, "results": out}, open(path, "w"), indent=1)
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
