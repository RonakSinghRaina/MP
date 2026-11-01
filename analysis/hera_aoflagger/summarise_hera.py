"""Print and save the HERA table: all runs of scripts/run_hera.sh next to Mesarcik et al.'s Table 1.
Every model is trained on AOFlagger labels (threshold 10) and scored on the simulator's labels,
135 leak-free test images. Prints what has finished so far, so it can be run at any time.
    python analysis/hera_aoflagger/summarise_hera.py [run_root]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "hera")
SEEDS = (0, 1, 2)
KEYS = ("max_f1", "pooled_f1", "roc_auc", "pr_auc")
# what every finished run must have been trained with
EXPECT = {"hybrid_b08_nocw_seed{}": dict(dataset="hera", base=8, norm="fixed", dice_weight=1.0,
                                         class_weights=[1.0, 1.0], batch_size=1, epochs=40,
                                         gradient_steps=28000, n_train=316),
          "tfunet_lr1e-4_seed{}": dict(dataset="hera", norm="fixed", class_weights=False,
                                       learning_rate=1e-4, batch_size=4, epochs=150, n_train=316),
          "mesarcik_unet_clean_seed{}": dict(dataset="hera", setup="clean", clip_version="hera",
                                             batch=1024, epochs=100, n_train=316),
          "hybrid_mesarcik_clean_bce_seed{}": dict(dataset="hera", setup="clean", clip_version="hera",
                                                   base=8, batch=1024, epochs=100, n_train=316)}


def collect(pattern):
    found, problems = [], []
    for s in SEEDS:
        f = os.path.join(RUNS, pattern.format(s), "eval_test", "metrics.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        problems += [f"{pattern.format(s)}: {k} = {m.get(k)!r}, expected {v!r}"
                     for k, v in EXPECT[pattern].items() if m.get(k) != v]
        found.append(m)
    return found, problems


def row(ms):
    out = {"runs": len(ms)}
    for k in KEYS:
        v = [m[k] for m in ms if m.get(k) is not None]
        if v:
            out[k] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    return out


def show(name, r):
    cells = []
    for k in KEYS:
        if k in r:
            mean, sd = r[k]
            cells.append(f"{mean:.4f}" + (f" +/- {sd:.4f}" if sd else "         "))
        else:
            cells.append("---".ljust(15))
    n = r.get("runs")
    print(f"  {name:<30}{str(n) if n else '-':>5}   " + "   ".join(c.ljust(15) for c in cells))


results, problems = {}, []
print("HERA: trained on AOFlagger labels, scored on simulator labels, 135 test images")
print(f"  {'':<30}{'runs':>5}   " + "   ".join(h.ljust(15) for h in ("max F1", "pooled F1", "AUROC", "AUPRC")))
print("Our settings")
for name, pat in (("tf_unet", "tfunet_lr1e-4_seed{}"), ("HybridRFINet", "hybrid_b08_nocw_seed{}")):
    ms, p = collect(pat); problems += p
    results["ours: " + name] = row(ms); show(name, results["ours: " + name])

print("Mesarcik et al.'s settings")
last_json = os.path.join(HERE, "tfunet_last_auc_hera.json")
last = json.load(open(last_json)) if os.path.exists(last_json) else {}
results["mesarcik: tf_unet closest (last epoch)"] = row([v["last"] for v in last.values()])
show("tf_unet, closest (last epoch)", results["mesarcik: tf_unet closest (last epoch)"])
for name, pat in (("Mesarcik U-Net", "mesarcik_unet_clean_seed{}"),
                  ("HybridRFINet", "hybrid_mesarcik_clean_bce_seed{}")):
    ms, p = collect(pat); problems += p
    results["mesarcik: " + name] = row(ms); show(name, results["mesarcik: " + name])

print("No training")
c = json.load(open(os.path.join(HERE, "check_aoflagger_hera.json")))["unique"]
results["AOFlagger, run by us"] = dict(max_f1=[c["f1"], 0.0], roc_auc=[c["auroc"], 0.0],
                                       pr_auc=[c["auprc_trapz"], 0.0])
show("AOFlagger, run by us", results["AOFlagger, run by us"])

print("Published, Mesarcik et al. Table 1 (their test set includes 31 copies of training images)")
published = {"AOFlagger": dict(max_f1=[0.8729, 0.0], roc_auc=[0.9736, 0.0], pr_auc=[0.8799, 0.0]),
             "U-Net": dict(max_f1=[0.9015, 0.0074], roc_auc=[0.9746, 0.0009], pr_auc=[0.8963, 0.0074]),
             "RFI-Net": dict(max_f1=[0.8950, 0.0068], roc_auc=[0.9728, 0.0007], pr_auc=[0.8898, 0.0086])}
for name, r in published.items():
    results["published: " + name] = r; show(name, r)

if problems:
    print("\nSETTINGS CHECK FAILED:")
    for p in problems:
        print("  " + p)
else:
    print("\nSettings check passed for every finished run.")
json.dump(results, open(os.path.join(HERE, "hera_results.json"), "w"), indent=2)
