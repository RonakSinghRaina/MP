"""Pilot: do a few expert labels raise the hybrid's F1? (precondition for topic 11)

Protocol, fixed BEFORE running. Split sizes mirror RFDL's expert-label setting
(van Zyl & Grobler 2024, MNRAS 530, 1907, Table 7: 20 train / 19 val / 70 test):

  * The 109 expert-labelled images are split 20 fine-tune / 19 validation /
    70 test, group-aware: expert masks with IoU > 0.3 (near-duplicate baselines)
    always land on the same side. 3 random splits (split seeds 0, 1, 2).
  * Base models: runs/lofar/hybrid_b08_nocw_seed{0,1,2}/best.pt. None of them
    ever saw any of the 109 images (clean_train_idx excludes their twins).
  * Arms, all scored on the same 70 test images of a split:
      base       no fine-tuning
      ft_aof     fine-tune on the 20 images with their AOFlagger masks (CONTROL:
                 same images, same steps, but the teacher's labels)
      ft_expert  fine-tune on the 20 images with the expert masks
  * Fine-tuning: Adam lr 1e-4, batch 1, 600 steps, loss CE + Dice exactly as in
    training (no class weights), random time and frequency flips (never a
    rotation: that would swap the axes), dropout on. Every 100 steps the max F1
    on the 19 validation images (expert labels) is measured and the best state
    kept; step 0 is a candidate, so selection may return the base model.
  * Test metrics: max F1 (threshold that is best on the test images -- the
    benchmark convention, RFDL's "Best F1") and F1 at the threshold that is best
    on the 19 VALIDATION images (no test information used).

3 models x 3 splits = 9 paired comparisons per arm (not 9 independent samples:
models share splits and splits share models -- read the t-test as approximate).

    ~/torch-env/bin/python analysis/active_learning/pilot_expert_finetune.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import precision_recall_curve

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "hybrid_rfi_package"))
from hybrid_model import HybridRFINet  # noqa: E402
from lofar_data import load_lofar      # noqa: E402

RUNS = os.path.join(ROOT, "runs", "lofar")
OUT = os.path.join(ROOT, "analysis", "active_learning", "pilot_expert_finetune.json")
N_FT, N_VAL, N_TEST = 20, 19, 70
STEPS, EVAL_EVERY, LR = 600, 100, 1e-4
GROUP_IOU = 0.3
SMOKE = os.environ.get("PILOT_SMOKE") == "1"          # 1 model, 1 split, 10 steps: checks the code only
if SMOKE:
    STEPS, EVAL_EVERY = 10, 5
    OUT = OUT.replace(".json", "_SMOKE.json")


def dice_loss(logits, target, eps=1.0):          # identical to experiments/lofar_hybrid.py
    probs = F.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (probs * t).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def normalise_fixed(img, lo, hi):
    d = np.asarray(img, dtype=np.float64)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def groups_from_masks(Y):
    """Union-find over expert masks with IoU > GROUP_IOU."""
    n = len(Y); parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            u = (Y[i] | Y[j]).sum()
            if u and (Y[i] & Y[j]).sum() / u > GROUP_IOU:
                parent[find(i)] = find(j)
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def split(groups, seed):
    rng = np.random.default_rng(seed)
    order = [groups[k] for k in rng.permutation(len(groups))]
    out = {"ft": [], "val": [], "test": []}
    for name, size in (("ft", N_FT), ("val", N_VAL)):
        rest = []
        for g in order:
            (out[name].extend(g) if len(out[name]) + len(g) <= size else rest.append(g))
        order = rest
    for g in order:
        out["test"].extend(g)
    assert (len(out["ft"]), len(out["val"]), len(out["test"])) == (N_FT, N_VAL, N_TEST), \
        {k: len(v) for k, v in out.items()}
    return {k: sorted(v) for k, v in out.items()}


@torch.no_grad()
def probs(model, X, idx, dev):
    model.eval()
    out = []
    for s in range(0, len(idx), 2):
        x = torch.from_numpy(X[idx[s:s + 2]]).unsqueeze(1).to(dev)
        out.append(F.softmax(model(x), 1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(out)


def f1_curve(y, p):
    pc, rc, th = precision_recall_curve(y.ravel(), p.ravel())
    f1 = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f1[:-1]))
    return float(f1[k]), float(th[k])


def f1_at(y, p, th):
    y = y.ravel(); q = p.ravel() > th
    tp = (q & y).sum(); fp = (q & ~y).sum(); fn = (~q & y).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return float(2 * pr * rc / max(pr + rc, 1e-12)), float(pr), float(rc)


def evaluate(model, X, Yexp, sp, dev):
    pv = probs(model, X, sp["val"], dev)
    _, th_val = f1_curve(Yexp[sp["val"]], pv)
    pt = probs(model, X, sp["test"], dev)
    yt = Yexp[sp["test"]]
    mx, _ = f1_curve(yt, pt)
    f1v, pr, rc = f1_at(yt, pt, th_val)
    return {"max_f1": mx, "f1_val_threshold": f1v, "precision": pr, "recall": rc, "val_threshold": th_val}


def finetune(model, X, Ytrain, sp, Yexp, dev, seed):
    rng = np.random.default_rng(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    def val_score():
        return f1_curve(Yexp[sp["val"]], probs(model, X, sp["val"], dev))[0]
    best = (val_score(), 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.train()
    for step in range(1, STEPS + 1):
        i = int(rng.choice(sp["ft"]))
        x = X[i]; y = Ytrain[i]
        if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]      # flip frequency
        if rng.random() < 0.5: x, y = x[::-1, :], y[::-1, :]      # flip time
        xt = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(dev)
        yt = torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))[None].to(dev)
        logits = model(xt)
        loss = F.cross_entropy(logits, yt) + dice_loss(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            v = val_score()
            if v > best[0]:
                best = (v, step, {k: t.detach().clone() for k, t in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[1], best[0]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_lofar()
    Yexp = np.asarray(d.test_masks[..., 0], bool)
    # AOFlagger mask of each expert image = the mask of its byte-identical training twin (PART 11.5)
    h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    twin = {h(d.train_images[int(i)]): int(i) for i in d.leak_train_idx}
    Yaof = np.stack([np.asarray(d.train_masks[twin[h(d.test_images[t])], :, :, 0], bool) for t in range(109)])
    groups = groups_from_masks(Yexp)
    print(f"{len(groups)} groups from 109 images (largest {max(map(len, groups))})", flush=True)
    splits = [split(groups, s) for s in (0, 1, 2)]

    R = {"protocol": {"n_ft": N_FT, "n_val": N_VAL, "n_test": N_TEST, "steps": STEPS, "lr": LR,
                      "eval_every": EVAL_EVERY, "group_iou": GROUP_IOU},
         "splits": splits, "runs": []}
    if os.path.exists(OUT):                                  # resumable: keep finished pairs
        old = json.load(open(OUT))
        if old["protocol"] == R["protocol"] and old["splits"] == splits:
            R["runs"] = old["runs"]
    done = {(r["model_seed"], r["split_seed"]) for r in R["runs"]}
    t0 = time.time()
    for mseed in ((0,) if SMOKE else (0, 1, 2)):
        if all((mseed, s) in done for s in range(len(splits))):
            continue
        run = os.path.join(RUNS, f"hybrid_b08_nocw_seed{mseed}")
        m = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
        lo, hi = m["fixed_range"]
        X = np.stack([normalise_fixed(d.test_images[i, :, :, 0], lo, hi) for i in range(109)])
        state = torch.load(os.path.join(run, "best.pt"), map_location="cpu")
        for sseed, sp in enumerate(splits[:1] if SMOKE else splits):
            if (mseed, sseed) in done:
                continue
            row = {"model_seed": mseed, "split_seed": sseed}
            for arm, Ytr in (("base", None), ("ft_aof", Yaof), ("ft_expert", Yexp)):
                torch.manual_seed(mseed * 10 + sseed)
                model = HybridRFINet(1, 2, base=m["base"], depth=m["depth"], dropout=m["dropout"])
                model.load_state_dict(state); model.to(dev)
                if Ytr is not None:
                    step, v = finetune(model, X, Ytr, sp, Yexp, dev, mseed * 10 + sseed)
                    row[arm + "_best_step"] = step; row[arm + "_val_max_f1"] = v
                row[arm] = evaluate(model, X, Yexp, sp, dev)
                del model; torch.cuda.empty_cache()
            R["runs"].append(row)
            print(f"model {mseed} split {sseed}: base {row['base']['max_f1']:.4f} | "
                  f"ft_aof {row['ft_aof']['max_f1']:.4f} (step {row['ft_aof_best_step']}) | "
                  f"ft_expert {row['ft_expert']['max_f1']:.4f} (step {row['ft_expert_best_step']})"
                  f"   [{(time.time() - t0) / 60:.1f} min]", flush=True)
            json.dump(R, open(OUT, "w"), indent=2)

    print("\n=== 70 test images, mean +/- sd over 9 (model, split) pairs ===")
    for key, lab in (("max_f1", "max F1 (best threshold on test)"),
                     ("f1_val_threshold", "F1 at validation-chosen threshold")):
        print(lab)
        vals = {a: np.array([r[a][key] for r in R["runs"]]) for a in ("base", "ft_aof", "ft_expert")}
        if len(R["runs"]) < 2:
            print("   (fewer than 2 runs -- no statistics)"); continue
        for a, v in vals.items():
            print(f"   {a:<10} {v.mean():.4f} +/- {v.std(ddof=1):.4f}")
        for a, b in (("ft_expert", "base"), ("ft_aof", "base"), ("ft_expert", "ft_aof")):
            diff = vals[a] - vals[b]; t, p = stats.ttest_rel(vals[a], vals[b])
            print(f"   {a} - {b}: {diff.mean():+.4f} (min {diff.min():+.4f}, max {diff.max():+.4f}), "
                  f"paired t = {t:.2f}, p = {p:.4f}")
    for a in ("ft_aof", "ft_expert"):
        pr = np.mean([r[a]["precision"] for r in R["runs"]]); rc = np.mean([r[a]["recall"] for r in R["runs"]])
        print(f"   {a}: precision {pr:.3f}, recall {rc:.3f} (at val threshold)")
    print(f"   base: precision {np.mean([r['base']['precision'] for r in R['runs']]):.3f}, "
          f"recall {np.mean([r['base']['recall'] for r in R['runs']]):.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
