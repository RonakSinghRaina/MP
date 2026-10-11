"""Topic 11 -- expert-label fine-tuning with active patch selection, scored on ALL 109
LOFAR expert images by 5-fold cross-validation (follow-up to PART 25).

BUILDS ON THE EXISTING HYBRID. Nothing in src/hybrid_rfi_package/hybrid_model.py or
experiments/lofar_hybrid.py is modified: the model class is imported unchanged and the
three headline checkpoints (runs/lofar/hybrid_b08_nocw_seed{0,1,2}/best.pt -- base 8,
fixed norm, no class weight, trained on AOFlagger labels only, never on any of the 109
expert images) are loaded and fine-tuned on expert labels.

Design (fixed before running)
-----------------------------
Cross-validation. The 109 expert images form 77 groups (expert-mask IoU > 0.3 =
near-duplicate baselines, PART 25.1); a group is never split. 5 folds of 21-22 test
images. For each fold, 19 other images (whole groups, random) are the VALIDATION set --
expert labels, used for early stopping and for the operating threshold -- and the
remaining ~68 images are the LABELLING POOL. Every image is tested exactly once, by a
model that never saw its expert label.

Arms (the pool is what an expert would label; validation is a fixed extra 19 images):
  base          no fine-tuning (pooled, it must reproduce the headline -- a built-in check)
  full          every pool image fully labelled (upper bound with all available labels)
  unc05/10/20   64x64 patches with the highest mean predictive entropy  (active learning)
  rfi05/10/20   64x64 patches with the most predicted RFI (sum of p)     [control]
  rand05/10/20  random 64x64 patches                                      [control]
  img05/10/20   random WHOLE images (how earlier papers used expert labels) [control]
Budgets are 5 %, 10 %, 20 % of the pool's patches (img: the same share of images).
They are nested rounds: each round adds to what is already labelled; unc and rfi
re-score the pool with the model fine-tuned in the previous round (the active loop);
every round re-fine-tunes from the base checkpoint on everything labelled so far.

Fine-tuning (identical for every arm). Adam, lr 1e-4, batch 1, loss = cross-entropy +
Dice exactly as in training (no class weights) but computed ONLY on labelled pixels
(with every pixel labelled it equals the training loss -- checked by --selftest);
random time/frequency flips (never a rotation, which would swap the axes); dropout on.
An image is drawn with probability proportional to its number of labelled patches.
Every 200 steps (and at step 0) the max F1 on the 19 validation images is measured
and the best state kept; training stops after 5 checks without improvement or at
4000 steps. "Train longer" is measured on the same runs: the best state within the
first 600 steps (the pilot's length) is scored too.

Scores, pooled over the 5 test folds (all 109 images):
  max_f1   best single threshold on the pooled 109 -- the benchmark convention
           (10,000-bin histograms; --selftest checks this against the exact value)
  val_f1   each fold thresholded at its own validation-chosen threshold, TP/FP/FN summed
           (no test information used)

    ~/torch-env/bin/python experiments/active_learning/al_lofar.py --selftest
    ~/torch-env/bin/python experiments/active_learning/al_lofar.py --smoke
    ~/torch-env/bin/python experiments/active_learning/al_lofar.py --seeds 0 1 2
    ~/torch-env/bin/python analysis/active_learning/summarise_al.py
Resumable: rerun the same command; an arm is finished once its .json exists.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "hybrid_rfi_package"))
from hybrid_model import HybridRFINet  # noqa: E402  -- the model, imported unchanged
from lofar_data import load_lofar      # noqa: E402

RUNS = os.path.join(ROOT, "runs", "lofar")
PATCH, GRID = 64, 8                      # 8 x 8 patches of 64 x 64 per 512 x 512 image
NB = 10000                               # histogram bins for the pooled max F1
GROUP_IOU = 0.3
CHAINS = ("base", "full", "unc", "rfi", "rand", "img")     # run order within a seed
CHAIN_ID = {c: i for i, c in enumerate(CHAINS)}


# ----------------------------------------------------------------------------- helpers
def arm_name(chain, b):
    return f"{chain}{int(round(100 * b)):02d}"


def n_patches(b, n_pool):
    return int(round(b * n_pool * GRID * GRID))


def n_images(b, n_pool):
    return max(1, int(round(b * n_pool)))


def normalise_fixed(img, lo, hi):        # = experiments/lofar_hybrid.py normalise(.., "fixed")
    d = np.asarray(img, dtype=np.float64)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def expand(lab_img):
    """(GRID, GRID) bool -> (512, 512) bool pixel mask."""
    return np.repeat(np.repeat(lab_img, PATCH, axis=0), PATCH, axis=1)


def per_patch(a, how):
    """(n, 512, 512) -> (n, GRID, GRID): sum or mean over each 64 x 64 patch."""
    r = a.reshape(a.shape[0], GRID, PATCH, GRID, PATCH)
    return r.sum(axis=(2, 4)) if how == "sum" else r.mean(axis=(2, 4))


def entropy(p):
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


# ----------------------------------------------------------------------------- losses
def masked_ce(logits, y, w):
    """Mean cross-entropy over labelled pixels (w = 1). Equals F.cross_entropy when w == 1
    everywhere, and lofar_hybrid.soft_ce(.., t = y, w, cw = [1, 1]) in general."""
    ce = F.cross_entropy(logits, y, reduction="none")
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def masked_dice(logits, y, w, eps=1.0):
    """Dice over labelled pixels. Equals lofar_hybrid.dice_loss when w == 1 everywhere,
    and lofar_hybrid.soft_dice(.., t = y, w) in general."""
    p = F.softmax(logits, dim=1)[:, 1]
    t = (y == 1).float()
    inter = (w * p * t).sum(dim=(1, 2))
    denom = (w * p).sum(dim=(1, 2)) + (w * t).sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ----------------------------------------------------------------------------- metrics
def best_f1_threshold(y, p):
    """Exact max F1 and its threshold -- the convention of experiments/lofar_hybrid.py."""
    pc, rc, th = precision_recall_curve(y.ravel(), p.ravel())
    f1 = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f1[:-1]))
    return float(f1[k]), float(th[k])


def counts_at(y, p, th):
    q = p > th
    return int((q & y).sum()), int((q & ~y).sum()), int((~q & y).sum())


def histograms(y, p):
    b = np.minimum((p * NB).astype(np.int64), NB - 1)
    return (np.bincount(b[y], minlength=NB).astype(np.int64),
            np.bincount(b[~y], minlength=NB).astype(np.int64))


def max_f1_from_hist(hp, hn):
    tp = np.cumsum(hp[::-1])[::-1].astype(np.float64)       # positives with bin >= k
    fp = np.cumsum(hn[::-1])[::-1].astype(np.float64)
    fn = float(hp.sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1.0)
    k = int(np.argmax(f1))
    return float(f1[k]), k / NB


# ----------------------------------------------------------------------------- splits
def make_groups(Y):
    """Union-find over expert masks with IoU > GROUP_IOU (same rule as the pilot)."""
    flat = Y.reshape(len(Y), -1).astype(np.float32)          # counts < 2**24: exact
    inter = (flat @ flat.T).astype(np.float64)
    s = flat.sum(1).astype(np.float64)
    union = s[:, None] + s[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)   # float64, as the pilot
    n = len(Y)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if iou[i, j] > GROUP_IOU:
                parent[find(i)] = find(j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in out.values()), key=lambda g: g[0])


def make_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    order = [groups[i] for i in rng.permutation(len(groups))]
    order.sort(key=len, reverse=True)                        # stable: random within a size
    folds = [[] for _ in range(k)]
    for g in order:
        j = min(range(k), key=lambda f: (len(folds[f]), f))  # smallest fold first
        folds[j].extend(g)
    return [sorted(f) for f in folds]


def make_val(groups, test, n_val, seed):
    test = set(test)
    cand = [g for g in groups if not set(g) & test]
    rng = np.random.default_rng(seed)
    val = []
    for i in rng.permutation(len(cand)):
        if len(val) + len(cand[i]) <= n_val:
            val.extend(cand[i])
        if len(val) == n_val:
            break
    assert len(val) == n_val, f"could not fill the validation set exactly ({len(val)})"
    return sorted(val)


def check_splits(groups, splits, n=109):
    tested = []
    for sp in splits:
        t, v, p = set(sp["test"]), set(sp["val"]), set(sp["pool"])
        assert not (t & v) and not (t & p) and not (v & p), "sets overlap"
        assert t | v | p == set(range(n)), "a fold does not cover all 109 images"
        for g in groups:
            sg = set(g)
            assert sg <= t or sg <= v or sg <= p, f"group {g} is split across sets"
        tested += sp["test"]
    assert sorted(tested) == list(range(n)), "every image must be tested exactly once"


def build_splits(Y, a):
    groups = make_groups(Y)
    folds = make_folds(groups, a.n_folds, a.split_seed)
    splits = []
    for k, test in enumerate(folds):
        val = make_val(groups, test, a.n_val, 1000 * a.split_seed + k)
        pool = sorted(set(range(len(Y))) - set(test) - set(val))
        splits.append({"test": test, "val": val, "pool": pool})
    check_splits(groups, splits, len(Y))
    return groups, splits


# ----------------------------------------------------------------------------- model
def new_model(S, state):
    m = HybridRFINet(1, 2, base=S["mcfg"]["base"], depth=S["mcfg"]["depth"],
                     dropout=S["mcfg"]["dropout"])
    m.load_state_dict(state)
    return m.to(S["dev"])


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def to_cpu(state):
    return {k: v.detach().cpu() for k, v in state.items()}


@torch.no_grad()
def predict(model, X, idx, dev):
    was = model.training
    model.eval()
    out = np.empty((len(idx), 512, 512), np.float32)
    for s in range(0, len(idx), 2):                          # 2 at a time (CLAUDE.md s8)
        sel = list(idx[s:s + 2])
        x = torch.from_numpy(X[sel]).unsqueeze(1).to(dev)
        out[s:s + len(sel)] = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    model.train(was)
    return out


def load_seed(seed, d, dev):
    run = os.path.join(RUNS, f"hybrid_b08_nocw_seed{seed}")
    m = json.load(open(os.path.join(run, "eval_test", "metrics.json")))
    assert m["norm"] == "fixed" and m["base"] == 8 and m["depth"] == 4, "not a headline run"
    lo, hi = m["fixed_range"]
    X = np.stack([normalise_fixed(d.test_images[i, :, :, 0], lo, hi) for i in range(109)])
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return {"X": X, "base_state": state, "dev": dev, "headline_max_f1": m["max_f1"],
            "mcfg": {"base": m["base"], "depth": m["depth"], "dropout": m["dropout"]}}


# ----------------------------------------------------------------------------- core
def select(chain, lab, target, sel_model, S, pool, rng):
    """Add labels until `target` patches (or whole images, for 'img') are labelled."""
    lab = lab.copy()
    n = len(pool)
    if chain == "img":
        touched = lab.reshape(n, -1).any(1)
        need = target - int(touched.sum())
        assert need >= 0
        pick = rng.choice(np.flatnonzero(~touched), size=need, replace=False)
        lab[pick] = True
        return lab
    flat = lab.reshape(-1).copy()
    need = target - int(flat.sum())
    assert need >= 0
    if chain == "rand":
        score = rng.random(flat.size)
    else:
        P = predict(sel_model, S["X"], pool, S["dev"])
        score = per_patch(entropy(P), "mean") if chain == "unc" else per_patch(P, "sum")
        score = score.reshape(-1).astype(np.float64)
    score[flat] = -np.inf                                    # never pick a labelled patch twice
    flat[np.argsort(-score, kind="stable")[:need]] = True
    return flat.reshape(lab.shape)


def finetune(S, Y, lab, pool, val, seedtuple, a):
    ss = np.random.SeedSequence(seedtuple)
    torch.manual_seed(int(ss.generate_state(1)[0]))
    rng = np.random.default_rng(ss)
    X, dev = S["X"], S["dev"]
    model = new_model(S, S["base_state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cnt = lab.reshape(len(pool), -1).sum(1)
    imgs = np.flatnonzero(cnt)
    prob = cnt[imgs] / cnt[imgs].sum()
    W = {int(j): expand(lab[j]) for j in imgs}
    yv = Y[val]

    f1, th = best_f1_threshold(yv, predict(model, X, val, dev))
    best = {"step": 0, "f1": f1, "th": th, "state": snapshot(model)}
    short = dict(best)
    curve, since, steps = [[0, round(f1, 6)]], 0, 0
    model.train()
    t0 = time.time()
    for step in range(1, a.max_steps + 1):
        j = int(imgs[rng.choice(len(imgs), p=prob)])
        g = pool[j]
        x, y, w = X[g], Y[g], W[j]
        if rng.random() < 0.5:
            x, y, w = x[:, ::-1], y[:, ::-1], w[:, ::-1]      # flip frequency
        if rng.random() < 0.5:
            x, y, w = x[::-1, :], y[::-1, :], w[::-1, :]      # flip time
        xt = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(dev)
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))[None].to(dev)
        wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None].to(dev)
        logits = model(xt)
        loss = masked_ce(logits, yt, wt) + masked_dice(logits, yt, wt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps = step
        if step % a.eval_every == 0:
            f1, th = best_f1_threshold(yv, predict(model, X, val, dev))
            curve.append([step, round(f1, 6)])
            if f1 > best["f1"]:
                best, since = {"step": step, "f1": f1, "th": th, "state": snapshot(model)}, 0
            else:
                since += 1
            if step == a.short_steps:
                short = dict(best)                            # best within the pilot's length
            if since >= a.patience:
                break
    if steps < a.short_steps:
        short = dict(best)
    return model, best, short, curve, steps, time.time() - t0


def score_test(model, state, S, Y, test, th):
    model.load_state_dict(state)
    p = predict(model, S["X"], test, S["dev"])
    y = Y[test]
    hp, hn = histograms(y, p)
    tp, fp, fn = counts_at(y, p, th)
    return hp, hn, {"test_tp": tp, "test_fp": fp, "test_fn": fn}


def is_done(odir, name):
    return all(os.path.exists(os.path.join(odir, name + e)) for e in (".json", ".npz"))


def save_result(odir, name, res, arrays):
    os.makedirs(odir, exist_ok=True)
    tmp = os.path.join(odir, name + ".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, os.path.join(odir, name + ".npz"))
    tmp = os.path.join(odir, name + ".json.tmp")              # written last = completion mark
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, os.path.join(odir, name + ".json"))


def run_arm(name, chain, b, lab, seed, k, sp, S, Y, a, odir, log, keep_state=None):
    """Fine-tune (unless base), score the test fold, save. If keep_state is a file name,
    the best state is written there BEFORE the .json completion mark."""
    pool, val, test = sp["pool"], sp["val"], sp["test"]
    if chain == "base":
        model = new_model(S, S["base_state"])
        f1, th = best_f1_threshold(Y[val], predict(model, S["X"], val, S["dev"]))
        best = {"step": 0, "f1": f1, "th": th, "state": S["base_state"]}
        short, curve, steps, secs = dict(best), [[0, round(f1, 6)]], 0, 0.0
    else:
        model, best, short, curve, steps, secs = finetune(
            S, Y, lab, pool, val, [seed, k, CHAIN_ID[chain], int(round(1000 * b)), 7], a)
    if keep_state:
        os.makedirs(odir, exist_ok=True)
        tmp = os.path.join(odir, keep_state + ".tmp")
        torch.save(to_cpu(best["state"]), tmp)
        os.replace(tmp, os.path.join(odir, keep_state))
    hp, hn, cb = score_test(model, best["state"], S, Y, test, best["th"])
    hps, hns, cs = score_test(model, short["state"], S, Y, test, short["th"])
    labpx = np.repeat(np.repeat(lab, PATCH, 1), PATCH, 2)
    res = {"seed": seed, "fold": k, "arm": name, "chain": chain, "budget": b,
           "n_pool": len(pool), "n_val": len(val), "n_test": len(test),
           "labelled_patches": int(lab.sum()),
           "labelled_images_touched": int(lab.reshape(len(pool), -1).any(1).sum()),
           "labelled_fraction_of_pool": float(lab.mean()),
           "labelled_rfi_pixels": int((Y[pool] & labpx).sum()),
           "best": {"step": best["step"], "val_max_f1": best["f1"], "val_threshold": best["th"], **cb},
           "short": {"step": short["step"], "val_max_f1": short["f1"], "val_threshold": short["th"], **cs},
           "steps_done": steps, "hit_step_cap": bool(steps == a.max_steps and best["step"] == steps),
           "val_curve": curve, "seconds": round(secs, 1)}
    save_result(odir, name, res, {"hp": hp, "hn": hn, "hp_short": hps, "hn_short": hns, "lab": lab})
    log(f"[seed {seed} fold {k}] {name:<7} labelled {int(lab.sum()):>4} patches "
        f"({lab.sum() / GRID ** 2:5.1f} image-equiv., {res['labelled_images_touched']:>2} images touched) | "
        f"val F1 {best['f1']:.4f} @ step {best['step']:>4} (<= {a.short_steps}: {short['f1']:.4f}) | "
        f"{steps:>4} steps, {secs:5.0f} s", counted=chain != "base")


def run_chain(chain, seed, k, sp, S, Y, a, log):
    odir = os.path.join(a.out, f"seed{seed}", f"fold{k}")
    pool = sp["pool"]
    if chain in ("base", "full"):
        if not is_done(odir, chain):
            lab = np.full((len(pool), GRID, GRID), chain == "full")
            run_arm(chain, chain, 1.0 if chain == "full" else 0.0, lab, seed, k, sp, S, Y, a, odir,
                    log, keep_state="full_state.pt" if chain == "full" else None)
        return
    lab = np.zeros((len(pool), GRID, GRID), bool)
    prev = None
    for b in a.budgets:
        name = arm_name(chain, b)
        if is_done(odir, name):
            lab = np.load(os.path.join(odir, name + ".npz"))["lab"].astype(bool)
            prev = name
            continue
        sel_model = None
        if chain in ("unc", "rfi"):                           # score with the previous round's model
            state = S["base_state"] if prev is None else torch.load(
                os.path.join(odir, prev + "_state.pt"), map_location="cpu")
            sel_model = new_model(S, state)
        target = n_images(b, len(pool)) if chain == "img" else n_patches(b, len(pool))
        rng = np.random.default_rng([seed, k, CHAIN_ID[chain], int(round(1000 * b))])
        lab = select(chain, lab, target, sel_model, S, pool, rng)
        run_arm(name, chain, b, lab, seed, k, sp, S, Y, a, odir, log,
                keep_state=name + "_state.pt" if chain in ("unc", "rfi") else None)
        prev = name
    if chain in ("unc", "rfi"):                               # intermediate models no longer needed
        for b in a.budgets[:-1]:
            p = os.path.join(odir, arm_name(chain, b) + "_state.pt")
            if os.path.exists(p):
                os.remove(p)


# ----------------------------------------------------------------------------- self-test
def selftest(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SELF-TEST")
    # 1. the masked losses equal the training code's losses
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import lofar_hybrid as LH                                 # imported, not modified
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 2, 512, 512, generator=g)
    y = torch.randint(0, 2, (1, 512, 512), generator=g)
    ones = torch.ones(1, 512, 512)
    ref = F.cross_entropy(logits, y) + LH.dice_loss(logits, y)
    ours = masked_ce(logits, y, ones) + masked_dice(logits, y, ones)
    assert abs(float(ref - ours)) < 1e-5, (float(ref), float(ours))
    lab = np.random.default_rng(0).random((GRID, GRID)) < 0.3
    w = torch.from_numpy(expand(lab).astype(np.float32))[None]
    cw = torch.tensor([1.0, 1.0])
    ref2 = LH.soft_ce(logits, y.float(), w, cw) + LH.soft_dice(logits, y.float(), w)
    ours2 = masked_ce(logits, y, w) + masked_dice(logits, y, w)
    assert abs(float(ref2 - ours2)) < 1e-5, (float(ref2), float(ours2))
    print(f"  [ok] losses: full mask {float(ours):.6f} = training loss {float(ref):.6f}; "
          f"partial mask {float(ours2):.6f} = lofar_hybrid soft loss {float(ref2):.6f}")
    # 2. patch geometry: expand() and per_patch() agree
    labs = np.random.default_rng(1).random((3, GRID, GRID)) < 0.4
    px = np.stack([expand(l) for l in labs]).astype(np.int64)
    assert (per_patch(px, "sum") == labs * PATCH * PATCH).all()
    assert (np.repeat(np.repeat(labs, PATCH, 1), PATCH, 2) == px.astype(bool)).all()
    print("  [ok] patch geometry")
    # 3. splits
    d = load_lofar()
    Y = np.asarray(d.test_masks[..., 0], bool)
    groups, splits = build_splits(Y, a)
    print(f"  [ok] {len(groups)} groups (largest {max(map(len, groups))}); folds: "
          + ", ".join(f"test {len(s['test'])}/val {len(s['val'])}/pool {len(s['pool'])}" for s in splits)
          + "; every image tested once; no group split")
    # 4. selection never re-picks, reaches its target exactly, and is nested
    S = load_seed(0, d, dev)
    sp = splits[0]
    rng = np.random.default_rng(0)
    l1 = select("rand", np.zeros((len(sp["pool"]), GRID, GRID), bool), 50, None, S, sp["pool"], rng)
    l2 = select("rand", l1, 120, None, S, sp["pool"], rng)
    assert l1.sum() == 50 and l2.sum() == 120 and (l2 >= l1).all()
    i1 = select("img", np.zeros_like(l1), 3, None, S, sp["pool"], rng)
    assert i1.reshape(len(sp["pool"]), -1).all(1).sum() == 3 and i1.sum() == 3 * GRID * GRID
    print("  [ok] selection: exact targets, nested rounds, whole images for 'img'")
    # 5. the base model + normalisation reproduce the headline; histogram max F1 = exact
    model = new_model(S, S["base_state"])
    P = predict(model, S["X"], list(range(109)), dev)
    exact, _ = best_f1_threshold(Y, P)
    approx, _ = max_f1_from_hist(*histograms(Y, P))
    print(f"  headline max F1 {S['headline_max_f1']:.6f} | recomputed exact {exact:.6f} | "
          f"histogram {approx:.6f}")
    assert abs(exact - S["headline_max_f1"]) < 2e-4, "model/normalisation does not reproduce the headline"
    assert 0 <= exact - approx < 5e-4, "histogram max F1 too far from the exact value"
    print("  [ok] base model reproduces the headline; histogram F1 within 5e-4 of exact")
    print("SELF-TEST PASSED")


# ----------------------------------------------------------------------------- main
def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="default: all")
    ap.add_argument("--chains", nargs="+", default=list(CHAINS), choices=list(CHAINS))
    ap.add_argument("--out", default=os.path.join(RUNS, "al"))
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_val", type=int, default=19)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--short_steps", type=int, default=600)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 fold, 20 steps: checks the code only")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.folds, a.max_steps, a.eval_every, a.patience, a.short_steps = [0], [0], 20, 10, 5, 10
        a.out = os.path.join(RUNS, "al_smoke")
    if a.folds is None:
        a.folds = list(range(a.n_folds))
    a.budgets = sorted(a.budgets)
    assert a.short_steps % a.eval_every == 0, "--short_steps must be a multiple of --eval_every"
    return a


def main():
    a = parse()
    if a.selftest:
        selftest(a)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_lofar()
    Y = np.asarray(d.test_masks[..., 0], bool)               # expert masks of the 109 images
    groups, splits = build_splits(Y, a)
    design = {"n_folds": a.n_folds, "n_val": a.n_val, "split_seed": a.split_seed,
              "group_iou": GROUP_IOU, "budgets": a.budgets, "lr": a.lr, "max_steps": a.max_steps,
              "eval_every": a.eval_every, "patience": a.patience, "short_steps": a.short_steps,
              "patch": PATCH, "hist_bins": NB, "splits": splits}
    design = json.loads(json.dumps(design))
    os.makedirs(a.out, exist_ok=True)
    cfg = os.path.join(a.out, "config.json")
    if os.path.exists(cfg):
        if json.load(open(cfg)) != design:
            sys.exit(f"{cfg} was written with different settings; use a new --out instead of mixing runs.")
    else:
        with open(cfg + ".tmp", "w") as fh:
            json.dump(design, fh, indent=1)
        os.replace(cfg + ".tmp", cfg)

    n_rounds = {"base": 0, "full": 1, "unc": len(a.budgets), "rfi": len(a.budgets),
                "rand": len(a.budgets), "img": len(a.budgets)}
    def pending():
        n = 0
        for s in a.seeds:
            for c in a.chains:
                names = [c] if c in ("base", "full") else [arm_name(c, b) for b in a.budgets]
                for k in a.folds:
                    odir = os.path.join(a.out, f"seed{s}", f"fold{k}")
                    n += sum(not is_done(odir, nm) for nm in names) * (1 if n_rounds[c] else 0)
        return n
    todo = pending()
    t_start, done = time.time(), [0]
    print(f"{len(groups)} groups; {len(splits)} folds; {todo} fine-tuning runs to do "
          f"(seeds {a.seeds}, chains {a.chains})", flush=True)

    def log(msg, counted=True):
        if counted:
            done[0] += 1
        el = time.time() - t_start
        eta = el / max(done[0], 1) * max(todo - done[0], 0) / 3600
        print(f"{msg} | {done[0]}/{todo}, ETA {eta:.1f} h", flush=True)

    for seed in a.seeds:
        S = load_seed(seed, d, dev)
        for chain in a.chains:
            for k in a.folds:
                run_chain(chain, seed, k, splits[k], S, Y, a, log)
        print(f"-- seed {seed} finished ({(time.time() - t_start) / 60:.0f} min so far)", flush=True)
    print("ALL DONE. Summarise with: ~/torch-env/bin/python analysis/active_learning/summarise_al.py"
          + ("" if a.out == os.path.join(RUNS, "al") else f" --out {a.out}"), flush=True)


if __name__ == "__main__":
    main()
