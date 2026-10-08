"""Where could a NEW method find F1? Diagnostics on the best hybrid's predictions.

Questions, all measured on hybrid base 8, no class weight, seed 0:
  Q1  How much would a per-image threshold help? (oracle upper bound)
  Q2  Are the missed RFI pixels inside frequency channels (columns) that the
      model already flags elsewhere in the same image? If so, channel-level
      context could recover them.
  Q3  Do the expert's masks flag WHOLE channels more than AOFlagger's do?
      (a style gap no model trained on AOFlagger labels can learn by itself)
  Q4  A crude channel-completion rule, tuned on VALIDATION (AOFlagger labels)
      and applied once to test.

Rows = time, columns = frequency (PART 11.2).
    ~/torch-env/bin/python analysis/diagnose_context_headroom.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from ablation_faint_recall import normalise_fixed, build, HybridRFINet  # noqa: E402
from lofar_data import load_lofar                                       # noqa: E402

RUN = os.path.join(ROOT, "runs", "lofar", "hybrid_b08_nocw_seed0")


def f1(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def counts(pred, y):
    return int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())


@torch.no_grad()
def predict(model, images, lo, hi, dev):
    out = np.zeros((len(images), 512, 512), np.float32)
    for i in range(0, len(images), 2):
        x = np.stack([normalise_fixed(im, lo, hi) for im in images[i:i + 2]])
        p = torch.softmax(model(torch.from_numpy(x).unsqueeze(1).float().to(dev)), 1)[:, 1]
        out[i:i + len(x)] = p.cpu().numpy()
    return out


def complete_columns(pred, frac):
    """Flag an entire column (frequency channel) of an image if at least
    `frac` of its pixels are already flagged."""
    colfrac = pred.mean(axis=1, keepdims=True)                # (N,1,512)
    return pred | (colfrac >= frac)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = json.load(open(os.path.join(RUN, "eval_test", "metrics.json")))
    lo, hi = m["fixed_range"]; th = float(m["val_selected_threshold"])
    model = HybridRFINet(1, 2, base=m["base"], depth=m["depth"], dropout=m["dropout"])
    ck = torch.load(os.path.join(RUN, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck)); model.to(dev).eval()

    d = load_lofar()
    Y = np.asarray(d.test_masks[..., 0], bool)
    P = predict(model, [d.test_images[i, :, :, 0] for i in range(109)], lo, hi, dev)
    pred = P > th
    tp, fp, fn = counts(pred, Y)
    base = f1(tp, fp, fn)
    R = {"threshold": th, "pooled_f1_test": base, "TP": tp, "FP": fp, "FN": fn}
    print(f"baseline pooled F1 (val threshold {th:.4f}): {base:.4f}   TP {tp:,} FP {fp:,} FN {fn:,}")

    # ---- Q1: per-image oracle threshold (upper bound, uses test labels) ----
    grid = np.linspace(0.02, 0.98, 49)
    TP = FP = FN = 0
    for i in range(109):
        best = max(grid, key=lambda t: f1(*counts(P[i] > t, Y[i])))
        a, b, c = counts(P[i] > best, Y[i]); TP += a; FP += b; FN += c
    R["Q1_per_image_oracle_threshold_f1"] = f1(TP, FP, FN)
    print(f"Q1 per-image ORACLE threshold F1: {R['Q1_per_image_oracle_threshold_f1']:.4f} "
          f"(upper bound, gain {R['Q1_per_image_oracle_threshold_f1'] - base:+.4f})")

    # ---- Q2: are missed pixels inside channels the model already flags? ----
    colfrac = pred.mean(axis=1, keepdims=True)
    missed = Y & ~pred
    R["Q2"] = {}
    for c in (0.05, 0.10, 0.25, 0.50):
        share = float((missed & (colfrac >= c)).sum() / max(missed.sum(), 1))
        R["Q2"][str(c)] = share
        print(f"Q2 missed RFI pixels in channels the model flags >= {int(c * 100):>2}%: {100 * share:5.1f}%")
    hcol = Y.mean(axis=1, keepdims=True)
    share_h = float((missed & (hcol >= 0.5)).sum() / max(missed.sum(), 1))
    R["Q2"]["in_channels_expert_flags_ge_50pct"] = share_h
    print(f"Q2 missed RFI pixels in channels the EXPERT flags >= 50%: {100 * share_h:5.1f}%")

    # ---- Q3: style -- share of flagged pixels that sit in mostly-flagged channels ----
    pairs = np.load(os.path.join(ROOT, "analysis", "lofar_analysis", "lofar_leak_train_idx.npy"))
    # map each test image to its byte-identical training twin (PART 11.5)
    import hashlib
    h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    tr_of = {h(d.train_images[int(i)]): int(i) for i in pairs}
    A = np.stack([np.asarray(d.train_masks[tr_of[h(d.test_images[t])], :, :, 0], bool) for t in range(109)])
    def full_channel_share(M, c=0.8):
        cf = M.mean(axis=1, keepdims=True)
        return float((M & (cf >= c)).sum() / max(M.sum(), 1))
    R["Q3"] = {"expert": full_channel_share(Y), "aoflagger": full_channel_share(A),
               "model": full_channel_share(pred)}
    print("Q3 share of flagged pixels lying in channels flagged >=80%: "
          + "  ".join(f"{k} {100 * v:.1f}%" for k, v in R["Q3"].items()))

    # ---- Q4: channel completion, fraction tuned on VALIDATION (AOFlagger labels) ----
    rng = np.random.default_rng(0); idx = d.clean_train_idx.copy(); rng.shuffle(idx)
    val = np.sort(idx[: int(len(idx) * 0.1)][:150])
    Pv = predict(model, [d.train_images[int(i), :, :, 0] for i in val], lo, hi, dev)
    Yv = np.asarray(d.train_masks[val, :, :, 0], bool)
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    val_f1 = {f_: f1(*counts(complete_columns(Pv > th, f_), Yv)) for f_ in fracs}
    best_f = max(val_f1, key=val_f1.get)
    test_f1 = {f_: f1(*counts(complete_columns(pred, f_), Y)) for f_ in fracs}
    R["Q4"] = {"val_f1_by_frac": val_f1, "chosen_on_val": best_f,
               "test_f1_at_chosen": test_f1[best_f], "test_f1_by_frac_DIAGNOSTIC_ONLY": test_f1}
    print(f"Q4 val-chosen completion frac {best_f}: test F1 {test_f1[best_f]:.4f} "
          f"({test_f1[best_f] - base:+.4f}); best on test (diagnostic) "
          f"{max(test_f1.values()):.4f} at {max(test_f1, key=test_f1.get)}")

    json.dump(R, open(os.path.join(ROOT, "analysis", "diagnose_context_headroom.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
