"""INDEPENDENT audit of the hybrid LOFAR result.

Deliberately does NOT import anything from experiments/lofar_hybrid.py except
the model class. Metrics are recomputed from scratch so a bug in the training
script's own metric code cannot hide here.
"""
import json, sys, hashlib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/ronaksingh/Documents/minor project/Minor Project"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/src/hybrid_rfi_package")
from hybrid_model import HybridRFINet
from lofar_data import load_lofar

RUN = f"{ROOT}/runs/lofar/hybrid_b08_nocw_seed0"
rep = json.load(open(f"{RUN}/eval_test/metrics.json"))
d = load_lofar()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 74)
print("A. SPLIT INTEGRITY — recomputed from scratch")
print("=" * 74)
rng = np.random.default_rng(rep["seed"])
idx = d.clean_train_idx.copy(); rng.shuffle(idx)
n_val = max(1, int(len(idx) * 0.1))
val_idx, tr_idx = idx[:n_val], idx[n_val:]
print(f"  train {len(tr_idx)}  val {len(val_idx)}  test 109")
print(f"  train n val overlap        : {len(set(tr_idx.tolist()) & set(val_idx.tolist()))}")

md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
test_h = {md5(d.test_images[t]) for t in range(109)}
tr_leak = sum(1 for i in tr_idx if md5(d.train_images[int(i)]) in test_h)
va_leak = sum(1 for i in val_idx if md5(d.train_images[int(i)]) in test_h)
print(f"  TRAIN images also in test  : {tr_leak}")
print(f"  VAL   images also in test  : {va_leak}")
print(f"  unique test images         : {len(test_h)} of 109")

# ---- rebuild normalisation from TRAIN ONLY, exactly as the script claims ----
lo_r, hi_r = rep["fixed_range"]
rng2 = np.random.default_rng(rep["seed"])
sel = rng2.choice(tr_idx, size=min(200, len(tr_idx)), replace=False)
los = [np.percentile(d.train_images[int(i), :, :, 0].astype(np.float64), 0.5) for i in sel]
his = [np.percentile(d.train_images[int(i), :, :, 0].astype(np.float64), 99.5) for i in sel]
lo, hi = float(np.mean(los)), float(np.mean(his))
print(f"\n  fixed range reported {lo_r:.6g}, {hi_r:.6g}")
print(f"  fixed range recomputed from TRAIN only {lo:.6g}, {hi:.6g}")
print(f"  match: {abs(lo-lo_r) < 1e-6 and abs(hi-hi_r) < 1e-6}  <- confirms no test data in calibration")

def norm(img):
    return np.clip((img.astype(np.float64) - lo) / (hi - lo), 0, 1).astype(np.float32)

# ---------------------------------------------------------------- inference
model = HybridRFINet(1, 2, base=rep["base"], depth=rep["depth"], dropout=rep["dropout"]).to(dev)
model.load_state_dict(torch.load(f"{RUN}/best.pt", map_location=dev))
model.eval()

@torch.no_grad()
def scores(images, masks, indices):
    yp, yt = [], []
    for s in range(0, len(indices), 2):
        sel = indices[s:s+2]
        x = torch.from_numpy(np.stack([norm(images[int(i), :, :, 0]) for i in sel])).unsqueeze(1).to(dev)
        yp.append(F.softmax(model(x), 1)[:, 1].cpu().numpy().reshape(len(sel), -1))
        yt.append(np.stack([masks[int(i), :, :, 0] for i in sel]).reshape(len(sel), -1))
    return np.concatenate(yt).astype(bool), np.concatenate(yp)

def f1_at(yt, yp, th):
    p = yp > th
    tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
    pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1)
    return 2*pr*rc/max(pr+rc,1e-12), pr, rc, tp, fp, fn

print("\n" + "=" * 74)
print("B. METRICS RECOMPUTED INDEPENDENTLY on the 109 test images")
print("=" * 74)
yt, yp = scores(d.test_images, d.test_masks, np.arange(109))
th = rep["val_selected_threshold"]
f1, pr, rc, tp, fp, fn = f1_at(yt.ravel(), yp.ravel(), th)
print(f"  pooled F1  reported {rep['pooled_f1']:.6f}   recomputed {f1:.6f}   diff {abs(f1-rep['pooled_f1']):.2e}")
print(f"  precision  reported {rep['pooled_precision']:.6f}   recomputed {pr:.6f}")
print(f"  recall     reported {rep['pooled_recall']:.6f}   recomputed {rc:.6f}")
print(f"  TP/FP/FN   reported {rep['TP']}/{rep['FP']}/{rep['FN']}   recomputed {tp}/{fp}/{fn}")

from sklearn.metrics import precision_recall_curve, roc_auc_score
pc, rcv, _ = precision_recall_curve(yt.ravel(), yp.ravel())
mf1 = float(np.max(2*pc*rcv/(pc+rcv+1e-10)))
roc = roc_auc_score(yt.ravel(), yp.ravel())
print(f"  max F1     reported {rep['max_f1']:.6f}   recomputed {mf1:.6f}   diff {abs(mf1-rep['max_f1']):.2e}")
print(f"  ROC AUC    reported {rep['roc_auc']:.6f}   recomputed {roc:.6f}   diff {abs(roc-rep['roc_auc']):.2e}")

print("\n" + "=" * 74)
print("C. WAS THE THRESHOLD REALLY CHOSEN ON VALIDATION?")
print("=" * 74)
ytv, ypv = scores(d.train_images, d.train_masks, val_idx[:150])
pcv, rcvv, thv = precision_recall_curve(ytv.ravel(), ypv.ravel())
f1v = 2*pcv*rcvv/(pcv+rcvv+1e-10)
best_val_th = float(thv[int(np.argmax(f1v[:-1]))])
print(f"  threshold used on test        : {th:.6f}")
print(f"  best threshold on VALIDATION  : {best_val_th:.6f}   match: {abs(best_val_th-th)<1e-5}")
best_test_th = 0.0
tf1 = 0.0
for t in np.linspace(0.01, 0.99, 99):
    v = f1_at(yt.ravel(), yp.ravel(), t)[0]
    if v > tf1: tf1, best_test_th = v, t
print(f"  best threshold ON TEST        : {best_test_th:.4f} giving F1 {tf1:.4f}")
print(f"  penalty for honest thresholding: {tf1-f1:+.4f}  (we report the LOWER number as pooled_f1)")

print("\n" + "=" * 74)
print("D. OVERFITTING CHECK — train vs val vs test")
print("=" * 74)
rng3 = np.random.default_rng(99)
tr_sample = rng3.choice(tr_idx, 109, replace=False)
ytt, ypt = scores(d.train_images, d.train_masks, tr_sample)
for name, (a, b) in [("TRAIN (109 sampled)", (ytt, ypt)), ("VAL   (150)", (ytv, ypv)), ("TEST  (109)", (yt, yp))]:
    pcx, rcx, _ = precision_recall_curve(a.ravel(), b.ravel())
    mx = float(np.max(2*pcx*rcx/(pcx+rcx+1e-10)))
    print(f"  {name:22} max-F1 {mx:.4f}   ROC {roc_auc_score(a.ravel(), b.ravel()):.4f}")
print("  NOTE train/val labels are AOFlagger; test labels are HUMAN -> not directly comparable,")
print("  but a large train>>val gap would still indicate memorisation.")
