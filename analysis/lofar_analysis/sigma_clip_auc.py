"""AUROC and AUPRC for the per-image sigma-clipping baseline on the 109 expert-labelled LOFAR test images.
Preprocessing is lofar_data.preprocess (clip at mu+20 sigma, PART 11.6b), as for the 0.4103 in PART 11.9. A pixel's score is how many standard deviations it sits
above its image's mean; flagging at score > 2.5 is the 2.5-sigma clip (F1 0.4103), and sweeping the cut-off gives
the curves. AUPRC is the trapezoid auc(recall, precision), as for the models and Mesarcik et al.
    ~/torch-env/bin/python analysis/lofar_analysis/sigma_clip_auc.py
"""
import json, os, sys
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lofar_data import load_lofar, preprocess


d = load_lofar()
z, y = [], []
for i in range(d.test_images.shape[0]):
    x = preprocess(d.test_images[i, :, :, 0].astype(np.float64))
    sd = x.std()
    z.append(((x - x.mean()) / sd if sd > 0 else np.zeros_like(x)).ravel())
    y.append(np.asarray(d.test_masks[i, :, :, 0]).ravel())
z, y = np.concatenate(z), np.concatenate(y)

p = z > 2.5
tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
f1_25 = 2 * tp / (2 * tp + fp + fn)
fpr, tpr, _ = roc_curve(y, z)
pc, rc, _ = precision_recall_curve(y, z)
f1s = 2 * pc * rc / np.maximum(pc + rc, 1e-12)
out = dict(f1_at_2p5_sigma=f1_25, auroc=float(auc(fpr, tpr)), auprc=float(auc(rc, pc)),
           max_f1_any_cutoff=float(f1s.max()), n_pixels=int(y.size))
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sigma_clip_auc.json"), "w"), indent=2)
