"""Stage 1 check, no training: run AOFlagger on the HERA test images exactly as Mesarcik et al.'s
code does (utils/flagging/ao_flagger.py, strategy hera_10.lua, raw values, before any clipping) and
score its masks against the simulator's masks in the pickle. Paper Table 1, AOFlagger row:
AUROC 0.9736, AUPRC 0.8799, F1 0.8729. Run with ~/aof-env/bin/python.
"""
import json, os, pickle, time
import numpy as np
import aoflagger as aof
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKL = os.path.join(ROOT, "data", "hera", "HERA_04-03-2022_all.pkl")
STRATEGY = os.path.join(HERE, "hera_10.lua")


def flag(data, strategy, ao):
    mask = np.empty(data[..., 0].shape, dtype=bool)
    for i in range(len(data)):
        s = ao.make_image_set(data.shape[1], data.shape[2], 1)
        s.set_image_buffer(0, data[i, ..., 0])
        mask[i] = strategy.run(s).get_buffer()
    return mask


def scores(pred, truth):
    p, t = pred.ravel(), truth.ravel()
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    pc, rc, _ = precision_recall_curve(t, p)
    return dict(f1=2 * prec * rec / (prec + rec), precision=prec, recall=rec, tp=tp, fp=fp, fn=fn,
                auroc=float(roc_auc_score(t, p)), auprc_trapz=float(auc(rc, pc)),
                auprc_ap=float(average_precision_score(t, p)),
                flagged_pct=float(p.mean() * 100), truth_pct=float(t.mean() * 100))


with open(PKL, "rb") as fh:
    train_x, train_y, test_x, test_y = pickle.load(fh)
print("train", train_x.shape, train_x.dtype, train_y.shape, train_y.dtype)
print("test ", test_x.shape, test_x.dtype, test_y.shape, test_y.dtype)
print("inf in test", int(np.isinf(test_x).sum()), "nan", int(np.isnan(test_x).sum()),
      "min", float(test_x.min()), "max", float(test_x.max()))
test_x = test_x.copy()
test_x[test_x == np.inf] = np.finfo(test_x.dtype).max
test_x = test_x.astype("float32")
truth = np.asarray(test_y)[..., 0].astype(bool)

ao = aof.AOFlagger()
print("AOFlagger version", ao.get_version_string() if hasattr(ao, "get_version_string") else "?")
strategy = ao.load_strategy_file(STRATEGY)
t0 = time.time()
pred = flag(test_x, strategy, ao)
sec = time.time() - t0
print(f"flagged {len(pred)} images in {sec:.1f} s ({sec / len(pred):.3f} s/image)")
out = dict(n_test=len(pred), seconds=sec, all_140=scores(pred, truth))

# the same score on the 135 unique test images (internal duplicates removed)
_, keep = np.unique(test_x.reshape(len(test_x), -1), axis=0, return_index=True)
keep = np.sort(keep)
out["unique_n"] = int(len(keep)); out["unique"] = scores(pred[keep], truth[keep])
print(json.dumps(out, indent=2))
print("paper Table 1 AOFlagger: AUROC 0.9736  AUPRC 0.8799  F1 0.8729")
json.dump(out, open(os.path.join(HERE, "check_aoflagger_hera.json"), "w"), indent=2)
print("DONE")
