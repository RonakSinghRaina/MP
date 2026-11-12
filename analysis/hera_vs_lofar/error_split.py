"""Split each model's mistakes by whether AOFlagger (the training labels) was right (PART 41).

For every test pixel in the centre 472x472, at the threshold that gives the model's
reported max F1 (vs the truth):
  own mistake    : the model is wrong on a pixel where AOFlagger's label was right
  copied mistake : the model is wrong on a pixel where AOFlagger's label was wrong too
                   (it repeats AOFlagger's false alarm, or misses what AOFlagger missed)
Also the border (the 20-pixel frame outside the centre, which tf_unet never outputs):
AOFlagger's precision/recall there, and Kestrel's F1 and copied false alarms there.

    ~/torch-env/bin/python analysis/hera_vs_lofar/error_split.py
Writes analysis/hera_vs_lofar/error_split.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "report", "Images", "Figures"))
from hera_data import load_hera      # noqa: E402
from lofar_data import load_lofar    # noqa: E402
from make_data_figures import lofar_twins   # noqa: E402

C0, C1 = 20, 492
K = {"hera": "runs/hera/hybrid_b08_nocw_seed{}", "lofar": "runs/lofar/hybrid_b08_nocw_seed{}"}
U = {"hera": "runs/hera/tfunet_lr1e-4_seed{}", "lofar": "runs/lofar/lr1e-4_seed{}"}


def thr_max_f1(p, t):
    pc, rc, th = precision_recall_curve(t, p)
    f1 = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f1[:-1]))
    return float(th[k]), float(f1[k])


def counts(flag, T, A):
    wrong = flag != T
    return dict(own=int((wrong & (A == T)).sum()), copied=int((wrong & (A != T)).sum()),
                copied_false_alarms=int((flag & ~T & A).sum()),
                copied_misses=int((~flag & T & ~A).sum()),
                own_false_alarms=int((flag & ~T & ~A).sum()),
                own_misses=int((~flag & T & A).sum()),
                fixed_teacher_errors=int(((flag == T) & (A != T)).sum()),
                tp=int((flag & T).sum()), fp=int((flag & ~T).sum()), fn=int((~flag & T).sum()))


def main():
    R = {}
    for ds in ("lofar", "hera"):
        d = load_hera() if ds == "hera" else load_lofar()
        Tf = np.asarray(d.test_masks[..., 0]).astype(bool)
        if ds == "hera":
            Af = np.asarray(d.test_masks_aof[..., 0]).astype(bool)
        else:
            Af = np.stack([np.asarray(d.train_masks[j, :, :, 0]) for j in lofar_twins(d)]).astype(bool)
        T, A = Tf[:, C0:C1, C0:C1], Af[:, C0:C1, C0:C1]
        frame = np.ones(Tf.shape[1:], bool); frame[C0:C1, C0:C1] = False
        Tb, Ab = Tf[:, frame], Af[:, frame]
        r = R[ds] = {"teacher_errors_crop": int((A != T).sum()), "true_rfi_crop": int(T.sum()),
                     "pixels_crop": int(T.size)}
        tpb = int((Ab & Tb).sum()); fpb = int((Ab & ~Tb).sum()); fnb = int((~Ab & Tb).sum())
        r["border"] = dict(true_rfi_frac=float(Tb.mean()), teacher_rfi_frac=float(Ab.mean()),
                           teacher_precision=tpb / max(tpb + fpb, 1), teacher_recall=tpb / max(tpb + fnb, 1),
                           teacher_f1=2 * tpb / max(2 * tpb + fpb + fnb, 1))
        r["centre"] = dict(true_rfi_frac=float(T.mean()), teacher_rfi_frac=float(A.mean()))
        for name, pat in (("kestrel", K[ds]), ("tf_unet", U[ds])):
            rows = []
            for s in (0, 1, 2):
                run = os.path.join(ROOT, pat.format(s), "eval_test")
                if name == "kestrel":
                    Pf = np.load(os.path.join(run, "test_pred_full512.npy"), mmap_mode="r")
                    P = np.asarray(Pf[:, C0:C1, C0:C1], dtype=np.float32)
                else:
                    P = np.load(os.path.join(run, "test_pred_crop472.npy"))
                th, f1 = thr_max_f1(P.ravel(), T.ravel())
                row = dict(seed=s, threshold=th, max_f1=f1, **counts(P > th, T, A))
                if name == "kestrel":       # the border, at the same threshold
                    Pb = np.asarray(Pf, dtype=np.float32)[:, frame] > th
                    tp = int((Pb & Tb).sum()); fp = int((Pb & ~Tb).sum()); fn = int((~Pb & Tb).sum())
                    row["border_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
                    row["border_false_alarms_also_flagged_by_teacher"] = int((Pb & ~Tb & Ab).sum()) / max(fp, 1)
                rows.append(row)
                print(ds, name, s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                      flush=True)
            r[name] = rows
        print(ds, json.dumps({k: v for k, v in r.items() if k in ("border", "centre", "teacher_errors_crop")}),
              flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "error_split.json"), "w"), indent=1)
    print("saved")


if __name__ == "__main__":
    main()
