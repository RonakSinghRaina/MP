"""Why is tf_unet ahead of Kestrel on HERA but behind it on LOFAR? (PART 41)

Everything is measured on the test images, on the centre 472x472 pixels that tf_unet
outputs (rows and columns 20..491), with the saved predictions of
predict_kestrel.py and predict_tfunet.py (best checkpoints, seeds 0-2).

  truth   : HERA = the simulator's exact masks; LOFAR = the human expert's masks
  teacher : AOFlagger's masks of the same test images, i.e. the kind of label the
            models were trained on. HERA: test_masks_aof. LOFAR: the AOFlagger mask of
            each test image's byte-identical twin in the training set (PART 11.6).

Writes analysis/hera_vs_lofar/compare_hera_lofar.json.

    ~/torch-env/bin/python analysis/hera_vs_lofar/compare_hera_lofar.py
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from hera_data import load_hera      # noqa: E402
from lofar_data import load_lofar    # noqa: E402

C0, C1 = 20, 492                      # tf_unet's 472x472 output window
OUT = os.path.join(os.path.dirname(__file__), "compare_hera_lofar.json")
K = {"hera": "runs/hera/hybrid_b08_nocw_seed{}", "lofar": "runs/lofar/hybrid_b08_nocw_seed{}"}
U = {"hera": "runs/hera/tfunet_lr1e-4_seed{}", "lofar": "runs/lofar/lr1e-4_seed{}"}


def prf(pred, lab):
    tp = int((pred & lab).sum()); fp = int((pred & ~lab).sum()); fn = int((~pred & lab).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12))


def best_f1(scores, lab):
    """Max F1 over thresholds (the reported 'max F1'), with its threshold, P and R."""
    pc, rc, th = precision_recall_curve(lab, scores)
    f1 = 2 * pc * rc / (pc + rc + 1e-10)
    k = int(np.argmax(f1[:-1]))
    return dict(max_f1=float(f1[k]), threshold=float(th[k]), precision=float(pc[k]),
                recall=float(rc[k]), auprc=float(auc(rc, pc)))


def lofar_teacher_masks(d):
    """AOFlagger mask of each LOFAR test image = training mask of its byte-identical twin."""
    leak = np.asarray(d.leak_train_idx)
    fp = {}
    for j in leak:
        fp.setdefault(float(d.train_images[int(j), :, :, 0].sum(dtype=np.float64)), []).append(int(j))
    n = d.test_images.shape[0]
    twin = np.full(n, -1)
    for i in range(n):
        hits = [j for j in fp.get(float(d.test_images[i, :, :, 0].sum(dtype=np.float64)), [])
                if np.array_equal(d.train_images[j], d.test_images[i])]
        assert len(hits) >= 1, "test image {} has no twin".format(i)
        masks = [np.asarray(d.train_masks[j, :, :, 0]) for j in hits]
        assert all(np.array_equal(masks[0], m) for m in masks), "twins disagree"
        twin[i] = hits[0]
    assert len(set(twin.tolist())) == n, "twin mapping is not one-to-one"
    return twin, np.stack([np.asarray(d.train_masks[j, :, :, 0]) for j in twin]).astype(bool)


def faint_share(images, truth):
    """Share of true RFI pixels fainter than the median clean pixel of their own image."""
    below = total = 0
    for x, t in zip(images, truth):
        if t.any() and (~t).any():
            med = np.median(x[~t])
            below += int((x[t] < med).sum()); total += int(t.sum())
    return below / max(total, 1)


def main():
    R = {}
    for ds in ("lofar", "hera"):
        print("=" * 70, "\n", ds, flush=True)
        d = load_hera() if ds == "hera" else load_lofar()
        n = d.test_images.shape[0]
        truth_full = np.asarray(d.test_masks[..., 0]).astype(bool)
        if ds == "hera":
            teach_full = np.asarray(d.test_masks_aof[..., 0]).astype(bool)
            twin = None
        else:
            twin, teach_full = lofar_teacher_masks(d)
        T = truth_full[:, C0:C1, C0:C1]
        A = teach_full[:, C0:C1, C0:C1]
        r = R[ds] = {"n_test": n, "n_train_used": int(len(d.clean_train_idx))}

        # ---- the training labels (teacher) against the truth -------------------------
        r["teacher_vs_truth_full512"] = prf(teach_full.ravel(), truth_full.ravel())
        r["teacher_vs_truth_crop"] = prf(A.ravel(), T.ravel())
        r["truth_rfi_frac_crop"] = float(T.mean())
        r["teacher_rfi_frac_crop"] = float(A.mean())
        r["teacher_over_truth_ratio_crop"] = float(A.mean() / T.mean())
        fpA = A & ~T                       # teacher's false alarms
        fnA = ~A & T                       # teacher's misses
        near_T1 = np.stack([binary_dilation(t, np.ones((3, 3), bool)) for t in T])
        near_T2 = np.stack([binary_dilation(t, np.ones((5, 5), bool)) for t in T])
        near_A1 = np.stack([binary_dilation(a, np.ones((3, 3), bool)) for a in A])
        r["teacher_false_alarm_px"] = int(fpA.sum())
        r["teacher_miss_px"] = int(fnA.sum())
        r["teacher_false_alarms_within_1px_of_true_rfi"] = float((fpA & near_T1).sum() / max(fpA.sum(), 1))
        r["teacher_false_alarms_within_2px_of_true_rfi"] = float((fpA & near_T2).sum() / max(fpA.sum(), 1))
        r["teacher_misses_within_1px_of_teacher_flags"] = float((fnA & near_A1).sum() / max(fnA.sum(), 1))
        print(json.dumps({k: v for k, v in r.items()}, indent=1), flush=True)

        # ---- the data themselves ------------------------------------------------------
        imgs = np.asarray(d.test_images[..., 0])
        r["faint_true_rfi_share_full512"] = faint_share(imgs, truth_full)
        if twin is not None:
            assert np.array_equal(np.asarray(d.train_images[int(twin[0])]), np.asarray(d.test_images[0]))
        r["train_teacher_rfi_frac"] = float(np.mean([np.asarray(d.train_masks[int(i), :, :, 0]).mean()
                                                     for i in d.clean_train_idx[:2000]]))

        # ---- each model -----------------------------------------------------------------
        for name, pat in (("kestrel", K[ds]), ("tf_unet", U[ds])):
            rows = []
            for s in (0, 1, 2):
                run = os.path.join(ROOT, pat.format(s), "eval_test")
                if name == "kestrel":
                    P = np.load(os.path.join(run, "test_pred_full512.npy"), mmap_mode="r")[:, C0:C1, C0:C1]
                else:
                    P = np.load(os.path.join(run, "test_pred_crop472.npy"), mmap_mode="r")
                P = np.asarray(P, dtype=np.float32)
                vt = best_f1(P.ravel(), T.ravel())           # the reported max F1
                va = best_f1(P.ravel(), A.ravel())           # agreement with the teacher
                flag = P > vt["threshold"]
                row = dict(seed=s, vs_truth=vt, vs_teacher=va,
                           f1_vs_teacher_at_truth_threshold=prf(flag.ravel(), A.ravel())["f1"],
                           flagged_over_truth=float(flag.mean() / T.mean()),
                           # how often the model repeats the teacher's two kinds of mistake
                           copies_teacher_false_alarms=float((flag & fpA).sum() / max(fpA.sum(), 1)),
                           copies_teacher_misses=float((~flag & fnA).sum() / max(fnA.sum(), 1)),
                           # the model's own false alarms: share that the teacher also flagged
                           false_alarms_also_flagged_by_teacher=float((flag & ~T & A).sum() /
                                                                      max((flag & ~T).sum(), 1)),
                           false_alarms_within_1px_of_true_rfi=float((flag & ~T & near_T1).sum() /
                                                                     max((flag & ~T).sum(), 1)),
                           misses_also_missed_by_teacher=float((~flag & T & ~A).sum() /
                                                               max((~flag & T).sum(), 1)))
                # threshold-free: can the model tell real RFI that the teacher flagged apart
                # from the teacher's false alarms? 0.5 = not at all, 1 = perfectly.
                pos = P[T & A]; neg = P[fpA]
                row["separates_real_from_teacher_false_alarms_auroc"] = float(roc_auc_score(
                    np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
                rows.append(row)
                print(name, s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in row.items() if not isinstance(v, dict)}),
                      "max F1 truth {:.4f} teacher {:.4f}".format(vt["max_f1"], va["max_f1"]), flush=True)
            r[name] = rows
            keys = [k for k, v in rows[0].items() if isinstance(v, float)]
            r[name + "_mean"] = {k: [float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows], ddof=1))]
                                 for k in keys}
            for sub in ("vs_truth", "vs_teacher"):
                for k in rows[0][sub]:
                    v = [x[sub][k] for x in rows]
                    r[name + "_mean"][sub + "." + k] = [float(np.mean(v)), float(np.std(v, ddof=1))]
        json.dump(R, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
