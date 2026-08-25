"""
Fine-tune the hybrid on HERA -- WITH the control that makes it meaningful.

THE EXPERIMENT
--------------
Run this script TWICE:

    --init pretrained   start from hybrid_run_paperdim/best.pt (our synthetic weights)
    --init scratch      start from random weights            <-- THE CONTROL

Everything else is identical: same data, same split, same seed, same epochs,
same optimiser, same evaluation. Only the starting point differs.

WHY THE CONTROL IS THE WHOLE POINT
----------------------------------
"We fine-tuned on HERA and got F1 0.7" means nothing on its own -- maybe HERA is
just easy and any model gets 0.7. The claim only becomes real if:

    pretrained  >  scratch

That difference is the transferable value of the synthetic dataset. It is the
single most publishable result available from this project, and it is the row a
reviewer will look for first. Without the scratch row, the pretrained row is
unfalsifiable.

WHAT WE ALREADY KNOW (zero-shot, no training)
    ROC AUC 0.8965 | F1 at our threshold 0.1415 | F1 recalibrated 0.4481
    no-model constant threshold on HERA: oracle F1 0.5665
So fine-tuning must clear 0.5665 to be worth anything at all.

PROTOCOL
--------
HERA's 420 training images are split 340 train / 80 val (fixed, deterministic).
The threshold is chosen on val and applied ONCE to the 140 test images. The test
split is never used for any decision. Same discipline as the rest of the project.

USAGE
    source ~/torch-env/bin/activate
    cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"

    python3 hera_transfer_test/finetune_on_hera.py --init pretrained
    python3 hera_transfer_test/finetune_on_hera.py --init scratch

Resumable: rerun the same command to continue. If VRAM is tight use --batch 1.
"""
import os, sys, glob, json, time, shutil, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "hybrid_rfi_package"))
sys.path.insert(0, _HERE)
from hybrid_model import HybridRFINet, count_parameters      # noqa: E402
from test_hybrid_on_hera import NORMS                        # same normalisations


class HeraSet(Dataset):
    def __init__(self, img_dir, msk_dir, norm, lo=None, hi=None):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.msks = sorted(glob.glob(os.path.join(msk_dir, "*.npy")))
        assert self.imgs and len(self.imgs) == len(self.msks), "no pairs in " + img_dir
        if lo is not None:
            self.imgs, self.msks = self.imgs[lo:hi], self.msks[lo:hi]
        self.fn = NORMS[norm]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        a = np.load(self.imgs[i]).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        m = np.load(self.msks[i])
        if m.ndim == 3:
            m = m[..., 0]
        return (torch.from_numpy(self.fn(a).astype(np.float32)).unsqueeze(0),
                torch.from_numpy(m.astype(np.int64)))


def dice_loss(logits, target, eps=1.0):
    probs = F.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (probs * t).sum(dim=(1, 2))
    den = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        ys.append((y.numpy() == 1).ravel())
        ps.append(p.ravel().astype(np.float32))       # float32: softmax saturates near 1
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser(description="Fine-tune the hybrid on HERA, with a from-scratch control")
    p.add_argument("--init", choices=["pretrained", "scratch"], required=True,
                   help="THE VARIABLE UNDER TEST. Run both.")
    p.add_argument("--data_dir", default=os.path.join(_HERE, "HERA_npy"))
    p.add_argument("--checkpoint", default=os.path.join(_ROOT, "hybrid_run_paperdim", "best.pt"))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--norm", default="log_per_image", choices=list(NORMS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2, help="drop to 1 if VRAM is tight")
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-4 for pretrained (gentle), 1e-3 for scratch")
    p.add_argument("--n_val", type=int, default=80, help="val images carved from the 420 train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    if a.out_dir is None:
        a.out_dir = os.path.join(_HERE, "runs", "hera_" + a.init)
    if a.lr is None:
        a.lr = 1e-4 if a.init == "pretrained" else 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.fresh:
        shutil.rmtree(a.out_dir, ignore_errors=True)
    os.makedirs(a.out_dir, exist_ok=True)
    pfile = os.path.join(a.out_dir, "progress.json")
    prog = json.load(open(pfile)) if os.path.exists(pfile) else \
        {"epochs_completed": 0, "best_f1": -1.0, "best_epoch": None, "best_threshold": 0.5}

    tr_i = os.path.join(a.data_dir, "train", "images")
    tr_m = os.path.join(a.data_dir, "train", "masks")
    n_all = len(glob.glob(os.path.join(tr_i, "*.npy")))
    n_tr = n_all - a.n_val
    train_ds = HeraSet(tr_i, tr_m, a.norm, 0, n_tr)
    val_ds = HeraSet(tr_i, tr_m, a.norm, n_tr, n_all)
    test_ds = HeraSet(os.path.join(a.data_dir, "test", "images"),
                      os.path.join(a.data_dir, "test", "masks"), a.norm)
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=1, shuffle=False)
    sl = DataLoader(test_ds, batch_size=1, shuffle=False)

    # class weights measured from HERA's OWN training split (it is 2.75% RFI,
    # not 15% like our synthetic data)
    pos = tot = 0
    for f in sorted(glob.glob(os.path.join(tr_m, "*.npy")))[:n_tr]:
        m = np.load(f); pos += int(m.sum()); tot += m.size
    frac = pos / tot
    cw = [0.5 / (1 - frac), 0.5 / frac]

    model = HybridRFINet(1, 2, base=32, depth=4, dropout=0.2).to(device)
    src = "random init"
    if a.init == "pretrained":
        ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        src = "{} (epoch {})".format(os.path.basename(a.checkpoint), ck.get("epoch", "?"))
    last = os.path.join(a.out_dir, "last.pt")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    if prog["epochs_completed"] > 0 and os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        src = "resumed from epoch {}".format(prog["epochs_completed"])

    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    print("=" * 76)
    print(" FINE-TUNE ON HERA   --   init = {}".format(a.init.upper()))
    print("=" * 76)
    print("  device        : {}".format(device))
    print("  starting from : {}".format(src))
    print("  data          : {} train / {} val / {} test  (512x512)".format(
        len(train_ds), len(val_ds), len(test_ds)))
    print("  HERA RFI frac : {:.4f}%   class weights [{:.3f}, {:.3f}]".format(100 * frac, cw[0], cw[1]))
    print("  normalisation : {}".format(a.norm))
    print("  epochs / lr   : {} / {}".format(a.epochs, a.lr))
    print("  parameters    : {:,}".format(count_parameters(model)))
    print("  output        : {}".format(a.out_dir))
    print("=" * 76, flush=True)

    csv = os.path.join(a.out_dir, "training_log.csv")
    if not os.path.exists(csv):
        open(csv, "w").write("epoch,train_loss,val_roc_auc,val_max_f1,is_best,minutes\n")

    t_start = time.time()
    while prog["epochs_completed"] < a.epochs:
        model.train()
        losses = []
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = ce(out, y) + dice_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach().item())
        sched.step()
        prog["epochs_completed"] += 1
        mins = (time.time() - t0) / 60.0

        vy, vp = collect(model, vl, device)
        fpr, tpr, _ = roc_curve(vy, vp)
        vroc = float(auc(fpr, tpr))
        pr, rc, th = precision_recall_curve(vy, vp)
        f1 = 2 * pr * rc / (pr + rc + 1e-10)
        bi = int(np.argmax(f1))
        vthr, vf1 = float(th[min(bi, len(th) - 1)]), float(f1[bi])
        is_best = vf1 > prog["best_f1"]
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": prog["epochs_completed"]}, last)
        if is_best:
            prog.update(best_f1=vf1, best_epoch=prog["epochs_completed"], best_threshold=vthr)
            shutil.copyfile(last, os.path.join(a.out_dir, "best.pt"))
        json.dump(prog, open(pfile, "w"), indent=2)
        open(csv, "a").write("{},{:.4f},{:.4f},{:.4f},{},{:.1f}\n".format(
            prog["epochs_completed"], np.mean(losses), vroc, vf1, int(is_best), mins))
        el = (time.time() - t_start) / 60.0
        done_now = prog["epochs_completed"]
        b = int(24 * done_now / a.epochs)
        print("  [{}{}] Epoch {}/{}  loss {:.4f}  |  val ROC {:.4f}  val F1 {:.4f}{}  "
              "|  {:.1f}m elapsed".format("#" * b, "." * (24 - b), done_now, a.epochs,
                                          np.mean(losses), vroc, vf1,
                                          "  <-- BEST" if is_best else "", el), flush=True)

    print("\nBest val F1 {:.4f} @ epoch {} (threshold {:.4f})".format(
        prog["best_f1"], prog["best_epoch"], prog["best_threshold"]), flush=True)
    print("Evaluating BEST checkpoint on HERA TEST (once) ...", flush=True)
    model.load_state_dict(torch.load(os.path.join(a.out_dir, "best.pt"),
                                     map_location=device, weights_only=False)["model"])
    ty, tp_ = collect(model, sl, device)
    fpr, tpr, _ = roc_curve(ty, tp_); roc = float(auc(fpr, tpr))
    pr2, rc2, _ = precision_recall_curve(ty, tp_); prauc = float(auc(rc2, pr2))
    f1c = 2 * pr2 * rc2 / (pr2 + rc2 + 1e-10)
    oracle = float(np.max(f1c))
    thr = prog["best_threshold"]
    pred = tp_ >= thr
    tn, fp, fn, tp = confusion_matrix(ty, pred, labels=[0, 1]).ravel()
    P = tp / (tp + fp + 1e-10); R = tp / (tp + fn + 1e-10)
    m = dict(init=a.init, roc_auc=roc, pr_auc=prauc, threshold=float(thr),
             f1=float(2 * P * R / (P + R + 1e-10)), precision=float(P), recall=float(R),
             iou=float(tp / (tp + fp + fn + 1e-10)), mcc=float(matthews_corrcoef(ty, pred)),
             oracle_f1=oracle, epochs=prog["epochs_completed"], lr=a.lr, norm=a.norm,
             best_val_f1=prog["best_f1"], best_epoch=prog["best_epoch"],
             n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
             note="threshold chosen on HERA val, applied once to HERA test")

    print("\n" + "=" * 76)
    print("  init = {}".format(a.init.upper()))
    print("  ROC AUC {:.4f}   PR AUC {:.4f}".format(roc, prauc))
    print("  F1 {:.4f}  (P {:.4f} / R {:.4f})  IoU {:.4f}  MCC {:.4f}  @ thr {:.4f}".format(
        m["f1"], P, R, m["iou"], m["mcc"], thr))
    print("  oracle F1 {:.4f}  (upper bound, not reportable)".format(oracle))
    print("-" * 76)
    print("  reference points on HERA:")
    print("    zero-shot, our threshold      F1 0.1415")
    print("    zero-shot, recalibrated       F1 0.4481")
    print("    constant threshold (no model) oracle F1 0.5665")
    print("  >> fine-tuning is only worth reporting if it clears 0.5665")
    print("=" * 76)
    os.makedirs(os.path.join(a.out_dir, "eval_test"), exist_ok=True)
    o = os.path.join(a.out_dir, "eval_test", "metrics.json")
    json.dump(m, open(o, "w"), indent=2)
    print("\nSaved: {}".format(o))
    other = os.path.join(_HERE, "runs", "hera_" + ("scratch" if a.init == "pretrained" else "pretrained"),
                         "eval_test", "metrics.json")
    if os.path.exists(other):
        om = json.load(open(other))
        d = m["f1"] - om["f1"]
        print("\n" + "=" * 76)
        print("  BOTH RUNS COMPLETE -- the comparison that matters")
        print("    pretrained F1 : {:.4f}".format(m["f1"] if a.init == "pretrained" else om["f1"]))
        print("    scratch    F1 : {:.4f}".format(om["f1"] if a.init == "pretrained" else m["f1"]))
        print("    difference    : {:+.4f}  <- the transferable value of the synthetic data".format(
            d if a.init == "pretrained" else -d))
        print("=" * 76)
    else:
        print("\nNow run the other one:")
        print("  python3 hera_transfer_test/finetune_on_hera.py --init {}".format(
            "scratch" if a.init == "pretrained" else "pretrained"))


if __name__ == "__main__":
    main()
