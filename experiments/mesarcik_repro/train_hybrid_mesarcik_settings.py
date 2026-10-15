"""HybridRFINet trained with Mesarcik et al.'s (2022) LOFAR training settings.

Fair-comparison partner of train_mesarcik_unet.py, which ran THEIR U-Net through
THEIR pipeline (PART 21, 30): released code -> max F1 0.6207 (`paper_code`),
leak-free -> 0.6166 (`clean`). This script puts HybridRFINet through the SAME
pipeline. Everything except the network is shared, and is not re-implemented:
the data preparation, clip bounds, 32x32 tiling, tile order, batch order, test
reassembly and the metric are imported from train_mesarcik_unet.py, and the
same seed gives the same split, the same normalisation cache and the same
sequence of training batches as the U-Net run.

Their settings, as reproduced (see train_mesarcik_unet.py for the line refs):
  * data     : `paper_code` = all 7500 training images, which include the 109
               test images, clip bounds from the TEST set's clean pixels
               (exactly their released code); `clean` = our 6621 training
               images, clip bounds from those images, 735 validation images
               only used to pick the pooled-F1 threshold
  * preproc  : clip [|mu - 3 sd|, mu + 95 sd] of clean pixels -> ln -> min-max
               (released "july" version; --clip june for the paper's text)
  * input    : non-overlapping 32x32 tiles (256 per image), test tiled the same
               way and reassembled to 512x512 before scoring
  * batch    : 1024 tiles
  * optimiser: Adam, lr 1e-3 (Keras default -- their code calls Adam()),
               betas (0.9, 0.999), eps 1e-7 (Keras default), constant lr,
               no weight decay, no lr schedule
  * loss     : binary cross-entropy, as their paper states (default, --loss bce).
               Their CODE has the arguments swapped, `bce(x_hat, y)` (~16 x L1,
               PART 21.4); --loss swapped reproduces it exactly (Keras 3
               formula, eps 1e-7). MEASURED 2026-10-15 (PART 31): with the
               swapped loss HybridRFINet dies inside the first epoch at lr 1e-3
               AND at lr 1e-4 -- every pixel's p_RFI is driven to ~1e-17, the
               gradient factor p(1-p) to ~1e-17, loss frozen at 0.1277, test
               max F1 0.015. With the correct BCE (lr 1e-3) it reaches max F1
               0.6537 after 2 epochs. So a fair comparison needs BOTH networks
               on the correct BCE: run their U-Net with
               train_mesarcik_unet.py --correct_loss (scripts/run_fair_mesarcik.sh).
  * no class weights, no Dice, no augmentation
  * epochs   : 100 full passes over all tiles, shuffled every epoch
  * model    : the model after the LAST epoch (no checkpoint selection)
  * metric   : max F1 over all thresholds on the 109 expert-labelled images,
               pooled over all pixels (their protocol); float16 scores too,
               as their infer() stores them

What stays HybridRFINet's own (the network being compared, exactly as their
paper does for every model: "we use all parameters specified by the authors
other than patch size"): architecture, GroupNorm, dropout 0.2, 2-class softmax
output (p_RFI = softmax[:, 1] = sigmoid(z1 - z0), so the loss sees the same
kind of probability as their sigmoid output). base 8 / depth 4 as everywhere
else in the report; it builds and trains on 32x32 (measured 2026-10-15:
2.87 GB, 131 ms/step at batch 1024 on the RTX 3060).

Only intended difference from the U-Net reference runs, unavoidable: PyTorch
instead of TensorFlow (Adam's eps sits inside vs outside the bias correction;
negligible). The `paper_code` setup trains on the test images, as their code
does -- use it ONLY as the like-for-like partner of 0.6207, never as a result
on its own. `clean` is the one to report as the model's score.

Resumable: rerun the identical command; model, optimiser AND the shuffle RNG
are restored, so a resumed run is the same run.

    ~/torch-env/bin/python experiments/mesarcik_repro/train_hybrid_mesarcik_settings.py \
        --setup clean --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "hybrid_rfi_package"))

import train_mesarcik_unet as R          # noqa: E402  shared pipeline (no TensorFlow at import)
from train_mesarcik_unet import load_lofar, P, TILES, CLIPS  # noqa: E402

# their U-Net through the same pipeline, seed 0 -- only valid for the SAME loss
REFERENCE = {("paper_code", "swapped"): 0.6207, ("clean", "swapped"): 0.6166}
PARTNER = "runs/lofar/mesarcik_unet_{setup}{bce}_seed{seed}  (train_mesarcik_unet.py{flag})"


def predict_images(model, X, device):
    """Torch twin of R.predict_images: tile, predict, reassemble (same transposes)."""
    import torch
    out = np.empty(X.shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            t = R.tiles_of(np.asarray(X[i:i + 1], dtype=np.float32))[0].reshape(-1, 1, P, P)
            logits = model(torch.from_numpy(np.ascontiguousarray(t)).to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            out[i] = p.reshape(TILES, TILES, P, P).transpose(0, 2, 1, 3).reshape(512, 512)
    model.train()
    return out


def mesarcik_loss(logits, y, kind, eps=1e-7):
    """kind='swapped': Keras 3 binary_crossentropy(target=p, output=mask), i.e.
    their bce(x_hat, y). kind='bce': binary_crossentropy(target=mask, output=p)."""
    import torch
    p = torch.softmax(logits, dim=1)[:, 1]
    if kind == "swapped":
        m = y.clamp(eps, 1.0 - eps)
        return -(p * torch.log(m) + (1.0 - p) * torch.log(1.0 - m)).mean()
    pc = p.clamp(eps, 1.0 - eps)
    return -(y * torch.log(pc) + (1.0 - y) * torch.log(1.0 - pc)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=["paper_code", "clean"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3, help="Keras Adam() default")
    ap.add_argument("--loss", choices=["bce", "swapped"], default="bce",
                    help="bce = their paper (default); swapped = their code's bug, kills this net")
    ap.add_argument("--clip", choices=["july", "june"], default="july")
    ap.add_argument("--shuffle_buffer", type=int, default=0,
                    help="0 = full shuffle (as the reference runs); 25000 = their tf.data buffer")
    ap.add_argument("--input_dtype", choices=["float16", "float32"], default="float16",
                    help="cache dtype; float16 = as the reference runs (shared cache files)")
    ap.add_argument("--base", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=10,
                    help="score the test set every K epochs into eval_test/trajectory.json")
    ap.add_argument("--limit_train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    out = a.output_dir or os.path.join(
        _ROOT, "runs", "lofar",
        f"hybrid_mesarcik_{a.setup}"
        f"{'_bce' if a.loss == 'bce' else ''}"
        f"{'_june' if a.clip == 'june' else ''}"
        f"{f'_buf{a.shuffle_buffer}' if a.shuffle_buffer else ''}"
        f"{'_f32' if a.input_dtype == 'float32' else ''}_seed{a.seed}")
    os.makedirs(os.path.join(out, "eval_test"), exist_ok=True)
    cache_dir = os.path.join(_ROOT, "data", "lofar", "mesarcik_cache")

    import torch
    from hybrid_model import HybridRFINet
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data: identical to train_mesarcik_unet.main() --------------------------
    d = load_lofar()
    if a.setup == "paper_code":
        tr_idx = np.arange(d.train_images.shape[0])                 # all 7500
        val_idx = None
        k = CLIPS[a.clip]
        te_lo, te_hi, _, _ = R.clip_bounds(d.test_images, d.test_masks, np.arange(109), k)
        if a.clip == "june":
            lo, hi, mu, sd = R.clip_bounds(d.train_images, d.train_masks, tr_idx, k)
        else:
            lo, hi, mu, sd = R.clip_bounds(d.test_images, d.test_masks, np.arange(109), k)
    else:
        rng_split = np.random.default_rng(a.seed)                    # = lofar_hybrid.py split
        idx = d.clean_train_idx.copy(); rng_split.shuffle(idx)
        n_val = max(1, int(len(idx) * 0.1))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        assert not set(tr_idx.tolist()) & set(d.leak_train_idx.tolist())
        assert not set(val_idx.tolist()) & set(d.leak_train_idx.tolist())
        lo, hi, mu, sd = R.clip_bounds(d.train_images, d.train_masks, tr_idx, CLIPS[a.clip])
        te_lo, te_hi = lo, hi                                        # no test statistics
    if a.limit_train:
        tr_idx = tr_idx[:a.limit_train]

    tag = (f"{a.setup}_seed{a.seed}" if a.setup == "clean" else a.setup) + \
          ("_june" if a.clip == "june" else "")
    cache = os.path.join(cache_dir, f"{tag}_train_{len(tr_idx)}"
                                    f"{'_f32' if a.input_dtype == 'float32' else ''}.npy")
    Xtr = R.prepare(d.train_images, tr_idx, lo, hi, cache,
                    dtype=np.float32 if a.input_dtype == "float32" else np.float16)
    if a.input_dtype == "float16":
        Xtr = np.asarray(Xtr)            # 3.9 GB: fits in RAM; float32 stays memory-mapped
    Ytr = np.asarray(d.train_masks[np.sort(tr_idx), ..., 0])[np.argsort(np.argsort(tr_idx))]
    Xte = R.transform(d.test_images[..., 0], te_lo, te_hi)
    Yte = np.asarray(d.test_masks[..., 0], dtype=bool)
    Ttr, Mtr = R.tiles_of(Xtr), R.tiles_of(Ytr)
    n_tiles = len(tr_idx) * TILES * TILES
    steps = int(np.ceil(n_tiles / a.batch))

    model = HybridRFINet(1, 2, base=a.base, depth=a.depth, dropout=a.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-7,
                           weight_decay=0.0)

    print("=" * 74)
    print(f"  HybridRFINet with Mesarcik et al.'s settings  |  setup = {a.setup}  |  seed {a.seed}")
    print("=" * 74)
    print(f"  device            : {device}")
    print(f"  training images   : {len(tr_idx)}"
          + ("  (includes the 109 test images -- as in their code)" if a.setup == "paper_code" else ""))
    print(f"  clip ({a.clip:<4})       : train [{lo:.6g}, {hi:.6g}] (mu {mu:.6g}, sd {sd:.6g})"
          f"   test [{te_lo:.6g}, {te_hi:.6g}]")
    print(f"  cache             : {os.path.basename(cache)}")
    print(f"  patches per epoch : {n_tiles:,}  ->  {steps:,} steps x {a.epochs} epochs, batch {a.batch}")
    print(f"  model             : HybridRFINet base {a.base} depth {a.depth} dropout {a.dropout}"
          f"  ({n_par:,} parameters)")
    print(f"  optimiser         : Adam lr {a.lr:g} constant, betas (0.9, 0.999), eps 1e-7")
    print(f"  loss              : {'bce(x_hat, y) [their code; effectively L1]' if a.loss == 'swapped' else 'bce(y, x_hat) [correct BCE]'}")
    print(f"  shuffle           : {f'tf.data buffer {a.shuffle_buffer:,}' if a.shuffle_buffer else 'full shuffle every epoch'}"
          f"  |  inputs {a.input_dtype}")
    ref = REFERENCE.get((a.setup, a.loss))
    partner = PARTNER.format(setup=a.setup, bce="_bce" if a.loss == "bce" else "", seed=a.seed,
                             flag=" --correct_loss" if a.loss == "bce" else "")
    print(f"  U-Net partner     : {partner}"
          + (f"  -> max F1 {ref}" if ref is not None else ""))
    print("=" * 74, flush=True)

    # ---- resumable training --------------------------------------------------
    rng = np.random.default_rng(a.seed)            # same stream as the U-Net run
    flat = np.arange(n_tiles)
    ckpt = os.path.join(out, "last.pt")
    prog_path = os.path.join(out, "progress.json")
    traj_path = os.path.join(out, "eval_test", "trajectory.json")
    start_ep, traj = 0, []
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)   # rng states must stay on CPU
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]; flat = st["flat"]
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        start_ep = st["epoch"]
        if os.path.exists(traj_path):
            traj = json.load(open(traj_path))
        print(f"  resumed after epoch {start_ep} (model, optimiser and shuffle state restored)",
              flush=True)

    model.train()
    t0 = time.time()
    for ep in range(start_ep, a.epochs):
        if a.shuffle_buffer:
            flat = R.buffer_order(n_tiles, a.shuffle_buffer, rng)
        else:
            rng.shuffle(flat)
        tot = 0.0
        for s in range(steps):
            b = flat[s * a.batch:(s + 1) * a.batch]
            im, rem = np.divmod(b, TILES * TILES); ty, tx = np.divmod(rem, TILES)
            x = torch.from_numpy(Ttr[im, ty, tx].astype(np.float32)[:, None]).to(device)
            y = torch.from_numpy(Mtr[im, ty, tx].astype(np.float32)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = mesarcik_loss(model(x), y, a.loss)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        ep_loss = tot / steps

        if a.eval_every and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs):
            p32 = predict_images(model, Xte, device)
            m32 = R.metrics(Yte, p32)
            m16 = R.metrics(Yte, p32.astype(np.float16).astype(np.float32))
            traj.append(dict(epoch=ep + 1, loss=ep_loss, max_f1=m32["max_f1"],
                             roc_auc=m32["roc_auc"], pr_auc=m32["pr_auc"],
                             max_f1_f16=m16["max_f1"], roc_auc_f16=m16["roc_auc"],
                             pr_auc_f16=m16["pr_auc"]))
            json.dump(traj, open(traj_path, "w"), indent=1)
            print(f"     test @ {ep + 1}: maxF1 {m32['max_f1']:.4f}  float16 scores -> "
                  f"AUROC {m16['roc_auc']:.4f} AUPRC {m16['pr_auc']:.4f}", flush=True)

        tmp = ckpt + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
                    "rng": rng.bit_generator.state, "flat": flat,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None},
                   tmp)
        os.replace(tmp, ckpt)
        json.dump({"epochs_completed": ep + 1, "loss": ep_loss}, open(prog_path, "w"))
        el = time.time() - t0
        print(f"  epoch {ep + 1:>3}/{a.epochs}  loss {ep_loss:.5f}  {el / 60:.1f} min  "
              f"ETA {el / (ep + 1 - start_ep) * (a.epochs - ep - 1) / 60:.0f} min", flush=True)

    # ---- evaluation: final model, as in their code ---------------------------
    pred = predict_images(model, Xte, device)
    th = None
    if val_idx is not None:
        Xv = R.transform(d.train_images[np.sort(val_idx), ..., 0], lo, hi)
        Yv = np.asarray(d.train_masks[np.sort(val_idx), ..., 0], dtype=bool)
        th = R.metrics(Yv, predict_images(model, Xv, device))["max_f1_threshold"]
    m = R.metrics(Yte, pred, threshold=th)
    m16 = R.metrics(Yte, pred.astype(np.float16).astype(np.float32))
    m.update(max_f1_f16=m16["max_f1"], roc_auc_f16=m16["roc_auc"], pr_auc_f16=m16["pr_auc"],
             model="HybridRFINet", base=a.base, depth=a.depth, dropout=a.dropout,
             parameters=int(n_par), setup=a.setup, seed=a.seed, epochs=a.epochs,
             batch=a.batch, patch=P, lr=a.lr, optimizer="Adam(0.9, 0.999, eps 1e-7), constant lr",
             loss="bce(x_hat, y) -- as in their code (effectively L1)" if a.loss == "swapped"
                  else "bce(y, x_hat) -- correct BCE",
             n_train=int(len(tr_idx)), clip_version=a.clip, clip_bounds_train=[lo, hi],
             clip_bounds_test=[te_lo, te_hi], shuffle_buffer=a.shuffle_buffer,
             input_dtype=a.input_dtype, model_selection="last epoch (their protocol)",
             unet_partner_run=partner, unet_partner_max_f1=ref,
             test_set="109 expert-labelled LOFAR baselines")
    json.dump(m, open(os.path.join(out, "eval_test", "metrics.json"), "w"), indent=2)
    np.save(os.path.join(out, "eval_test", "test_pred.npy"), pred.astype(np.float16))
    print("=" * 74)
    print(f"  HybridRFINet, Mesarcik settings ({a.setup}) : max F1 {m['max_f1']:.4f}")
    print(f"  their U-Net, same pipeline and loss      : "
          + (f"max F1 {ref}" if ref is not None else f"see {partner}"))
    if th is not None:
        print(f"  pooled F1 (threshold from validation)    : {m['pooled_f1']:.4f}")
    print(f"  ROC AUC / PR AUC                         : {m['roc_auc']:.4f} / {m['pr_auc']:.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
