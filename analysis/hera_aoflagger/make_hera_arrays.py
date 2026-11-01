"""Build the leak-free HERA arrays with AOFlagger training labels, once. No GPU, about a minute.

Mesarcik et al. (2022), Table 1: trained on AOFlagger masks at threshold 10, tested on the
simulator's masks. Their code flags the raw values (data.py load_hera -> ao_flagger.py flag_data,
strategy hera_10.lua), keeps the pickle's masks as test truth, then clips and logs.

Leakage: 31 of the pickle's 140 test images are copies of training images, and each split has
internal duplicates (PART 2, 3). Same rule as analysis/hera_transfer_test/dedupe_hera.py: first
copy kept in each split, images also in test dropped from train -> 351 train, 135 test.

Output in data/hera/HERA_arrays/, the same layout lofar_data.py reads (N, 512, 512, 1):
    train_images     float32  raw values
    train_masks      bool     AOFlagger, threshold 10   <- what models train on
    train_masks_sim  bool     simulator, never trained on; kept for checking only
    test_images      float32  raw values
    test_masks       bool     simulator                 <- what models are scored on
    test_masks_aof   bool     AOFlagger, for the AOFlagger row

Run with ~/aof-env/bin/python. Re-running overwrites the arrays with identical content.
"""
import hashlib, json, os, pickle, time
import numpy as np
import aoflagger as aof

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKL = os.path.join(ROOT, "data", "hera", "HERA_04-03-2022_all.pkl")
OUT = os.path.join(ROOT, "data", "hera", "HERA_arrays")
STRATEGY = os.path.join(HERE, "hera_10.lua")


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def flag(data, strategy, ao):
    """ao_flagger.py flag_data, unchanged in substance."""
    mask = np.empty(data[..., 0].shape, dtype=bool)
    for i in range(len(data)):
        s = ao.make_image_set(data.shape[1], data.shape[2], 1)
        s.set_image_buffer(0, data[i, ..., 0])
        mask[i] = strategy.run(s).get_buffer()
    return mask


def f1(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-12), precision=p, recall=r,
                flagged_pct=float(pred.mean() * 100), truth_pct=float(truth.mean() * 100))


def main():
    with open(PKL, "rb") as fh:
        tr_x, tr_y, te_x, te_y = pickle.load(fh)
    # hera.py get_hera_data: inf -> dtype max, then float32
    tr_x, te_x = tr_x.copy(), te_x.copy()
    tr_x[tr_x == np.inf] = np.finfo(tr_x.dtype).max
    te_x[te_x == np.inf] = np.finfo(te_x.dtype).max
    tr_x, te_x = tr_x.astype("float32"), te_x.astype("float32")
    tr_y, te_y = np.asarray(tr_y, dtype=bool), np.asarray(te_y, dtype=bool)

    # duplicates, as dedupe_hera.py: first copy in each split, then train minus test
    seen, te_keep = set(), []
    for i in range(len(te_x)):
        h = digest(te_x[i])
        if h not in seen:
            te_keep.append(i)
        seen.add(h)
    te_hashes, seen, tr_keep = seen, set(), []
    for i in range(len(tr_x)):
        h = digest(tr_x[i])
        if h not in seen and h not in te_hashes:
            tr_keep.append(i)
        seen.add(h)
    tr_keep, te_keep = np.array(tr_keep), np.array(te_keep)
    print(f"pickle: train {len(tr_x)}, test {len(te_x)}  ->  clean: train {len(tr_keep)}, test {len(te_keep)}")
    if (len(tr_keep), len(te_keep)) != (351, 135):
        raise SystemExit("expected 351 train and 135 test images (PART 3)")
    tr_x, tr_y, te_x, te_y = tr_x[tr_keep], tr_y[tr_keep], te_x[te_keep], te_y[te_keep]
    if {digest(x) for x in tr_x} & {digest(x) for x in te_x}:
        raise SystemExit("a test image is still in the training set")

    ao = aof.AOFlagger()
    strategy = ao.load_strategy_file(STRATEGY)
    t0 = time.time()
    tr_aof = flag(tr_x, strategy, ao)
    te_aof = flag(te_x, strategy, ao)
    print(f"AOFlagger {ao.get_version_string()}: {len(tr_x) + len(te_x)} images in {time.time() - t0:.1f} s")

    os.makedirs(OUT, exist_ok=True)
    for name, arr in (("train_images", tr_x), ("train_masks", tr_aof[..., None]),
                      ("train_masks_sim", tr_y), ("test_images", te_x), ("test_masks", te_y),
                      ("test_masks_aof", te_aof[..., None])):
        np.save(os.path.join(OUT, name + ".npy"), arr)
        print(f"  {name:16s} {arr.shape} {arr.dtype}")

    report = dict(n_train=len(tr_x), n_test=len(te_x), strategy="hera_10.lua",
                  aoflagger_version=ao.get_version_string(),
                  raw_range_train=[float(tr_x.min()), float(tr_x.max())],
                  raw_range_test=[float(te_x.min()), float(te_x.max())],
                  train_aoflagger_vs_simulator=f1(tr_aof, tr_y[..., 0]),
                  test_aoflagger_vs_simulator=f1(te_aof, te_y[..., 0]),
                  kept_train_indices=tr_keep.tolist(), kept_test_indices=te_keep.tolist())
    json.dump(report, open(os.path.join(HERE, "hera_arrays_report.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if "indices" not in k}, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
