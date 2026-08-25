# Why the tf_unet baseline failed — measured, not guessed

Written 2026-08-24. Four controlled runs on the **same** dataset
(`Synthetic Dataset 276x600`, seed 42), the **same** unmodified `tf_unet`
(layers=3, features_root=32), the **same** optimiser (Adam @ 1e-3), the **same**
batch size (4). Only the listed variables changed.

## The measurements

| # | Normalisation | Class weights | Epochs | ROC AUC | max F1 | What the model did |
|---|---|---|---:|---:|---:|---|
| 1 | per-image | **ON** | 22 | 0.6681 | **0.3879** | froze at 81.1% error for 15 epochs |
| 2 | per-image | **ON** | 60 (died at 5) | 0.5677 | **0.3064** | **collapsed** — constant 0.537 everywhere |
| 3 | per-image | OFF | 60 | 0.8747 | **0.7191** | trained normally |
| 4 | **fixed range** | OFF | 60 | 0.9908 | **0.9317** | trained well |

## The decomposition

| Step | Change | Gain in F1 | Share |
|---|---|---:|---:|
| 2 → 3 | **Turn class weights OFF** | **+0.413** | **66%** |
| 3 → 4 | **Fixed-range normalisation** | **+0.213** | **34%** |
| — | More epochs (22 → 60) | ≈0 | ≈0% |
| | **Total recovery** | **+0.625** | |

**Class weighting was the primary cause. Normalisation was secondary but real.
Epoch count was irrelevant.**

Run 2 settles the epoch question on its own: with class weights ON the network
collapsed at **epoch 5**, so the extra epochs never had a chance to help. Runs 1
and 2 both land near F1 0.31–0.39 regardless of whether they were allowed 22
epochs or 60.

## Why class weighting destroys this particular network

Class weighting is normally the *correct* remedy for an 85/15 imbalance. It is
harmful here because of a specific quirk of `tf_unet`'s architecture.

`tf_unet` applies a **ReLU to its output logits** before the softmax
(`tf_unet/unet.py`, line 145: `output_map = tf.nn.relu(conv)`). A weight of 3.45
on the RFI class pushes hard enough that the "clean" logit is driven negative;
ReLU clamps it to exactly zero; and **the gradient of ReLU at a negative input is
exactly zero**. The network can never recover from that state, no matter how long
it trains.

The signature is unmistakable in run 2: `pred range [0.537, 0.537]`,
`std 0.0000` — one identical number for all 24,840,000 test pixels.

So the finding is sharper than "the baseline was undertrained". It is:

> **The plain U-Net is fragile enough that the standard textbook remedy for class
> imbalance — class weighting — drives it into an unrecoverable dead state.**

## Why normalisation still matters (+0.213)

`RFIPatchDataset` and `RFINpyDataProvider` scale each image by its own minimum and
maximum. Measured across this dataset, the per-image maximum ranges from **27 to
222 — an 8.2× swing** — so identical physical noise lands at different values in
different images and no single decision threshold generalises.

The authors avoided this: their own `scripts/rfi_launcher.py` passes
`a_min=30, a_max=210`, clipping every image into one fixed physical range. That
parameter pair was not carried over into this project's provider.

`tf_unet` has **no normalisation layers** (no BatchNorm, no GroupNorm), so it
cannot compensate. The hybrid's **GroupNorm** re-standardises activations at every
layer, which is why the identical loader never hurt it.

## Corrected claim for the write-up

> A faithful reproduction of Akeret et al.'s U-Net reaches only F1 0.39 on our
> benchmark. Controlled ablation shows this is not an architectural limit: class
> weighting drives the network's ReLU-gated output layer into a constant-output
> dead state (+0.413 F1 on removal), and per-image intensity normalisation costs a
> further 0.213 F1 because the architecture has no internal normalisation to
> absorb an 8.2× per-image scale variation. Correcting both recovers the baseline
> to **F1 0.93** with no change to the published architecture. Our hybrid, whose
> GroupNorm confers robustness to both failure modes, reaches F1 0.98.

## An earlier hypothesis that was wrong

An initial diagnosis attributed the failure primarily to normalisation, at
"~85% confidence". Runs 2 and 3 refuted it: normalisation accounts for about a
third of the recovery, not the bulk. Two pieces of evidence had been
misinterpreted:

- *"It froze at 81% for 15 epochs"* — that run also had class weights ON. Run 3
  (per-image normalisation, weights OFF) did not freeze. The freezing was the
  weights.
- *"The authors' own recipe collapsed to a constant"* — that run used momentum at
  lr=0.2. Run 3 used Adam at 1e-3 with the same per-image normalisation and
  trained fine. That collapse was the learning rate.

Both were confounded. The controlled runs are the first clean measurement.

## Reproducing this table

```bash
source ~/tf-env/bin/activate
cd "/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project"

# run 2
python3 experiments/normalisation_control/run_control.py --norm per_image --class_weights on
# run 3
python3 experiments/normalisation_control/run_control.py --norm per_image --class_weights off
# run 4
python3 experiments/normalisation_control/run_control.py --norm fixed      --class_weights off
```

Results in `unet_run_control_*/eval_test/metrics.json`.

**Caveat:** one seed per cell. Repeat with ≥3 seeds before publishing the exact
numbers. The ordering and the order-of-magnitude are unlikely to change; the
third decimal place will.
