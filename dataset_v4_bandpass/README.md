# Dataset v4 — instrument bandpass, 1024 × 265

Generator, verifier and notes for `../Synthetic Dataset 1024x265/`.

v4 is **v3 plus a receiver bandpass**. Everything else — the six RFI
morphologies, the density tiers, the fractional-size scaling, the split sizes,
the file layout — is unchanged, so any difference measured between v3 and v4
results is attributable to the bandpass and the shape change, not to drift in the
tuning.

```bash
# regenerate (42 seconds)
python3 generate_dataset_v4.py --output_dir "../Synthetic Dataset 1024x265" --seed 42

# verify, and prove it regenerates bit-exactly
python3 generate_dataset_v4.py --output_dir /tmp/repro --seed 42
python3 verify_dataset_v4.py --dataset_dir "../Synthetic Dataset 1024x265" \
    --compare_dir /tmp/repro
```

---

## 1. Headline numbers

| | v3 (276×600) | **v4 (1024×265)** |
|---|---:|---:|
| Splits | 700 / 150 / 150 | **700 / 150 / 150** |
| Mean RFI fraction | 14.67% | **14.19%** |
| Images with no RFI | 9 / 150 test | 13 / 150 test |
| Per-image max spread | 13.19× *(measured)* | **14.49×** |
| Fixed-range `LO`, `HI` | −9.69, 47.40 | **−2.89, 44.89** |
| Class weights | — | **[0.5816, 3.5633]** |
| Difficulty floor (global threshold) | 0.7411 *(measured)* | **0.7668** |
| Train/test leakage | 0 | **0** |
| Bit-exact from seed 42 | yes | **yes — 4025/4025 files** |
| Generation time | — | **42 s** |
| Disk | 1.2 GB | **1.9 GB** |

`LO`/`HI` and the class weights are measured from the **train split only**. The
test split is read for structural checks and reported statistics, never used to
choose a constant.

---

## 2. What changed, and why

### 2.1 A real multiplicative bandpass

v3's `generate_bandpass_gain_curve` was not a bandpass despite the name. It set
the *additive* background pedestal (`pure = gain[:,None] + sigma[:,None]*noise`)
and left the RFI untouched. v4 keeps that curve as the sky/system pedestal and
adds a genuine multiplicative gain `B(f)`, drawn fresh per image, applied to sky,
noise **and** RFI alike — one signal chain, nothing exempt.

Three components, each independently flagged:

| Component | Flag | Default | What it models |
|---|---|---|---|
| Passband envelope | `--edge_frac` | 0.08 | Tukey raised-cosine roll-off over 8% of the band at each edge |
| Slow ripple | `--ripple_amp`, `--ripple_modes` | 0.15 p-p, 2–4 modes | imperfect filters |
| Standing wave | `--sw_period_channels`, `--sw_amp` | 128 ch, 0.05 p-p | reflections in the signal path |
| Per-image variation | `--bandpass_jitter` | 0.25 | each amplitude scaled by `U(0.75, 1.25)`; phases always random |

Every image gets its own draw. Verified: **0 identical pairs**, roll-off knee
jitters between channel 60 and 95, per-channel sd across images 0.044 in
mid-band. Saved to `<split>/bandpass/bandpass_NNNN.npy` so any later analysis can
recover exactly what was applied.

### 2.2 The post-gain noise floor — this is not optional

**A purely multiplicative bandpass cannot bury any RFI.** If `B` multiplies the
RFI and its noise equally it cancels out of their ratio:

```
(B · A_rfi) / (B · sigma) = A_rfi / sigma
```

Detectability would be preserved *exactly*, at every channel, for every value of
`B` — including `B = 0.001`. This was verified numerically before the design was
settled. It is also why bandpass calibration works in real astronomy: dividing
out `B` recovers uniform sensitivity.

Real instruments still discard edge channels, because amplifier and digitiser
noise enter **after** the gain stage and do not scale with it. Where `B` is small
the amplified sky sinks toward that fixed floor:

```
SNR(f) = snr_injected · B(f)·sigma_sky / sqrt( B(f)²·sigma_sky² + sigma_rx² )
```

flat in mid-band, falling to zero at the edges. `--rx_noise_frac` (default 0.10,
as a fraction of the median sky noise) sets that floor. **Setting it to 0
recovers the pure-radiometer model in which the bandpass changes brightness but
never detectability, and the labelling rule below becomes a no-op.**

### 2.3 The label is now a threshold, not a union

v3's rule was `mask = hard_geometric_footprint OR (blurred_rfi > 0.5σ)`. Because
it was a union it could only ever *add* pixels — a pixel could be labelled RFI
while carrying almost no RFI power. (Measured on the v3 train split: 0.06% of
labelled pixels sit below 0.5σ; worst image 4.49%.)

v4 evaluates the threshold on the **post-bandpass** amplitude against the
**local total noise** of that channel:

```python
sigma_total = sqrt((B * sky_sigma_eff)**2 + sigma_rx**2)
strength    = (B * rfi_layer) / sigma_total
mask        = strength > 0.5            # --mask_sigma_threshold
```

`sky_sigma_eff` is the *realised* noise, not the nominal one: the correlation
smoothing in `generate_pure_signal` reduces per-channel noise by a known factor,
and the temporal drift modulates it. Both are corrected analytically. Validated
against RFI-free images — **realised / predicted noise = 0.9997**.

`strength/` is therefore also post-bandpass now, so `strength > 0.5` *is* the
mask condition. Verified on disk: **0 disagreeing pixels across all 1000 images.**

### 2.4 v3's edge roll-off is off by default

v3 already darkened the band edges via `generate_edge_rolloff`. Running that
*and* the Tukey taper would attenuate the edges twice and make `--edge_frac` mean
something other than what it says. Available as `--v3_edge_rolloff` if you want
v3's behaviour back.

---

## 3. What the bandpass cost

Of **37,073,066** injected RFI pixels, **54,403 (0.147%)** fell below threshold
and are not labelled.

| Channels losing… | Count | Share of band |
|---|---:|---:|
| >90% of injected RFI | 6 | 0.59% |
| >50% | 9 | 0.88% |
| >10% | 21 | 2.05% |

By band position:

| Channels | Exclusion rate |
|---|---:|
| 0–101 | 2.02% |
| 102–920 (all eight middle deciles) | ≤0.01% |
| 921–1023 | 1.39% |

**Recommendation: leave `--edge_frac` at 0.08 and do not trim any channels.** The
loss is confined to roughly the outermost 10 channels per side — under 1% of the
band. That is a real effect (band-edge RFI genuinely does become undetectable)
without being a large enough fraction to distort a score. There is no case here
for a gentler taper or for discarding edge channels.

The cost is small because the *noise* at the edges is attenuated alongside the
RFI. Only where `B` falls low enough for the fixed receiver floor to dominate —
about the outer 1% of the band — does the ratio actually collapse.

---

## 4. The difficulty floor went UP, not down. Here is why.

A plain global brightness threshold, no model at all, measured identically on
both datasets over 200 train images:

| | Floor F1 |
|---|---:|
| v3, 276×600 | 0.7411 *(project's documented figure: 0.7421 — reproduced)* |
| **v4, 1024×265** | **0.7668** |
| | **+0.0257 — easier** |

The bandpass did not make this benchmark harder by this metric. Two reasons, and
the second is the interesting one.

**a) The roll-off deletes v3's hardest region.** v3's noise is inversely tied to
the sky pedestal (`sigma = sigma_bg / sqrt(pedestal_norm)`), which is the
*opposite* of a radiometer. That makes v3's band edges its **noisiest** channels
— measured at **1.83× mid-band**. Bright noise excursions on a dark background
are exactly what caps a global threshold. The roll-off multiplies those channels
by ≈0, taking them to **0.28× mid-band**. No value of `--edge_frac` fixes this; a
gentler taper simply does less of it.

**b) The mid-band ripple's difficulty is entirely removable.** Raising
`--ripple_amp` does lower the raw floor — 0.15 → 0.844, 1.0 → 0.745, 1.5 → 0.656
(25-image samples). But giving the same naive baseline a per-channel median/MAD
calibration first:

| `--ripple_amp` | raw floor | after per-channel calibration |
|---:|---:|---:|
| 0.15 | 0.8440 | 0.2867 |
| 1.0 | 0.7452 | 0.2868 |
| 1.5 | 0.6559 | 0.2868 |

The calibrated column **does not move**. The ripple is exactly, completely
removable by dividing out per-channel gain — it defeats a baseline that does not
calibrate, and nothing else. A CNN with normalisation layers learns it for free.
Tuning `--ripple_amp` upward would buy a headline number that would not survive a
reviewer asking what it means.

**The floor is set by the label definition, not by gain structure.** The label is
a deterministic threshold on a smooth latent field, so brightness is a
near-sufficient statistic for it. That is the finding already recorded in
`AUDIT_REPORT.md`, and no bandpass touches it. If this benchmark needs to be
harder, the lever is the injected SNR distribution (more RFI near the threshold)
or label ambiguity — not instrument response.

> **Side finding worth keeping.** Per-channel calibration makes the naive
> baseline dramatically *worse* — 0.84 → 0.29 — because narrowband and
> persistent-band RFI occupy an entire channel for the whole observation. The
> channel's median **is** the RFI, so subtracting it erases the signal.
> Per-channel normalisation is actively harmful on this data.

### A correction to the project's documented figures

Two numbers in `RFI-project-context.md` are understated, both because they were
measured on small samples:

- **Per-image max spread.** Documented as 8.2× (27 → 222) from 40 images.
  Measured over 200 v3 train images it is **13.19×**; over 120 it was 12.17×. The
  documented figure is a lower bound, not the spread. v4 is **14.49×** — so
  per-image min-max normalisation is *more* dangerous here, not less.
- The difficulty floor of 0.7421 **does** reproduce exactly (0.7411 on 200 train
  images). That one is sound.

---

## 5. RFI geometry at the new dimensions, and the strip kernels

The image went from **landscape to portrait**: 276×600 is 2.17 : 1 (time : freq);
1024×265 is 0.26 : 1. The anisotropy of the data has flipped.

| | v3 276×600 | v4 1024×265 |
|---|---|---|
| Narrowband (runs along **time**) | 1 channel × 600 samples | **1–5 channels × 265 samples** |
| Wideband burst (runs along **freq**) | 138–276 ch × 1 sample | **512–1024 ch × 1 sample** |
| Persistent band width | 3–26 ch | **10–100 ch** |
| Broadband block width | 8–80 ch | **30–300 ch** |
| Blob | 1–10 ch × 2–11 bins | **5–39 ch × 1–4 bins** |

Fractionally these are equivalent — `_scale_range` preserves fractional size —
but two things genuinely change. At `n_freq = 1024` the scale factor is exactly
1.0, so frequency morphology returns to the values it was originally tuned for:
narrowband lines go from a degenerate always-1-channel to a real 1–5, and blobs
become much more elongated along frequency.

### Do the (7, 11, 21) strip kernels still look sensible?

`MultiScaleStrip` runs `(1,k)` along time and `(k,1)` along frequency at encoder
depths 0–2 and at the bottleneck. Effective coverage of the largest kernel
(k = 21), as a fraction of each axis:

| Stage | resolution | v3: freq / time | **v4: freq / time** |
|---|---|---|---|
| depth 0 | ÷1 | 7.6% / 3.5% | **2.1% / 7.9%** |
| depth 1 | ÷2 | 15.2% / 7.0% | **4.1% / 15.8%** |
| depth 2 | ÷4 | 30.4% / 14.0% | **8.2% / 31.7%** |
| bottleneck | ÷16 | **121.7%** / 56.0% | 32.8% / **126.8%** |

**It is a clean reversal.** In v3 the *vertical* (frequency) kernels reached full
coverage of the axis by the bottleneck while the horizontal ones reached 56%. In
v4 that is exactly inverted: the horizontal (time) kernels now span the whole
axis, and the vertical ones reach only a third of it.

Read: the kernels are **still reasonable but no longer well matched**. They are
now generous in the direction that needs least help — narrowband RFI already
spans all 265 time samples, so a kernel covering 127% of the time axis is
redundant — and short in the direction where wideband bursts run 512–1024
channels but the longest vertical strip reaches 336. The U-Net's pooling still
supplies a large global receptive field, and wideband bursts are strong
(SNR 3–20) and easy, so this is unlikely to be a large effect. The matched-budget
ablation already put strip convolutions at only **+0.023 F1**.

If the kernels are ever revisited, the case at this shape is for **asymmetric
kernel sets** — longer along frequency than along time — rather than the shared
(7, 11, 21). Note MARS (arXiv:2608.05546) widens to 1×31 / 31×1 in its decoder.
**No model change has been made. This is a flag, not a patch.**

---

## 6. Runtime and disk

| | Value |
|---|---|
| Generation, 1000 images | **42 s** (16 ms compute + I/O per image) |
| Verification, incl. bit-exactness | 35 s |
| Disk, final | **1.9 GB** (v3: 1.2 GB) |
| Peak during reproducibility check | 3.8 GB |

Per image: 1.09 MB image (f32) + 0.27 MB mask (u8) + 0.54 MB strength (f16)
+ 4 KB bandpass (f32) ≈ **1.9 MB**. 1024×265 is 1.64× the pixels of 276×600, and
the extra 0.7 GB over v3 is that ratio plus the new bandpass arrays.

---

## 7. Verification

`verify_dataset_v4.py` — exits non-zero on any failure, writes
`verification/report.json` plus two PNGs.

```
1. STRUCTURE      shape (1024,265) f32/u8/f16 + (1024,) f32 across all 1000
                  mask == (strength > 0.5) everywhere: 0 disagreeing pixels
2. METADATA       1000 rows; rfi_fraction matches the masks exactly (drift 0.0)
3. LEAKAGE        0 duplicates within any split; 0 shared images between splits
4. RFI FRACTION   train 14.03% / val 14.89% / test 14.24%; overall 14.19%
5. DYNAMIC RANGE  per-image max spread 14.49x
6. CALIBRATION    LO -2.8883  HI 44.8850  weights [0.5816, 3.5633]  (train only)
7. FLOOR          0.7668 vs v3's 0.7421
8. BANDPASS       54,403 of 37,073,066 injected px excluded (0.147%)
9. PREVIEWS       previews_images_masks.png, bandpass_curves.png
10. BIT-EXACT     4025 / 4025 files identical, 0 differing
```

Reproducibility is total — **including the 20 preview PNGs**, which are also
byte-identical between runs. The generator seeds numpy's legacy `RandomState`
once and draws in a fixed call order; `RandomState` carries numpy's stream
compatibility guarantee, which `default_rng` does not.

---

## 8. Two traps carried over from v3

**`--patch_size` is now twice as dangerous.** `RFIPatchDataset` crops with plain
numpy slicing, which truncates silently rather than erroring. At the old default
of 512:

- 276×600 → 276×512, dropping 14.7% (already documented)
- **1024×265 → 512×265, dropping 50% of the frequency axis**

`--patch_size 0` was already mandatory. Forgetting it now costs half the image
instead of a seventh, and still fails without a warning.

**VRAM.** 1024×265 is 1.64× the pixels of 276×600. The hybrid at `batch_size 8`
fit "comfortably" in this GPU's real ~3.5 GB at the old shape; that headroom is
now largely spent. Run the preflight check with the **real** image shape before
assuming batch 8 still fits — per `CLAUDE.md` §5, that check has had the
square-shape bug twice.

---

## 9. Flags

Every constant is a named flag with a documented default — there are no bare
literals in the generator. `python3 generate_dataset_v4.py --help` lists all of
them, grouped. Multi-valued flags keep the count manageable: Poisson rates are
one flag per RFI type taking four values, one per density tier
(`--lam_narrowband 0.0 2.0 6.0 11.0`), and ranges are two-value flags
(`--nb_snr 1.5 12.0`).

Defaults reproduce v3 exactly for everything that carries over. The new flags are
the bandpass group, `--rx_noise_frac`, and `--mask_sigma_threshold`.

`validate()` rejects parameter combinations that would produce a silently wrong
dataset — a band too narrow for the blobs it must place, non-positive SNR bounds
for a log-uniform draw, splits that leave no test set.

---

## 10. Judgement calls

Flagged so they can be revisited rather than inherited silently.

1. **Kept v3's inverted noise model** (`sigma = sigma_bg / sqrt(gain)`, noise
   rises where the pedestal is low). It is the opposite of a radiometer and it is
   why the roll-off deletes the hardest channels. Kept for comparability: fixing
   it would move every image statistic and v4 would stop being "v3 plus a
   bandpass". Documented rather than corrected.

2. **Shipped `--ripple_amp` at the specified 0.15** rather than tuning it to
   1.0 to force the difficulty floor down. The extra difficulty would have been
   provably removable (§4b).

3. **`--rx_noise_frac 0.10`** was not measured from anything — no real receiver
   was characterised. It is a plausible round number that produces a defensible
   exclusion rate (0.147%). A larger value would deepen the band-edge gradient;
   the sweep in §3 shows it barely moves before 0.5.

4. **Blob injection keeps v3's nested Python loop** rather than being vectorised.
   It draws no random numbers, so the cost is only speed (negligible at 16 ms per
   image), and keeping it identical makes the v3 → v4 diff auditable.

5. **The difficulty floor is computed on `train`**, not test, to honour the
   "test is never touched" rule. It is a descriptive statistic, so either split
   would be defensible; train was chosen to keep the rule absolute.
