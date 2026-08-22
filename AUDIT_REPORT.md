# Publication-Readiness Audit — RFI Detection Project

**Audited:** `RonakSinghRaina/MP` @ `9b8e16e` · **Date:** 2026-08-22
**Scope:** every file in the repository, plus independent regeneration of the
dataset and independent re-training of controlled model variants.

---

## Verdict up front

**The hybrid model's reported numbers are almost certainly honest. The
comparison built on top of them is not publishable as written.**

I could not find fabrication, hardcoding, or test-set leakage in the code paths
that produced the reported hybrid result. I regenerated the dataset from the
documented seed and it reproduces **bit-exactly**, which let me check the
reported figures against the same data the model was trained on.

But the paper's central claim —

> *"the performance gap is attributable to architectural limitations of the plain
> U-Net rather than training configuration"*

— is **contradicted by direct measurement**. On the identical test set:

| Method | Params | ROC AUC | F1 |
|---|---:|---:|---:|
| A single constant threshold on the raw pixel value | **0** | **0.9308** | **0.7421** |
| A 3-layer toy CNN, 200 gradient steps | **2,578** | **0.9574** | **0.7823** |
| A plain U-Net with the hybrid's own training recipe | 1,942,306 | **0.9838** | **0.8814** |
| **`tf_unet` baseline, as reported in this repo** | ~2.0 M | **0.6681** | **0.3879** |

The reported baseline loses to a constant. It is a broken run, not an
architectural limit, and no claim about architecture can rest on it.

I then ran the ablation that this project never ran — same data, same order,
same seed, same loss, same optimiser, same step budget, **only the architecture
changing**. Turning the hybrid's three claimed components on and off moves F1
between **0.8785 and 0.9117**. The full architecture beats a plain U-Net of the
same family by **+0.011 F1**. Of the three components, only the strip
convolutions show a real effect (**+0.023**); residual blocks contribute
**+0.002** and ECA **−0.005** — the single best variant in the grid is the full
model *with ECA removed*.

So there is a genuine finding here: **multiscale anisotropic strip convolutions
help.** It is a modest, defensible, publishable result. It is not the +0.59 gap
currently being attributed to architecture.

This is all fixable. Nothing below requires abandoning the work — it requires
running controls that were never run, correcting documentation that currently
contradicts itself, and making a smaller, true claim instead of a large,
unsupported one.

---

## 1. What I verified, and what held up

I regenerated the dataset with the documented command and `--seed 42` and
recomputed the underlying statistics independently.

| Claim | Source | My measurement | |
|---|---|---|---|
| Test set = 24,840,000 pixels | prior audit | 24,840,000 | ✅ exact |
| Test RFI fraction 15.0808% | prior audit | 15.0808% (3,746,075 px) | ✅ exact |
| Dataset mean RFI 14.67% | `dataset_statistics.txt` | 14.67% | ✅ exact |
| 9 test images contain no RFI | prior audit | 9 / 150 | ✅ exact |
| `HybridRFINet` = 9,304,186 parameters | comparison doc | 9,304,186 | ✅ exact |
| Splits 700 / 150 / 150 | comparison doc | 700 / 150 / 150 | ✅ |
| Masks are strictly binary | — | all values ∈ {0,1} | ✅ |
| Splits are disjoint and generator-assigned | — | `metadata.jsonl` carries `split`; indices 0–699 / 700–849 / 850–999 | ✅ |
| No duplicate or shared images across splits | prior audit | 0 shared SHA-256 hashes across any pair; 0 internal duplicates | ✅ |
| The model cannot see the label | prior audit | `forward(x)` takes only `x`; first conv is `(32, 1, 3, 3)` → `in_channels=1`; no training or eval code reads `strength/` | ✅ |
| "all-RFI" degenerate baseline F1 = 0.2621 | prior audit | 0.2621 (my `logistic_pixel` collapses to exactly this) | ✅ exact |
| `tf_unet` applies ReLU to its output logits | `CLAUDE.md` §7 | confirmed, `tf_unet/unet.py` line 145: `output_map = tf.nn.relu(conv)` | ✅ |
| The generator is deterministic given a seed | — | confirmed | ✅ |

**The dataset is genuinely reproducible.** That is a real strength and worth
stating in the paper.

One correction to the *existing* `rfi_model_audit_report.md`: its
strength-stratified recall table — described there as *"the strongest single
piece of evidence in the whole audit"* — does not reconcile with its own
confusion matrix. The table's buckets sum to **1,191,617** RFI pixels, but the
same report's TP + FN is **3,746,075** (which I confirm exactly). The table
therefore covers about **32%** of the test RFI pixels, with no statement that it
is a subset. Recompute it over all pixels before relying on it.

---

## 2. BLOCKING — these prevent publication

### A1. The baseline is broken, so the headline comparison establishes nothing

Reported: `tf_unet` ROC AUC 0.6681, F1 0.3879.

Measured by me on the identical test set, using the same protocol (threshold
chosen on validation, test touched once):

- **A constant global threshold on the raw pixel value**: ROC AUC 0.9308, F1 0.7421, MCC 0.7092.
- **A 2,578-parameter, 3-layer CNN** (no downsampling, no skips, no attention), 200 gradient steps: ROC AUC 0.9574, F1 0.7823, MCC 0.7493.

A U-Net that scores below a constant threshold has not demonstrated an
architectural limitation — it has failed to train. Every sentence of the form
*"the hybrid architecture outperforms the plain U-Net"* currently rests on this
number, and a referee will find the problem in minutes.

**What to do:** either get `tf_unet` to a defensible score (it should
comfortably clear 0.9 ROC AUC on this data) and report that, or drop the
`tf_unet` comparison and compare against a U-Net you can actually train — see A2.
Report the constant-threshold number in the paper regardless; it is the floor
every method must clear.

Run: `python3 experiments/classical_baselines.py`

### A2. No ablation exists, so the architecture claim is unsupported anyway

`hybrid_model.py`'s own docstring says the excluded extras *"belong in the
ablation table as separate rows, not in v1"*. **There is no ablation table
anywhere in this project.** Nothing establishes that residual blocks, strip
convolutions or ECA contribute anything.

The hybrid differs from `tf_unet` in at least **eight** ways simultaneously:

| | `tf_unet` baseline | Hybrid |
|---|---|---|
| Architecture | plain U-Net | residual + strip conv + ECA |
| Framework | TensorFlow 1.x (compat shim) | PyTorch |
| Loss | weighted cross-entropy | weighted CE **+ Dice** |
| Normalisation | none | GroupNorm |
| Padding | valid | same |
| Output activation | **ReLU on logits** | raw logits |
| Batch size | 4 | 8 |
| Width | `features_root=32` | `base=32`, depth 4 |

Attributing the gap to "the architecture" when eight variables moved together is
not supportable.

I ran the missing experiment — every variant with identical data, identical
order, identical seed, identical loss, identical optimiser, identical gradient-step
budget, identical evaluation protocol; only the architecture changes.

**Setup.** Identical for every row: the same 400 training images in the same
order, seed 42, the project's own weighted CE + Dice loss imported from
`train_hybrid.py`, Adam @ 1e-3 with cosine annealing, **200 gradient steps**,
batch 4, `base=16`, whole 276×600 images, threshold picked on 80 validation
images and applied once to all 150 test images. Reduced budget so seven variants
fit on 2 CPU cores — absolute F1 is therefore below the full-budget 0.98.

| Variant | Residual | Strip conv | ECA | Params | ROC AUC | F1 | MCC |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| `logistic_pixel` (1×1 conv) | – | – | – | 4 | 0.3165 | 0.2621 | 0.0000 |
| `tiny_cnn` (3 layers) | – | – | – | 2,578 | 0.9574 | 0.7823 | 0.7493 |
| `plain_unet` | ✗ | ✗ | ✗ | 1,942,306 | 0.9838 | 0.8814 | 0.8614 |
| `no_strip` | ✓ | ✗ | ✓ | 2,029,386 | 0.9859 | 0.8785 | 0.8579 |
| `no_res` | ✗ | ✓ | ✓ | 2,255,418 | 0.9900 | 0.9033 | 0.8867 |
| `no_eca` | ✓ | ✓ | ✗ | 2,342,450 | **0.9919** | **0.9117** | **0.8963** |
| `hybrid_full` | ✓ | ✓ | ✓ | 2,342,474 | 0.9889 | 0.8928 | 0.8737 |
| *reported `tf_unet` baseline* | – | – | – | ~2.0 M | *0.6681* | *0.3879* | *–* |

**Marginal effect of each component** (mean F1 with it, minus mean F1 without):

| Component | Present | Absent | Δ F1 |
|---|---:|---:|---:|
| Multiscale strip convolutions | 0.9026 | 0.8800 | **+0.0226** |
| Residual blocks | 0.8944 | 0.8924 | +0.0020 |
| Efficient channel attention | 0.8916 | 0.8966 | **−0.0050** |

**Read it carefully — three things follow.**

1. **All five U-Net variants land between F1 0.8785 and 0.9117.** The full
   architecture beats a plain U-Net of the same family by **+0.011 F1**
   (0.8928 vs 0.8814). The paper attributes a gap of **+0.59** to architecture.
   Whatever produced that gap, it is not these three components.

2. **Only the strip convolutions show a real effect** (+0.023 F1), and it is
   consistent across both comparisons that isolate it (`no_eca` 0.9117 vs
   `no_strip` 0.8785; `no_res` 0.9033 vs `plain_unet` 0.8814). That *is* a
   genuine, defensible, publishable contribution — it is just two orders of
   magnitude smaller than the claim currently being made.

3. **ECA does nothing, and may cost you.** Its marginal effect is −0.005, and
   the best-scoring variant in the whole grid is `no_eca` — the full model with
   ECA removed. Dropping it would simplify the architecture at no measured cost.

**Caveats, stated plainly.** One seed per cell, at a reduced budget. Differences
smaller than about 0.03 F1 are **not** established by this run and could be seed
noise — that includes the residual-block and ECA effects, and arguably the strip
effect too. What the reduced budget *cannot* explain is the difference in order
of magnitude: every learned variant here, down to a 2,578-parameter CNN, scores
far above the reported baseline. Rerun with `--base 32 --n_train 700 --batch 8
--steps 1936 --seeds 0 1 2` on the GPU for the publication table; the script also
reports recall stratified by injected RFI strength, which is the precise test of
the strip convolutions' stated mechanism (they exist so that *"a 1.5-sigma line
becomes detectable"*, so any real benefit should concentrate in the weakest bins).

**What to do:** rerun `experiments/run_ablation.py` at full budget on the GPU
with error bars —

```bash
python3 experiments/run_ablation.py \
    --dataset_dir "Synthetic Dataset 276x600" \
    --base 32 --n_train 700 --n_val 150 --batch 8 --steps 1936 \
    --seeds 0 1 2 --out results/ablation_full.json
```

— and put that table in the paper in place of the current baseline comparison.
Expect it to confirm the shape of the reduced-budget result: a modest,
strip-convolution-driven gain. Then consider dropping ECA from the model
entirely; it costs parameters and a paragraph of justification and buys nothing
measurable.

### A3. The documented "fair comparison" configuration is self-contradictory

`RFI_Project_Model_Comparison.md` §3 describes Model 2 as **batch size 4**,
**`features_root=32`**, and **`training_iters=87`**.

Those cannot all be true. `run_fair_comparison.py` computes
`training_iters = n_train / batch_size`; at batch 4 that is **175**, not 87.
87 is `700/8`. So either:

- `training_iters=87` really was used at batch 4 → the baseline saw **348 of 700
  training images per epoch, half the training set**, an enormous unfair
  handicap that by itself could explain the gap; or
- the documented 87 is wrong and 175 was used → the baseline got **3,850**
  gradient steps against the hybrid's 1,936, i.e. the epoch counts were never
  matched either.

Both readings are fatal to a "matched conditions" claim, and **the repository
cannot disambiguate**, because no run output is committed (see A5).

Compounding it: `run_fair_comparison.py` defaults to `--features_root 64` while
the doc reports 32, and `CLAUDE.md` §12's command passed neither — so following
the documented instructions trains a 64-wide model that
`evaluate_test_set.py --features_root 32` then cannot even restore.

**What to do:** rerun the baseline with an explicit, recorded configuration. The
script now writes `run_config.json`; quote that file in the paper, not memory.

### A4. The two models in the summary table are not the same measurement

| | Hybrid | Baseline |
|---|---|---|
| Metric | F1 at a **fixed** validation-selected threshold | `max_f1` — **maximum F1 over all thresholds, computed on the test set** |
| Pixels scored | full 276 × 600 = 165,600 / image | **236 × 560 = 132,160 / image** |

The baseline number is an *oracle* threshold tuned on test. And because
`tf_unet` uses valid padding, its output is smaller than its input — I computed
the offset for `layers=3` as **40 pixels**, so 20 rows/columns are dropped from
every edge. That crop removes the entire band-edge rolloff region (rolloff width
is only 5–15 channels at 276), which is the noisiest, hardest part of each image.

Both biases favour the baseline, so the true gap is if anything *larger* than
reported — but the table as printed compares two different quantities over two
different pixel populations and will not survive review.

**What to do:** report both models at a fixed validation-selected threshold over
an identical pixel region, and say which region.

### A5. Nothing needed to verify a single reported number is in the repository

Absent: the dataset, all checkpoints, `training_log.csv`, `progress.json`,
`eval_test/metrics.json`, `dataset_statistics.txt`, and every figure.
`.gitignore` excluded all of it.

The dataset is regenerable, which is good. The run evidence is not. As it
stands, **no reader can check any number in `RFI_Project_Model_Comparison.md`.**

**What to do:** the evidence files are kilobytes — commit them. `.gitignore` has
been updated with negation rules so `training_log.csv`, `progress.json`,
`run_config.json` and `eval_*/metrics.json` are kept while the heavy binaries
stay excluded. Archive the checkpoints on Zenodo/OSF and cite the DOI.

---

## 3. SERIOUS — fix before submission

### B1. N = 1. No repeats, no error bars

One run per model. Reporting `F1 = 0.9808` to four decimal places from a single
seed is not defensible, and the hybrid-vs-baseline gap has no uncertainty
attached. Run ≥3 seeds per configuration and report mean ± sd.
`experiments/run_ablation.py --seeds 0 1 2` does this.

### B2. Training was stopped by hand, and the baseline was matched to that

`RFI_Project_Model_Comparison.md` §3: *"Training was stopped manually at epoch 22
(of a planned 40) once score gains had clearly flattened."* No pre-declared
criterion; not reproducible; and the arbitrary 22 then became the epoch budget
the baseline was "matched" to. A pre-declared `--early_stop_patience` has been
added — use it, and report the criterion.

### B3. A silent test-set-as-validation fallback was still live in the code

`train_unet_rfi_gpu.py` contained:

```python
if not os.path.exists(val_img_dir):
    val_img_dir = os.path.join(args.dataset_dir, "test", "images")   # silent
```

`CLAUDE.md` §10 records that this caused real leakage once. It was still there,
still silent. **Fixed:** it now refuses to run and tells you how to make a val
split. Confirm which runs, if any, were produced before this existed.

### B4. Validation used *random* crops, making checkpoint selection noisy

`RFINpyDataProvider._post_process` takes a random crop whenever `patch_size` is
set — regardless of `shuffle_data=False`. So with `--patch_size 512` every
checkpoint was scored on a *different random window*. This is the mechanism
behind the documented "winner's curse" incident (an inflated best F1 of 0.8138
that collapsed to 0.34). It affects Model 1, which used `patch_size=512`.
**Fixed:** validation now always uses whole images.

### B5. The documented evaluation command silently cropped the test set

`evaluate_hybrid_test.py --patch_size` defaulted to **512**, and `CLAUDE.md`
§12's evaluation command omitted the flag. At 276 × 600 that centre-crops to
276 × 512, discarding **88 of 600 time bins — 14.7% of every image** — and
yields numbers that do not match the published ones. `PAPER_DIMENSIONS.md` got
this right; `CLAUDE.md` did not. **Fixed:** the default is now 0, the value used
is recorded in `metrics.json`, and a mismatch against the training config warns.

### B6. The "validation-selected threshold" could silently be a number nothing selected

```python
best_thr = float(thr[bi]) if bi < len(thr) else 0.5
```

`precision_recall_curve` returns one more precision/recall point than
thresholds. If the argmax landed on that degenerate endpoint, the code
substituted **0.5** and reported it as the validation-selected threshold.
**Fixed:** clamps to the last real threshold and warns loudly.

### B7. The overfitting argument compares two different things

`progress.json`'s best val F1 (0.9802) is the **oracle**, threshold-tuned score
on validation; the test F1 (0.9808) is at a fixed threshold. "Test beat
validation, therefore no overfitting" is a comparison deliberately stacked in
validation's favour. That makes the conclusion *conservative*, which is fine —
but the write-up presents it as evidence without saying the comparison is
biased. Say it.

### B8. No determinism controls

No `cudnn.deterministic`, no `torch.use_deterministic_algorithms`, no seed
recorded in any output. Exact reproduction was impossible by construction.
**Fixed:** `--deterministic` flag added; `run_config.json` now records the seed
and full configuration.

### B9. The benchmark's difficulty was never characterised — and it is low

This is the most important scientific gap after A1/A2.

The generator defines the ground truth as

```python
mask = (hard injected footprint) | (rfi_layer > 0.5 * local_sigma)
```

I measured the consequence directly: **exactly 0 clean-labelled pixels in the
test set carry more than 0.5σ of injected RFI.** The label is a deterministic,
noise-free threshold on a smooth latent field — there is **zero label
ambiguity**. Real telescope flags are nothing like this.

Strength distribution of the labelled RFI (units of local noise σ):

| Strength | RFI pixels | Share | Cumulative |
|---|---:|---:|---:|
| 0–0.5 σ | 1,616 | 0.04% | 0.04% |
| 0.5–1 σ | 338,098 | 9.03% | 9.07% |
| 1–2 σ | 826,867 | 22.07% | 31.14% |
| 2–4 σ | 1,116,319 | 29.80% | 60.94% |
| 4–8 σ | 1,034,988 | 27.63% | 88.57% |
| > 8 σ | 428,187 | 11.43% | 100.00% |

**90.93%** of the target sits at or above 1σ; **68.86%** at or above 2σ. A
detector that flagged everything ≥ 1σ and nothing else would already reach 91%
recall before any learning.

F1 ≈ 0.98 is therefore an unremarkable score *for this benchmark*, not evidence
of a strong method. State these numbers next to any headline result.

Run: `python3 experiments/dataset_difficulty.py`

### B10. No comparison against methods the field actually uses

Nobody flags RFI with a plain U-Net; they use sigma clipping and SumThreshold
(Offringa et al. 2010). I implemented both:

| Method | ROC AUC | PR AUC | F1 | MCC |
|---|---:|---:|---:|---:|
| Constant global threshold | 0.9308 | 0.8257 | 0.7421 | 0.7092 |
| Per-channel sigma clipping | 0.5787 | 0.2169 | 0.2788 | 0.0851 |
| SumThreshold-lite | 0.6432 | 0.3608 | 0.3460 | 0.2507 |

Note the striking result: the *classical, physically-motivated* methods do
**badly**, while the naive constant threshold does well. That is diagnostic of
the dataset, not of the methods — per-channel normalisation fails here because
the synthetic narrowband and persistent-band RFI occupies **100% of the time
axis** in the channels it contaminates, so each channel's own median is the RFI
level. Real narrowband RFI does this too, which is exactly why real flaggers fit
a bandpass model rather than per-channel medians.

This is worth a paragraph in the paper either way, and it is a fair-minded thing
to include.

### B11. In-distribution generalisation only

Train, val and test are i.i.d. draws from one generator, one seed, one parameter
set. The splits are provably disjoint — this is *not* leakage — but 0.98
measures in-distribution generalisation and nothing else. It says nothing about
real observations, or even about RFI morphologies this generator cannot produce.
The existing write-up says this; keep saying it, prominently.

### B12. Unequal tuning effort

The hybrid gets a carefully justified configuration; the baseline gets one
configuration, at a reduced width forced by VRAM. Unequal tuning effort between
"ours" and "theirs" is the single most common source of inflated gaps in the
literature. Report what search, if any, each model received.

---

## 4. REPRODUCIBILITY AND REPO HYGIENE

### C1. Copyright violation — publisher PDF redistributed ⚠

`RFI_Using_UNET.pdf` is the **full Elsevier-published article**: Akeret et al.
(2017), *Astronomy and Computing* 18, 35–39, doi:10.1016/j.ascom.2017.01.002
(PDF metadata: `/Creator: Elsevier`, `PDF/A-1b`). Redistributing it in a public
repository infringes copyright.

**Removed from the working tree.** It is **still in git history** (present since
the initial commit), so history must be rewritten before the repository is made
public:

```bash
pip install git-filter-repo
git filter-repo --path RFI_Using_UNET.pdf --invert-paths
git push --force-with-lease origin main       # coordinate with any collaborator first
```

Cite the DOI, or link arXiv:1609.09077.

### C2. Licensing — GPL-3.0 code vendored with no licence declared

`tf_unet` is **GPL-3.0** (`setup.py`: `license='GPLv3'`) and is vendored
**twice**. The repository had no root `LICENSE`. Distributing a work containing
GPL-3.0 code means the combined work must be GPL-3.0 and must carry the licence.
**Fixed:** GPL-3.0 `LICENSE` added at root; `README.md` and `CITATION.cff` state
it. Alternative: remove the vendored copy and pin `tf_unet` as a dependency.

### C3. A folder literally named `unet_rfi_package copy` is the canonical code

Every reported baseline number came from scripts that exist **only** in
`unet_rfi_package copy/`. `unet_rfi_package/` holds older, different scripts and
a byte-identical duplicate of `tf_unet`. A reviewer cannot tell which is real.

I did **not** rename it — that would break the paths in `CLAUDE.md` and in your
local workflow, and it is your call. Before release, rename it to something like
`unet_rfi_package_faircompare/`, delete the superseded copy, and update the
documented paths.

### C4. Unrelated material at the top level

`get_transcript.py` (a YouTube transcript scraper), `video_transcript.txt`, and
`Machine_Learning_Explained_Notes.{md,html,pdf}` (the same personal revision
notes in three formats). **Moved to `notes/`** with a README explaining what
they are; consider deleting before release. The transcript is third-party
content reproduced verbatim — the same copyright issue as C1, smaller.

### C5. Editor auto-approve settings committed

`.vscode/settings.json` set `autoApprove: true`, `nonWorkspaceFileAccess: allow`,
`internetAccessPolicy: allow`. Machine-local agent permissions do not belong in a
shared repository. **Removed**, and `.vscode/` added to `.gitignore`.

### C6. No README, no dependency manifest, no citation metadata

`CLAUDE.md` described the environment narratively (two venvs, cuDNN 9.24 vs 9.1)
with nothing machine-readable. **Fixed:** root `README.md`, `requirements-tf.txt`,
`requirements-torch.txt`, `CITATION.cff` added.

### C7. Git history

Four commits, two of them titled `c`. Not disqualifying, but it signals a
repository that was never prepared for release. Squash or rewrite before
publishing the link.

### C8. Hard-coded absolute Windows paths

`CLAUDE.md` documents `/mnt/c/Users/RONAK SINGH/Documents/Coding/Minor Project/...`.
Fine as private notes; not usable as instructions for anyone else. The new root
`README.md` gives repository-relative commands.

---

## 5. DOCUMENTATION CONTRADICTIONS

Each of these is a question a referee will ask.

| # | Contradiction | Status |
|---|---|---|
| D1 | **Train size 700 vs 550.** The comparison doc says 700/150/150. `make_val_split.py` and the hybrid README say val is carved out of train, leaving **550**. The generator already emits 700/150/150, so `make_val_split.py` is a no-op — but both READMEs told you to run it. Nothing states which happened. | Clarified in `hybrid_rfi_package/README.md`; **you must confirm which applies to the reported run** |
| D2 | **RFI fraction 12.4% vs 14.67%.** `PAPER_DIMENSIONS.md` claimed 12.4% at 276×600. Regeneration with the documented seed gives **14.67%**. | Corrected |
| D3 | **`features_root` 64 vs 32.** Script default 64; doc reports 32; `CLAUDE.md` passed neither to training but 32 to evaluation — a combination that cannot restore. | Corrected in both files |
| D4 | **Batch size 8 vs 4.** Script default 8; doc reports 4. Interacts with A3. | Flagged in-script |
| D5 | **The hybrid README told you to break the project.** It said `source ~/tf-env/bin/activate` then `pip install torch` — the exact action `CLAUDE.md` §3 documents as having broken the TensorFlow environment (torch downgrades cuDNN). | Fixed |
| D6 | **`PAPER_DIMENSIONS.md`'s baseline command omits `--training_iters`**, so an "epoch" would be tf_unet's default 64 steps, not a full pass — producing a run comparable to nothing. It also uses batch 32 / features 64 / momentum 0.2, a different configuration from the one reported. | Fixed |
| D7 | **`evaluate_test_set.py`'s own docstring warns** the checkpoint may have been selected on the test set when no `val/` exists. If that applied to Model 1, Model 1's number is not held-out — yet the summary table presents all three as *"never-trained-on test sets"*. | **You must confirm Model 1's provenance** |
| D8 | **`train_hybrid.py`'s header** claimed `batch_size = 1, patch_size = 512` and *"same 1024×1024 dataset"* — stale; the reported run used batch 8, patch 0, 276×600. Argparse defaults are still the 1024 ones, so running with no flags reproduces nothing published. | Corrected |
| D9 | `generate_spectrogram`'s docstring listed 3 return values; it returns 4. | Corrected |
| D10 | The existing `rfi_model_audit_report.md`'s strength-recall table covers ~32% of test RFI pixels while presenting itself as complete (see §1). | **Recompute** — `--strength_report` now does this correctly |

---

## 6. MINOR CODE ISSUES

- `train_hybrid.py` logged one CSV row per **chunk** (default 2 epochs) while the
  `epoch` column names a single epoch, and `train_loss` was the 2-epoch mean. A
  22-epoch run yields 11 rows. Easy to misread as a per-epoch curve. *Fixed:*
  column renamed `train_loss_chunk_mean`, `epochs_in_chunk` added.
- `train_hybrid.py` called `input()` when no GPU was found — hangs forever in any
  unattended run. *Fixed:* `--allow_cpu`, plus a non-TTY guard.
- `run_fair_comparison.py` used `n_train // batch_size` (**floor**). PyTorch's
  DataLoader keeps the short final batch, so the hybrid's epoch is
  `ceil(700/8) = 88`, not 87 — the baseline was given one step per epoch fewer.
  *Fixed:* `math.ceil`. (88 × 22 = 1936, matching the Adam step count in the
  hybrid checkpoint.)
- `RFIPatchDataset` passes `seed=None` for validation → a non-deterministic
  `RandomState`. Harmless only because `random_crop=False`.
- `evaluate()`'s `max_images` counts **batches**, not images. Equal today because
  the loader uses `batch_size=1`; fragile if that changes.
- `compute_class_weights` silently returns `[1.0, 1.0]` if `metadata.jsonl` is
  missing, quietly disabling class weighting on an 85/15 imbalanced problem.
- `generate_spectrogram`'s `rfi_density` parameter is never passed by
  `generate_dataset` — dead in practice.
- **Per-image min-max normalisation is outlier-sensitive.** `RFIPatchDataset`
  does `img -= img.min(); img /= img.max()`, so the dynamic range every image is
  squeezed into is set by its single brightest pixel. On this dataset the effect
  is mild (median max/median ratio 1.98, worst 4.63; in the worst image 99.9% of
  pixels still sit below 53.5% of the range) because the generator caps RFI at
  SNR ≈ 20. On real data, where a single saturated sample is routine, this would
  compress everything else toward zero and destroy sensitivity to faint RFI. A
  percentile-based normalisation (e.g. clip at the 99.5th percentile) is the
  standard fix and is worth a sentence in the paper's limitations.

---

## 7. What I changed in this repository

**Added**

| File | Why |
|---|---|
| `AUDIT_REPORT.md` | this document |
| `README.md` | there was none |
| `LICENSE` (GPL-3.0) | required by vendored `tf_unet` (C2) |
| `requirements-tf.txt`, `requirements-torch.txt` | no dependency manifest existed (C6) |
| `CITATION.cff` | citation metadata, credits Akeret et al. and Offringa et al. |
| `experiments/dataset_difficulty.py` | measures B9 |
| `experiments/classical_baselines.py` | measures A1 / B10 |
| `experiments/models_ablation.py`, `experiments/run_ablation.py` | measures A2 |
| `experiments/README.md`, `notes/README.md` | orientation |

**Fixed**

- `train_unet_rfi_gpu.py` — removed the silent test-set-as-validation fallback (B3); validation no longer random-crops (B4).
- `evaluate_hybrid_test.py` — `--patch_size` default 0 (B5); provenance recorded in `metrics.json`; `--per_image_csv`; `--strength_report` (D10); warns on train/eval config mismatch.
- `train_hybrid.py` — threshold-fallback bug (B6); `--deterministic` (B8); `--early_stop_patience` (B2); `--allow_cpu`; `run_config.json`; CSV columns clarified; stale docstring corrected (D8).
- `run_fair_comparison.py` — `ceil` not floor; `run_config.json`; `features_root` mismatch documented (A3/D3).
- `dataset_generator_v3_strength.py` — docstring corrected, label construction documented (D9/B9).
- `.gitignore` — keeps small evidence files, ignores `.vscode/` (A5, C5).
- Docs — `CLAUDE.md`, `PAPER_DIMENSIONS.md`, `hybrid_rfi_package/README.md` (D2, D3, D5, D6).

**Removed / moved**

- `RFI_Using_UNET.pdf` — deleted (C1). **History rewrite still required.**
- `.vscode/settings.json` — deleted (C5).
- `notes/` — personal notes, transcript and scraper moved out of the root (C4).

**Deliberately not changed** — your call: renaming `unet_rfi_package copy/` (C3),
rewriting git history (C1), deleting `notes/` entirely, and every reported
number, which I have not touched.

---

## 8. Priority order

**Before you write another word of the paper**

1. **A1** — fix or drop the `tf_unet` baseline. It loses to a constant threshold.
2. **A2** — rerun the ablation at full budget with 3 seeds. A reduced-budget run
   is already in `results/ablation_reduced_budget.json`; confirm it at scale,
   then **rewrite the central claim around the strip convolutions** rather than
   around the baseline gap.
3. **A3** — rerun the baseline with a recorded config; resolve the 87-vs-175 contradiction.
4. **D1, D7** — confirm the training-set size, and whether Model 1 was selected on test.

**Before submission**

5. **B1** — ≥3 seeds, mean ± sd on every reported number.
6. **A4** — one metric, one pixel region, both models.
7. **B9 + B10** — report benchmark difficulty and classical baselines.
8. **A5** — commit the evidence files; archive checkpoints with a DOI.
9. **B2** — pre-declared stopping criterion, rerun.

**Before making the repository public**

10. **C1** — rewrite history to remove the Elsevier PDF.
11. **C3** — rename the `copy` folder; delete the superseded duplicate.
12. **C4, C7** — clear `notes/`; tidy the commit history.

---

## 9. How to state the result honestly, today

Given only what is currently verified, the defensible claim is narrow — and it
is a real claim, not a retreat:

> On a synthetic 276×600 RFI benchmark generated by a rule-based simulator, a
> U-Net variant trained with GroupNorm, raw output logits and a combined
> cross-entropy + Dice objective reaches F1 = 0.98 at a threshold selected on a
> held-out validation split. A controlled ablation holding data, loss,
> optimiser and gradient-step budget fixed attributes **+0.023 F1 to multiscale
> anisotropic strip convolutions**, while residual blocks (+0.002) and efficient
> channel attention (−0.005) show no measurable benefit. We report reference
> points on the same benchmark: a constant intensity threshold reaches F1 = 0.74
> and a 2,578-parameter CNN reaches F1 = 0.78, so the benchmark is not difficult
> in absolute terms — 90.9% of the labelled RFI lies at or above the local noise
> level and the labels are a deterministic function of the injected amplitude.
> We make no claim about performance on real telescope observations.

That is a paper. It is a smaller paper than the one currently drafted, and it is
one a referee can check.

What you **cannot** claim is that the ~0.6 F1 gap over `tf_unet` measures an
architectural limitation of the plain U-Net. On the evidence, it measures a
`tf_unet` run that failed to train.
