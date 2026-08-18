Verdict up front: the result is real.

I could not find any fabrication, hardcoding, or leakage. Details below, including the two things I'd still flag.

Note: hybrid_rfi_package/best.pt does not exist. The only checkpoint is hybrid_run_paperdim/best.pt. That's what I audited.

---
1. Checkpoint forensics — genuinely trained

┌────────────────────────────────────────────┬─────────────────────────────────────────────┐
│                  Evidence                  │                   Finding                   │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Stored epoch                               │ 22                                          │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Adam optimizer step                        │ 1936                                        │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ ceil(700 train / batch 8) × 22 epochs      │ 1936 ✅ exact                               │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ GroupNorm weights still ==1.0 (init value) │ 0 / 22                                      │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ GroupNorm biases still ==0.0 (init value)  │ 0 / 22                                      │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Tensors identical to a fresh random init   │ 0 / 117                                     │
├────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Adam exp_avg_sq                            │ 4.8e-04 (nonzero ⇒ gradients really flowed) │
└────────────────────────────────────────────┴─────────────────────────────────────────────┘

The step-count arithmetic is the strongest single item: 1936 is not a round number, and it independently reconstructs "700 images, batch 8, 22 epochs" from optimizer state alone.

Timestamps are consistent and correctly ordered: last.pt 05:25:11 → best.pt 05:25:15 → progress.json/training_log.csv 05:25:15 → eval_test/metrics.json 05:52. Training finished, then evaluation ran.

best.pt and last.pt are byte-identical (same SHA-256). That's expected, not suspicious: best_epoch == epochs_completed == 22, so the final epoch was also the best.

2. Pixel accounting — exact

TP =  3,676,613
TN = 21,019,233
FP =     74,692
FN =     69,462
        ----------
SUM= 24,840,000  ==  150 × 276 × 600  ✅ EXACT
All 150 masks contain only values {0, 1} — no third class being silently dropped.

One honest discrepancy: my confusion matrix differs from the reported one by exactly ±1 pixel (TP +1, FN −1, FP +1, TN −1). I chased this down: exactly one pixel in the whole test set sits within 1e-6 of the threshold 0.6112715602. cuDNN selects convolution algorithms non-deterministically, so bit-exact reproduction across two runs isn't expected.

This is evidence for authenticity, not against. A hardcoded metrics file would have matched perfectly. A ±1 float-boundary flip is the signature of an honest re-run.

3. Every metric recomputed by hand

┌───────────┬────────────────┬───────────┬─────────┐
│  Metric   │ Mine (by hand) │ Reported  │ |diff|  │
├───────────┼────────────────┼───────────┼─────────┤
│ ROC AUC   │ 0.9995474      │ 0.9995474 │ 5.0e-10 │
├───────────┼────────────────┼───────────┼─────────┤
│ F1        │ 0.980773       │ 0.980773  │ 5.2e-09 │
├───────────┼────────────────┼───────────┼─────────┤
│ Precision │ 0.980089       │ 0.980089  │ 2.6e-07 │
├───────────┼────────────────┼───────────┼─────────┤
│ Recall    │ 0.981457       │ 0.981457  │ 2.7e-07 │
├───────────┼────────────────┼───────────┼─────────┤
│ IoU       │ 0.962271       │ 0.962271  │ 9.9e-09 │
├───────────┼────────────────┼───────────┼─────────┤
│ MCC       │ 0.977356       │ 0.977356  │ 5.2e-09 │
└───────────┴────────────────┴───────────┴─────────┘

Largest discrepancy anywhere: 2.7e-07 — fully accounted for by that one pixel.

Two internal cross-checks passed: F1 via 2PR/(P+R) equals F1 via 2TP/(2TP+FP+FN) to 1e-12, and I computed ROC AUC with a rank-based Mann-Whitney U implementation (a genuinely different algorithm from sklearn's trapezoid-over-ROC-curve). They agree to 1.1e-16.

4. RFI fraction — three independent code paths agree exactly

┌───────────────────────────────────────────────────────────────┬──────────────┐
│                            Source                             │ RFI fraction │
├───────────────────────────────────────────────────────────────┼──────────────┤
│ (a) My confusion matrix (TP+FN)/total                         │ 15.0808%     │
├───────────────────────────────────────────────────────────────┼──────────────┤
│ (b) metadata.jsonl, written by the generator at creation time │ 15.0808%     │
├───────────────────────────────────────────────────────────────┼──────────────┤
│ (c) Direct recount from the 150 mask .npy files               │ 15.0808%     │
└───────────────────────────────────────────────────────────────┴──────────────┘

Agreement to 4+ decimals across paths that share no code. dataset_statistics.txt reports 14.67% mean over all 1000 images — correct ballpark for a 150-image subset.

5. Leakage — none found

- test appears in train_hybrid.py only inside print/help strings, never in a path that loads data.
- Splits are disjoint by filename (0 shared), by index (train 0–699, val 700–849, test 850–999), and by SHA-256 content hash (0 byte-identical images across any pair). No internal duplicates either.
- Near-duplicate check: min L2 fingerprint distance test→val = 0.7355, zero exact matches.
- Splits were assigned by the generator itself (metadata.jsonl carries a split field per record) — make_val_split.py wasn't needed here.
- Threshold provenance traced: 0.6112715602 appears in training_log.csv at epoch 22 as the validation PR-curve optimum, is copied to progress.json, and is what metrics.json used. It was not re-optimized on test.

6. Overfitting — evidence against it

┌─────────────────────────────────┬────────────┬──────────┐
│                                 │ Validation │   Test   │
├─────────────────────────────────┼────────────┼──────────┤
│ F1 @ fixed threshold 0.6113     │ 0.9802     │ 0.9808   │
├─────────────────────────────────┼────────────┼──────────┤
│ Precision                       │ 0.9840     │ 0.9801   │
├─────────────────────────────────┼────────────┼──────────┤
│ Recall                          │ 0.9764     │ 0.9815   │
├─────────────────────────────────┼────────────┼──────────┤
│ ROC AUC                         │ 0.999406   │ 0.999547 │
├─────────────────────────────────┼────────────┼──────────┤
│ Oracle (threshold-tuned) max-F1 │ 0.9802     │ 0.9810   │
└─────────────────────────────────┴────────────┴──────────┘

I re-ran validation through my own pipeline so both are measured identically. Test came out marginally higher than validation.

Why this argues against overfitting: progress.json's 0.9802 is the oracle F1 — the best achievable over all thresholds on val. Test's 0.9808 is at a fixed, pre-committed threshold. The comparison is deliberately stacked in validation's favor, and test still won. Overfitting to the validation set would show the opposite: a val number inflated by selection that collapses on truly held-out data.

The threshold also transferred well — test oracle F1 (0.9810) is only +0.0003 above the fixed-threshold result, so no meaningful tuning advantage is being left on the table.

Divergence analysis of training_log.csv: train loss decreased monotonically across all 11 logged points. Epochs where loss fell while val F1 fell: epochs 8 and 18 only — non-consecutive, longest run = 1. That's noise, not the multi-epoch divergence signature of real overfitting. Training also stopped at 22/40 planned epochs while still improving, so it never ran into a divergent regime.

Additionally: because best_epoch == 22 == final epoch, checkpoint "selection" effectively selected nothing. The winner's-curse failure mode documented in CLAUDE.md §10 structurally cannot apply here.

7. Degenerate strategies ruled out

┌────────────────────┬───────────┬────────┬────────┬────────┐
│      Strategy      │ Precision │ Recall │   F1   │  MCC   │
├────────────────────┼───────────┼────────┼────────┼────────┤
│ All-clean          │ 0         │ 0.0000 │ 0.0000 │ 0      │
├────────────────────┼───────────┼────────┼────────┼────────┤
│ All-RFI            │ 0.1508    │ 1.0000 │ 0.2621 │ 0      │
├────────────────────┼───────────┼────────┼────────┼────────┤
│ Random @ base rate │ 0.1508    │ 0.1510 │ 0.1509 │ ~0     │
├────────────────────┼───────────┼────────┼────────┼────────┤
│ Actual model       │ 0.9801    │ 0.9815 │ 0.9808 │ 0.9774 │
└────────────────────┴───────────┴────────┴────────┴────────┘

The decisive number is |precision − recall| = 0.0014. A model exploiting the 84.9/15.1 class imbalance is necessarily lopsided — it buys recall with precision or vice versa. Both being ~0.98 simultaneously cannot be achieved by any imbalance trick. MCC 0.977 vs 0.0 for both degenerate baselines confirms it, and FPR is 0.0035 (vs 1.0 all-RFI).

8. Visual spot-check — looks like real detection

panel (spotcheck_panel.png)

Columns: input spectrogram | ground truth (green) | prediction (yellow) | error map (white=TP, red=FP, blue=FN).

Predictions track visible structure in the input — horizontal bands, thin narrowband lines, a vertical burst, blobs. Crucially, the predicted masks are visibly rougher than the ground truth: ground-truth rectangles have crisp geometric edges, predictions have ragged, noisy boundaries, and the error maps show red/blue fringing concentrated at those edges. That's the fingerprint of learned segmentation. A leaked mask would be pixel-perfect.

Quantitatively:
- Images with prediction pixel-for-pixel identical to ground truth: 9/150 — and all 9 are images with no RFI at all, where the model correctly predicted nothing.
- Images with IoU exactly 1.0: 0.
- Fully-saturated (all-RFI) predictions: 0.
- Per-image IoU: mean 0.920, median 0.966.

The model has 3 genuine total misses (0927, 0949, 0988 — 15, 5, and 30 RFI pixels respectively). It is not perfect, and the failures are on physically tiny/faint targets.

The strongest single piece of evidence in the whole audit — pixel-level recall bucketed by the ground-truth RFI strength map:

┌──────────────┬────────────┬────────┐
│ RFI strength │ RFI pixels │ Recall │
├──────────────┼────────────┼────────┤
│ < 1 σ        │ 91,229     │ 0.8575 │
├──────────────┼────────────┼────────┤
│ 1–2 σ        │ 284,273    │ 0.9933 │
├──────────────┼────────────┼────────┤
│ 2–4 σ        │ 378,808    │ 0.9990 │
├──────────────┼────────────┼────────┤
│ 4–8 σ        │ 312,663    │ 0.9997 │
├──────────────┼────────────┼────────┤
│ > 8 σ        │ 124,644    │ 1.0000 │
└──────────────┴────────────┴────────┘

Perfectly monotonic. The model's accuracy degrades precisely where the physics says detection gets hard. A leaked or copied label would give ~1.0 uniformly across every bucket. This pattern can only arise from actually reading signal amplitude out of the spectrogram.

9. No architectural shortcut

- First conv weight shape: (32, 1, 3, 3) → in_channels = 1.
- HybridRFINet.forward(self, x) — takes only x. No mask parameter exists.
- grep strength across train_hybrid.py, hybrid_model.py, evaluate_hybrid_test.py: zero matches. The test/strength/ directory exists on disk but no code loads it.
- Empirical: identical input → identical output; pixel-scrambled input → output changes (mean |diff| 0.068).

The decisive proof is my own script. My inference loop loaded only test/images/*.npy and never opened a mask until after the forward pass was complete — and it reproduced F1 = 0.9808. The mask cannot be an input to a pipeline that hadn't read it yet.

---
10. Final verdict

I believe this result is genuine. The reported ROC AUC 0.9995 and F1 ~0.98 reproduce independently.

The findings that convinced me, in order of weight:

1. The strength-stratified recall curve (85.75% → 100% monotonic). No leakage or hardcoding mechanism produces graceful physics-consistent degradation.
2. Adam step 1936 = 88 × 22 exactly — optimizer state independently reconstructs the training configuration.
3. Precision and recall both ≈0.98 simultaneously, with MCC 0.977 against 0.0 for every degenerate baseline.
4. Three independent code paths agree on 15.0808% RFI to four decimals.
5. Zero content-hash overlap between splits, and the threshold traced to validation epoch 22.
6. Test ≥ validation under a comparison deliberately biased toward validation.
7. The ±1 pixel discrepancy itself — reproduction-level noise, not fabrication.

What I'd flag honestly

Two caveats, neither of which is a fraud finding:

1. I did not audit the dataset generator. I verified the number is honest; I did not verify the task is hard. dataset_generator_v3_strength.py (41KB) creates RFI as geometric primitives — lines, rectangles, blobs — additively over Gaussian noise. Looking at the panel, most RFI is high-contrast and geometrically simple. F1 = 0.98 is a real measurement on this benchmark, but the benchmark may be substantially easier than real telescope data. If you want, I can audit the generator to characterize difficulty.

2. Train and test are i.i.d. from one generator under one seed. This is normal for synthetic data and is not leakage — the splits are provably disjoint. But it means 0.98 measures in-distribution generalization only. It does not predict performance on real observations, or even on RFI morphologies the generator doesn't produce.

One thing to be careful about in how you write this up: the baseline-vs-hybrid gap (0.39 vs 0.98) is a valid controlled comparison on this dataset, but both caveats above apply equally to it. The honest claim is "the hybrid architecture substantially outperforms tf_unet on this synthetic benchmark under matched conditions" — not a general claim about RFI detection.

All verification scripts are in the scratchpad if you want to re-run or extend them.

✻ Baked for 12m 34s · 1 shell still running

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── Verify RFI detection model results independently ──
❯
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ auto mode on · ← for agents · 1 shell · ↓ to manage