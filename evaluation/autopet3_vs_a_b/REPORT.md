# Lymphoma lesion segmentation on FDG PET/CT: A vs B vs the autoPET III reference (10 examinations)

**Date:** 2026-10-19 · **Data:** autoPET FDG-PET-CT-Lesions, lymphoma subset · **n = 10 examinations of 10 patients**

> **Disclosure: training-data overlap.** Model C is the official pretrained autoPET III checkpoint. It was
> trained on autoPET data that **include these 10 examinations**. C has therefore seen these images and their
> reference masks during training. Its results are **not an independent generalization benchmark**, and they
> are not comparable with the out-of-fold results of A and B as a held-out test. C serves only as an
> orientational reference: an optimistic estimate of what a mature, large-scale pipeline reaches on data it
> was fitted to. No statistical significance is reported: the sample is small (n = 10), and the comparison
> is confounded by the training exposure.

## 1. Models

| | Model | How the 10 predictions were obtained |
|---|---|---|
| **A** | Microsoft `lymphoma-segmentation-dnn` UNet, 10 epochs, 5-fold patient-level CV | Out-of-fold: each examination was predicted by the fold model that did not train on it |
| **B** | `testTraining` pipeline, 500-epoch budget with early stopping, 5-fold patient-level CV | Out-of-fold, as for A, with its own seed-20261006 patient folds |
| **C** | autoPET III winning submission (MIC-DKFZ), `autoPET3_Trainer`, ResEnc-UNet with organ head (`ResidualEncoderUNetOrgan`, MultiTalent plans), patch 192³ | Official inference: ensemble of all 5 folds, mirroring TTA, step 0.5. Run unmodified from `checkpoints/autopet3/` |

A and B were trained on the 145 lymphoma examinations only. C was trained on the full autoPET training data
(including these lymphoma cases), so it differs from A/B in data volume, architecture, training length and data exposure.

## 2. Examinations

Ten patients were drawn from the 144 sorted lymphoma patients with `random.Random(42).sample`, taking the first
examination per patient (`experiments/autopet3_subset10/manifest.json`; the draw was recomputed and matches).
The reference contains 348 lesions (18-connected), median 8.5 per examination. Reference TMTV ranges from 6.7 to 2481 ml.
The subset matches the full cohort in median lesion count (8.5 vs 9). Its median reference TMTV is somewhat larger
(206 vs 163 ml). A scores about the same on the subset as on all 145 (median Dice 0.26 vs 0.27). B scores somewhat higher
on the subset (0.36 vs 0.26). Both models have slightly higher mean Dice here, consistent with the larger tumour burden. See
`results_tables.md`, section "Is the 10-case subset representative".

## 3. Methods

All metrics come from the same code as the completed A vs B evaluation (`evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py`,
unmodified, imported). That code uses the definitions of Microsoft's `metrics/metrics.py`, and an earlier check against the
original Microsoft functions matched 176/176 values. It evaluates on the native PET grid with 18-connectivity:

- **Dice** on binary masks; **FPV** = volume of predicted components that touch no reference lesion; **FNV** = volume of reference lesions touched by no prediction.
- **Lesion detection**, three criteria: C1 any overlap, C2 IoU ≥ 0.5, C3 the prediction covers the SUVmax voxel of the reference lesion.
  Recall = share of reference lesions found; precision = TP / (TP + FP) over predicted components.
- **TMTV** (voxels × voxel volume) and **SUVmax** inside the predicted and the reference mask, plus their errors.

**Matching and validation.** The 10 examination IDs come from the manifest. For each examination, A, B and C are scored against
the same reference mask and SUV image. Every mask was checked to lie on the SUV grid (size, spacing, origin, direction)
and to be strictly binary (40/40 passed). A and B rows were taken from the stored `per_exam_metrics.csv`.
As a consistency check, **all 20 A/B evaluations were recomputed; 480/480 values match exactly.** All 10 C predictions
were verified to be complete (full gzip stream readable) before scoring.

## 4. Results

### 4.1 Overlap and volume errors (per examination, median [IQR])

| Metric | A | B | C* |
|---|---|---|---|
| Dice, mean (range) | 0.35 (0.01–0.75) | 0.33 (0.02–0.65) | 0.92 (0.85–0.98) |
| Dice, median | 0.26 [0.15, 0.62] | 0.36 [0.16, 0.45] | 0.93 [0.91, 0.95] |
| False-positive volume (ml) | 590 [340, 983] | 184 [106, 280] | 0.14 [0, 1.4] |
| False-negative volume (ml) | 0.42 [0.003, 1.7] | 1.3 [0.09, 8.3] | 0.11 [0.003, 0.42] |
| TMTV absolute error (ml) | 626 [453, 1010] | 259 [177, 470] | 10 [0.6, 16] |
| TMTV signed relative error | +4.55 | +1.08 | +0.01 |
| TMTV Spearman ρ (pred vs ref) | 0.70 | 0.67 | 1.00 |
| SUVmax: predicted = reference | 1/10 | 1/10 | 8/10 |
| SUVmax: predicted > reference | 9/10 | 9/10 | 2/10 |
| Predicted lesions per exam | 71.5 | 47 | 9.5 |

\* C has seen these examinations during training.

### 4.2 Lesion-level detection (pooled over 348 reference lesions)

| Criterion | A recall / precision / F1 | B recall / precision / F1 | C* recall / precision / F1 |
|---|---|---|---|
| C1 any overlap | 0.52 / 0.13 / 0.21 | 0.39 / 0.45 / 0.42 | 0.74 / 0.89 / 0.81 |
| C2 IoU ≥ 0.5 | 0.24 / 0.05 / 0.09 | 0.13 / 0.09 / 0.10 | 0.53 / 0.62 / 0.57 |
| C3 SUVmax voxel | 0.44 / 0.10 / 0.16 | 0.36 / 0.25 / 0.29 | 0.68 / 0.78 / 0.72 |
| False-positive lesions (C1), total | 1355 | 278 | 33 |

Plots: `plots/dice_per_exam.png`, `plots/fpv_fnv_per_exam.png`, `plots/detection_pooled.png`, `plots/tmtv_suvmax_agreement.png`.
Per-examination values: `per_exam_metrics_10.csv`; all tables: `results_tables.md`.

### 4.3 Observations

1. **C is almost uniformly high.** Dice is 0.85–0.98 in every examination, independent of tumour burden. A and B
   range from near 0 (small-burden cases) to 0.65–0.75 (high-burden cases).
2. **The A/B vs C gap is driven mainly by false positives.** FPV differs by about three orders of magnitude
   (A 590 ml, B 184 ml, C 0.14 ml median). FNV differs far less (all three medians are below 1.5 ml). Every predicted
   component of A/B that touches no lesion adds directly to TMTV, which is why the TMTV errors of A and B are large.
3. **SUVmax.** For C, the predicted SUVmax equals the reference SUVmax in 8/10 examinations. For A and B it does so in
   1/10, and in 9/10 it is higher. This means the hottest predicted voxel lies outside the reference lesions, which is
   consistent with false positives in physiologically high-uptake tissue. The anatomical location was not checked.
4. **Missed lesions remain for C as well.** In the two high-burden examinations (157 and 125 reference lesions),
   C finds 68% and 81% of lesions (C1). Its errors there concentrate on small lesions, while the missed volume stays
   small (FNV 4.2 and 2.2 ml).
5. **The A vs B trade-off on these 10 cases agrees with the 145-case evaluation.** A is more sensitive (C1 recall 0.52 vs 0.39)
   but produces about 5× more false-positive lesions. B has lower FPV and TMTV error. Mean Dice is similar (0.35 vs 0.33).

## 5. Interpretation and limitations

- **Training-data overlap (dominant caveat).** C's near-perfect agreement can be explained at least in part by having been
  trained on these exact examinations and reference masks. With the available data it is impossible to separate how much
  of C's advantage comes from better methodology (architecture, organ supervision, more data, longer training) and how much
  from exposure. C's numbers are an upper reference for orientation, not a fair competitor to A and B.
- **Not a statistical comparison.** With n = 10 and confounded exposure, no tests, p-values or confidence intervals
  are reported. Even the A vs B differences here are descriptive; the 145-case evaluation (answer 27) is the reference for A vs B.
- **Different training regimes.** A and B used 145 lymphoma scans. C used the full autoPET training set (lymphoma, melanoma, lung
  cancer and negative controls), a larger network and much longer training.
- **One training run per configuration** for A and B. Training variance is not captured.

## 6. Conclusions for the supervisor

1. Our two baselines (A, B) are far from the autoPET III reference on the same 10 examinations (mean Dice about 0.34 vs 0.92).
   Because C was trained on these cases, the gap overstates what a fair held-out comparison would show. It is an orientation, not a benchmark.
2. The main failure mode of A and B is **false-positive segmentation**, not missed tumour. Their FPV is hundreds of ml, while C's is
   below 1 ml. Their FNV is of the same order as C's. This false-positive volume is what inflates TMTV and corrupts SUVmax.
   **Suppressing false positives** is therefore the most direct way to improve the clinically relevant measures. Possible routes, in our
   reading of the C pipeline: more training data including negative scans, anatomical/organ context, larger patch/model, and longer training.
3. Lesion delineation at the single-lesion level (criterion C2, IoU ≥ 0.5) is weak for A and B (recall 0.13–0.24), while C reaches 0.53.
   Fine delineation of small lesions remains hard even for C.
4. For a fair comparison with a strong pretrained model, it would need to be evaluated on lymphoma examinations it was **not**
   trained on (external data), or the autoPET III pipeline would need to be retrained on our patient-level folds. Neither was done here.

## 7. Reproducibility

- Inference: `autopet_run_subset.py` (copied unchanged from branch `baseline-10-epoch-run`), log in `experiments/autopet3_subset10/inference.log`,
  predictions in `experiments/autopet3_subset10/predictions/`. Official repository and checkpoint unmodified.
- Evaluation: `~/venvs/lymphoma_seg/bin/python evaluation/autopet3_vs_a_b/evaluate_autopet3_subset.py` (refuses to overwrite).
  `compute_manifest.json` records script hashes (this script, `evaluate_a_vs_b.py`, `summarize_a_vs_b.py`, Microsoft `metrics.py`),
  the input CSV hash, the SHA-256 of all 10 C predictions and the package versions. Console output: `compute_log.txt`.
