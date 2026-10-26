# architecture-cleanup-and-pilot-results — experiment summary

This combines the completed 96³/128³ five-fold training comparison (reviewed
2026-10-25) and the subsequent 128³ sampling/loss pilot (reviewed 2026-10-26).
The five-fold results are held-out evaluations across 145 examinations; the pilot
uses 23 fold-0 inner-validation examinations. Their scores are not directly
comparable. Sizes refer to training patches; inference covers full scans.

The 128³ five-fold run improved mean Dice from 0.4865 to 0.5577, while small-lesion
recall remained weak. The subsequent sampling/loss candidates improved pilot Dice
and reduced unmatched false-positive volume, but both failed their predeclared
screens because small-lesion recall fell. Retain the original completed 128³
five-fold experiment as the baseline.

This positive-only cohort does not establish performance on lesion-negative
examinations or patient-level lymphoma diagnosis. These comparisons do not
establish statistical significance.

## Full training runs: 96³ versus 128³

The runs use patient-level five-fold cross-validation with separate inner
validation for checkpoint selection and shared patient assignments. Outputs are
under `runs/patch-comparison/patch96/` and `runs/patch-comparison/patch128/`.
The configured recipe uses a 500-epoch cosine schedule with early stopping;
completion does not imply every fold trained for all 500 epochs. See
[training setup and commands](README.md#compare-96³-and-128³-patches-across-all-five-folds).

Both CPU evaluation manifests are complete: 145 examinations / 144 patients per model. Case IDs and reference lesion counts, TMTV and SUVmax agree between reports. This is a descriptive comparison of one training realization per configuration; no significance claim is made.

| Measure | 96³ | 128³ |
|---|---:|---:|
| Mean Dice | 0.4865 | 0.5577 |
| Mean unmatched FP volume (mL) | 70.7267 | 47.6451 |
| Mean completely missed lesion volume (mL) | 7.7478 | 8.8304 |
| Mean voxelwise FP volume (mL) | 107.6297 | 83.9711 |
| Mean voxelwise FN volume (mL) | 121.8691 | 118.1372 |
| Mean absolute TMTV error (mL) | 143.7949 | 135.9273 |
| Mean absolute SUVmax error | 43.5081 | 44.2584 |
| Any-overlap lesion recall | 44.98% | 44.43% |
| Any-overlap predicted-component precision | 38.42% | 44.65% |
| IoU ≥0.5 lesion recall | 17.55% | 19.52% |

| Reference component volume | Number | Recall 96³ (any overlap) | Recall 128³ (any overlap) |
|---|---:|---:|---:|
| <1 mL | 1957 | 19.37% | 19.16% |
| 1–<10 mL | 980 | 76.22% | 75.20% |
| ≥10 mL | 367 | 98.09% | 97.00% |

128³ has higher Dice in 89 examinations, lower Dice in 55, and a tie in 1. Mean difference: +0.0712.

Interpretation: 128³ improves overlap and reduces false-positive burden. It does not improve overall any-overlap lesion recall or small-lesion recall. SUVmax mean absolute error is slightly worse, although its median is lower. TMTV absolute error improves modestly. A better Dice score should not be described as better lesion detection across the board.

The dominant remaining detection weakness is the <1 mL group: 128³ detects 375/1957 reference components by any overlap (19.16%) and only 114/1957 at IoU ≥0.5 (5.83%). These are exploratory volume bins on native reference components, not clinical lesion categories.

FPV/FNV count whole unmatched components; voxelwise FP/FN quantify additional delineation errors. A low unmatched FNV does not mean most tumour voxels were segmented.

The next steps identified on 2026-10-25 were to inspect small-lesion misses, severe Dice declines and SUVmax outliers, run the training-only sampling audit, and design a separate lesion-balanced sampling pilot using inner validation. The resulting completed pilot is documented below. Existing outer results inform development and must not become repeated tuning targets.

### Worst 128³ cases by Dice

| Case | Dice 96³ | Dice 128³ |
|---|---:|---:|
| PETCT_0223010e46/09-04-2003-NA-PET-CT Ganzkoerper  primaer mit KM-25360 | 0.0000 | 0.0000 |
| PETCT_1f6b6b0548/10-16-2003-NA-PET-CT Ganzkoerper  primaer mit KM-05707 | 0.0124 | 0.0000 |
| PETCT_15f4b7254f/10-11-2003-NA-PET-CT Ganzkoerper  primaer mit KM-99526 | 0.0050 | 0.0021 |
| PETCT_0f44cec2e6/09-07-2003-NA-PET-CT Ganzkoerper  primaer mit KM-99224 | 0.0028 | 0.0049 |
| PETCT_74bbceaeeb/06-03-2005-NA-PET-CT Ganzkoerper  primaer mit KM-94697 | 0.1191 | 0.0072 |

### Largest Dice declines

| Case | Dice 96³ | Dice 128³ |
|---|---:|---:|
| PETCT_f9e0c504af/06-29-2003-NA-PET-CT Ganzkoerper  primaer mit KM-14170 | 0.6707 | 0.1455 |
| PETCT_97320b0b58/04-17-2005-NA-PET-CT Ganzkoerper  primaer mit KM-10200 | 0.9064 | 0.6155 |
| PETCT_8b74860ce6/03-10-2005-NA-PET-CT Ganzkoerper  primaer mit KM-00433 | 0.6630 | 0.4070 |
| PETCT_0beb67c923/07-25-1999-NA-PET-CT Ganzkoerper  primaer mit KM-37911 | 0.8110 | 0.6271 |
| PETCT_ae3074834a/07-26-2003-NA-PET-CT Ganzkoerper  primaer mit KM-46217 | 0.9191 | 0.7421 |

Source reports: [96³ evaluation](runs/analysis/evaluation96/summary.json) and [128³ evaluation](runs/analysis/evaluation128/summary.json). No training or inference was launched during this comparison.

## Sampling and loss pilot: 128³

Reviewed from local artifacts on 2026-10-26. The user completed training and
evaluation. This records a completed bounded experiment, not a successful
promotion of either candidate to five-fold training.

### Verified scope

All three fold-0 histories contain epochs 1–50. Best and last checkpoints are
present. Prediction and evaluation manifests report completion on the same 23
inner-validation examinations/patients, with split identity
`a506bc3420cb53717fe841944953335c2cde1a8f5c6f9e407d2260864324a021`.
The original 500-epoch cosine schedule was retained; training paused at epoch 50.
No 96³ rerun was required. This evaluation is inner validation, not outer-test
performance or a completed new five-fold experiment.

A uses original voxel sampling and Dice. B uses lesion-balanced sampling and
Dice. C uses lesion-balanced sampling and 1 × Dice + 1 × cross-entropy.
Selected checkpoints were A epoch 46, B epoch 50, C epoch 50. Selection uses
2 mm inner-validation Dice; the table below reports native-grid evaluation.

### Results

| Metric | A: Original | B: Balanced | C: Balanced + Dice/CE |
|---|---:|---:|---:|
| Mean foreground Dice ↑ | 0.5161 | 0.5308 | 0.5651 |
| Mean unmatched FP volume (mL) ↓ | 55.4972 | 46.3168 | 37.0412 |
| Mean unmatched FN volume (mL) ↓ | 6.5018 | 10.5467 | 11.4868 |
| Mean absolute TMTV error (mL) ↓ | 254.0932 | 268.4129 | 248.4840 |
| Mean absolute SUVmax error ↓ | 123.6972 | 120.9520 | 122.3647 |
| Small (<1 mL) lesions detected ↑ | 37/348 (10.63%) | 30/348 (8.62%) | 17/348 (4.89%) |
| Medium (1–<10 mL) lesions detected ↑ | 104/140 (74.29%) | 92/140 (65.71%) | 90/140 (64.29%) |
| Large (≥10 mL) lesions detected ↑ | 55/56 (98.21%) | 52/56 (92.86%) | 55/56 (98.21%) |
| All-lesion any-overlap recall ↑ | 36.03% | 31.99% | 29.78% |
| All-lesion IoU ≥0.5 recall ↑ | 15.07% | 12.87% | 11.03% |

Lesions are 18-connected components. C1 detection means any overlap. FP/FN
volume counts entire unmatched components, not all voxelwise errors. Pooled
recall weights lesions rather than patients. Higher Dice is not a percentage
of lesions detected. SUVmax errors are in the native SUV scale.

### Predeclared screens and interpretation

- **B versus A (sampling): failed.** Dice increased by 0.01465 and mean FP volume
  fell from 55.50 to 46.32 mL, but small-lesion recall fell by 2.01 percentage
  points instead of achieving the required 5-point gain. Both guardrails passed.
- **C versus B (loss): failed.** Mean FP volume fell by 20.03% and Dice increased
  by 0.03432. Small-lesion recall fell by 3.74 percentage points, exceeding the
  allowed 2-point drop. The FP-volume and Dice criteria passed.
- **C versus A (combined, descriptive):** Dice increased by 0.04897 and FP volume
  fell by 33.26%, but small lesions detected fell from 37 to 17. Unmatched
  predicted component counts increased from 363 to 412: lower FP volume does
  not imply fewer false-positive components.

Neither change met its frozen screen. Retain the original completed 128³
five-fold experiment as the baseline; do not promote a candidate based on Dice
alone. This one-fold, one-seed, 50-epoch comparison does not establish statistical
significance or final generalization. B and C selected the last available epoch,
and recent validation scores fluctuate, so convergence is not established.
Scores here should not be directly compared with the earlier 145-case outer-fold
scores as though they used the same patients and training budget.

### Next work (proposed, not launched or implemented)

1. Review paired predictions for lesions A detects but B/C miss, plus false
   positives. Record representative overlays and whether failures cluster by size.
2. Consider a fourth matched 128³ arm: original voxel sampling + Dice/CE. This
   would test the loss without the balanced sampler; the current C/B comparison
   only establishes its effect under balanced sampling. Freeze its protocol before
   launch rather than editing the completed three-arm protocol.
3. Reassess whether a separately specified longer pilot is justified before any
   new five-fold training. Preserve the completed checkpoints, schedule and reports.

### Local evidence

- [Frozen protocol](runs/sampling-loss-pilot/pilot_protocol.json)
- [Comparison and criteria](runs/sampling-loss-pilot/inner-review/comparison.json)
- [Evaluation manifest](runs/sampling-loss-pilot/inner-review/manifest.json)
- [A metrics](runs/sampling-loss-pilot/inner-review/voxel-metrics/summary.json)
- [B metrics](runs/sampling-loss-pilot/inner-review/lesion-balanced-metrics/summary.json)
- [C metrics](runs/sampling-loss-pilot/inner-review/lesion-balanced-dice-ce-metrics/summary.json)

Artifacts remain local under `runs/`; no GitHub publication or new training was
performed during this documentation update.
