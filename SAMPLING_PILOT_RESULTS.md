# Completed 128³ sampling and loss pilot review

Reviewed from local artifacts on 2026-10-26. The user completed training and
evaluation. This records a completed bounded experiment, not a successful
promotion of either candidate to five-fold training.

## Verified scope

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

## Results

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

## Predeclared screens and interpretation

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

## Next work (proposed, not launched or implemented)

1. Review paired predictions for lesions A detects but B/C miss, plus false
   positives. Record representative overlays and whether failures cluster by size.
2. Consider a fourth matched 128³ arm: original voxel sampling + Dice/CE. This
   would test the loss without the balanced sampler; the current C/B comparison
   only establishes its effect under balanced sampling. Freeze its protocol before
   launch rather than editing the completed three-arm protocol.
3. Reassess whether a separately specified longer pilot is justified before any
   new five-fold training. Preserve the completed checkpoints, schedule and reports.

## Local evidence

- [Frozen protocol](runs/sampling-loss-pilot/pilot_protocol.json)
- [Comparison and criteria](runs/sampling-loss-pilot/inner-review/comparison.json)
- [Evaluation manifest](runs/sampling-loss-pilot/inner-review/manifest.json)
- [A metrics](runs/sampling-loss-pilot/inner-review/voxel-metrics/summary.json)
- [B metrics](runs/sampling-loss-pilot/inner-review/lesion-balanced-metrics/summary.json)
- [C metrics](runs/sampling-loss-pilot/inner-review/lesion-balanced-dice-ce-metrics/summary.json)

Artifacts remain local under `runs/`; no GitHub publication or new training was
performed during this documentation update.
