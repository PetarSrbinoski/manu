# Sampling and segmentation-error follow-up — 25 October 2026

The four follow-up recommendations have implementation and analysis deliverables:
the training-only sampling audit was executed, eight error examples were reviewed,
SUVmax outliers were traced to original images, and an opt-in component-balanced
sampler plus a controlled pilot/scoring workflow were implemented. The user subsequently completed the three-arm real-cohort pilot and evaluation.
The 2026-10-26 review found improved Dice/FP volume but worse small-lesion recall;
neither candidate passed its predefined screen. See
[the completed results](SAMPLING_PILOT_RESULTS.md) and the status update below.

## 1. Completed training-only sampling audit (ticket 07)

The reference audit selected six fold-0 **training** examinations with seed
20261006 and drew 50 patches from each at 64³, 96³ and 128³. The final verified
128³ comparison uses the same six cases and 300 draws for each policy. The
2 mm training-grid references contain 118 18-connected components: 53 <1 mL,
49 at 1–<10 mL and 16 ≥10 mL. Original images and saved models were read-only.

| 128³ audit measure | Original voxel sampler | Component-balanced sampler |
|---|---:|---:|
| Total augmented draws | 300 | 300 |
| Intended foreground draws | 204 | 207 |
| Foreground draws targeting <1 mL components | 3 (1.47%) | 83 (40.10%) |
| Foreground draws targeting 1–<10 mL components | 49 (24.02%) | 108 (52.17%) |
| Foreground draws targeting ≥10 mL components | 152 (74.51%) | 16 (7.73%) |
| Small components visible at least once | 53/53 | 53/53 |
| Mean fraction of draws containing each small component | 52.75% | 54.04% |
| Draws containing background SUV ≥5 | 247/300 | 252/300 |

The important distinction is **deliberate targeting versus incidental visibility**.
Large patches already contain many nearby small components. The original sampler
rarely chooses a small component as its foreground target because it samples
foreground voxels, so large components dominate that selection. The new sampler
changes target allocation strongly, but actual small-component visibility rises
only modestly in this sample. This justifies a controlled test, not a promised
recall improvement. Selection is uniform over components within an examination,
not over all cohort components pooled together.

MONAI may shift a proposed centre to keep a crop inside the image. The verified
audit records both the intended target and the corrected centre; they should not
be conflated. Background SUV ≥5 is only a proxy for difficult uptake, not an organ
label or a clinical rule. The six-case sample is not a cohort-wide coverage audit.

The reference patch-size audit found small-component mean visibility of 27.47%
at 64³, 43.02% at 96³ and 52.75% at 128³. This measures patch exposure rather than
the model's ability to recognize or delineate those components.

An implementation defect was discovered during audit expansion: on uncached
inputs, MONAI regenerated both candidate lists if background indices were absent,
discarding the intended balanced foreground candidate. It was fixed before any
real training. Tests now verify real MONAI component selection on cached and
uncached inputs. The earlier balanced/comparison diagnostic directories contain
explicit `SUPERSEDED.json` notices and were retained for the audit trail; their
balanced results must not be used. The replacement is the verified report below.

Artifacts:

- [Verified sampling summary](runs/analysis/sampling-comparison-20261025-verified/summary.json)
- [Sampling comparison figure](runs/analysis/sampling-comparison-20261025-verified/comparison.png)
- [Per-draw audit](runs/analysis/sampling-comparison-20261025-verified/draws.csv)
- [Per-component exposure](runs/analysis/sampling-comparison-20261025-verified/lesions.csv)
- [Reference 64³/96³/128³ summary](runs/analysis/sampling-reference-20261025/summary.json)

## 2. Completed missed-lesion and regression review

Eight examinations were chosen reproducibly: the two lowest 128³ Dice cases,
the two largest Dice declines, the two largest SUVmax errors and the two largest
counts of missed components <1 mL. Review figures show CT, original SUV, reference,
96³ and 128³ outputs, with a zoomed missed component, largest missed-voxel slice,
largest false-positive slice and predicted-SUVmax slice where applicable. All
images are displayed after lossless canonical orientation; predictions are not
resampled. PET display clipping at SUV 20 changes only visualization.

| Example | Observed evidence | Interpretation |
|---|---|---|
| PETCT_0223010e46 | Reference TMTV ~0.10 mL, two tiny components; both Dice scores zero. The zoom shows a missed superficial peripheral focus. Predicted volume is ~6.81/18.23 mL for 96³/128³. | A tiny reference burden can be entirely missed while background predictions dominate. |
| PETCT_1f6b6b0548 | Reference TMTV ~1.22 mL. 128³ misses the reference focus visible near the head; the SUVmax comes from the volume boundary. | Complete reference misses and boundary false positives both need review. |
| PETCT_f9e0c504af | Dice falls 0.671→0.146. Reference volume ~19.73 mL; predicted volume rises 37.36→247.30 mL; voxelwise FP volume rises 18.21→227.87 mL. | The major decline is driven by false-positive expansion in this case, not a uniform benefit from a larger patch. |
| PETCT_97320b0b58 | Dice falls 0.906→0.616. Reference volume ~504.06 mL; predictions fall 430.66→238.27 mL. The displayed main disease region is undersegmented by 128³. | A second regression mechanism is loss of reference coverage. |
| PETCT_ea051a3e6c | 200/251 reference components <1 mL missed at 128³; the zoom shows a small rib/chest-wall-region annotation missed by both models. | Improved whole-mask Dice can coexist with many small-component misses. |
| PETCT_8a52353a72 | 149/172 reference components <1 mL missed at 128³; small head/neck and skeletal-region coverage remains incomplete on the selected slices. | Lesion count and burden should accompany Dice. |

These descriptions are exploratory image observations, not radiologist-validated
anatomical diagnoses. Review the tiny and low-uptake annotations with Igor or an
experienced reader before changing sampling or labels. Neither annotations nor
model outputs were edited. The current outer results are now development evidence;
further settings should be selected with inner validation.

Artifacts: [selection/slice inventory](runs/analysis/error-review-20261025-final/examples.json),
[tiny missed focus](runs/analysis/error-review-20261025-final/case01.png),
[false-positive regression](runs/analysis/error-review-20261025-final/case03.png),
[undersegmentation regression](runs/analysis/error-review-20261025-final/case04.png),
[many small misses](runs/analysis/error-review-20261025-final/case07.png).

## 3. Completed SUVmax source review

| All 145 examinations | 96³ | 128³ |
|---|---:|---:|
| Mean absolute SUVmax error | 43.51 | 44.26 |
| Median absolute SUVmax error | 13.24 | 11.41 |
| Reference SUVmax reproduced within 1e-6 | 31 | 43 |
| Largest error's share of total absolute error | 24.29% | 23.88% |
| Mean absolute error excluding that one case, descriptive only | 33.17 | 33.93 |

In PETCT_642d6c78d6, both masks include a peripheral limb-region hotspot with
predicted SUVmax ~1571.45, versus reference SUVmax ~39.26. The scalar is present
in the original SUV file at the recorded native voxel; the image proxy uses
slope 1/intercept 0 and the file stores float64 values. Therefore our mask metric
or model-input preprocessing did not create that extreme value. The displayed
location is compatible with injection-related uptake, but neither its clinical
cause nor the original SUV calibration is established by this review. It should
be checked with the data provenance/Igor rather than silently clipped or deleted.

In PETCT_f2f28337ba, 128³ includes a pelvic urinary-region hotspot outside the
reference. In the largest-Dice-regression case, the hottest predicted voxel is
also in a pelvic urinary region. Thus the reviewed errors have more than one
location; a blanket organ-removal rule would be poorly justified and could remove
real disease. Other reviewed cases have maxima at the acquisition boundary.

The median and exact-match counts improved with 128³ even though its mean error
slightly worsened. Report all of these; do not remove outliers from the main
score. The leave-one-case-out value above is only a sensitivity description.

Artifacts: [source coordinates/scaling](runs/analysis/error-review-20261025-final/suvmax_sources.csv),
[SUVmax summary](runs/analysis/error-review-20261025-final/suvmax_summary.json),
[largest outlier figure](runs/analysis/error-review-20261025-final/case05.png),
[second outlier figure](runs/analysis/error-review-20261025-final/case06.png).

## 4. Implemented next experiment (ticket 08)

The original voxel sampler remains the default. The opt-in lesion-balanced
sampler, bounded derived-index cache, scientific metadata and strict compatibility
checks are implemented. Existing 96³/128³ metadata passes the exact known-source
voxel continuation check without saved-file changes. Unknown source histories,
changed samplers and changed scientific settings still fail compatibility checks.

The paired pilot uses fresh 128³ fold-0 models, unchanged patient assignments,
the original 500-epoch schedule and the same initialization seed. Both stop for
inspection at epoch 50 before outer inference. A frozen protocol records the
primary endpoint and guards before training; a separate command exports complete
inner-validation masks, scores both arms and applies the screen. See
[the pilot plan](SAMPLING_PILOT_PLAN.md) for definitions and commands.

**Subsequent completion update (reviewed 2026-10-26):** The user completed the
three-arm 128³ pilot and evaluation. All arms reached epoch 50; all evaluation
manifests report completion on 23 inner-validation examinations. The original
implementation-only status is superseded by [the results review](SAMPLING_PILOT_RESULTS.md).
Neither sampling nor loss met its predeclared screen because small-lesion recall
fell, despite higher Dice and lower unmatched FP volume. No new five-fold recipe
has been selected. The CPU audits and prior 96³/128³ review above remain separate.

**Subsequent requested extension:** ticket 12 adds a third arm with the same
lesion-balanced sampler and 1 × Dice + 1 × cross-entropy. The launcher now defaults
to all three arms at 128³, under a separate sampling/loss pilot root; it does not
repeat 96³. B/A measures sampling and C/B measures loss, with loss-specific guards
frozen before launch. The pilot plan contains the current commands and output
layout; the completed CPU audits and error-review findings above are unchanged.
