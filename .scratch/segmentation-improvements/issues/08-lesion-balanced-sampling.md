# 08: Evaluate lesion-balanced sampling

**What to build:** Train and evaluate a configurable sampler that selects reference lesions before choosing foreground patch centres.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 07: Audit training-patch coverage.

**Status:** completed bounded pilot and evaluation — candidate failed the predefined screen

- [x] Preserve background sampling, paired spatial transforms and the current sampler as a selectable reference.
- [x] Verify component selection and patch targeting on synthetic cases containing lesions of different sizes.
- [x] Run an authorized bounded inner-validation comparison with recorded update/time budgets and unchanged non-sampling settings.
- [x] Report small-lesion recall, false positives and whole-mask metrics; reject unsupported claims based solely on sampling frequencies.

**Implementation evidence (2026-10-25):** The new policy is opt-in and recorded
as a scientific setting. It has bounded per-process index caching and strict
resume rejection when switching policies. The exact pre-change source may
continue only with voxel sampling; default seeded outputs are bitwise unchanged.
Tests cover real MONAI cached/uncached selection, the 2:1 background policy,
paired geometry, cache limits, CPU training/resume and frozen pilot criteria.

A matched 128³ fold-0 launcher now includes control, sampling and sampling+loss
arms and pauses all three at epoch 50 of
the unchanged 500-epoch schedule. A separate evaluation command exports/scores
the complete inner-validation sets and applies a predeclared screen (+5 percentage
points small-component recall, at most +10% mean FPV, at most 0.02 Dice loss).
The protocol is frozen before training; outer inference is excluded at this
inspection point. See the pilot plan. The real-cohort comparison has now completed; see the outcome below.

**Completed outcome (reviewed 2026-10-26):** The user trained all three arms
to epoch 50 and evaluated the same 23 inner-validation examinations.
B/A increased mean Dice from 0.51613 to 0.53078 and reduced mean FPV
from 55.50 to 46.32 mL. Small-lesion recall fell from 37/348 to 30/348
(−2.01 percentage points), failing the required +5-point gain.
Ticket completion records the experiment, not acceptance of the candidate.
See [results and limitations](../../../SAMPLING_PILOT_RESULTS.md).
No new five-fold run is selected.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
