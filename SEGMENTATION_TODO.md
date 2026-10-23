# Segmentation improvement TODO

Proposed experiments following the October 2026 baseline review. Unchecked items remain proposals; completion updates appear below and in the
local tracker. Listing a proposal does not authorize a training launch. Our main observed weaknesses
are false positives in regions of physiological PET uptake and missed small
lesions. No proposed change is guaranteed to improve segmentation.

## Rules for every experiment

- Preserve the existing runs, checkpoints and original images. Use a separate
  output directory and record the source commit, configuration and split identity.
- Keep configurable defaults in `config.py`. Record scientific changes explicitly;
  hardware changes must not silently change the recipe.
- Screen ideas using one outer fold's training and separate inner-validation
  patients. Never select settings using that fold's outer test patients. Keep
  repeat visits together. A one-fold pilot screens ideas; it is not a final result.
- Change one factor at a time initially. Compare optimizer updates and wall-clock
  time as well as epochs, especially when changing patch size or sampling.
- Preserve quantitative SUV, CT/PET alignment, full field of view and native-grid
  prediction export. Keep numerical recovery and strict resume checks.
- Decide acceptance criteria before each experiment. Track foreground Dice,
  lesion recall by size, false-positive counts/volumes and TMTV/SUVmax errors.
  A Dice improvement must not hide unacceptable loss of small-lesion recall.
- Current outer-fold results have already informed these ideas. Treat subsequent
  improvements as development; reserve untouched external data for stronger
  generalization claims.

## 1. Establish a trustworthy reference

- [ ] **Recover the completed B experiment from the colleague.** Obtain `run.json`,
  `splits.json`, all five histories, completion records and metrics, exact source
  identity, migration records, checkpoint checksums and 145 native-grid predictions.
  Keep it separate from this machine's unfinished `runs/full-training` experiment.

- [ ] **Verify the completed experiment.** Check patient separation, checkpoint
  selection using inner validation only, consistent scientific settings across
  folds, and exactly one correctly aligned prediction per examination. Recompute
  summary metrics before using this run as the reference for improvements.

- [ ] **Review learning curves and representative errors.** Plot training loss,
  inner-validation Dice and learning rate with best/stopping epochs marked. Review
  inner-validation overlays for small lesions, extensive disease and physiological
  uptake. Check whether the evidence suggests inadequate learning, unstable
  validation or overfitting rather than assuming a cause.

- [ ] **Make the evaluation reusable and test its edge cases.** Bring the shared
  evaluator from the colleague's branches into the training workflow with explicit
  input paths. Add synthetic known-volume, empty-mask, split/merge matching and
  SUVmax-tie cases. Document 18-connectivity and distinguish unmatched-component
  FPV/FNV from voxelwise errors; partially detected lesions can have zero FNV.

## 2. Give the model more anatomical context

- [ ] **Benchmark larger patches.** Compare memory and throughput for 64³, 96³ and
  128³ patches with the same 2 mm spacing and architecture. Use bounded diagnostics
  in separate directories. Test 160³/192³ only if suitable hardware is available;
  do not assume they fit the 6 GB laptop GPU or reduce spacing resolution to fit.

- [ ] **Run a controlled larger-patch pilot.** Compare the current 64³ recipe with
  one feasible larger patch on the same training/inner-validation patients. Record
  update budget, time and sampled voxel exposure. Test whether the extra context
  reduces organ-region false positives without worsening small-lesion recall.

## 3. Improve training examples and sampling

- [ ] **Audit what the current sampler sees.** On training patients only, measure
  how often individual lesions and lesion-size groups appear in patches, and how
  often background patches contain difficult physiological uptake. Retain the
  current 2:1 foreground/background sampler as the reference.

- [x] **Test lesion-balanced sampling.** Sample reference connected components
  before selecting a foreground patch centre, so large lesions do not dominate
  all foreground draws. Preserve background sampling and compare small-lesion
  recall and false positives against the existing sampler.

- [ ] **Test hard-negative sampling separately.** Use predictions on training
  patients to identify non-reference regions the model mistakes for tumour. Mix
  patches from those locations with ordinary foreground/background patches.
  Audit suspicious labels and never mine examples from outer test patients.

- [ ] **Test multiple patches per examination.** Draw several independently
  augmented patches from each loaded volume to improve lesion/background coverage.
  Keep caching deterministic preprocessing only. Record the increased number of
  optimizer updates; equal epoch counts would no longer mean equal training work.

## 4. Test training schedule and loss

- [ ] **Investigate the stopping/schedule interaction.** After reviewing the B
  histories, test whether a revised learning-rate schedule or stopping policy
  improves inner-validation results. Specify each change before launching and
  change one factor at a time. Start a new experiment: changing the total epoch
  budget changes the cosine schedule and is not a compatible continuation.

- [x] **Compare Dice with Dice plus cross-entropy.** Add a configurable loss choice
  and explicit weighting while retaining stable float32 loss reductions and AMP
  recovery. Hold data, sampling, patch and optimization settings fixed. Check
  whether the extra voxelwise penalty reduces false positives without suppressing
  small lesions; do not assume the combined loss will win.

## 5. Add anatomical information if simpler changes are insufficient

- [ ] **Prepare and validate organ information.** Evaluate an organ-segmentation
  tool on training/inner-validation CT images and verify output alignment. Record
  model provenance, label definitions and extraction failures. This is additional
  preprocessing work, not an already available feature.

- [ ] **Test one anatomy-aware model variant.** Compare either organ-mask input
  channels or an auxiliary organ-segmentation output with the strongest simple
  baseline. For an auxiliary output, define the additional loss and weighting;
  for input channels, supply the same information at inference. Measure both
  false-positive reduction and retained lesion recall.

- [ ] **Assess postprocessing cautiously, only as a separate experiment.** If
  testing organ-aware or component-size rules, choose them on inner validation
  and audit removed true lesions. Do not blindly delete all predictions inside
  organs or all small components: both can contain genuine disease.

## 6. Establish a stronger comparator and final evaluation

- [ ] **Train an nnU-Net comparator on our patient assignments.** Configure CT/PET
  inputs and use the same training, inner-validation and outer-test patients.
  Preserve independent checkpoint selection and audit preprocessing for retained
  lesions. Record architecture, compute and training differences; treat this as
  a competing baseline rather than a guaranteed upgrade.

- [ ] **Consider lesion-negative examinations as a separate data experiment.**
  First define the expanded cohort, patient splits, empty-mask behavior and
  negative-case sampling. Report false positives on lesion-negative scans. Such
  examinations are not necessarily healthy patients, and adding them requires
  explicit dataset/download authorization.

- [ ] **Freeze and evaluate the selected recipe across five folds.** Once pilot
  choices are settled, run all folds consistently in a new output directory.
  Verify checkpoints and 145 native-grid out-of-fold predictions. Report Dice,
  lesion-size results, false positives, TMTV/SUVmax errors and patient-aware
  uncertainty. Repeat seeds if resources permit to assess training variability.

- [ ] **Plan an independent comparison with the pretrained checkpoint.** Obtain
  an untouched external cohort with audited training-data non-overlap, or train
  the comparison method within our patient splits. Do not interpret the current
  overlapping autoPET III results as an independent target score.

## Suggested execution order

Artifact handoff and curve review → patch-size benchmark and pilot → sampling
audit and separate sampling pilots → schedule/loss pilots → anatomy-aware variant
if needed → stronger comparator → frozen five-fold evaluation.

GenAI remains a later meeting decision, not a proposed change to this training
loop. Define its role, data, comparator and measurable endpoint with Igor before
implementing it.

## References

- `future.md`: earlier local proposals; some historical status/resource notes are
  stale, so use current configuration and actual run artifacts.
- [Official autoPET III implementation](https://github.com/MIC-DKFZ/autopet-3-submission):
  large contextual patches and joint organ/lesion supervision motivate controlled
  experiments, but do not establish the cause of our observed performance gap.
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet): comparator implementation.

## Completed pilot outcome (reviewed 2026-10-26)

The user completed three 128³ fold-0 pilots through epoch 50 and evaluated all
23 inner-validation examinations. Tickets 08 and 12 are complete as bounded
comparisons, but neither candidate passed its predefined screen. Dice and FP
volume improved while small-lesion recall declined. Checked experiment items
do not mean adoption of their candidate recipes. See
[results, limitations and proposed next work](SAMPLING_PILOT_RESULTS.md).
No 96³ rerun or new five-fold candidate experiment was performed.
