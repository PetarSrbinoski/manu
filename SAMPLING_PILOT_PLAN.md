# Matched lesion-sampling pilot

The completed patch comparison improved Dice and false-positive burden with
128³ patches, but pooled any-overlap recall of reference components <1 mL remained
about 19%. This is a proposal to test one additional factor: how foreground
patch candidates are selected. It is not evidence that the new sampler improves
segmentation. Original images and the completed experiments are preserved.

## Implemented sampler

`--sampling voxel` retains the original behavior. `--sampling lesion-balanced`
selects an 18-connected reference component uniformly within the currently
selected examination, then selects one of that component's voxels uniformly.
The MONAI foreground/background decision retains the configured 2:1 weights;
background candidates remain GT=0 and PET above the existing threshold. The
original boundary correction, padding and paired affine augmentation still apply.

Both foreground and background index lists are supplied explicitly. This matters:
MONAI regenerates both lists when either is absent, which would otherwise discard
the chosen component on uncached volumes. Regression tests exercise the real
MONAI crop for both cached and uncached inputs, with a one-voxel lesion and a
1,000-voxel lesion. Target selection is approximately equal, rather than
proportional to volume. The original voxel outputs match four pinned seeded
crop/augmentation snapshots bit-for-bit.

Component indices use a per-loader-process byte-bounded RAM cache (default
0.25 GiB) containing foreground indices and offsets, not full images. This is
additional to the existing preprocessing cache. Random crops/augmentation remain
uncached. MONAI's transform seeding integrates with the existing epoch-derived
seeds; CPU integration verifies identical resumed and uninterrupted training.
The component-index algorithm/version and cc3d version are recorded for the new
sampling policy. GPU/DDP training of the new policy is not yet measured.

Sampling is a scientific setting in checkpoint metadata. The exact known
pre-sampler source hash can migrate to current **voxel** behavior; changed patch
size, sampler, data, versions or unknown source histories still fail resume
checks. Completed 96³/128³ run metadata passes this compatibility check without
modifying its saved files. The default sampling policy remains voxel.

## Frozen comparison

| Item | Both arms |
|---|---|
| Patch | 128³, 2 mm grid, full field of view |
| Fold | 0, existing patient-level train/inner-validation assignments |
| Network/loss/optimizer | Existing residual U-Net, background-inclusive Dice, AdamW |
| Schedule | Original 500-epoch cosine schedule, starting LR 2e-4 |
| Inspection point | Pause after epoch 50; early stopping cannot trigger before epoch 100 |
| Update budget | One patch per training examination per epoch, batch 1, accumulation 1 |
| Selection | Highest inner-validation foreground Dice within the first 50 epochs |
| Outer test | No outer inference at the pilot pause or during pilot evaluation |
| Changed factor | Foreground candidate distribution: voxel versus uniform component |

Both arms start from scratch with the same seed. A new 50-epoch control is needed:
the completed 128³ run retained best/last checkpoints, not an arbitrary historical
epoch-50 model. This is not a fresh cosine schedule ending at 50; at epoch 50 the
500-epoch schedule still uses about 97.6% of its initial learning rate.

The first full pilot launch writes a protocol containing commands, split/source/
configuration/metric hashes, inspection budget and success criteria. Later resume
and evaluation reject a changed protocol. Smoke outputs are separate and do not
count as pilot results. Do not edit settings after launch; use a fresh root for a
changed experiment. Further continuation after the inspection point requires a
separate agreed protocol, not silently changing this pilot's stopping point.

## Predefined exploratory screen

The primary endpoint is the **pooled native-grid C1 any-overlap recall of reference
components <1 mL** on the complete inner-validation set. Advance the new sampler
to further validation only if all three checks pass:

1. Small-component recall improves by at least **0.05 absolute** (5 percentage points).
2. Mean unmatched-component false-positive volume increases by no more than **10%**.
3. Mean foreground Dice decreases by no more than **0.02**.

These are explicit research screening choices, not validated clinical benefit
thresholds. Report IoU≥0.5 recall, voxelwise errors, tumour-volume/SUVmax errors,
large-lesion performance, runtime and representative failures as secondary
evidence. A one-fold, one-seed, 50-epoch result cannot establish a final gain or
justify tuning repeatedly against outer scores. If the endpoint has no reference
small components, the screen is inconclusive and does not advance automatically.

## Commands

From the project root:

```bash
# Preview only; prints commands and starts nothing.
.venv/bin/python scripts/train_sampling_pilot.py

# Optional short execution checks for both policies.
.venv/bin/python scripts/train_sampling_pilot.py --smoke

# Two sequential fold-0 pilots, each pausing at epoch 50/500.
.venv/bin/python scripts/train_sampling_pilot.py --run

# Resume after interruption with unchanged settings/protocol.
.venv/bin/python scripts/train_sampling_pilot.py --run --resume

# Once BOTH pilots reach epoch 50, export their complete inner-validation masks,
# compute common native-grid metrics and apply the predefined screen.
.venv/bin/python scripts/train_sampling_pilot.py --evaluate
```

Outputs: `runs/sampling-pilot/voxel/`, `lesion-balanced/`, `pilot_protocol.json`
and `inner-review/`. The last contains the two prediction/evaluation packages
and `comparison.json`. The evaluator checks the two arms' source/settings and
case/reference identities before reporting a comparison. Repeated evaluation
uses a fresh `--review-output`. Keep the terminal/WSL alive during GPU workloads.

No real-cohort pilot training or GPU inference was launched while implementing
this plan. The next training action belongs to the user, as requested earlier.
The only training performed during implementation used tiny synthetic CPU
fixtures to verify sampling, checkpointing and resume.

## Reproduce the completed CPU diagnostics

```bash
.venv/bin/python scripts/audit_sampling.py \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma --fold 0 --cases 6 --draws 50 \
  --patches 128 --suv-threshold 5 --sampling both \
  --output runs/analysis/new-sampling-comparison --execute

.venv/bin/python scripts/review_patch_errors.py \
  --evaluation96 runs/analysis/evaluation96 \
  --evaluation128 runs/analysis/evaluation128 \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/new-error-review --per-category 2

OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The sampling audit now distinguishes the intended foreground target from the
boundary-corrected crop centre. A crop centred away from the selected lesion can
still contain it, so these quantities must not be conflated. The high-SUV
background threshold is a diagnostic proxy; it is not an organ annotation or
an automatic rule for suppressing predictions.
