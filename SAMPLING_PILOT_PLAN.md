# Three matched 128³ sampling/loss pilots

The completed patch comparison improved Dice and false-positive burden with
128³ patches, but pooled any-overlap recall of reference components <1 mL remained
about 19%. This pilot tests sampling and loss in separate matched comparisons.
Every arm uses 128³; the completed 96³ experiment is not rerun. Neither proposed
change has yet demonstrated a trained-model improvement. Original images and
completed experiments are preserved.

| Arm / directory | Foreground sampling | Training loss |
|---|---|---|
| A / `voxel` | Original voxel sampling | Original Dice |
| B / `lesion-balanced` | Uniform reference component, then voxel | Original Dice |
| C / `lesion-balanced-dice-ce` | Same as B | 1 × Dice + 1 × cross-entropy |

**B versus A** tests sampling. **C versus B** tests loss. C versus A is reported
descriptively as the combination of both changes. There is no automatic winner
selection from these three pilot runs.

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

## Combined loss

The trainer now accepts `--loss dice-ce --dice-weight 1 --ce-weight 1`. MONAI's
DiceCELoss computes the weighted sum of background-inclusive softmax Dice and
cross-entropy on raw two-class logits. Loss/reduction computation stays in float32
inside the existing AMP recovery path. The default remains the original Dice
loss; changing the objective or either positive weight fails resume compatibility.
Combined-loss training history uses a distinct `dice_ce_loss_*` key. Its magnitude
must not be compared directly with Dice-only training loss or interpreted as
segmentation accuracy. Checkpoint selection still uses inner-validation Dice.

The exact known pre-loss source can continue an unchanged Dice experiment with
either existing sampling policy. Other source migrations remain narrowly
restricted. Existing frozen two-arm protocols cannot silently become three-arm
protocols; the new default output root is separate.

## Frozen comparison

| Item | All three arms |
|---|---|
| Patch | 128³, 2 mm grid, full field of view |
| Fold | 0, existing patient-level train/inner-validation assignments |
| Network/optimizer | Existing residual U-Net and AdamW; loss differs only for C |
| Schedule | Original 500-epoch cosine schedule, starting LR 2e-4 |
| Inspection point | Pause after epoch 50; early stopping cannot trigger before epoch 100 |
| Update budget | One patch per training examination per epoch, batch 1, accumulation 1 |
| Selection | Highest inner-validation foreground Dice within the first 50 epochs |
| Outer test | No outer inference at the pilot pause or during pilot evaluation |
| Comparisons | B/A changes sampling only; C/B changes loss only |

All three arms start from scratch with the same seed. A new 50-epoch control is needed:
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

For the **C versus B loss comparison**, the primary endpoint is mean native-grid
unmatched-component FP volume. Advance the combined loss to further validation
only if mean FP volume falls by at least **10%**, small-component C1 recall falls
by no more than **0.02 absolute**, and mean Dice falls by no more than **0.02**.
These additional research criteria are frozen before launch. They reflect the
loss hypothesis (better background rejection) and do not replace the sampling
comparison's recall endpoint. Report all outcomes even when a screen fails.

## Commands

From the project root:

```bash
# Preview only; prints commands and starts nothing.
.venv/bin/python scripts/train_sampling_pilot.py

# Optional short execution checks for all three 128³ arms.
.venv/bin/python scripts/train_sampling_pilot.py --smoke

# Three sequential fold-0 pilots, each pausing at epoch 50/500.
.venv/bin/python scripts/train_sampling_pilot.py --run

# Resume after interruption with unchanged settings/protocol.
.venv/bin/python scripts/train_sampling_pilot.py --run --resume

# Once ALL THREE reach epoch 50, score the complete inner-validation sets.
.venv/bin/python scripts/train_sampling_pilot.py --evaluate
```

Outputs are under **`runs/sampling-loss-pilot/`**: `voxel/`, `lesion-balanced/`,
`lesion-balanced-dice-ce/`, `pilot_protocol.json` and `inner-review/`. The last
contains all three prediction/evaluation packages and `comparison.json`, with
separate `sampling_B_vs_A`, `loss_C_vs_B` and `combined_C_vs_A_descriptive` entries.
The evaluator rejects differences beyond the intended factor in each comparison
and checks all three case/reference identities. Repeated evaluation
uses a fresh `--review-output`. Keep the terminal/WSL alive during GPU workloads.

To run just one arm, add `--arm voxel`, `--arm lesion-balanced`, or
`--arm lesion-balanced-dice-ce`. Resume later with the same root and protocol.
Evaluation requires all three arms. `--strategy` remains an alias for `--arm`;
legacy `--arm both` selects A/B only. The default is all three arms, all at 128³.

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
