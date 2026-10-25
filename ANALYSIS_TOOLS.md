# Segmentation analysis tools

These commands implement the tooling for local tickets 02 (evaluation), 05 (patch
benchmark) and 07 (sampling audit). They use separate reports and preserve source
images, predictions, checkpoints and the existing training implementation.
Benchmark and sampling commands preview their inputs unless `--execute` is given.
Evaluation reads existing masks and does not train a model.

Install the optional small connected-component dependency if needed:

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
```

## Evaluate the completed 96³ and 128³ experiments

Run each command from the project root. They are CPU analyses and can take time
to read and label all native volumes. Use them sequentially to limit RAM and disk
contention. Output directories must not already exist.

```bash
.venv/bin/python scripts/evaluate_segmentation.py \
  --run runs/patch-comparison/patch96 \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/evaluation96

.venv/bin/python scripts/evaluate_segmentation.py \
  --run runs/patch-comparison/patch128 \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/evaluation128
```

Use `--fold 0` to evaluate just one completed fold. The run mode validates the
split identity, patient routing and exact prediction case inventory. It does not
certify checkpoint training provenance; that is a separate experiment audit.

Each report contains `per_case.csv`, `reference_lesions.csv`, `metrics.json`,
`summary.json` and `manifest.json`. The manifest records source/configuration
hashes, dependency versions, input-mask/image hashes and completion state.
An interrupted report remains explicitly incomplete; choose a fresh output to retry.
The summary includes per-examination means/medians and pooled detection counts,
with reference lesion-size strata. It does not supply inferential confidence
intervals or claim statistical significance.

For masks from another workflow, supply `--manifest inputs.json` instead of
`--run`. The input is a JSON list with `case_id` (`patient/examination`), `patient`,
`reference`, `pet` and `prediction`. Paths may be absolute or relative to the JSON
file. One prediction path cannot represent multiple cases; patient/case prefixes
and case uniqueness are checked. Explicit manifests rely on the caller to assert
patient routing and image identity; file names alone cannot prove provenance.

Example:

```json
[
  {
    "case_id": "patientA/examinationA",
    "patient": "patientA",
    "reference": "/absolute/path/SEG.nii.gz",
    "pet": "/absolute/path/SUV.nii.gz",
    "prediction": "/absolute/path/SEG_pred.nii.gz"
  }
]
```

### Fixed metric definitions

- Arrays use native x,y,z PET geometry. PET must declare millimetres; masks may
  declare millimetres or unknown units after affine agreement. Shape/affine must
  match; there is no automatic resampling. Voxel mL is `abs(det(affine[:3,:3]))/1000`.
- Masks must be finite binary 0/1; SUV must be finite and nonnegative. Empty mask
  TMTV/SUVmax are zero. Relative errors with zero reference denominator are null.
- Dice is foreground overlap; both masks empty gives 1, exactly one empty gives 0.
- Lesions are 18-connected components, labelled from Fortran-order x,y,z arrays
  to match the inspected Microsoft reader's memory order.
- **FPV/FNV** count whole unmatched predicted/reference components. Additional
  `voxel_FP_ml`/`voxel_FN_ml` count individual misclassified voxels. A touched but
  badly undersegmented lesion can have zero unmatched-component FNV.
- **C1:** each predicted component overlapping any reference is a predicted TP;
  reference recall counts each touched reference once. Splits/merges mean that
  predicted TP and detected-reference counts need not agree.
- **C2:** choose the reference with highest IoU for each prediction, first label
  on ties, then accept IoU ≥0.5. **C3:** use that same best-IoU reference and require
  the prediction to contain its first C-order maximum-SUV voxel. At zero SUV the
  maximum is selected inside the reference component, never in background.
- For C2/C3, a reference can be accepted once; a repeated accepted match is FP,
  logged in `notes`. The upstream code can raise on such cases. We do not rematch
  an already consumed best reference to a second-best reference.
- Recall with zero reference lesions and precision with zero predictions are
  null. Pooled counts are summed before division; F1 is zero when both defined
  precision/recall are zero and otherwise null if either is undefined.
- Reference size strata are **<1 mL**, **1–<10 mL**, **≥10 mL**. These are explicit
  exploratory volume bins, not clinical size categories. Native-grid strata are
  used by evaluation; the sampling audit labels components on the training grid.

Definitions were inspected against Microsoft's pinned implementation at
`81129243db3c868370a7664c9adc2ad52609484f`. Twelve synthetic cases preserve 192
values obtained by executing its original functions; tests compare against those
snapshots. This does not repeat the colleague's real-case 176-value comparison
or establish universal equivalence on exception/empty cases. The fixture records
the upstream source SHA-256. No SimpleITK dependency is required to use our tools.

## Inspect selected checkpoints using inner validation

This optional command exports inner-validation masks from a trusted local
checkpoint without updating the model. It previews unless `--execute` is supplied.
GPU inference is a separate workload: run after other GPU jobs finish.

```bash
.venv/bin/python scripts/predict_inner_validation.py \
  --checkpoint runs/patch-comparison/patch128/fold_0/best.pt \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/inner128-fold0 --cases 3 --execute

.venv/bin/python scripts/evaluate_segmentation.py \
  --manifest runs/analysis/inner128-fold0/evaluation_inputs.json \
  --output runs/analysis/inner128-fold0-evaluation
```

Omit `--cases` for the entire inner-validation set. Checkpoint source/split
identity and the current scientific preprocessing/model recipe must match.
The checkpoint is trusted pickle input; use only your known local experiment.
Predictions use CT/PET; labels participate only in source verification and later
evaluation. Raw case order is deterministic; a short subset is a diagnostic and
does not replace full inner validation for decisions.

## Benchmark patch memory and throughput

This is useful before another experiment, or for tuning inference resources.
The completed 96³/128³ experiments already demonstrate that their configured
settings executed; a new benchmark is not required to evaluate their masks.

```bash
.venv/bin/python scripts/benchmark_patches.py \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/patch-benchmark \
  --patches 64 96 128 --sw-batches 1 2 4 --steps 3
```

Add `--execute` only when ready to run the bounded GPU workers. Do not overlap
this with training or other GPU jobs. Default fold 0 supplies the largest native
training examination and largest native inner-validation examination; held-out
patients are never used. Each patch runs in a fresh process for ordinary AMP and
forced recovery: an AMP-only invalid-output hook exercises the existing buffer/
RNG rollback and float32 retry in both training and inference. The training code
itself is unchanged. Inputs, augmentation and the model follow the existing recipe.

Workers use training batch 1, accumulation 1 and independently initialized model
weights. One cold optimizer update and the requested number of subsequent updates
are timed separately, including AdamW state allocation. Full-volume sliding-window
validation measures each specified window batch while retaining optimizer state/
gradients as the trainer does. Preprocessing time is recorded separately. This is
a capacity/throughput diagnostic, not segmentation-quality training or proof that
all cases at all future weights will fit. Runtime precision failures, gradient
skips and OOMs remain visible.

Reports include isolated worker logs/specifications/results, a flattened
`measurements.csv`, manifest and `recommendations.json`. A window batch is eligible
only when both precisions completed training updates and validation at that batch.
Among eligible batches, the fastest measured AMP validation time is recommended.
No settings are automatically applied. Failed jobs remain recorded. An interrupted
benchmark uses a new output directory on retry.

## Audit sampling coverage

```bash
.venv/bin/python scripts/audit_sampling.py \
  --splits runs/patch-comparison/splits.json \
  --data-root data/autopet-v2-lymphoma \
  --output runs/analysis/sampling-fold0 \
  --fold 0 --cases 3 --draws 50 --patches 64 96 128 --suv-threshold 5
```

Add `--execute` to perform the CPU audit. It randomly selects a bounded number of
training examinations with the recorded seed, verifies their source hashes and
uses the actual training crop/pad/affine transforms. Reference component IDs are
carried through nearest-neighbour label transforms, so the report counts which
components remain visible after augmentation. GT positivity and the reference
2:1 foreground/background sampler remain unchanged. Random augmentation is never
cached; this command performs no shared preprocessing-cache writes.

The report contains each draw's crop centre, repeated-centre flag, reference
coverage and high-SUV background volume; per-lesion appearance frequency and
size strata; and aggregate coverage summaries. Size/volume measurements refer
to the 2 mm training grid. Crop-centre occupancy is observed occupancy, not the
sampler's intended foreground/background coin flip after boundary correction.
Sampling frequencies across these independent seeded draws are a diagnostic,
not a reconstruction of all historical training epochs.

`--sampling both` compares the original voxel sampler and the opt-in
lesion-balanced sampler on the same cases/seed. It adds intended foreground
targets, their reference component IDs, and corrected-centre counts. Intended
targets and corrected centres differ near boundaries. For a single policy,
use `--sampling voxel` or `--sampling lesion-balanced`. The updated audit
precomputes the same deterministic candidate indices used by the training cache.

The supplied SUV threshold (5 in this example) is a **high-uptake background
proxy**, not an anatomical label, malignancy threshold or a mined hard negative.
It can include physiological uptake or missing annotations. Review cases before
using it to change sampling. This audit requires CPU RAM for full preprocessing;
run it separately from large training/validation jobs.

## Verification and ticket status

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p test_analysis_tools.py -v
```

The CPU suite checks metric geometry/known-volume and matching contracts,
upstream snapshots, report preservation, patient routing, deterministic sampling
and conservative benchmark recommendations. It needs no pytest and launches no
GPU workload. Ticket 07 has now been executed on six training examinations; see
the follow-up review and local issue status. Ticket 05 GPU timings remain pending.
Tools and tests alone are not completed experimental findings.
