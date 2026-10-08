# MANU — lymphoma PET/CT project

Each examination contains:

| File | Purpose |
|---|---|
| `CTres.nii.gz` | CT anatomy, resampled to the PET image grid |
| `SUV.nii.gz` | PET tracer uptake, expressed as SUV |
| `SEG.nii.gz` | Reference mask marking the lesions |

## How it works

From the [official FDAT version-2 ZIP](https://doi.org/10.57754/FDAT.8f14a-pf846):

1. Find lymphoma examinations in the [official metadata](https://www.cancerimagingarchive.net/wp-content/uploads/Clinical-Metadata-FDG-PET_CT-Lesions.csv).
2. Read the ZIP index to find each file's location and size.
3. Select only CTres, SUV and SEG for those examinations.
4. Request only those bytes.
5. Extract and save the `.nii.gz` files into patient/examination folders.

Downloaded: 145 examinations from 144 patients, totaling 16.62 GB. Negative examinations and other cancer diagnoses are excluded.

## Training baseline

`train.py` adapts the professor's [Microsoft baseline](https://github.com/microsoft/lymphoma-segmentation-dnn/tree/81129243db3c868370a7664c9adc2ad52609484f), inspected at commit `81129243db3c868370a7664c9adc2ad52609484f`. This is public-data cross-validation, not reproduction of the paper's private-cohort experiment. Original images are read only.

Use Python 3.12 in a separate environment. These direct dependency versions were exercised together using an existing PyTorch installation and temporary MONAI/NiBabel packages; no large dependency installation was performed. CUDA wheel choices are listed in [PyTorch's installation archive](https://pytorch.org/get-started/previous-versions/).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m mypy --check-untyped-defs --follow-imports=skip --ignore-missing-imports train.py
```

Running `python train.py` only prints help. Start by auditing the 145 examinations from 144 patients and creating a shared, immutable split manifest. The audit reads all files, checks aligned grids and positive binary labels, rejects duplicate SUV voxel content, and preserves the complete field of view. It can take several minutes. Checksums are rechecked on subsequent launches. CT/SEG's unspecified spatial units are interpreted as millimetres only after checking their grids against SUV.

```bash
python train.py --prepare --splits runs/splits.json
# Two training examinations, 20 updates, one complete inner-validation volume.
python train.py --smoke --patch 64 --output runs/smoke64 --splits runs/splits.json
# Benchmark separately; changing patch size changes the experiment.
python train.py --smoke --patch 96 --output runs/smoke96 --splits runs/splits.json
# One fold, short execution pilot; not a final accuracy experiment.
python train.py --fold 0 --epochs 2 --patch 64 --output runs/pilot --splits runs/splits.json
# Only after choosing and freezing the scientific settings and epoch budget:
python train.py --fold all --epochs 500 --patch 64 --output runs/baseline --splits runs/splits.json
python train.py --fold all --epochs 500 --patch 64 --output runs/baseline --splits runs/splits.json --resume
```

Each patient belongs to one of five outer folds; both visits of the repeated patient stay together. For each fold, 20% of the remaining patients are reserved for checkpoint selection. Outer patients never train or select checkpoints. Smoke outputs contain no outer-fold predictions and must use a separate output directory. Do not tune settings using outer-fold scores.

The defaults are deliberately conservative: one GPU, batch 1, 64³ patches, zero loader workers, zero cache, four CPU threads and one inference window at a time. No hardware detection changes patch size, learning rate, batch size, resolution or architecture. An out-of-memory error stops the run. Benchmark resource changes before increasing them. `--data-root`, `--output`, and `--splits` make relocation possible without source edits; examination paths in the manifest are relative.

### What follows upstream, and what changes

Inspected source: [initialization](https://github.com/microsoft/lymphoma-segmentation-dnn/blob/81129243db3c868370a7664c9adc2ad52609484f/segmentation/initialize_train.py), [training loop](https://github.com/microsoft/lymphoma-segmentation-dnn/blob/81129243db3c868370a7664c9adc2ad52609484f/segmentation/trainddp.py), [inference](https://github.com/microsoft/lymphoma-segmentation-dnn/blob/81129243db3c868370a7664c9adc2ad52609484f/segmentation/inference.py), [dataset contract](https://github.com/microsoft/lymphoma-segmentation-dnn/blob/81129243db3c868370a7664c9adc2ad52609484f/documentation/dataset_format.md), and [dependencies](https://github.com/microsoft/lymphoma-segmentation-dnn/blob/81129243db3c868370a7664c9adc2ad52609484f/environment.yml). Adapted code retains attribution in `train.py` and `THIRD_PARTY_LICENSE`.

| Decision | Behavior and reason |
|---|---|
| Preserve architecture | MONAI residual 3D U-Net, CT then PET, 2 classes, widths 16/32/64/128/256/512, five stride-2 stages, two residual units, BatchNorm. |
| Preserve intensity and geometry recipe | Clip CT −154…325 HU to 0…1; PET SUV unchanged; RAS, 2 mm; linear image and nearest mask interpolation. |
| **Remove foreground cropping for safety** | The cohort audit found upstream’s CT>0 bounding box discards lesion voxels in `PETCT_0cda25453b` (12 October 2003 examination). Keep the full field of view for every exam, without using labels to choose bounds. This increases CPU memory and inference time. |
| Preserve sampling/augmentation | One patch per exam per epoch, lesion/background 2:1 with PET>0 background candidates; pad small patches; affine probability 0.5, translation 10 voxels, axial rotation π/15, scale range 0.1. Smoke repeats two cases to bound updates. |
| Preserve optimization | Background-inclusive softmax Dice, AdamW lr 2e-4, weight decay 1e-5, cosine to zero over the explicit epoch budget. Foreground Dice is reported separately. |
| Smaller patches **change the published configuration** | Upstream uses 192³; 64³/96³ reduce anatomical context. Patch dimensions must be divisible by 32 and at least 64. Resolution and network widths are unchanged. |
| Replace unsafe evaluation defaults | Patient-level outer folds plus separate inner validation; select `best.pt` on inner foreground Dice only. Always validate the final epoch. Predict only the selected model's outer fold. |
| Reduce GPU memory | CUDA float16 autocast and scaling, float32 loss; full-volume inputs and stitched float32 logits stay on CPU. Inference ROI equals the explicit training patch, replacing upstream's larger ROI mapping. |
| Restore geometry from inputs | Invert continuous logits through PET input history, then argmax to uint8 on the native SUV affine/grid. Inference needs no reference mask; native-grid Dice is computed afterward. |
| Modern dependencies and execution | Minimal pinned packages instead of the old full Conda environment. Optional DDP and bounded per-process caching, epoch seeds, resumable checkpoints and atomic output writes. |

### Outputs and resume

The output root has `run.json` (scientific settings, resolved resources, software/hardware and source hash) and, unless external, `splits.json` (patient assignments, content hashes and split identity). Each `fold_N` has only two model checkpoints: `best.pt` and `last.pt`, plus `history.json`, completion metadata, native-grid predictions under `predictions/<patient>/<exam>/SEG_pred.nii.gz`, and per-case `metrics.json`. A completed sequential five-fold run checks for exactly 145 unique predictions and writes root `metrics.json`. Missing, extra or misrouted predictions cause an error. Training, validation and inference memory/timing are measured separately; process peak RSS is cumulative.

`last.pt` records the actual completed epoch, model, optimizer, cosine scheduler, scaler, every rank's random states and history. It also retains the selected weights so resume can repair a `best.pt` write interrupted before the epoch was committed. Random transforms, shuffling and workers are reseeded by epoch/rank; workers restart each epoch. Resume keeps the original total epoch budget. Changes to GPU count, effective batch, sampler, software/code or recorded resource settings are rejected. Data/output roots and physical CUDA device indices may change. Bit-identical CUDA results across different hardware are not promised. Only load trusted local checkpoints.

To test a deliberate interruption without changing the schedule:

```bash
python train.py --smoke --epochs 2 --smoke-steps 2 --stop-after-epoch 1 --output runs/resume-check --splits runs/splits.json
python train.py --smoke --epochs 2 --smoke-steps 2 --output runs/resume-check --splits runs/splits.json --resume
```

A completed fold is checked and skipped with `--resume`. An incomplete prediction pass restarts from the selected checkpoint and atomically replaces its partial outputs. Concurrent launches cannot write the same fold. An interruption before the first epoch checkpoint requires a new output directory.

### Institute hardware

First benchmark a larger single GPU explicitly. The following are examples to measure, not claims that they fit:

```bash
python train.py --smoke --device cuda:0 --patch 192 --batch-size 2 --sw-batch-size 2 --output runs/large-gpu-smoke --splits runs/splits.json
```

For independent folds, prepare splits once and launch each fold on a distinct GPU. Use one shared output root to enforce identical settings, with separate `fold_N` outputs and locks. These commands can run in separate terminals; repeat for folds 2–4:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 --epochs 500 --output runs/institute --splits runs/splits.json
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 --epochs 500 --output runs/institute --splits runs/splits.json
# After all independent folds complete, verify/skip them and assemble the OOF summary:
python train.py --fold all --epochs 500 --output runs/institute --splits runs/splits.json --resume
```

For multiple GPUs training one fold, explicitly request DDP. The bounded integration check uses the same model and training loop, sharding, full-volume validation, synchronization and checkpoint code:

```bash
torchrun --standalone --nproc-per-node=2 train.py --ddp --smoke --epochs 2 --smoke-steps 2 --stop-after-epoch 1 --output runs/ddp-check --splits runs/splits.json
torchrun --standalone --nproc-per-node=2 train.py --ddp --smoke --epochs 2 --smoke-steps 2 --output runs/ddp-check --splits runs/splits.json --resume
# After benchmarking, an explicit full-fold example:
torchrun --standalone --nproc-per-node=2 train.py --ddp --fold 0 --epochs 500 --patch 96 --batch-size 1 --accumulate 2 --output runs/ddp-baseline --splits runs/splits.json
```

Effective global batch is per-GPU batch × GPU count × accumulation (4 in the last example). The final partial update is normalized by its actual sample count. Learning rate does not scale automatically. Training's shuffled distributed sampler pads to equal rank lengths, so a few training exams may repeat; this behavior is recorded and fixed. Validation and prediction run only on global rank zero without a padded sampler. Other ranks wait; only rank zero writes. BatchNorm stays local per GPU, as upstream; its running buffers are synchronized from rank zero at epoch boundaries. It is not converted to SyncBatchNorm. More GPUs do not pool VRAM: each must fit its own model and batch.

`--workers`, `--threads`, `--cache-gib`, and `--sw-batch-size` control resources. Cache limits are per rank, hold only deterministic tensors, and require zero workers to prevent worker copies; transient preprocessing/augmentation arrays and model memory are additional. Aggregate caches may not exceed half the currently available RAM. With workers, full-volume preprocessing also multiplies by workers/processes; benchmark conservatively.

### Verification status

Synthetic integration tests cover repeat visits and patient isolation, immutable splits, original-geometry restoration without labels, preservation of lesions outside CT foreground, actual optimizer/scheduler/RNG continuation, incompatible resume rejection, and unique held-out export across all five folds. These do not establish segmentation accuracy. Multi-GPU execution remains unverified on this one-GPU laptop; run the two `ddp-check` commands above at the institute. No full training experiment has been started.

Verified on 8 October 2026: **7 tests passed**, type checking passed, and the complete real cohort passed geometry, positive-label, file-integrity and duplicate-SUV checks (145 exams / 144 patients; both repeat visits retained). Two real-data smoke measurements exercised the final, uncropped `train_fold` path on the RTX 4050, each with two updates from two training exams and one separate complete inner-validation exam. CLI orchestration and resume were tested with synthetic data. Measured PyTorch GPU peaks:

| Patch | Training allocated / reserved | Validation allocated / reserved | Process peak RAM |
|---|---:|---:|---:|
| 64³ | 382.5 / 420.0 MiB | 261.8 / 300.0 MiB | 12.75 GiB |
| 96³ | 388.7 / 444.0 MiB | 273.7 / 340.0 MiB | 12.86 GiB |

Training included a 400×400×619 scan; complete validation used a separate 400×400×284 scan. These settings fit for the measured operations; this is not a full-cohort memory guarantee. Logs/configurations are in the local, ignored `runs/full-field-smoke64/` and `runs/full-field-smoke96/` directories. They are diagnostic runs, not cross-validation results or accuracy evidence. Larger batches, caches, workers, real-data geometry export, full-volume validation of the largest scan, and multi-GPU execution remain to be benchmarked. The pinned MONAI release emits PyTorch indexing deprecation warnings; the tested operations pass. No existing Python environment or original image was modified.
