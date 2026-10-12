# MANU — lymphoma PET/CT project

Automatic segmentation of lymphoma lesions in FDG PET/CT, evaluated locally on the 145 lymphoma examinations of the autoPET dataset.

Two models are run on the same examinations:

1. **Microsoft baseline** — a U-Net trained from scratch with the unmodified [microsoft/lymphoma-segmentation-dnn](https://github.com/microsoft/lymphoma-segmentation-dnn), 5-fold cross-validation split by patient, 10 epochs per fold.
2. **Pretrained autoPET III** — the published checkpoint of the winning autoPET III solution, [MIC-DKFZ/autopet-3-submission](https://github.com/MIC-DKFZ/autopet-3-submission), used as it is, without training. This comparison is for orientation only (see [Results and limitations](#results-and-limitations)).

Both are scored with the same three measures: Dice, TMTV and SUVmax.

## Contents

- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Dataset preparation](#dataset-preparation)
- [Microsoft baseline](#microsoft-baseline)
- [Pretrained autoPET III](#pretrained-autopet-iii)
- [The 10-patient subset](#the-10-patient-subset)
- [Results and limitations](#results-and-limitations)
- [Collaboration](#collaboration)

## Repository structure

Our own scripts, all in the project root:

| Script | Runs on | Purpose |
|---|---|---|
| `download_lymphoma.py` | Windows or WSL | Downloads the lymphoma subset (CTres, SUV, SEG) into `data/autopet-v2-lymphoma` |
| `preparation_for_training.py` | Windows or WSL | Organizes the data into `data/lymphoma-baseline` and creates the patient-level 5-fold split |
| `make_datalist.py` | WSL | Creates the datalist (`data_split/train_filepaths.csv`) expected by the Microsoft training code and verifies the split |
| `run_cv_pipeline.py` | WSL | Runs the 5-fold training/inference pipeline, fold by fold |
| `evaluate_cv_light.py` | WSL | Evaluates the baseline's held-out predictions using Dice, TMTV and SUVmax |
| `cv_lesion_measures.py` | WSL | Additional lesion-measure helper (SUVmean, SUVmax, lesion count, TMTV, TLG); called by `run_cv_pipeline.py` only in its full mode |
| `download_autopet_checkpoint.py` | Windows or WSL | Downloads the autoPET III checkpoint from Zenodo, verifies it and extracts the files needed for inference into `checkpoints/autopet3` |
| `autopet_setup_env.sh` | WSL | Creates the separate `~/venvs/autopet3` environment and installs the official autoPET III code into it |
| `autopet_check_setup.py` | WSL | Loads the five checkpoints through the official code to confirm that code and checkpoint fit. Runs no inference |
| `autopet_smoke_one_case.sh` | WSL | Runs the pretrained model on one examination |
| `autopet_run_subset.py` | WSL | Selects 10 patients with seed 42, writes the manifest and runs the pretrained model on them |
| `evaluate_autopet_light.py` | WSL | Scores autoPET III predictions with the same Dice / TMTV / SUVmax code as `evaluate_cv_light.py` |

Directories:

| Directory | In Git | Contents |
|---|---|---|
| `experiments/unet_10ep/` | results and logs only | Baseline experiment: `manifest.json`, `pipeline.log`, `logs/`, `summary/`, `light_eval/` are tracked; `work/` (checkpoints, predictions) is not |
| `experiments/autopet3_subset10/` | no | Manifest and predictions of the 10-patient subset |
| `data/` | no | Downloaded and prepared dataset |
| `checkpoints/` | no | autoPET III checkpoint |
| `lymphoma-segmentation-dnn/` | no | Clone of the Microsoft repository, unmodified |
| `autopet-3-submission/` | no | Clone of the MIC-DKFZ repository, unmodified |

Neither cloned repository is edited. Everything we add lives in the project root.

## Environment setup

The work was done on Windows 11 with WSL2 (Ubuntu) and an NVIDIA RTX 5060 Laptop GPU with 8 GB of memory. Data download and preparation use only the Python standard library and can run on Windows. Training, inference and evaluation run inside WSL.

**Requirements**

- Windows with WSL2 and an NVIDIA GPU that `nvidia-smi` shows inside WSL.
- Python 3.11. The scripts were run with 3.11; `download_lymphoma.py`, `preparation_for_training.py` and `download_autopet_checkpoint.py` need no extra packages.
- A PyTorch build that supports your GPU. The RTX 50 series needs a CUDA 12.8 build (`cu128`); both environments below use `torch 2.11.0+cu128`.
- Disk space: about 17 GB for the dataset, the same again for its copy inside WSL, and about 8 GB for the autoPET III ZIP plus the extracted checkpoints.

**1. Clone this repository and the two third-party repositories into it**

```
git clone <url of this repository> manu
cd manu
git clone https://github.com/microsoft/lymphoma-segmentation-dnn.git
git -C lymphoma-segmentation-dnn checkout 81129243db3c868370a7664c9adc2ad52609484f
git clone https://github.com/MIC-DKFZ/autopet-3-submission.git
git -C autopet-3-submission checkout 257afc3b197f8648ab5ba5e1755adabf45a3028a
```

These are the commits our results were produced with.

**2. Baseline environment** (inside WSL, uses [uv](https://docs.astral.sh/uv/)):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 ~/venvs/lymphoma_seg
uv pip install --python ~/venvs/lymphoma_seg/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python ~/venvs/lymphoma_seg/bin/python monai==1.2.0 numpy==1.24.3 pandas==2.0.3 nibabel==5.1.0 itk==5.3.0 matplotlib
uv pip install --python ~/venvs/lymphoma_seg/bin/python SimpleITK==2.3.0 connected-components-3d==3.12.3
```

The Microsoft repository ships a conda `environment.yml` for Python 3.8 and CUDA 11.3. It was not used, because that PyTorch build does not support the RTX 5060.

**3. autoPET III environment** (inside WSL, from the project folder):

```
bash autopet_setup_env.sh
```

This creates `~/venvs/autopet3` with the same PyTorch build and installs the official repository with `pip install -e .`, as its readme says. It is a separate environment because that repository contains its own modified `nnunetv2` and pulls different versions of numpy and pandas. `~/venvs/lymphoma_seg` is not touched.

**Paths.** In the commands below, replace `/mnt/c/path/to/manu` with the location of your clone. Several scripts default to the dataset location `/home/eva25/data/lymphoma-baseline`. If your WSL user name is different, pass `--dataset-root` where the script has that option. `autopet_run_subset.py` has no such option: change the `DATASET_ROOT` constant at its top.

## Dataset preparation

Each examination contains:

| File | Purpose |
|---|---|
| `CTres.nii.gz` | CT anatomy, resampled to the PET image grid |
| `SUV.nii.gz` | PET tracer uptake, expressed as SUV |
| `SEG.nii.gz` | Reference mask marking the lesions |

### How the download works

From the [official FDAT version-2 ZIP](https://doi.org/10.57754/FDAT.8f14a-pf846):

1. Find lymphoma examinations in the [official metadata](https://www.cancerimagingarchive.net/wp-content/uploads/Clinical-Metadata-FDG-PET_CT-Lesions.csv).
2. Read the ZIP index to find each file's location and size.
3. Select only CTres, SUV and SEG for those examinations.
4. Request only those bytes.
5. Extract and save the `.nii.gz` files into patient/examination folders.

Downloaded: 145 examinations from 144 patients, totaling 16.62 GB. Negative examinations and other cancer diagnoses are excluded.

### Steps

**1. Download and organize** (from the project folder):

```
python download_lymphoma.py
python preparation_for_training.py
```

`download_lymphoma.py` writes one folder per patient and examination:

```
data/autopet-v2-lymphoma/
    PETCT_<patient>/
        <examination>/
            CTres.nii.gz
            SUV.nii.gz
            SEG.nii.gz
```

`preparation_for_training.py` turns this into the layout used by both models. Files are hard-linked where possible, not copied:

```
data/lymphoma-baseline/
    imagesTr/
        <case_id>_0000.nii.gz      CT  (CTres.nii.gz)
        <case_id>_0001.nii.gz      PET (SUV.nii.gz)
    labelsTr/
        <case_id>.nii.gz           SEG.nii.gz
    cases.json                     case_id, patient, exam, fold for all 145 examinations
    splits_patient_level.json      training / validation case ids for each of the 5 folds
```

`<case_id>` is `<patient>_<examination>`, for example `PETCT_04ab5c61c9_11-22-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42241`. Case ids contain spaces, so quote paths in the shell.

The split shuffles the 144 patients with seed 42 and assigns them to 5 folds, so both examinations of the one patient who has two land in the same fold. The script prints the fold sizes and checks that no patient appears in more than one fold; it must end with `NO PATIENT LEAKAGE` and `DATASET READY FOR 5-FOLD TRAINING`.

**2. Copy the dataset into WSL** (training reads it from there, it is much faster):

```
cd "/mnt/c/path/to/manu"
mkdir -p ~/data && cp -r data/lymphoma-baseline ~/data/
```

Everything from here on runs inside WSL, from the project folder.

## Microsoft baseline

Training uses the unmodified [microsoft/lymphoma-segmentation-dnn](https://github.com/microsoft/lymphoma-segmentation-dnn), cloned into `lymphoma-segmentation-dnn/`. `run_cv_pipeline.py` refuses to start if tracked files of that repository are modified.

Only two training arguments are passed, `--epochs 10` and `--num_workers 6`. Everything else stays at Microsoft's defaults (U-Net, patch size 192, batch size 1, learning rate 2e-4, validation every 2 epochs). The full list, the exact commands and the software versions are recorded in `experiments/unet_10ep/manifest.json`.

**1. Create the datalist:**

```
~/venvs/lymphoma_seg/bin/python make_datalist.py --dataset-root /home/eva25/data/lymphoma-baseline
```

This writes `lymphoma-segmentation-dnn/data_split/train_filepaths.csv` with absolute paths and the fold of every case, then reads the split back through Microsoft's own loader and checks that no patient is in both training and validation. It must end with `DATALIST OK`.

**2. Dry run** (checks everything and prints the plan, runs nothing):

```
~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --train-inference-only --dry-run
```

**3. Train + inference** (10 epochs per fold):

```
~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --train-inference-only
```

For each fold the model is trained on four folds and then predicts the examinations of the fold that was held out. Completed folds/stages are skipped automatically, so the same command is used to resume. Results go to `experiments/unet_10ep/`.

**4. Evaluate**, after all folds finish:

```
~/venvs/lymphoma_seg/bin/python evaluate_cv_light.py
```

This calculates Dice, TMTV and SUVmax over the held-out predictions and writes a CSV and a summary to `experiments/unet_10ep/light_eval/`. To evaluate only finished folds, add e.g. `--folds 0`.

### Measures

Defined in `evaluate_cv_light.py` and used for both models:

| Measure | Definition |
|---|---|
| Dice (DSC) | `2*|P & G| / (|P| + |G|)` on binary masks, one value per examination |
| TMTV | Number of positive voxels times the voxel volume, in ml |
| SUVmax | Maximum of the SUV image inside the mask |
| Errors | Absolute error `|Pred - GT|`; relative error `(Pred - GT) / GT`, signed |

There is no lesion matching and no connected-component analysis. Prediction, ground truth and SUV image must lie on the same grid; the scripts stop instead of resampling.

## Pretrained autoPET III

The official repository is a modified nnU-Net 2.5.1, and the checkpoint needs its custom trainer and network, so the code in `autopet-3-submission/` is required. Inference is the standard command named in its readme:

```
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

Our scripts call exactly this, with the official defaults: the five folds are ensembled, mirroring test-time augmentation is on, step size 0.5. The model expects CT as channel `_0000` and PET (SUV) as channel `_0001`, which is the layout `preparation_for_training.py` produces.

**1. Download the checkpoint** (from the project folder, Windows or WSL):

```
python download_autopet_checkpoint.py
```

It downloads `autoPET-3-LesionTracer.zip` (3.55 GB) from [Zenodo record 14007247](https://zenodo.org/records/14007247), checks size and MD5 against the record, and extracts only `dataset.json`, `plans.json` and the five `fold_X/checkpoint_final.pth` into:

```
checkpoints/autopet3/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3/
```

An interrupted download is resumed when the script is started again.

**2. Create the environment** (inside WSL, once):

```
cd "/mnt/c/path/to/manu"
bash autopet_setup_env.sh
```

**3. Check the setup** (loads the checkpoints, runs no inference):

```
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/autopet3/bin/python autopet_check_setup.py
```

It must end with `OK - checkpoint and code are compatible.`

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is required: the official code calls `torch.load` in a way that PyTorch 2.6 and newer reject for these checkpoints. Setting the variable avoids editing the official repository. `autopet_smoke_one_case.sh` and `autopet_run_subset.py` set it themselves.

**4. Smoke test on one examination:**

```
bash autopet_smoke_one_case.sh
```

It predicts `PETCT_04ab5c61c9_11-22-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42241`, one of the shortest examinations, and writes the mask and `smoke.log` to `~/autopet3_runs/smoke/`. This took about 10 minutes on the RTX 5060 Laptop GPU. Another case can be given as the first argument, in quotes.

**5. Score the smoke test:**

```
~/venvs/lymphoma_seg/bin/python evaluate_autopet_light.py
```

Without arguments it scores `~/autopet3_runs/smoke/output` and writes `~/autopet3_runs/smoke/light_metrics_smoke.csv`. Before scoring it prints size, spacing, origin and value counts of the SUV image, the ground truth and the prediction.

## The 10-patient subset

With all five folds and test-time augmentation, one examination takes several minutes on an 8 GB laptop GPU, so the pretrained model is run on a fixed subset of 10 patients instead of all 145 examinations.

**Selection.** `autopet_run_subset.py` sorts the patients of `cases.json`, draws 10 with `random.Random(42).sample`, and takes one examination per patient (the first by name if a patient has several). The result is written to `experiments/autopet3_subset10/manifest.json` with `case_id`, `patient`, `exam` and `baseline_fold` for each case. If the manifest already exists, a later run must reproduce the same selection or it stops.

The 10 selected examinations:

| case_id | Baseline fold |
|---|---|
| `PETCT_0b57b247b6_05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966` | 2 |
| `PETCT_1fe6e48293_06-26-2003-NA-PET-CT Ganzkoerper  primaer mit KM-45853` | 1 |
| `PETCT_22d07bdc49_01-05-2003-NA-PET-CT Ganzkoerper  primaer mit KM-77260` | 2 |
| `PETCT_249dd35d0c_05-20-2005-NA-PET-CT Ganzkoerper  primaer mit KM-75049` | 3 |
| `PETCT_30001118d0_04-03-1999-NA-PET-CT Ganzkoerper  primaer mit KM-62786` | 3 |
| `PETCT_5ef99b0b6b_09-07-2000-NA-PET-CT Ganzkoerper  primaer mit KM-44649` | 4 |
| `PETCT_62405d117c_03-11-2001-NA-PET-CT Ganzkoerper  primaer mit KM-26235` | 0 |
| `PETCT_6a3477cd9a_01-15-2005-NA-PET-CT Ganzkoerper  primaer mit KM-52826` | 1 |
| `PETCT_c4a686881e_04-23-1999-NA-PET-CT Ganzkoerper  primaer mit KM-83878` | 0 |
| `PETCT_f47e31ceb5_07-26-2003-NA-PET-CT Ganzkoerper  primaer mit KM-36586` | 0 |

**1. Prepare** (select, write the manifest, link the inputs; no inference):

```
~/venvs/autopet3/bin/python autopet_run_subset.py --prepare-only
```

The CT and SUV files are linked, not copied, into `~/autopet3_runs/autopet3_subset10/input`.

**2. Run inference:**

```
~/venvs/autopet3/bin/python autopet_run_subset.py 2>&1 | tee -a ~/autopet3_runs/autopet3_subset10/run.log
```

Predictions are written to `experiments/autopet3_subset10/predictions/`, one `<case_id>.nii.gz` per examination. The expected run time is roughly 2 hours on the RTX 5060 Laptop GPU; this is an estimate from the speed of the smoke test.

**Resuming.** The script passes `--continue_prediction`, so examinations whose prediction already exists are skipped. After an interruption, run the same command again.

**3. Check completion:**

```
~/venvs/autopet3/bin/python autopet_run_subset.py --status
```

Each case is listed as `done`, `missing` or `CORRUPT`, followed by `10 of 10 predictions complete` when everything is there. A prediction counts as done only if the whole file can be read.

**4. Evaluate** (Dice, TMTV, SUVmax):

```
~/venvs/lymphoma_seg/bin/python evaluate_autopet_light.py \
    --pred-dir experiments/autopet3_subset10/predictions \
    --out experiments/autopet3_subset10/light_metrics.csv
```

**5. Baseline results for the same 10 examinations.** `case_id` in the manifest is `ImageID` in the baseline results, so the matching rows are selected like this:

```
~/venvs/lymphoma_seg/bin/python - <<'EOF'
import json
import pandas as pd

manifest = json.load(open("experiments/autopet3_subset10/manifest.json"))
ids = [c["case_id"] for c in manifest["cases"]]
baseline = pd.read_csv("experiments/unet_10ep/light_eval/light_metrics_all_folds.csv").set_index("ImageID").loc[ids]
print(baseline[["DSC", "TMTV_absolute_error_ml", "SUVmax_absolute_error"]].describe().loc[["count", "mean", "50%"]])
EOF
```

Compare the autoPET III subset with these 10 rows, not with the baseline's mean over all 145 examinations.

## Results and limitations

### Microsoft baseline: 5-fold cross-validation, 145 examinations

Every examination is predicted once, by the model of the fold in which it was held out. Full tables: `experiments/unet_10ep/light_eval/light_summary_all_folds.md` and `light_metrics_all_folds.csv`.

| Measure | Value |
|---|---|
| DSC mean | 0.3014 |
| DSC median | 0.2672 |
| DSC std | 0.2369 |
| DSC min / max | 0.0000 / 0.8699 |
| TMTV mean absolute error | 751.754 ml |
| TMTV median absolute error | 572.498 ml |
| TMTV median relative error (signed) | +3.689 |
| SUVmax mean absolute error | 51.057 |
| SUVmax median absolute error | 21.770 |
| SUVmax median relative error (signed) | +1.354 |
| Empty predictions | 0 of 145 |

| Fold | Examinations | DSC mean | DSC median |
|---|---|---|---|
| 0 | 30 | 0.2714 | 0.2742 |
| 1 | 29 | 0.2917 | 0.2520 |
| 2 | 29 | 0.3318 | 0.3437 |
| 3 | 29 | 0.1843 | 0.1526 |
| 4 | 28 | 0.4332 | 0.4299 |

This is a short run: 10 epochs per fold. The positive relative errors show that the model segments far too much volume. The run used `--train-inference-only`, so Microsoft's own metric scripts (false-positive / false-negative volume, lesion detection) were not run for all folds.

On the 10 examinations of the subset, the same baseline results give a mean DSC of 0.3544 and a median DSC of 0.2605.

### Pretrained autoPET III: orientation only

- **Smoke test, one examination** (`PETCT_04ab5c61c9_…KM-42241`): DSC 0.8894, TMTV 45.87 ml against 46.50 ml in the ground truth, SUVmax 17.84 for both. The baseline's result for this same examination is DSC 0.1243. One case is not a result for the model.
- **10-patient subset:** prepared, as described above. No results are recorded in this repository yet.

### Limitations

- **Training-data overlap.** The autoPET III checkpoint was trained on the autoPET dataset (its `dataset.json` lists 1611 training cases), which is the source of our lymphoma examinations. Its predictions here are therefore not held-out predictions, and its scores are expected to be optimistic. The baseline's scores are true cross-validation scores. The two are not a like-for-like comparison.
- **Different amounts of training.** The baseline was trained for 10 epochs on 115–117 examinations per fold. The autoPET III model is a five-model ensemble from a challenge-winning pipeline.
- **Subset size.** Ten examinations give an impression, not a statistic.
- **Three whole-mask measures only.** No per-lesion detection results are reported.

## Collaboration

**Ignored by Git** (see `.gitignore`):

| Path | Why |
|---|---|
| `data/` | Dataset, about 17 GB |
| `checkpoints/` | autoPET III ZIP and extracted checkpoints |
| `lymphoma-segmentation-dnn/` | Third-party clone |
| `autopet-3-submission/` | Third-party clone |
| `experiments/` | Model checkpoints, predictions and working files |
| `results/`, `answers/` | Local output and notes |
| `.venv/`, `.idea/`, `.mypy_cache/`, `__pycache__/` | Local tooling |

`experiments/` is ignored as a whole. The small result files of the baseline that are in the repository (`manifest.json`, logs, `summary/`, `light_eval/`) were added explicitly with `git add -f`. Do the same for a new result file you want to share, and never for `work/`, predictions or checkpoints.

**To obtain separately**

| What | How |
|---|---|
| Dataset | `python download_lymphoma.py`, then `python preparation_for_training.py` |
| Microsoft code | `git clone`, see [Environment setup](#environment-setup) |
| autoPET III code | `git clone`, see [Environment setup](#environment-setup) |
| autoPET III checkpoint | `python download_autopet_checkpoint.py` |
| Python environments | Commands in [Environment setup](#environment-setup) |
| Baseline model checkpoints and predictions | Not shared; reproduce with `run_cv_pipeline.py` |

**Reproducing from a fresh clone**

1. [Environment setup](#environment-setup): clone the two repositories, create both environments.
2. [Dataset preparation](#dataset-preparation): download, organize, copy into WSL.
3. [Microsoft baseline](#microsoft-baseline): datalist, dry run, train + inference, evaluate.
4. [Pretrained autoPET III](#pretrained-autopet-iii): download the checkpoint, check the setup, smoke test.
5. [The 10-patient subset](#the-10-patient-subset): prepare, run, check, evaluate.

The fold assignment and the subset selection are both seeded with 42, so a fresh preparation gives the same folds and the same 10 examinations. Trained baseline weights will differ slightly between runs.
