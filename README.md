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

## Scripts

| Script | Purpose |
|---|---|
| `download_lymphoma.py` | Downloads the lymphoma subset (CTres, SUV, SEG) into `data/autopet-v2-lymphoma` |
| `preparation_for_training.py` | Organizes the data into `data/lymphoma-baseline` and creates the patient-level 5-fold split |
| `make_datalist.py` | Creates the datalist (`data_split/train_filepaths.csv`) expected by the Microsoft training code and verifies the split |
| `run_cv_pipeline.py` | Runs the 5-fold training/inference pipeline, fold by fold |
| `evaluate_cv_light.py` | Evaluates the predictions using Dice, TMTV and SUVmax |
| `cv_lesion_measures.py` | Additional lesion-measure helper (SUVmean, SUVmax, lesion count, TMTV, TLG); called by `run_cv_pipeline.py` only in its full mode |

Training uses the unmodified [microsoft/lymphoma-segmentation-dnn](https://github.com/microsoft/lymphoma-segmentation-dnn), cloned into `lymphoma-segmentation-dnn/`.

## How to run

**1. Prepare the data** (from the project folder):

```
python download_lymphoma.py
python preparation_for_training.py
```

**2. Copy the dataset into WSL** (training reads it from there, it is much faster):

```
cd "/mnt/c/Users/eva/Desktop/6th semester/Projects/manu"
mkdir -p ~/data && cp -r data/lymphoma-baseline ~/data/
```

Everything from here on runs inside WSL, from the project folder, with the `~/venvs/lymphoma_seg` environment.

**3. Create the datalist:**

```
~/venvs/lymphoma_seg/bin/python make_datalist.py --dataset-root /home/eva25/data/lymphoma-baseline
```

**4. Dry run** (checks everything and prints the plan, runs nothing):

```
~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --train-inference-only --dry-run
```

**5. Train + inference** (10 epochs per fold):

```
~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --train-inference-only
```

Completed folds/stages are skipped automatically, so the same command is used to resume. Results go to `experiments/unet_10ep/`.

**6. Evaluate**, after all folds finish:

```
~/venvs/lymphoma_seg/bin/python evaluate_cv_light.py
```

This calculates Dice, TMTV and SUVmax over the held-out predictions and writes a CSV and a summary to `experiments/unet_10ep/light_eval/`. To evaluate only finished folds, add e.g. `--folds 0`.
