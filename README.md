# MANU — lymphoma PET/CT segmentation

Train a model to locate and segment lymphoma lesions in PET/CT scans. The baseline adapts [Microsoft’s MIT-licensed implementation](https://github.com/microsoft/lymphoma-segmentation-dnn/tree/81129243db3c868370a7664c9adc2ad52609484f).

## Data

The local [autoPET v2 cohort](https://doi.org/10.57754/FDAT.8f14a-pf846) contains **145 lesion-positive examinations from 144 patients**, under `data/autopet-v2-lymphoma/`. Each examination has:

- `CTres.nii.gz`: CT anatomy aligned to PET.
- `SUV.nii.gz`: quantitative PET uptake.
- `SEG.nii.gz`: reference lesion mask used for supervision, not as an input channel.

Original images stay unchanged. Both visits of the repeated patient stay in the same fold.

## Setup

Use Python 3.12. The commands below start from the project root. From `scripts/`, use `python train.py` instead; data, output, split and cache paths always resolve relative to the project root. Absolute paths are also accepted.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

For an existing environment, just activate it with `source .venv/bin/activate`.

## Run

```bash
# Audit the data and save patient splits; no training.
python scripts/train.py --prepare

# Small execution check in a separate output directory.
python scripts/train.py --smoke --output runs/smoke

# Train one fold, or all five sequentially.
python scripts/train.py --fold 0
python scripts/train.py --fold all

# Continue the existing experiment from its last saved epoch.
python scripts/train.py --fold all --resume
```

Running without arguments prints help. The smoke run uses two training examinations and one separate validation examination; it does not produce held-out results.

## Configuration and resume

Edit [config.py](config.py) before a new experiment. CLI flags override its defaults. Current defaults are 64³ patches, training batch 1, learning rate `2e-4`, up to 500 epochs, validation every 2 epochs, 16 CPU threads and 8 inference windows per batch. The shared disk cache has a 160 GiB limit and keeps at least 40 GiB free; it fills as needed.

Early stopping starts no earlier than epoch 100, after 25 validation checks without a Dice improvement greater than 0.001. Resume preserves its history and the original learning-rate schedule. The best checkpoint is still selected on every score improvement.

Keep the same scientific and resource settings across all five folds and on resume. Use a new output directory for changed settings. To pause at a chosen epoch without changing the schedule or running held-out inference:

```bash
python scripts/train.py --fold 0 --resume --stop-after-epoch 100
```

Resume rejects incompatible settings, data, software versions and unknown source changes. Only the exact known AMP fixes, pilot stopping-policy upgrade, refactor, directory move and project-path fix are allowed by [training_runtime.py](scripts/training_runtime.py); accepted upgrades write `*-resume-migration.json` records.

## Results

Outputs default to `runs/full-training/`:

| File | Contents |
|---|---|
| `run.json`, `splits.json` | Configuration, provenance and patient assignments |
| `logs/` | Live progress, timings, memory and errors |
| `fold_N/history.json` | Training loss and inner-validation Dice by epoch |
| `fold_N/last.pt`, `fold_N/best.pt` | Resume state and selected checkpoint |
| `fold_N/predictions/`, `fold_N/metrics.json` | Native-grid held-out masks and per-case Dice |
| `fold_N/complete.json` | Fold completion record |
| `metrics.json` | Summary after all five folds, with one prediction per examination |

Check these files for current progress. Inner-validation Dice selects checkpoints; it is not the final held-out score or the percentage of lesions detected.

## Research notes

The residual 3D U-Net takes CT and PET as inputs. It uses one random patch per training examination per epoch, preserves quantitative PET values and the full field of view, and exports masks in the original scan geometry. Patient-level five-fold splitting, separate inner validation and smaller patches are adaptations of the upstream baseline, not an exact paper reproduction.

Mixed-precision recovery skips overflowing gradient updates and retries overflowing forward passes in float32 after restoring model buffers and random state. Persistent invalid results stop training. Reproduction evidence is saved locally under `runs/amp-reproduction/` and `runs/loss-reproduction/`; the pilot review is under `runs/pilot-review/`.

This positive-only cohort cannot establish performance on lesion-negative examinations or patient-level lymphoma diagnosis. Lesion-level metrics, TMTV/SUVmax, the exploratory autoPET III comparison and a defined GenAI follow-up remain separate project tasks.

## Tests and code

```bash
python -m pip install pytest==9.1.1 mypy==2.1.0
python -m pytest -q
python -m mypy --check-untyped-defs --follow-imports=skip --ignore-missing-imports scripts config.py
```

The latest code verification passed 37 tests and type checking. Tests cover patient splits, geometry, caching, numerical recovery, resume and held-out routing; they do not establish segmentation quality. CUDA multi-GPU execution remains unverified; use `--help` for optional DDP and resource settings.

The training loop is in `scripts/train.py`. Its helpers are `training_data.py`, `training_options.py`, `training_runtime.py` and `preprocessing_cache.py`, all under `scripts/`. Defaults stay in `config.py`.
