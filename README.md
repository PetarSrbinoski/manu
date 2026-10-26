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

## Compare 96³ and 128³ patches across all five folds

Use the existing trainer through the comparison launcher. It runs **all five folds
at 96³, then all five at 128³**, sequentially on the selected GPU. Each size starts
from scratch, with its own checkpoints and predictions. Both share the same
patient/inner-validation splits. Defaults remain a 500-epoch cosine schedule with
early stopping, training batch 1, 2 mm spacing and 16 CPU threads. Inference uses
one window at a time to limit memory with the larger patches.

```bash
# From the project root; no environment activation needed.
# Preview the commands without auditing data or launching training.
.venv/bin/python scripts/train_patch_comparison.py

# Recommended first: bounded training + full-volume validation for BOTH sizes.
# GPU fit is not yet verified. These checks do not establish segmentation quality.
.venv/bin/python scripts/train_patch_comparison.py --smoke

# Train BOTH sizes, five folds each (10 fold runs in total).
.venv/bin/python scripts/train_patch_comparison.py --run

# Continue after interruption; verify/skip completed folds, resume partial folds.
.venv/bin/python scripts/train_patch_comparison.py --run --resume

# Alternatively, run one size at a time.
.venv/bin/python scripts/train_patch_comparison.py --run --patch 96
.venv/bin/python scripts/train_patch_comparison.py --run --patch 128
```

Outputs are under `runs/patch-comparison/`: shared `splits.json`, separate
`smoke96/` and `smoke128/` diagnostics, and full `patch96/` and `patch128/` runs.
Each full run has its own `run.json`, `logs/`, five fold directories and final
`metrics.json`. The original `runs/full-training/` is preserved. The first launch
audits the data and prepares splits if needed; auditing can take several minutes.

The launcher stops on any failed child command. It does not silently reduce patch
size or change the recipe after an out-of-memory error. Smoke checks are a separate
command, not automatically launched by `--run`. Repeating smoke checks requires a
fresh `--output-root`, since completed diagnostics are protected from overwrite.
For full runs, use the same output root and unchanged settings when resuming.
An interruption before the first checkpoint may require a new output root; the
trainer will report that explicitly. Do not run the two sizes simultaneously on
the laptop GPU. Keep the terminal/WSL session alive, or resume after interruption.

Comparison defaults live in `config.py` under `PATCH_COMPARISON_*`; the remaining
settings come from the existing trainer defaults. Do not edit them between folds.
After both experiments finish, compare held-out results descriptively; use inner
validation, not repeated outer-fold inspection, to choose further changes.

## Configuration and resume

Edit [config.py](config.py) before a new experiment. CLI flags override its defaults. Current defaults are 64³ patches, training batch 1, learning rate `2e-4`, up to 500 epochs, validation every 2 epochs, 16 CPU threads and 8 inference windows per batch. The shared disk cache has a 160 GiB limit and keeps at least 40 GiB free; it fills as needed.

Early stopping starts no earlier than epoch 100, after 25 validation checks without a Dice improvement greater than 0.001. Resume preserves its history and the original learning-rate schedule. The best checkpoint is still selected on every score improvement.

Keep the same scientific and resource settings across all five folds and on resume. Use a new output directory for changed settings. To pause at a chosen epoch without changing the schedule or running held-out inference:

```bash
python scripts/train.py --fold 0 --resume --stop-after-epoch 100
```

Resume rejects incompatible settings, data, software versions and unknown source changes. Only the exact known AMP fixes, pilot stopping-policy upgrade, refactor, directory move and project-path fix are allowed by [training_runtime.py](scripts/training_runtime.py); accepted upgrades write `*-resume-migration.json` records.

The architecture cleanup also permits the exact pre-cleanup trainer hash
`4bcbfa2cc1457a49e1d4cee3deaaba133ababe842f870b28f933a91e8cdbc972`:
only unused forwarding imports changed in the training loop module. Scientific,
resource and version settings must still match; continuation records a
`cleanup-resume-migration.json`. The pilot launcher accepts the exact paired
pre-cleanup trainer/launcher hashes with all other protocol fields unchanged,
retains the original `pilot_protocol.json`, and records a separate
`cleanup-protocol-migration.json`. Review and pilot comparisons share completion,
unique-examination and reference/PET identity checks in `analysis_common.py`.
Verification covered identical training function bodies, 10,368 unchanged pilot
scoring outputs and synthetic checkpoint continuation for all three pilot arms.

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

Analysis tools for native-grid lesion/volume/SUV evaluation, patch memory/throughput
benchmarks, sampling coverage and saved-checkpoint inner-validation predictions
are described in [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md). Each writes an isolated
report and preserves its inputs.

The opt-in lesion-balanced sampler, Dice+cross-entropy loss and three matched
128³ inner-validation pilot arms are
described in the [experiment summary](architecture-cleanup-and-pilot-results-summary.md#sampling-and-loss-pilot-128³). The original voxel
sampler and Dice loss remain the defaults. Exact recorded source migrations
allow unchanged older runs to resume; changing sampling/loss requires a fresh
experiment directory. The pilot launcher runs only 128³, never the 96³ experiment.

The user completed all three 50-epoch pilots and evaluation on 23 inner-validation
examinations. Dice improved (0.516 → 0.531 → 0.565), but small-lesion detection
declined (37 → 30 → 17 of 348); neither candidate passed the frozen screen.
See [completed pilot results and next steps](architecture-cleanup-and-pilot-results-summary.md).
A new five-fold candidate run and a fourth loss-only arm have not been launched.

```bash
python -m pip install pytest==9.1.1 mypy==2.1.0
python -m pytest -q
python -m mypy --check-untyped-defs --follow-imports=skip --ignore-missing-imports scripts config.py
```

The CPU regression suite covers patient splits, geometry, caching, numerical
recovery, resume, held-out routing, evaluation and opt-in sampling. These checks
do not establish segmentation quality. CUDA multi-GPU execution remains
unverified; use `--help` for optional DDP and resource settings.

The training loop is in `scripts/train.py`. Its helpers are `training_data.py`, `training_options.py`, `training_runtime.py` and `preprocessing_cache.py`, all under `scripts/`. Defaults stay in `config.py`.
