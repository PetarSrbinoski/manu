"""Edit training defaults here. CLI options override corresponding values.

Running train.py without a mode still only prints help. Scientific changes need
new experiment outputs; resume checks reject incompatible settings. Keep the
same scientific settings across the five folds of a reported experiment.
"""
import math
from pathlib import Path

# Experiment and data. --smoke uses one epoch unless --epochs is specified.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data/autopet-v2-lymphoma'
OUTPUT_ROOT = PROJECT_ROOT / 'runs/full-training'
EXPECTED_EXAMS = 145
EXPECTED_PATIENTS = 144
EPOCHS = 500
SEED = 20261006
INNER_VALIDATION_FRACTION = 0.20
VALIDATION_INTERVAL = 2
# Inner-validation checks, not epochs. Zero patience disables automatic stopping.
EARLY_STOPPING_PATIENCE = 25         # 50 epochs without meaningful improvement at interval 2.
EARLY_STOPPING_MIN_DELTA = 0.001     # Absolute foreground Dice improvement.
EARLY_STOPPING_MIN_EPOCHS = 100      # Allow an initial learning period; retain the 500-epoch schedule.

# Scientific settings: the upstream model/recipe with a smaller explicit patch.
PATCH_SIZE = 64
BATCH_SIZE = 1                      # Per GPU.
GRADIENT_ACCUMULATION = 1   
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
MIN_LEARNING_RATE = 0.0             # Cosine schedule's final rate.
LOSS_INCLUDE_BACKGROUND = True
MODEL_CHANNELS = (16, 32, 64, 128, 256, 512)
MODEL_STRIDES = (2, 2, 2, 2, 2)
RESIDUAL_UNITS = 2
NORMALIZATION = 'BATCH'
SPACING_MM = (2.0, 2.0, 2.0)
CT_CLIP_HU = (-154, 325)
POSITIVE_SAMPLE_WEIGHT = 2
NEGATIVE_SAMPLE_WEIGHT = 1
BACKGROUND_PET_THRESHOLD = 0.0
AFFINE_PROBABILITY = 0.5
TRANSLATION_RANGE = (10, 10, 10)
ROTATION_RANGE = (0, 0, math.pi / 15)
SCALE_RANGE = (0.1, 0.1, 0.1)
INFERENCE_OVERLAP = 0.25

# Resource settings: these do not automatically change the scientific settings.
DEVICE = 'cuda:0'
WORKERS = 0
CPU_THREADS = 16                    # 22 threads slowed CPU stitching substantially on this laptop.
RAM_CACHE_GIB = 0.0
SLIDING_WINDOW_BATCH_SIZE = 8        # Faster than 1/4/16/32/64 in the measured 64-cubed sweep.
DISK_CACHE_DIR = PROJECT_ROOT / 'runs/preprocessed-cache'  # Set to None to disable.
DISK_CACHE_GIB = 160.0               # Cohort estimate <138 GiB; shared bounded cache across folds.
MIN_FREE_DISK_GIB = 40.0             # Include Windows C: free space when running under WSL.

# Diagnostic run and console progress.
SMOKE_STEPS = 20
LOG_EVERY = 10
