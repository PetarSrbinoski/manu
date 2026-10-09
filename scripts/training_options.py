"""Command-line validation and the frozen scientific/resource run description."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import config as settings


def scientific_recipe() -> dict:
    names = ('INNER_VALIDATION_FRACTION', 'WEIGHT_DECAY', 'MIN_LEARNING_RATE',
             'LOSS_INCLUDE_BACKGROUND', 'MODEL_CHANNELS', 'MODEL_STRIDES', 'RESIDUAL_UNITS',
             'NORMALIZATION', 'SPACING_MM', 'CT_CLIP_HU', 'POSITIVE_SAMPLE_WEIGHT',
             'NEGATIVE_SAMPLE_WEIGHT', 'BACKGROUND_PET_THRESHOLD', 'AFFINE_PROBABILITY',
             'TRANSLATION_RANGE', 'ROTATION_RANGE', 'SCALE_RANGE', 'INFERENCE_OVERLAP')
    return json.loads(json.dumps({name.lower(): getattr(settings, name) for name in names}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Patient-level autoPET v2 adaptation of Microsoft's lymphoma baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Audit data and persist splits; no training')
    mode.add_argument('--smoke', action='store_true', help='Bounded training + one full-volume inner validation')
    mode.add_argument('--fold', choices=['0', '1', '2', '3', '4', 'all'])
    parser.add_argument('--data-root', type=Path, default=settings.DATA_ROOT)
    parser.add_argument('--output', type=Path, default=settings.OUTPUT_ROOT)
    parser.add_argument('--splits', type=Path, help='Existing shared manifest; create with --prepare first')
    parser.add_argument('--epochs', type=int, help='Total cosine-schedule budget (default: config.EPOCHS; smoke: 1)')
    parser.add_argument('--patch', type=int, default=settings.PATCH_SIZE, help='Scientific choice; multiple of 32, at least 64')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE, help='Training patches per GPU')
    parser.add_argument('--accumulate', type=int, default=settings.GRADIENT_ACCUMULATION, help='Microbatches per optimizer update')
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE)
    parser.add_argument('--seed', type=int, default=settings.SEED)
    parser.add_argument('--val-interval', type=int, default=settings.VALIDATION_INTERVAL)
    parser.add_argument('--early-stopping-patience', type=int, default=settings.EARLY_STOPPING_PATIENCE,
                        help='Validation checks without meaningful improvement; 0 disables')
    parser.add_argument('--early-stopping-min-delta', type=float, default=settings.EARLY_STOPPING_MIN_DELTA)
    parser.add_argument('--early-stopping-min-epochs', type=int, default=settings.EARLY_STOPPING_MIN_EPOCHS)
    parser.add_argument('--device', default=settings.DEVICE, help='Explicit single device; use cpu only for tests')
    parser.add_argument('--ddp', action='store_true', help='Explicit torchrun one-process-per-GPU execution')
    parser.add_argument('--workers', type=int, default=settings.WORKERS, help='Data-loader workers per rank')
    parser.add_argument('--threads', type=int, default=settings.CPU_THREADS, help='CPU compute threads per process')
    parser.add_argument('--cache-gib', type=float, default=settings.RAM_CACHE_GIB, help='Deterministic tensor cache per rank; workers must be 0')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--disk-cache', type=Path, default=settings.DISK_CACHE_DIR,
                       help='Shared deterministic preprocessing cache (memory-mapped arrays)')
    cache.add_argument('--no-disk-cache', action='store_const', dest='disk_cache', const=None,
                       help='Disable the configured disk cache for a baseline comparison')
    parser.add_argument('--disk-cache-gib', type=float, default=settings.DISK_CACHE_GIB)
    parser.add_argument('--min-free-disk-gib', type=float, default=settings.MIN_FREE_DISK_GIB)
    parser.add_argument('--sw-batch-size', type=int, default=settings.SLIDING_WINDOW_BATCH_SIZE)
    parser.add_argument('--smoke-steps', type=int, default=settings.SMOKE_STEPS, help='Optimizer updates per smoke epoch')
    parser.add_argument('--log-every', type=int, default=settings.LOG_EVERY,
                        help='Training batch progress interval; first/last and updates after 30s also logged')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, help='Save then stop early without changing scheduler budget')
    parser.add_argument('--expected-exams', type=int, default=settings.EXPECTED_EXAMS)
    parser.add_argument('--expected-patients', type=int, default=settings.EXPECTED_PATIENTS)
    args = parser.parse_args(argv)
    # Relative paths belong to the project, regardless of the launch directory.
    for name in ('data_root', 'output', 'splits', 'disk_cache'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (settings.PROJECT_ROOT / value).resolve())
    if not (args.prepare or args.smoke or args.fold is not None):
        parser.print_help()
        return None
    if args.smoke:
        args.epochs = 1 if args.epochs is None else args.epochs
        if args.output == (settings.PROJECT_ROOT / settings.OUTPUT_ROOT).resolve():
            parser.error('--smoke requires its own explicit --output directory')
    if not args.smoke and args.epochs is None:
        args.epochs = settings.EPOCHS
    if not args.prepare and (args.epochs is None or args.epochs < 1):
        parser.error('training requires a positive epoch budget in config.py or --epochs')
    if (not settings.MODEL_STRIDES or any(not isinstance(s, int) or s < 1 for s in settings.MODEL_STRIDES)
            or any(not isinstance(c, int) or c < 1 for c in settings.MODEL_CHANNELS)):
        parser.error('Model channels and strides must be positive integers')
    stride_product = math.prod(settings.MODEL_STRIDES)
    if args.patch < 2 * stride_product or args.patch % stride_product:
        parser.error(f'--patch must be a multiple of {stride_product} and at least {2 * stride_product}')
    if len(settings.MODEL_CHANNELS) != len(settings.MODEL_STRIDES) + 1:
        parser.error('config.MODEL_CHANNELS needs one more entry than MODEL_STRIDES')
    if (not 0 < settings.INNER_VALIDATION_FRACTION < 1 or not 0 <= settings.INFERENCE_OVERLAP < 1
            or any(not math.isfinite(v) or v <= 0 for v in settings.SPACING_MM)
            or settings.CT_CLIP_HU[0] >= settings.CT_CLIP_HU[1]
            or settings.POSITIVE_SAMPLE_WEIGHT < 0 or settings.NEGATIVE_SAMPLE_WEIGHT < 0
            or settings.POSITIVE_SAMPLE_WEIGHT + settings.NEGATIVE_SAMPLE_WEIGHT <= 0
            or not 0 <= settings.AFFINE_PROBABILITY <= 1):
        parser.error('Invalid scientific settings in config.py')
    if any(v < 1 for v in (args.batch_size, args.accumulate, args.val_interval, args.sw_batch_size,
                           args.smoke_steps, args.expected_exams, args.expected_patients, args.threads, args.log_every)):
        parser.error('batch sizes, accumulation, intervals, steps and cohort sizes must be positive')
    if args.workers < 0 or not math.isfinite(args.cache_gib) or args.cache_gib < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('invalid resource budget or learning rate')
    if (args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 1
            or not math.isfinite(args.early_stopping_min_delta) or not 0 <= args.early_stopping_min_delta < 1):
        parser.error('Invalid early-stopping policy: patience >= 0, minimum epochs >= 1, and 0 <= delta < 1 required')
    if args.workers and args.cache_gib:
        parser.error('bounded caching requires --workers 0 to prevent worker cache copies')
    if args.disk_cache and args.cache_gib:
        parser.error('Choose disk caching or RAM caching, rather than duplicating cached volumes')
    if (not math.isfinite(args.disk_cache_gib) or args.disk_cache_gib <= 0
            or not math.isfinite(args.min_free_disk_gib) or args.min_free_disk_gib < 0):
        parser.error('Disk cache size must be positive and free-space reserve nonnegative')
    if args.disk_cache and (args.disk_cache.resolve() == args.data_root.resolve()
                           or args.data_root.resolve() in args.disk_cache.resolve().parents):
        parser.error('Disk cache must be outside the original image directory')
    if not 0 <= args.seed < 2**32 - 1000000:
        parser.error('seed out of supported range')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error('--stop-after-epoch must be within the total epoch budget')
    if args.ddp and (int(os.environ.get('WORLD_SIZE', '1')) < 2 or 'LOCAL_RANK' not in os.environ):
        parser.error('--ddp requires torchrun with at least two GPU processes')
    if not args.ddp and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('torchrun requires explicit --ddp; refusing duplicate single-GPU runs')
    if args.prepare and args.ddp:
        parser.error('run --prepare once before distributed execution')
    return args


def stopping_policy(args) -> dict:
    return {'patience': args.early_stopping_patience,
            'min_delta': args.early_stopping_min_delta, 'min_epochs': args.early_stopping_min_epochs}


def resolved_config(args, split_id: str, environment: dict, device, world: int) -> tuple[dict, dict]:
    """Keep scientific choices fixed across folds and hardware; preserve resume keys."""
    scientific = {**scientific_recipe(), 'patch': args.patch, 'channels': list(settings.MODEL_CHANNELS),
                  'input_order': ['CT', 'PET_SUV'],
                  'foreground_crop': 'none-full-field-of-view',
                  'loss': 'softmax-Dice', 'lr': args.lr,
                  'epochs': args.epochs, 'seed': args.seed, 'val_interval': args.val_interval,
                  'early_stopping': stopping_policy(args),
                  'batch_size_per_gpu': args.batch_size, 'accumulate': args.accumulate,
                  'effective_batch_size': args.batch_size * args.accumulate * world,
                  'world_size': world, 'smoke': args.smoke,
                  'smoke_steps': args.smoke_steps if args.smoke else None}
    config = {'scientific': scientific, 'split_id': split_id,
              'versions': environment['packages'], 'script_sha256': environment['script_sha256'],
              'precision': 'cuda-amp-fp16' if device.type == 'cuda' else 'fp32',
              'sampler': 'DistributedSampler-pad-shuffle-epoch-v1',
              'workers': args.workers, 'threads': args.threads, 'cache_gib': args.cache_gib,
              'sw_batch_size': args.sw_batch_size, 'disk_cache_enabled': args.disk_cache is not None,
              'disk_cache_gib': args.disk_cache_gib, 'min_free_disk_gib': args.min_free_disk_gib}
    resources = {'device': str(device), 'workers_per_rank': args.workers, 'cache_gib_per_rank': args.cache_gib,
                 'aggregate_cache_gib': args.cache_gib * world,
                 'sw_batch_size': args.sw_batch_size, 'world_size': world, 'threads_per_process': args.threads,
                 'disk_cache': str(args.disk_cache) if args.disk_cache else None,
                 'disk_cache_gib': args.disk_cache_gib, 'min_free_disk_gib': args.min_free_disk_gib}
    return config, resources
