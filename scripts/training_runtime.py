"""Training execution: progress logs, numerical recovery and checkpoint integrity.

Model recipe adapted from Microsoft Corporation, MIT licensed; see README.
Heavy dependencies stay lazy so command-line help needs only the standard library.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import config as settings

UPSTREAM = '81129243db3c868370a7664c9adc2ad52609484f'
FORMAT_VERSION = 1
# The exact pre-fix implementation is compatible up to its first AMP overflow:
# it aborted before updating weights. Do not extend this to arbitrary revisions.
AMP_OVERFLOW_PREDECESSOR = 'd78ee89f3258ae33552ca1d13e6d2aa8876c3fb1170c035aa1093572648e30ca'
AMP_FORWARD_PREDECESSOR = '09702c7fa41f85fb99aef4b99d02a6fc8cf011ab3a28fa0f22ecb67136464ba4'
PILOT_PREDECESSOR = '2bced3078a3051563f7fe9c6c7bb25479611beecdd1ae09004371d0572253858'
# Exact source before the behavior-preserving module extraction.
REFACTOR_PREDECESSOR = '9237f2bec8a77a782c62a183cb52c289e996d461c11e75f3c3e7a342c3b8a312'
# Exact implementation before moving the modules under scripts/.
SCRIPTS_PREDECESSOR = 'cf129b8b458bcdcfdbe963cae3d3f39d344c9277ed44b93e3135636b91e42dbd'
# Exact implementation before anchoring paths to the project root.
PATHS_PREDECESSOR = '9d836647be078742b8d9a615c98387e95027a890d2a74b7bc426744b30a0f214'
# Fixed migration contract, independent of user-editable defaults. The pilot had
# no stopping policy; this adoption cannot stop it before the completed epoch 50.
PILOT_STOPPING_POLICY = {'patience': 25, 'min_delta': 0.001, 'min_epochs': 100}
LOGGER = logging.getLogger('lymphoma')
LOGGER.addHandler(logging.NullHandler())


@contextmanager
def run_logging(args):
    """One flushed log per launch/rank; independent folds never share a file."""
    rank = int(os.environ.get('RANK', '0')) if args.ddp else 0
    mode = 'prepare' if args.prepare else ('smoke' if args.smoke else f'fold_{args.fold}')
    directory = args.output / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path = directory / f'{mode}-{stamp}-rank{rank}-pid{os.getpid()}.log'
    formatter = logging.Formatter(f'%(asctime)s | %(levelname)s | rank {rank} | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logfile = logging.FileHandler(path, mode='x', encoding='utf-8')
    logfile.setLevel(logging.INFO)
    old_handlers, old_level, old_propagate = LOGGER.handlers[:], LOGGER.level, LOGGER.propagate
    LOGGER.handlers = [console, logfile]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers:
        handler.setFormatter(formatter)
    started = time.perf_counter()
    try:
        LOGGER.info('Starting %s%s | data=%s | output=%s', mode,
                    ' (resume requested)' if args.resume else '', args.data_root, args.output)
        LOGGER.info('Live log: %s', path.resolve())
        LOGGER.info('Loading PyTorch, MONAI and imaging dependencies...')
        yield
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted. Only completed epoch checkpoints can be resumed; current work is not committed.')
        raise
    except BaseException:
        LOGGER.exception('Run failed; see the error below. Earlier completed epoch checkpoints are retained.')
        raise
    else:
        LOGGER.info('Run finished successfully | elapsed=%s', elapsed_time(time.perf_counter() - started))
    finally:
        for handler in LOGGER.handlers:
            handler.close()
        LOGGER.handlers, LOGGER.propagate = old_handlers, old_propagate
        LOGGER.setLevel(old_level)


def elapsed_time(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def log_progress(phase: str, completed: int, total: int, started: float, detail: str = '') -> None:
    elapsed = time.perf_counter() - started
    remaining = elapsed * (total - completed) / completed if completed else 0
    LOGGER.info('%s | %d/%d (%.0f%%) | elapsed=%s | phase ETA~%s%s', phase,
                completed, total, 100 * completed / total, elapsed_time(elapsed),
                elapsed_time(remaining), f' | {detail}' if detail else '')


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_model():
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3, in_channels=2, out_channels=2,
                channels=settings.MODEL_CHANNELS, strides=settings.MODEL_STRIDES,
                num_res_units=settings.RESIDUAL_UNITS, norm=settings.NORMALIZATION)


def finite_tensors(tensors, synchronize: bool = False) -> bool:
    """All training ranks must agree to retry or fail before the next backward."""
    import torch
    import torch.distributed as dist
    status = torch.stack([torch.isfinite(t.detach()).all() for t in tensors]).all().to(torch.int32)
    if hasattr(status, 'as_tensor'):
        status = status.as_tensor()
    if synchronize and dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return bool(status)


def training_loss(model, inputs, labels, loss_function, amp_enabled: bool):
    """Retry an overflowing forward in float32, rolling back mutable model state."""
    import torch
    if not finite_tensors((inputs, labels), synchronize=True):
        raise RuntimeError('Nonfinite training input or label; no checkpoint written')
    # Forward passes update BatchNorm even when no backward/optimizer step runs.
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()] if amp_enabled else []
    rng = random_state() if amp_enabled else None

    def rollback():
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        if rng is not None:
            set_random_state(rng)

    for use_amp in ((True, False) if amp_enabled else (False,)):
        with torch.autocast(device_type=inputs.device.type, enabled=use_amp):
            logits = model(inputs.float())
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            loss = loss_function(logits.float(), labels)
        if finite_tensors((loss, logits, *model.buffers()), synchronize=True):
            return loss, amp_enabled and not use_amp
        del loss, logits  # Discard the failed graph before restoring saved buffers.
        rollback()
        if use_amp:
            LOGGER.warning('AMP forward overflow | restoring model buffers and RNG; retrying this batch in float32')
        else:
            raise RuntimeError('Nonfinite training loss, output or model buffers in float32; no checkpoint written')
    raise AssertionError('unreachable')


def inference_forward(model, inputs, amp_enabled: bool):
    """Keep invalid window logits out of validation metrics and exported masks."""
    import torch
    if not finite_tensors((inputs,)):
        raise RuntimeError('Nonfinite inference input')
    for use_amp in ((True, False) if amp_enabled else (False,)):
        with torch.autocast(device_type=inputs.device.type, enabled=use_amp):
            result = model(inputs.float()).float()
        if finite_tensors((result,)):
            return result
        del result
        if use_amp:
            LOGGER.warning('AMP inference overflow | retrying these windows in float32')
        else:
            raise RuntimeError('Nonfinite inference output in float32; refusing invalid predictions')
    raise AssertionError('unreachable')


def early_stopping_status(history: list, patience: int, min_delta: float, min_epochs: int) -> dict:
    """Reconstruct patience from committed inner validation; resume never resets it."""
    best, bad_checks, last_improvement, last_validation = -1.0, 0, 0, 0
    for row in history:
        score = row['inner_foreground_dice_2mm']
        if score is None:
            continue
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite inner validation score')
        last_validation = row['epoch']
        if score > best + min_delta:
            best, bad_checks, last_improvement = score, 0, row['epoch']
        else:
            bad_checks += 1
    return {'should_stop': patience > 0 and last_validation >= min_epochs and bad_checks >= patience,
            'bad_checks': bad_checks, 'last_improvement_epoch': last_improvement,
            'reference_score': best, 'patience': patience, 'min_delta': min_delta, 'min_epochs': min_epochs}


def random_state() -> dict:
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def set_random_state(state: dict) -> None:
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def atomic_torch_save(path: Path, state: dict) -> None:
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int,
                    best: float, history: list, config: dict, split_id: str,
                    rank_states: list | None, selected_checkpoint: dict | None = None) -> None:
    atomic_torch_save(path, {'version': FORMAT_VERSION, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best,
                            'history': history, 'config': config, 'split_id': split_id,
                            'selected_checkpoint': selected_checkpoint,
                            'rng': rank_states if rank_states is not None else [random_state()]})


def restore_checkpoint(path: Path, model, optimizer, scheduler, scaler,
                       config: dict, split_id: str, rank: int) -> dict:
    import torch
    # Only load locally generated, trusted checkpoints (RNG state needs pickle).
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['version'] != FORMAT_VERSION or not compatible_config(state['config'], config) or
            state['split_id'] != split_id or rank >= len(state['rng'])):
        raise ValueError('incompatible resume: configuration, split, versions or GPU count changed')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    set_random_state(state['rng'][rank])
    return state


def implementation_hash() -> str:
    return identity({name: file_hash(Path(__file__).with_name(name))
                     for name in ('train.py', 'training_data.py', 'training_runtime.py',
                                  'training_options.py', 'preprocessing_cache.py', '__init__.py')})


def compatible_config(saved: dict, current: dict) -> bool:
    """Allow exact known fixes, pilot policy adoption and the module refactor only."""
    if saved == current:
        return True
    if (saved.get('script_sha256') not in (AMP_OVERFLOW_PREDECESSOR, AMP_FORWARD_PREDECESSOR, PILOT_PREDECESSOR, REFACTOR_PREDECESSOR, SCRIPTS_PREDECESSOR, PATHS_PREDECESSOR)
            or current.get('script_sha256') != implementation_hash()):
        return False
    migrated = {**saved, 'script_sha256': current['script_sha256']}
    if saved['script_sha256'] not in (REFACTOR_PREDECESSOR, SCRIPTS_PREDECESSOR, PATHS_PREDECESSOR) and 'early_stopping' not in saved.get('scientific', {}):
        migrated['scientific'] = {**saved.get('scientific', {}), 'early_stopping': PILOT_STOPPING_POLICY}
    return migrated == current


def hardware_info() -> dict:
    import torch
    available = None
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
    return {'cpu_count': os.cpu_count(), 'available_ram_bytes': available,
            'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                      'vram_bytes': torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]}


def environment_info() -> dict:
    import torch
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'nibabel')}
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'upstream_commit': UPSTREAM, 'script_sha256': implementation_hash(),
            'amp_forward_policy': 'retry-float32-with-buffer-and-rng-rollback',
            'config_sha256': file_hash(Path(settings.__file__))}


def gpu_peak(device) -> dict:
    import torch
    if device.type != 'cuda':
        return {'allocated_bytes': 0, 'reserved_bytes': 0}
    torch.cuda.synchronize(device)
    return {'allocated_bytes': torch.cuda.max_memory_allocated(device),
            'reserved_bytes': torch.cuda.max_memory_reserved(device)}


def reset_peak(device) -> None:
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def seed_epoch(seed: int, dataset=None) -> None:
    from monai.utils import set_determinism
    set_determinism(seed)
    if dataset is not None:
        dataset.transform.set_random_state(seed=seed)


def run_on_rank_zero(action, rank: int, world: int):
    """Publish one writer's decision or error to every distributed process."""
    if world == 1:
        return action()
    import torch.distributed as dist
    message: list[Any] = [None, None]
    if rank == 0:
        try:
            message[0] = action()
        except Exception as error:
            message[1] = f'{type(error).__name__}: {error}'
    dist.broadcast_object_list(message, src=0)
    if message[1] is not None:
        raise ValueError(message[1])
    return message[0]


def save_run_record(output: Path, record: dict, resume: bool) -> None:
    """Serialize root publication and retain an audit trail for exact source migrations."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'run.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['config'] == record['config']:
                return
            if not resume or not compatible_config(previous['config'], record['config']):
                raise ValueError('incompatible run configuration; use a new output root')
            source = previous['config']['script_sha256']
            migrations = {
                AMP_OVERFLOW_PREDECESSOR: ('amp', 'Applying verified AMP-overflow resume fix',
                                           'Mixed-precision forward/gradient overflow recovery'),
                AMP_FORWARD_PREDECESSOR: ('amp-forward', 'Applying verified AMP-overflow resume fix',
                                          'Mixed-precision forward/gradient overflow recovery'),
                PILOT_PREDECESSOR: ('pilot', 'Applying verified pilot upgrade',
                                    'Exact binary-mask optimization and validation-based early stopping'),
                REFACTOR_PREDECESSOR: ('refactor', 'Applying verified training-script refactor',
                                       'Behavior-preserving extraction of data, options and runtime helpers'),
                SCRIPTS_PREDECESSOR: ('scripts', 'Applying verified scripts-directory move',
                                      'Relocation of training modules to scripts/'),
                PATHS_PREDECESSOR: ('paths', 'Applying verified project-path fix',
                                    'Resolve relative paths from the project root'),
            }
            name, message, reason = migrations[source]
            LOGGER.warning('%s | %s | remaining scientific/resource settings, data identity and versions unchanged',
                           message, reason)
            migration = output / f'{name}-resume-migration.json'
            if not migration.exists():
                atomic_json(migration, {'reason': reason, 'previous_run': previous, 'resumed_run': record})
        atomic_json(path, record)
