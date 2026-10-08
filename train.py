"""Patient-level autoPET v2 adaptation of Microsoft's lymphoma baseline.

Model and transform recipe adapted from Microsoft Corporation, MIT licensed.
See THIRD_PARTY_LICENSE and README for the inspected source and deviations.
Heavy dependencies are loaded only when needed; --help and split tests use stdlib.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any

UPSTREAM = '81129243db3c868370a7664c9adc2ad52609484f'
FILES = ('CTres.nii.gz', 'SUV.nii.gz', 'SEG.nii.gz')
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Case:
    patient: str
    exam: str
    hashes: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f'{self.patient}/{self.exam}'

    def inputs(self, root: Path, labels: bool = True) -> dict:
        keys = ('CT', 'PT', 'GT') if labels else ('CT', 'PT')
        return {k: str(root / self.id / f) for k, f in zip(keys, FILES)}


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


def discover_cases(root: Path, verify: bool = True) -> list[Case]:
    """Audit aligned, positive binary triplets without changing source images."""
    folders = sorted({p.parent for name in FILES for p in root.glob(f'*/*/{name}')})
    if not folders:
        raise ValueError(f'No examinations in {root}')
    cases = []
    seen: dict[str, str] = {}
    for folder in folders:
        paths = [folder / name for name in FILES]
        if any(not p.is_file() or p.is_symlink() for p in paths):
            raise ValueError(f'Missing files or symlink examination: {folder}')
        hashes = tuple(file_hash(p) for p in paths) if verify else ()
        case = Case(folder.parent.name, folder.name, hashes)
        if verify:
            import nibabel as nib
            import numpy as np

            images = [nib.load(p) for p in paths]
            pet = images[1]
            if len(pet.shape) != 3 or pet.header.get_xyzt_units()[0] != 'mm':
                raise ValueError(f'Expected 3D millimetre SUV grid: {case.id}')
            for image in images:
                if (image.shape != pet.shape or not np.allclose(image.affine, pet.affine, atol=1e-4, rtol=0)
                        or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8
                        or image.header.get_xyzt_units()[0] not in ('unknown', 'mm')):
                    raise ValueError(f'Geometry mismatch: {case.id}')
            # One image at a time: no full-cohort arrays or float64 image cache.
            for index, image in enumerate(images):
                array = image.get_fdata(dtype=np.float32, caching='unchanged')
                if not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite image: {case.id}/{FILES[index]}')
                if index == 2 and (not np.isin(array, [0, 1]).all() or not array.any()):
                    raise ValueError(f'Expected positive binary labels: {case.id}')
                if index == 1:
                    digest = hashlib.sha256(array.tobytes(order='C')).hexdigest()
                    if digest in seen:
                        raise ValueError(f'duplicate SUV examination: {seen[digest]} and {case.id}')
                    seen[digest] = case.id
                del array
        cases.append(case)
    if len({c.id for c in cases}) != len(cases):
        raise ValueError('duplicate examination IDs')
    return cases


def load_or_create_splits(cases: list[Case], path: Path, seed: int = 20261006) -> dict:
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate examinations')
    patients = sorted({c.patient for c in cases})
    if len(patients) < 10:
        raise ValueError('At least 10 patients required for five outer folds and inner validation')
    random.Random(seed).shuffle(patients)
    assignments = {p: i % 5 for i, p in enumerate(patients)}
    folds = []
    for fold in range(5):
        remaining = sorted(p for p in patients if assignments[p] != fold)
        random.Random(seed + fold + 1).shuffle(remaining)
        nval = max(1, round(len(remaining) * 0.2))
        folds.append({'train': sorted(remaining[nval:]), 'validation': sorted(remaining[:nval]),
                      'held_out': sorted(p for p in patients if assignments[p] == fold)})
    manifest = {'version': FORMAT_VERSION, 'seed': seed,
                'cases': [asdict(c) for c in sorted(cases, key=lambda c: c.id)],
                'patient_folds': assignments, 'folds': folds}
    # JSON normalization makes tuple hashes round-trip without changing equality.
    manifest = json.loads(json.dumps(manifest))
    manifest['identity'] = identity(manifest)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Split manifest differs: seed, cohort, file contents or assignments changed')
    else:
        # Immutable publication; concurrent independent folds cannot overwrite it.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    for fold in range(5):
        fold_cases(cases, manifest, fold)
    return manifest


def fold_cases(cases: list[Case], manifest: dict, fold: int) -> dict[str, list[Case]]:
    roles = manifest['folds'][fold]
    groups = {r: set(roles[r]) for r in ('train', 'validation', 'held_out')}
    if (any(not g for g in groups.values()) or
            sum(len(g) for g in groups.values()) != len(set.union(*groups.values())) or
            set.union(*groups.values()) != {c.patient for c in cases}):
        raise ValueError('Patient overlap, missing patients, or empty split')
    return {role: [c for c in cases if c.patient in patients] for role, patients in groups.items()}


def build_transforms(labels: bool = True):
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                  ScaleIntensityRanged, Orientationd, Spacingd, ConcatItemsd)
    keys = ['CT', 'PT', 'GT'] if labels else ['CT', 'PT']
    modes = ('bilinear', 'bilinear', 'nearest') if labels else ('bilinear', 'bilinear')
    return Compose([
        LoadImaged(keys=keys, image_only=True, dtype='float32'),
        EnsureChannelFirstd(keys=keys),
        # Upstream CT>0 bounding crops remove real cohort lesions. Keep the full
        # field of view for every exam; never choose bounds from reference labels.
        ScaleIntensityRanged(keys=['CT'], a_min=-154, a_max=325, b_min=0, b_max=1, clip=True),
        Orientationd(keys=keys, axcodes='RAS', labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        Spacingd(keys=keys, pixdim=(2, 2, 2), mode=modes),
        ConcatItemsd(keys=['CT', 'PT'], name='image', dim=0),
    ])


def training_transforms(patch: int):
    from monai.transforms import (Compose, RandCropByPosNegLabeld, ResizeWithPadOrCropd,
                                  RandAffined, ConcatItemsd, DeleteItemsd)
    keys = ['CT', 'PT', 'GT']
    size = (patch,) * 3
    return Compose([
        DeleteItemsd(keys=['image']),
        RandCropByPosNegLabeld(keys=keys, label_key='GT', spatial_size=size,
                              pos=2, neg=1, num_samples=1, image_key='PT',
                              image_threshold=0, allow_smaller=True),
        ResizeWithPadOrCropd(keys=keys, spatial_size=size, mode='constant'),
        RandAffined(keys=keys, mode=('bilinear', 'bilinear', 'nearest'), prob=0.5,
                    spatial_size=size, translate_range=(10, 10, 10),
                    rotate_range=(0, 0, math.pi / 15), scale_range=(0.1, 0.1, 0.1)),
        ConcatItemsd(keys=['CT', 'PT'], name='image', dim=0),
        DeleteItemsd(keys=['CT', 'PT']),
    ])


class TrainingDataset:
    """A byte-bounded deterministic cache; random patches are never cached.

    Caching requires workers=0, so it cannot multiply across loader processes.
    Every DDP rank has its own explicitly budgeted cache.
    """
    def __init__(self, cases: list[Case], root: Path, patch: int, cache_gib: float = 0):
        self.cases, self.root = cases, root
        self.prepare = build_transforms()
        self.transform = training_transforms(patch)
        self.limit = int(cache_gib * 1024**3)
        self.cache: OrderedDict[str, tuple[dict, int]] = OrderedDict()
        self.bytes = 0

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        if case.id in self.cache:
            data, _ = self.cache[case.id]
            self.cache.move_to_end(case.id)
        else:
            data = self.prepare(case.inputs(self.root))
            size = sum(v.numel() * v.element_size() for v in data.values() if hasattr(v, 'numel'))
            if self.limit and size <= self.limit:
                while self.cache and self.bytes + size > self.limit:
                    _, (_, old_size) = self.cache.popitem(last=False)
                    self.bytes -= old_size
                self.cache[case.id] = (data, size)
                self.bytes += size
        # Spatial transforms must not mutate cached tensors or their histories.
        return self.transform(deepcopy(data) if case.id in self.cache else data)[0]


def build_model():
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3, in_channels=2, out_channels=2,
                channels=(16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2),
                num_res_units=2, norm='BATCH')


def predict_volume(model, image, device, patch: int, sw_batch_size: int):
    """CPU volume and float32 stitched logits; only patches visit the GPU."""
    import torch
    from monai.inferers import sliding_window_inference

    model.eval()
    def predictor(window):
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            return model(window).float()
    with torch.inference_mode():
        return sliding_window_inference(image.as_tensor().cpu().float().unsqueeze(0),
                                        (patch,) * 3, sw_batch_size, predictor,
                                        overlap=0.25, mode='constant', sw_device=device,
                                        device=torch.device('cpu'))[0]


def export_prediction(logits, data: dict, transforms, original: Path, target: Path) -> None:
    import nibabel as nib
    import numpy as np
    from monai.transforms import Invertd
    from monai.data import MetaTensor

    # PT carries the input history even when there is no GT at inference time.
    restored = Invertd(keys='prediction', transform=transforms, orig_keys='PT',
                       nearest_interp=False, to_tensor=True, device='cpu')(
                           {**data, 'prediction': MetaTensor(logits)})['prediction']
    source = nib.load(original)
    binary = restored.argmax(dim=0).cpu().numpy().astype(np.uint8)
    if binary.shape != source.shape or not np.allclose(restored.affine, source.affine, atol=1e-4):
        raise ValueError('Prediction inversion failed to restore original SUV geometry')
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(binary, source.affine, header)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('SEG_pred.tmp.nii.gz')
    nib.save(result, temporary)
    temporary.replace(target)


def foreground_dice(prediction, target) -> float:
    prediction, target = prediction > 0, target > 0
    denominator = int(prediction.sum()) + int(target.sum())
    return 2 * int((prediction & target).sum()) / denominator if denominator else 1.0


def predict_held_out(model, cases: list[Case], manifest: dict, fold: int,
                     root: Path, output: Path, device, patch: int, sw_batch_size: int) -> list[dict]:
    import nibabel as nib
    import numpy as np

    selected = fold_cases(cases, manifest, fold)['held_out']
    transforms = build_transforms(labels=False)
    results = []
    for case in selected:
        start = time.perf_counter()
        data = transforms(case.inputs(root, labels=False))
        logits = predict_volume(model, data['image'], device, patch, sw_batch_size)
        target = output / 'predictions' / case.id / 'SEG_pred.nii.gz'
        export_prediction(logits, data, transforms, root / case.id / 'SUV.nii.gz', target)
        # Labels participate only in evaluation AFTER prediction and restoration.
        pred = np.asarray(nib.load(target).dataobj) > 0
        gt = np.asarray(nib.load(root / case.id / 'SEG.nii.gz').dataobj) > 0
        results.append({'case': case.id, 'fold': fold, 'foreground_dice_native': foreground_dice(pred, gt),
                        'seconds': time.perf_counter() - start})
        del data, logits, pred, gt
    atomic_json(output / 'metrics.json', results)
    return results


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
    if (state['version'] != FORMAT_VERSION or state['config'] != config or
            state['split_id'] != split_id or rank >= len(state['rng'])):
        raise ValueError('incompatible resume: configuration, split, versions or GPU count changed')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    set_random_state(state['rng'][rank])
    return state


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
            'upstream_commit': UPSTREAM, 'script_sha256': file_hash(Path(__file__))}


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


def train_fold(args, cases: list[Case], manifest: dict, fold: int, config: dict,
               device, rank: int, world: int) -> None:
    import torch
    import torch.distributed as dist
    from monai.data import DataLoader
    from monai.losses import DiceLoss
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data.distributed import DistributedSampler

    output = args.output / f'fold_{fold}'
    checkpoint_config = {**config, 'fold': fold}
    lock = None

    def prepare_output():
        nonlocal lock
        import fcntl
        output.mkdir(parents=True, exist_ok=True)
        lock = (output / '.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'complete.json').exists():
            if not args.resume:
                raise ValueError(f'{output} already completed; use --resume to verify/skip')
            verify_completed_fold(output, cases, manifest, fold, config, args.smoke)
            return 'skip'
        if any((output / name).exists() for name in ('last.pt', 'started.json', 'best.pt', 'predictions')) and not args.resume:
            raise ValueError(f'{output} exists; use --resume or a new output')
        if args.resume and not (output / 'last.pt').exists() and (output / 'started.json').exists():
            raise ValueError('Interrupted before first checkpoint; use a new output directory')
        atomic_json(output / 'started.json', {'config': config, 'split_id': manifest['identity']})
        return 'run'

    try:
        # Only the writer checks/publishes output state. All ranks receive the same
        # proceed/skip/error decision, so they cannot race the new started.json.
        if run_on_rank_zero(prepare_output, rank, world) == 'skip':
            return
        roles = fold_cases(cases, manifest, fold)
        training, validation = roles['train'], roles['validation']
        if args.smoke:
            training, validation = training[:max(2, world)], validation[:1]
            # Exactly smoke_steps optimizer updates per epoch on a bounded cohort.
            count = args.smoke_steps * args.batch_size * args.accumulate * world
            training = [training[i % len(training)] for i in range(count)]
        seed_epoch(args.seed + fold)
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
        start_epoch, best, history = 0, -1.0, []
        selected_checkpoint = None
        if args.resume and (output / 'last.pt').exists():
            state = restore_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler,
                                       checkpoint_config, manifest['identity'], rank)
            start_epoch, best, history = state['epoch'], state['best'], state['history']
            selected_checkpoint = state['selected_checkpoint']
            # last.pt is the authoritative committed epoch. Repair a best.pt write
            # interrupted before last.pt publication, even if replay differs.
            if rank == 0 and selected_checkpoint is not None:
                atomic_torch_save(output / 'best.pt', selected_checkpoint)
        wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
        dataset = TrainingDataset(training, args.data_root, args.patch, args.cache_gib)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed + fold, drop_last=False)
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=False, generator=generator,
                            persistent_workers=False)
        valid_transforms = build_transforms()
        end_epoch = min(args.epochs, args.stop_after_epoch or args.epochs)
        for epoch in range(start_epoch, end_epoch):
            # Epoch-derived transform/worker seeds make resume independent of cache hits
            # and of how many random draws validation happened to make.
            epoch_seed = args.seed + fold * 100000 + epoch * world + rank
            seed_epoch(epoch_seed, dataset)
            generator.manual_seed(epoch_seed)
            sampler.set_epoch(epoch)
            wrapped.train()
            reset_peak(device)
            started = time.perf_counter()
            total_loss, examples = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                inputs, labels = batch['image'].to(device), batch['GT'].to(device)
                group_start = (step // args.accumulate) * args.accumulate
                # Normalize partial final accumulation groups by actual examples.
                group_examples = min(args.accumulate * args.batch_size,
                                     len(sampler) - group_start * args.batch_size)
                synchronize = (step + 1) % args.accumulate == 0 or step + 1 == len(loader)
                context = wrapped.no_sync() if world > 1 and not synchronize else nullcontext()
                with context:
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        logits = wrapped(inputs)
                    loss = loss_function(logits.float(), labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    scaler.scale(loss * len(inputs) / group_examples).backward()
                if synchronize:
                    scaler.unscale_(optimizer)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('Nonfinite gradients; no checkpoint written')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item() * len(inputs)
                examples += len(inputs)
                del inputs, labels, logits, loss, batch
            scheduler.step()
            train_memory = gpu_peak(device)
            train_seconds = time.perf_counter() - started
            if world > 1:
                totals = torch.tensor([total_loss, examples], device=device, dtype=torch.float64)
                dist.all_reduce(totals)
                total_loss, examples = totals.tolist()
                # Preserve ordinary per-rank BatchNorm; canonicalize running buffers
                # to rank zero before validation/checkpointing and the next epoch.
                for buffer in model.buffers():
                    dist.broadcast(buffer, src=0)
            reset_peak(device)
            validation_start = time.perf_counter()
            score = None
            if rank == 0 and ((epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs):
                scores = []
                for case in validation:
                    data = valid_transforms(case.inputs(args.data_root))
                    logits = predict_volume(model, data['image'], device, args.patch, args.sw_batch_size)
                    scores.append(foreground_dice(logits.argmax(0), data['GT'][0]))
                    del data, logits
                score = sum(scores) / len(scores)
            validation_memory = gpu_peak(device)
            validation_seconds = time.perf_counter() - validation_start
            row = {'epoch': epoch + 1, 'loss_including_background': total_loss / examples,
                   'inner_foreground_dice_2mm': score, 'next_lr': scheduler.get_last_lr()[0],
                   'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'train_gpu_peak': train_memory, 'validation_gpu_peak': validation_memory,
                   'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
            states: list = [random_state()]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, random_state())
                rank_memory = [None] * world
                dist.all_gather_object(rank_memory, row)
                row['per_rank_resources'] = rank_memory
            if rank == 0:
                history.append(row)
                if score is not None and score > best:
                    best = score
                    selected_checkpoint = {'model': {k: v.detach().cpu().clone()
                                                     for k, v in model.state_dict().items()},
                                           'epoch': epoch + 1, 'score': score, 'config': checkpoint_config,
                                           'split_id': manifest['identity']}
                    atomic_torch_save(output / 'best.pt', selected_checkpoint)
                save_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler,
                                epoch + 1, best, history, checkpoint_config, manifest['identity'], states, selected_checkpoint)
                atomic_json(output / 'history.json', history)
                print(json.dumps(row), flush=True)
            if world > 1:
                dist.barrier()
        if end_epoch < args.epochs:
            return  # Explicit epoch-boundary interruption for resume verification.
        if rank == 0:
            selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
            if selected['config'] != checkpoint_config or selected['split_id'] != manifest['identity']:
                raise ValueError('incompatible selected checkpoint (including fold identity)')
            model.load_state_dict(selected['model'])
            reset_peak(device)
            inference_start = time.perf_counter()
            results = [] if args.smoke else predict_held_out(
                model, cases, manifest, fold, args.data_root, output, device, args.patch, args.sw_batch_size)
            atomic_json(output / 'complete.json', {'config': config, 'split_id': manifest['identity'],
                        'best_epoch': selected['epoch'], 'inner_foreground_dice_2mm': selected['score'],
                        'held_out_cases': [r['case'] for r in results],
                        'inference_seconds': time.perf_counter() - inference_start,
                        'inference_gpu_peak': gpu_peak(device), 'smoke': args.smoke})
        if world > 1:
            dist.barrier()
    finally:
        if lock is not None:
            lock.close()


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


def verify_completed_fold(folder: Path, cases: list[Case], manifest: dict, fold: int,
                          config: dict, smoke: bool = False) -> list[dict]:
    complete = json.loads((folder / 'complete.json').read_text())
    if complete['config'] != config or complete['split_id'] != manifest['identity']:
        raise ValueError('incompatible completed fold')
    if smoke:
        return []
    metrics = json.loads((folder / 'metrics.json').read_text())
    expected = {c.id for c in fold_cases(cases, manifest, fold)['held_out']}
    if ({r['case'] for r in metrics} != expected or len(metrics) != len(expected)
            or any(r['fold'] != fold for r in metrics)):
        raise ValueError('Incorrect held-out prediction routing')
    actual = {str(p.parent.relative_to(folder / 'predictions'))
              for p in (folder / 'predictions').glob('*/*/SEG_pred.nii.gz')}
    if actual != expected:
        raise ValueError('Missing or extra prediction files')
    return metrics


def summarize_folds(output: Path, cases: list[Case], manifest: dict, config: dict) -> None:
    """Called by sequential --fold all; independent jobs keep per-fold summaries."""
    rows = []
    for fold in range(5):
        rows.extend(verify_completed_fold(output / f'fold_{fold}', cases, manifest, fold, config))
    if len(rows) != len(cases) or len({r['case'] for r in rows}) != len(cases):
        raise ValueError('Expected exactly one out-of-fold prediction per examination')
    atomic_json(output / 'metrics.json', {'cases': rows, 'examinations': len(rows),
                'mean_foreground_dice_native': sum(r['foreground_dice_native'] for r in rows) / len(rows),
                'split_id': manifest['identity']})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Audit data and persist splits; no training')
    mode.add_argument('--smoke', action='store_true', help='Bounded training + one full-volume inner validation')
    mode.add_argument('--fold', choices=['0', '1', '2', '3', '4', 'all'])
    parser.add_argument('--data-root', type=Path, default=Path('data/autopet-v2-lymphoma'))
    parser.add_argument('--output', type=Path, default=Path('runs/baseline'))
    parser.add_argument('--splits', type=Path, help='Existing shared manifest; create with --prepare first')
    parser.add_argument('--epochs', type=int, help='Total cosine-schedule budget (required outside smoke)')
    parser.add_argument('--patch', type=int, default=64, help='Scientific choice; multiple of 32, at least 64')
    parser.add_argument('--batch-size', type=int, default=1, help='Training patches per GPU')
    parser.add_argument('--accumulate', type=int, default=1, help='Microbatches per optimizer update')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--val-interval', type=int, default=2)
    parser.add_argument('--device', default='cuda:0', help='Explicit single device; use cpu only for tests')
    parser.add_argument('--ddp', action='store_true', help='Explicit torchrun one-process-per-GPU execution')
    parser.add_argument('--workers', type=int, default=0, help='Data-loader workers per rank')
    parser.add_argument('--threads', type=int, default=4, help='CPU compute threads per process')
    parser.add_argument('--cache-gib', type=float, default=0, help='Deterministic tensor cache per rank; workers must be 0')
    parser.add_argument('--sw-batch-size', type=int, default=1)
    parser.add_argument('--smoke-steps', type=int, default=20, help='Optimizer updates per smoke epoch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, help='Save then stop early without changing scheduler budget')
    parser.add_argument('--expected-exams', type=int, default=145)
    parser.add_argument('--expected-patients', type=int, default=144)
    args = parser.parse_args(argv)
    if not (args.prepare or args.smoke or args.fold is not None):
        parser.print_help()
        return None
    if args.smoke:
        args.epochs = args.epochs or 1
        if args.output == Path('runs/baseline'):
            parser.error('--smoke requires its own explicit --output directory')
    if not args.prepare and (args.epochs is None or args.epochs < 1):
        parser.error('training requires an explicit positive --epochs budget')
    if args.patch < 64 or args.patch % 32:
        parser.error('--patch must be a multiple of 32 and at least 64 (BatchNorm needs spatial support)')
    if any(v < 1 for v in (args.batch_size, args.accumulate, args.val_interval, args.sw_batch_size,
                           args.smoke_steps, args.expected_exams, args.expected_patients, args.threads)):
        parser.error('batch sizes, accumulation, intervals, steps and cohort sizes must be positive')
    if args.workers < 0 or not math.isfinite(args.cache_gib) or args.cache_gib < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('invalid resource budget or learning rate')
    if args.workers and args.cache_gib:
        parser.error('bounded caching requires --workers 0 to prevent worker cache copies')
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


def main(argv=None) -> None:
    args = parse_args(argv)
    if args is None:
        return
    try:
        import torch
        import monai
        import nibabel
        import numpy
    except ImportError as error:
        raise SystemExit(f'Missing dependency: {error}. Follow README setup; no packages installed automatically.')
    import torch.distributed as dist
    torch.set_num_threads(args.threads)
    world = int(os.environ['WORLD_SIZE']) if args.ddp else 1
    rank = int(os.environ['RANK']) if args.ddp else 0
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if args.ddp else args.device)
    if device.type not in ('cpu', 'cuda') or (args.ddp and device.type != 'cuda'):
        raise ValueError('Only CPU tests or CUDA training are supported')
    if not args.prepare and device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; check environment. No automatic CPU fallback.')
        torch.cuda.set_device(device)
    if args.ddp:
        # Rank-zero full-volume validation can take longer than NCCL's default timeout.
        dist.init_process_group('nccl', timeout=timedelta(hours=24))
    try:
        manifest_path = args.splits or args.output / 'splits.json'
        if args.splits and not manifest_path.exists() and not args.prepare:
            raise ValueError('Shared --splits must exist; run --prepare first')
        cases = None
        manifest = None
        if rank == 0:
            print('Auditing examination files, geometry, positive labels and duplicate SUV content...', flush=True)
            cases = discover_cases(args.data_root)
            if len(cases) != args.expected_exams or len({c.patient for c in cases}) != args.expected_patients:
                raise ValueError('Unexpected cohort size; inspect data or explicitly configure expected counts')
            manifest = load_or_create_splits(cases, manifest_path, args.seed)
            print(f'Audited {len(cases)} examinations / {len({c.patient for c in cases})} patients; '
                  f'split {manifest["identity"]}', flush=True)
        if world > 1:
            payload: list[Any] = [cases, manifest]
            dist.broadcast_object_list(payload, src=0)
            cases, manifest = payload
        assert cases is not None and manifest is not None
        if args.prepare:
            return
        hardware, environment = hardware_info(), environment_info()
        # Scientific settings stay fixed across all folds and hardware choices.
        scientific = {'patch': args.patch, 'spacing_mm': [2, 2, 2], 'channels': [16, 32, 64, 128, 256, 512],
                      'input_order': ['CT', 'PET_SUV'], 'normalization': 'BatchNorm-local',
                      'foreground_crop': 'none-full-field-of-view',
                      'loss': 'softmax-Dice-including-background', 'lr': args.lr, 'weight_decay': 1e-5,
                      'epochs': args.epochs, 'seed': args.seed, 'val_interval': args.val_interval,
                      'batch_size_per_gpu': args.batch_size, 'accumulate': args.accumulate,
                      'effective_batch_size': args.batch_size * args.accumulate * world,
                      'world_size': world, 'smoke': args.smoke,
                      'smoke_steps': args.smoke_steps if args.smoke else None}
        config = {'scientific': scientific, 'split_id': manifest['identity'],
                  'versions': environment['packages'], 'script_sha256': environment['script_sha256'],
                  'precision': 'cuda-amp-fp16' if device.type == 'cuda' else 'fp32',
                  'sampler': 'DistributedSampler-pad-shuffle-epoch-v1',
                  'workers': args.workers, 'threads': args.threads, 'cache_gib': args.cache_gib, 'sw_batch_size': args.sw_batch_size}
        resources = {'device': str(device), 'workers_per_rank': args.workers, 'cache_gib_per_rank': args.cache_gib,
                     'aggregate_cache_gib': args.cache_gib * world,
                     'sw_batch_size': args.sw_batch_size, 'world_size': world, 'threads_per_process': args.threads}
        available = hardware['available_ram_bytes']
        if available and args.cache_gib * world * 1024**3 > available / 2:
            raise ValueError('Cache budget exceeds half available RAM; leave room for transforms and volumes')
        if rank == 0:
            record = {'config': config, 'resources': resources, 'hardware': hardware, 'environment': environment,
                      'data_root': str(args.data_root.resolve()), 'split_path': str(manifest_path.resolve()),
                      'command': sys.argv}
            print(json.dumps(record, indent=2), flush=True)
            args.output.mkdir(parents=True, exist_ok=True)
            # Serialize only root metadata creation; independent folds may run concurrently.
            import fcntl
            with (args.output / '.run.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = args.output / 'run.json'
                if path.exists():
                    if json.loads(path.read_text())['config'] != config:
                        raise ValueError('incompatible run configuration; use a new output root')
                else:
                    atomic_json(path, record)
        if world > 1:
            dist.barrier()
        folds = range(5) if args.fold == 'all' else [int(args.fold or 0)]
        for fold in folds:
            train_fold(args, cases, manifest, fold, config, device, rank, world)
        if rank == 0 and args.fold == 'all' and not args.smoke and not args.stop_after_epoch:
            summarize_folds(args.output, cases, manifest, config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
