"""Patient splits, paired PET/CT preprocessing and native-grid prediction export.

Transform recipe adapted from Microsoft Corporation, MIT licensed; see README.
Original images are read-only, and outer-fold labels are used only after prediction.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import config as settings
from scripts.preprocessing_cache import PreprocessingCache
from scripts.training_runtime import (FORMAT_VERSION, LOGGER, atomic_json, compatible_config,
                              elapsed_time, file_hash, identity, inference_forward, log_progress)

FILES = ('CTres.nii.gz', 'SUV.nii.gz', 'SEG.nii.gz')


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


def discover_cases(root: Path, verify: bool = True) -> list[Case]:
    """Audit aligned, positive binary triplets without changing source images."""
    folders = sorted({p.parent for name in FILES for p in root.glob(f'*/*/{name}')})
    if not folders:
        raise ValueError(f'No examinations in {root}')
    cases = []
    seen: dict[str, str] = {}
    started = time.perf_counter()
    LOGGER.info('Data audit: %d examinations found; checking files, geometry, labels and duplicates', len(folders))
    for case_index, folder in enumerate(folders, 1):
        LOGGER.info('Audit %d/%d: %s/%s | %s', case_index, len(folders), folder.parent.name,
                    folder.name, 'hashing files, then checking image contents' if verify else 'checking file presence')
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
        log_progress('Data audit', case_index, len(folders), started, case.id)
    if len({c.id for c in cases}) != len(cases):
        raise ValueError('duplicate examination IDs')
    return cases


def load_or_create_splits(cases: list[Case], path: Path, seed: int = settings.SEED) -> dict:
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
        nval = max(1, round(len(remaining) * settings.INNER_VALIDATION_FRACTION))
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
        LOGGER.info('Reusing verified patient splits: %s', path)
    else:
        # Immutable publication; concurrent independent folds cannot overwrite it.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        LOGGER.info('Saved patient splits: %s', path)
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


def prepare_case(case: Case, root: Path, transforms, disk_cache: PreprocessingCache | None = None) -> dict:
    if disk_cache is None:
        return transforms(case.inputs(root))
    if len(case.hashes) != len(FILES):
        raise ValueError('Disk caching requires audited source-file hashes')
    fingerprint = {'case': case.id, 'source_hashes': list(case.hashes),
                   'preprocessing': {'spacing_mm': list(settings.SPACING_MM),
                                     'ct_clip_hu': list(settings.CT_CLIP_HU),
                                     'background_threshold': settings.BACKGROUND_PET_THRESHOLD,
                                     'orientation': 'RAS', 'foreground_crop': False,
                                     'versions': {p: importlib.metadata.version(p)
                                                  for p in ('monai', 'torch', 'numpy', 'nibabel')}}}
    return disk_cache.get(fingerprint, lambda: transforms(case.inputs(root)))


def build_transforms(labels: bool = True, concatenate: bool = True):
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                  ScaleIntensityRanged, Orientationd, Spacingd, ConcatItemsd)
    keys = ['CT', 'PT', 'GT'] if labels else ['CT', 'PT']
    modes = ('bilinear', 'bilinear', 'nearest') if labels else ('bilinear', 'bilinear')
    transforms = [
        LoadImaged(keys=keys, image_only=True, dtype='float32'),
        EnsureChannelFirstd(keys=keys),
        # Upstream CT>0 bounding crops remove real cohort lesions. Keep the full
        # field of view for every exam; never choose bounds from reference labels.
        ScaleIntensityRanged(keys=['CT'], a_min=settings.CT_CLIP_HU[0], a_max=settings.CT_CLIP_HU[1], b_min=0, b_max=1, clip=True),
        Orientationd(keys=keys, axcodes='RAS', labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        Spacingd(keys=keys, pixdim=settings.SPACING_MM, mode=modes),
    ]
    if concatenate:
        transforms.append(ConcatItemsd(keys=['CT', 'PT'], name='image', dim=0))
    return Compose(transforms)


def training_transforms(patch: int):
    from monai.transforms import (Compose, RandCropByPosNegLabeld, ResizeWithPadOrCropd,
                                  RandAffined, ConcatItemsd, DeleteItemsd)
    keys = ['CT', 'PT', 'GT']
    size = (patch,) * 3
    return Compose([
        DeleteItemsd(keys=['image']),
        RandCropByPosNegLabeld(keys=keys, label_key='GT', spatial_size=size,
                              pos=settings.POSITIVE_SAMPLE_WEIGHT, neg=settings.NEGATIVE_SAMPLE_WEIGHT, num_samples=1, image_key='PT',
                              image_threshold=settings.BACKGROUND_PET_THRESHOLD, allow_smaller=True,
                              fg_indices_key='GT_fg_indices', bg_indices_key='GT_bg_indices'),
        ResizeWithPadOrCropd(keys=keys, spatial_size=size, mode='constant'),
        RandAffined(keys=keys, mode=('bilinear', 'bilinear', 'nearest'), prob=settings.AFFINE_PROBABILITY,
                    spatial_size=size, translate_range=settings.TRANSLATION_RANGE,
                    rotate_range=settings.ROTATION_RANGE, scale_range=settings.SCALE_RANGE),
        ConcatItemsd(keys=['CT', 'PT'], name='image', dim=0),
        DeleteItemsd(keys=['CT', 'PT']),
    ])


class TrainingDataset:
    """A byte-bounded deterministic cache; random patches are never cached.

    RAM caching requires workers=0 and has a per-rank budget. Disk caching uses
    shared memory-mapped entries and one budget across processes and folds.
    """
    def __init__(self, cases: list[Case], root: Path, patch: int, cache_gib: float = 0, disk_cache: PreprocessingCache | None = None):
        self.cases, self.root = cases, root
        self.prepare = build_transforms(concatenate=False)
        self.disk_cache = disk_cache
        self.transform = training_transforms(patch)
        self.limit = int(cache_gib * 1024**3)
        self.cache: OrderedDict[str, tuple[dict, int]] = OrderedDict()
        self.bytes = 0

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        from torch.utils.data import get_worker_info

        case = self.cases[index]
        # Workers must not share their parent's console/file handlers. The parent
        # reports loader waits; detailed case messages apply to the default workers=0.
        report = get_worker_info() is None
        started = time.perf_counter()
        if report:
            LOGGER.info('Training input: %s | %s', case.id,
                        'using RAM cache' if case.id in self.cache else ('checking disk cache' if self.disk_cache else 'loading and resampling full volume on CPU'))
        if case.id in self.cache:
            data, _ = self.cache[case.id]
            self.cache.move_to_end(case.id)
        else:
            data = prepare_case(case, self.root, self.prepare, self.disk_cache)
            size = sum(v.numel() * v.element_size() for v in data.values() if hasattr(v, 'numel'))
            if self.limit and size <= self.limit:
                while self.cache and self.bytes + size > self.limit:
                    _, (_, old_size) = self.cache.popitem(last=False)
                    self.bytes -= old_size
                self.cache[case.id] = (data, size)
                self.bytes += size
        # Spatial transforms must not mutate cached tensors or their histories.
        result = self.transform(deepcopy(data) if case.id in self.cache else data)[0]
        if report:
            LOGGER.info('Training patch ready: %s | shape=%s | %.1fs | cache=%.2f GiB',
                        case.id, tuple(result['image'].shape), time.perf_counter() - started, self.bytes / 1024**3)
        return result


def predict_volume(model, image, device, patch: int, sw_batch_size: int):
    """CPU volume and float32 stitched logits; only patches visit the GPU."""
    import torch
    from monai.inferers import sliding_window_inference

    model.eval()
    started = time.perf_counter()
    # Match MONAI's configured overlap scan grid, including short-dimension padding.
    stride = max(int(patch * (1 - settings.INFERENCE_OVERLAP)), 1)
    total = math.prod(math.ceil(max(int(size) - patch, 0) / stride) + 1 for size in image.shape[-3:])
    completed, last_report = 0, started
    LOGGER.info('Sliding-window inference: shape=%s | %d windows | patch=%d | window batch=%d | stitching on CPU',
                tuple(image.shape), total, patch, sw_batch_size)
    def predictor(window):
        nonlocal completed, last_report
        result = inference_forward(model, window, amp_enabled=device.type == 'cuda')
        completed += len(window)
        now = time.perf_counter()
        if completed == len(window) or completed >= total or now - last_report >= 30:
            log_progress('Inference windows', completed, total, started)
            last_report = now
        return result
    with torch.inference_mode():
        result = sliding_window_inference(image.as_tensor().cpu().float().unsqueeze(0),
                                        (patch,) * 3, sw_batch_size, predictor,
                                        overlap=settings.INFERENCE_OVERLAP, mode='constant', sw_device=device,
                                        device=torch.device('cpu'))[0]
    LOGGER.info('Volume inference complete | elapsed=%s', elapsed_time(time.perf_counter() - started))
    return result


def export_prediction(logits, data: dict, transforms, original: Path, target: Path) -> None:
    import nibabel as nib
    import numpy as np
    from monai.transforms import Invertd
    from monai.data import MetaTensor

    LOGGER.info('Restoring original geometry and saving prediction: %s', target)
    # PT carries the input history even when there is no GT at inference time.
    restored = Invertd(keys='prediction', transform=transforms, orig_keys='PT',
                       nearest_interp=False, to_tensor=True, device='cpu')(
                           {**data, 'prediction': MetaTensor(logits)})['prediction']
    source = nib.load(original)
    binary = binary_prediction(restored).cpu().numpy().astype(np.uint8)
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
    LOGGER.info('Prediction saved: %s | shape=%s', target, binary.shape)


def binary_prediction(logits):
    """Exact two-class argmax for finite logits, including background on ties.

    A direct comparison avoids the slow strided CPU argmax and an int64 volume.
    Call after stitching (and after continuous inversion for native export).
    """
    if logits.shape[0] != 2:
        raise ValueError('Binary prediction requires exactly two class channels')
    return logits[1] > logits[0]


def foreground_dice(prediction, target) -> float:
    import numpy as np
    import torch
    prediction, target = prediction > 0, target > 0
    count = torch.count_nonzero if torch.is_tensor(prediction) else np.count_nonzero
    denominator = int(count(prediction)) + int(count(target))
    return 2 * int(count(prediction & target)) / denominator if denominator else 1.0


def predict_held_out(model, cases: list[Case], manifest: dict, fold: int,
                     root: Path, output: Path, device, patch: int, sw_batch_size: int) -> list[dict]:
    import nibabel as nib
    import numpy as np

    selected = fold_cases(cases, manifest, fold)['held_out']
    transforms = build_transforms(labels=False)
    results = []
    started = time.perf_counter()
    LOGGER.info('Fold %d: predicting %d held-out examinations with the selected checkpoint', fold, len(selected))
    for case_index, case in enumerate(selected, 1):
        start = time.perf_counter()
        LOGGER.info('Held-out %d/%d: %s | loading and resampling on CPU', case_index, len(selected), case.id)
        data = transforms(case.inputs(root, labels=False))
        logits = predict_volume(model, data['image'], device, patch, sw_batch_size)
        target = output / 'predictions' / case.id / 'SEG_pred.nii.gz'
        export_prediction(logits, data, transforms, root / case.id / 'SUV.nii.gz', target)
        # Labels participate only in evaluation AFTER prediction and restoration.
        pred = np.asarray(nib.load(target).dataobj) > 0
        gt = np.asarray(nib.load(root / case.id / 'SEG.nii.gz').dataobj) > 0
        results.append({'case': case.id, 'fold': fold, 'foreground_dice_native': foreground_dice(pred, gt),
                        'seconds': time.perf_counter() - start})
        log_progress(f'Fold {fold} held-out predictions', case_index, len(selected), started,
                     f"native foreground Dice={results[-1]['foreground_dice_native']:.4f}")
        del data, logits, pred, gt
    atomic_json(output / 'metrics.json', results)
    LOGGER.info('Held-out metrics saved: %s', output / 'metrics.json')
    return results


def verify_completed_fold(folder: Path, cases: list[Case], manifest: dict, fold: int,
                          config: dict, smoke: bool = False) -> list[dict]:
    complete = json.loads((folder / 'complete.json').read_text())
    if not compatible_config(complete['config'], config) or complete['split_id'] != manifest['identity']:
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
    LOGGER.info('Five-fold run complete: verified exactly %d unique out-of-fold predictions | summary: %s',
                len(rows), output / 'metrics.json')
