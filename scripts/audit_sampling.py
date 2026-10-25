"""Audit actual MONAI crop/augmentation coverage on training patients only."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings
from scripts.analysis_common import load_splits, new_report, provenance, verify_case, write_csv, case_record
from scripts.training_data import build_transforms, training_transforms, prepare_case, fold_cases
from scripts.training_options import scientific_recipe
from scripts.training_runtime import atomic_json, file_hash


def select_training(cases, manifest, fold, count, seed):
    training = fold_cases(cases, manifest, fold)['train']
    if count < 1 or count > len(training):
        raise ValueError(f'--cases must be between 1 and {len(training)}')
    return sorted(random.Random(seed).sample(training, count), key=lambda c: c.id)


def audit_volume(data, patch, draws, seed, suv_threshold, sampling='voxel'):
    """Carry component IDs through the exact nearest-neighbour label transforms.

    Positive-vs-background sampling depends only on GT>0, so replacing positive
    binary values by component IDs preserves crop selection and spatial operations.
    Coverage counts refer to components on the 2 mm training grid, not native lesions.
    """
    import numpy as np
    import torch
    from scripts.segmentation_metrics import components, size_bin

    gt = np.asarray(data['GT'][0].cpu())
    from monai.transforms import FgBgToIndicesd
    # Same deterministic candidates as production's preprocessing cache. This
    # also avoids rebuilding millions of background indices for every draw.
    if data.get('GT_fg_indices') is None or data.get('GT_bg_indices') is None:
        data = FgBgToIndicesd(keys='GT', image_key='PT',
                              image_threshold=settings.BACKGROUND_PET_THRESHOLD)(data)
    labels, number = components(gt)
    counts = np.bincount(labels.ravel(), minlength=number + 1)
    voxel_ml = abs(float(np.linalg.det(np.asarray(data['GT'].affine)[:3, :3]))) / 1000
    tagged = data['GT'].clone().float()
    tagged[0] = torch.from_numpy(labels.astype(np.float32))
    tagged_data = {**data, 'GT': tagged}
    if sampling == 'lesion-balanced':
        from scripts.lesion_sampling import lesion_index
        tagged_data['_lesion_index'] = lesion_index(gt)
    transform = training_transforms(patch, sampling)
    transform.set_random_state(seed=seed)
    cropper = transform.transforms[1].cropper
    appearances = Counter()
    centre_hits = Counter()
    target_hits = Counter()
    centres = Counter()
    rows = []
    fg_indices = np.asarray(data['GT_fg_indices'])
    pos_ratio = settings.POSITIVE_SAMPLE_WEIGHT / (settings.POSITIVE_SAMPLE_WEIGHT + settings.NEGATIVE_SAMPLE_WEIGHT)
    if not len(data['GT_bg_indices']):
        pos_ratio = 1.0
    probe = np.random.RandomState()
    for draw in range(draws):
        probe.set_state(cropper.R.get_state())
        intended_foreground = bool(probe.rand() < pos_ratio)
        reference_candidate = int(fg_indices[probe.randint(len(fg_indices))]) if intended_foreground and sampling == 'voxel' else None
        sample = transform(tagged_data)[0]
        if intended_foreground and sampling == 'lesion-balanced':
            reference_candidate = transform.transforms[1].foreground_candidate
        target_component = int(labels[np.unravel_index(reference_candidate, labels.shape)]) if reference_candidate is not None else 0
        target_hits[target_component] += 1
        centre = tuple(int(x) for x in cropper.centers[0])
        centres[centre] += 1
        centre_hits[int(labels[centre])] += 1
        lab = sample['GT'][0].cpu().numpy()
        if not np.equal(lab, np.rint(lab)).all():
            raise ValueError('Component IDs lost to non-nearest label interpolation')
        ids = np.unique(lab).astype(int)
        appearances.update(int(g) for g in ids if g)
        pet = sample['image'][1].cpu().numpy()
        high_background = (lab == 0) & (pet >= suv_threshold)
        rows.append({'draw': draw, 'centre_x': centre[0], 'centre_y': centre[1], 'centre_z': centre[2],
            'centre_in_reference': int(labels[centre] > 0), 'repeat_centre': int(centres[centre] > 1),
            'centre_reference_component': int(labels[centre]),
            'intended_foreground': int(intended_foreground), 'target_reference_component': target_component,
            'reference_components_visible': int((ids > 0).sum()),
            'reference_voxels': int((lab > 0).sum()), 'background_voxels': int((lab == 0).sum()),
            'high_suv_background_voxels': int(high_background.sum()),
            'high_suv_background_ml': float(high_background.sum() * voxel_ml)})
    lesions = [{'lesion': g, 'volume_ml': float(counts[g] * voxel_ml),
                'size_bin': size_bin(counts[g] * voxel_ml), 'patches_visible': appearances[g],
                'fraction_draws_visible': appearances[g] / draws, 'draws_centred_on_component': centre_hits[g],
                'fraction_draws_centred_on_component': centre_hits[g] / draws,
                'foreground_targets': target_hits[g]} for g in range(1, number + 1)]
    return rows, lesions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--cases', type=int, required=True, help='Bounded training-examination sample')
    parser.add_argument('--draws', type=int, required=True, help='Patches per selected examination and patch size')
    parser.add_argument('--patches', nargs='+', type=int, default=[settings.PATCH_SIZE])
    parser.add_argument('--suv-threshold', type=float, required=True, help='Background uptake proxy, not an organ/tumour classifier')
    parser.add_argument('--seed', type=int, default=settings.SEED)
    parser.add_argument('--sampling', choices=['voxel', 'lesion-balanced', 'both'], default='voxel')
    parser.add_argument('--execute', action='store_true', help='Without this, print selected training cases only')
    args = parser.parse_args(argv)
    import math
    if (args.draws < 1 or args.cases < 1 or not math.isfinite(args.suv_threshold) or args.suv_threshold <= 0
            or not 0 <= args.seed < 2**32 - 10000 or any(p < 64 or p % 32 for p in args.patches)
            or len(set(args.patches)) != len(args.patches)):
        parser.error('Positive bounds/SUV threshold, valid seed and stride-compatible patches >=64 required')
    cases, manifest = load_splits(args.splits)
    selected = select_training(cases, manifest, args.fold, args.cases, args.seed)
    print(json.dumps({'role': 'train', 'cases': [c.id for c in selected], 'patches': args.patches,
                      'draws_per_case': args.draws, 'sampling': args.sampling, 'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    output = new_report(args.output, [args.data_root, args.splits])
    import torch
    torch.set_num_threads(min(settings.CPU_THREADS, 4))
    record = provenance(split_id=manifest['identity'], split_sha256=file_hash(args.splits),
                        role='train', fold=args.fold, seed=args.seed, draws=args.draws,
                        patches=args.patches, suv_threshold=args.suv_threshold,
                        sampling=args.sampling,
                        scientific_recipe=scientific_recipe(), cases=[case_record(c) for c in selected],
                        coverage_grid='preprocessed training grid; 18-connected components', status='running')
    atomic_json(output / 'manifest.json', record)
    all_draws, all_lesions = [], []
    strategies = ['voxel', 'lesion-balanced'] if args.sampling == 'both' else [args.sampling]
    transforms = build_transforms(concatenate=False)
    for index, case in enumerate(selected):
        print(f'Audit {index + 1}/{len(selected)} training case: {case.id}', flush=True)
        verify_case(case, args.data_root)
        # No shared cache writes: safe to inspect while a training job uses its cache.
        data = prepare_case(case, args.data_root, transforms)
        for patch in args.patches:
            for strategy in strategies:
                print(f'  {patch}³ {strategy}: {args.draws} independent augmented draws', flush=True)
                rows, lesions = audit_volume(data, patch, args.draws, args.seed + index, args.suv_threshold, strategy)
                all_draws.extend({'case_id': case.id, 'patient': case.patient, 'patch': patch, 'sampling': strategy, **r} for r in rows)
                all_lesions.extend({'case_id': case.id, 'patient': case.patient, 'patch': patch, 'sampling': strategy, **r} for r in lesions)
        del data
    write_csv(output / 'draws.csv', all_draws)
    if all_lesions:
        write_csv(output / 'lesions.csv', all_lesions)
    summary = {}
    from scripts.segmentation_metrics import SIZE_BINS
    for patch in args.patches:
        for strategy in strategies:
            key = f'{patch}:{strategy}' if args.sampling == 'both' else str(patch)
            rows = [r for r in all_draws if r['patch'] == patch and r['sampling'] == strategy]
            summary[key] = {'draws': len(rows), 'repeat_centres': sum(r['repeat_centre'] for r in rows),
                'draws_with_high_suv_background': sum(r['high_suv_background_voxels'] > 0 for r in rows),
                'observed_positive_centres': sum(r['centre_in_reference'] for r in rows),
                'intended_foreground_draws': sum(r['intended_foreground'] for r in rows), 'size_strata': {}}
            for group in SIZE_BINS:
                lesions = [r for r in all_lesions if r['patch'] == patch and r['sampling'] == strategy and r['size_bin'] == group]
                summary[key]['size_strata'][group] = {'reference_components': len(lesions),
                    'seen_at_least_once': sum(r['patches_visible'] > 0 for r in lesions),
                    'observed_centre_hits': sum(r['draws_centred_on_component'] for r in lesions),
                    'foreground_targets': sum(r['foreground_targets'] for r in lesions),
                    'mean_draw_fraction': sum(r['fraction_draws_visible'] for r in lesions) / len(lesions) if lesions else None}
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete'})
    print(f'Complete: {output}. High uptake outside reference is a proxy, not confirmed physiological anatomy.', flush=True)


if __name__ == '__main__':
    main()
