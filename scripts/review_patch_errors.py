"""Read-only review figures for missed small lesions, regressions and SUVmax outliers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.analysis_common import new_report, provenance, write_csv
from scripts.training_runtime import atomic_json, file_hash


def load_evaluation(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows = json.loads((folder / 'metrics.json').read_text())
    if manifest['status'] != 'complete' or len(rows) != len({r['case_id'] for r in rows}):
        raise ValueError('Evaluation must be complete with unique case identities')
    return {r['case_id']: r for r in rows}, {r['case_id']: r for r in manifest['cases']}


def choose_examples(a, b, count):
    reasons = {}
    def select(keys, reason):
        for key in keys[:count]:
            reasons.setdefault(key, []).append(reason)
    select(sorted(b, key=lambda k:b[k]['DSC']), 'lowest Dice at 128³')
    select(sorted(b, key=lambda k:b[k]['DSC'] - a[k]['DSC']), 'largest Dice decline')
    select(sorted(b, key=lambda k:-b[k]['SUVmax_absolute_error']), 'largest SUVmax error')
    select(sorted(b, key=lambda k:-(b[k]['small_lt1ml_reference'] - b[k]['small_lt1ml_found_C1'])),
           'most missed reference components <1 mL')
    return reasons


def canonical_arrays(paths):
    import nibabel as nib
    images = {key: nib.load(path) for key, path in paths.items()}
    pet = images['pet']
    for image in images.values():
        if image.shape != pet.shape or not np.allclose(image.affine, pet.affine, atol=1e-4, rtol=0):
            raise ValueError('Native geometry differs')
    canonical = {key: nib.as_closest_canonical(image) for key, image in images.items()}
    arrays = {key: image.get_fdata(dtype=np.float32, caching='unchanged') for key, image in canonical.items()}
    for key in ('gt', 'p96', 'p128'):
        if not np.isin(arrays[key], [0, 1]).all():
            raise ValueError('Nonbinary prediction/reference')
        arrays[key] = arrays[key].astype(bool)
    return images, canonical['pet'].affine, arrays


def suv_source(case, model, mask, gt, pet, canonical_affine, native_pet):
    import nibabel as nib
    if not mask.any():
        return {'case_id': case, 'model': model, 'predicted_SUVmax': 0., 'inside_reference': None,
                'maximum_voxels': 0, 'maximum_voxels_inside_reference': 0,
                'native_voxel_xyz': '', 'world_RAS_mm': '', 'original_PET_value': None,
                'PET_global_max': float(pet.max()), 'PET_proxy_slope': float(native_pet.dataobj.slope),
                'PET_proxy_intercept': float(native_pet.dataobj.inter), 'PET_storage_dtype': str(native_pet.get_data_dtype())}
    maximum = pet[mask].max()
    xyz = np.array(np.unravel_index(np.argmax(np.where(mask, pet, -np.inf)), pet.shape))
    world = nib.affines.apply_affine(canonical_affine, xyz)
    native = np.rint(nib.affines.apply_affine(np.linalg.inv(native_pet.affine), world)).astype(int)
    return {'case_id': case, 'model': model, 'predicted_SUVmax': float(maximum),
            'maximum_voxels': int((mask & (pet == maximum)).sum()),
            'maximum_voxels_inside_reference': int((mask & gt & (pet == maximum)).sum()),
            'inside_reference': bool(gt[tuple(xyz)]), 'native_voxel_xyz': str(native.tolist()),
            'world_RAS_mm': str(world.round(3).tolist()), 'original_PET_value': float(native_pet.dataobj[tuple(native)]),
            'PET_global_max': float(pet.max()), 'PET_proxy_slope': float(native_pet.dataobj.slope),
            'PET_proxy_intercept': float(native_pet.dataobj.inter), 'PET_storage_dtype': str(native_pet.get_data_dtype())}


def review_case(case, paths, title, metrics_a, metrics_b, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.segmentation_metrics import components

    images, affine, data = canonical_arrays(paths)
    gt, pet, p96, p128 = (data[k] for k in ('gt', 'pet', 'p96', 'p128'))
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    labels, count = components(gt)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    hit = set(np.unique(labels[p128]).tolist())
    small_missed = [g for g in range(1, count + 1) if g not in hit and sizes[g] * voxel_ml < 1]
    small_missed.sort(key=lambda g:(-sizes[g], g))
    slice_labels = []
    zoom = None
    if small_missed:
        positions = np.argwhere(labels == small_missed[0])
        xyz = positions[np.argmin(((positions - positions.mean(axis=0)) ** 2).sum(axis=1))]
        zoom = xyz
        slice_labels.append((int(xyz[2]), f'missed <1 mL component ({sizes[small_missed[0]] * voxel_ml:.3f} mL)'))
    missed_by_slice = (gt & ~p128).sum(axis=(0, 1))
    slice_labels.append((int(missed_by_slice.argmax()), 'most reference voxels missed by 128³'))
    fp_by_slice = (p128 & ~gt).sum(axis=(0, 1))
    slice_labels.append((int(fp_by_slice.argmax()), 'most false-positive voxels at 128³'))
    if p128.any():
        hottest = np.unravel_index(np.argmax(np.where(p128, pet, -np.inf)), pet.shape)
        slice_labels.append((int(hottest[2]), 'predicted SUVmax at 128³'))
    else:
        slice_labels.append((int(gt.sum(axis=(0, 1)).argmax()), 'largest reference slice'))
    fig, axes = plt.subplots(len(slice_labels), 5, figsize=(15, 3.1 * len(slice_labels)), squeeze=False)
    for row, (z, label) in enumerate(slice_labels):
        ct = data['ct'][:, :, z].T
        suv = pet[:, :, z].T
        ref = gt[:, :, z].T
        for col, name in enumerate(('CT', 'PET SUV', 'Reference', '96³', '128³')):
            ax = axes[row, col]
            ax.imshow(suv if col == 1 else ct, cmap='gray_r' if col == 1 else 'gray',
                      vmin=0 if col == 1 else -160, vmax=20 if col == 1 else 240,
                      origin='lower', aspect=abs(affine[1, 1] / affine[0, 0]))
            if col == 2:
                overlay = np.zeros((*ref.shape, 4)); overlay[ref] = [.1, .8, .3, .7]
                ax.imshow(overlay, origin='lower')
            if col in (3, 4):
                pred = (p96 if col == 3 else p128)[:, :, z].T
                overlay = np.zeros((*ref.shape, 4))
                overlay[ref & pred] = [.1, .8, .3, .75]
                overlay[~ref & pred] = [1., .35, .05, .75]
                overlay[ref & ~pred] = [.6, .25, 1., .8]
                ax.imshow(overlay, origin='lower')
            ax.set_title(name if row == 0 else '', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(.02, .96, 'L', color='white', transform=ax.transAxes, va='top', fontsize=8)
            ax.text(.96, .96, 'R', color='white', transform=ax.transAxes, va='top', fontsize=8)
            ax.set_aspect(abs(affine[1, 1] / affine[0, 0]))
            if row == 0 and zoom is not None:
                ax.set_xlim(max(0, int(zoom[0]) - 24), min(gt.shape[0], int(zoom[0]) + 25))
                ax.set_ylim(max(0, int(zoom[1]) - 24), min(gt.shape[1], int(zoom[1]) + 25))
                ax.plot(zoom[0], zoom[1], marker='+', color='#ffff00', markersize=8, markeredgewidth=.6)
        axes[row, 0].set_ylabel(f'RAS axial z={z}\n{label}', fontsize=8)
    fig.suptitle(f'{case}\n{title}\nDice 96³={metrics_a["DSC"]:.3f}; 128³={metrics_b["DSC"]:.3f}', fontsize=10)
    fig.text(.5, .01, 'Green: overlap; orange: false-positive voxels; violet: missed reference voxels. '
             'Neurological view; small-lesion row zoomed with yellow target. PET display clipped at SUV 20 only.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .91))
    fig.savefig(target, dpi=140)
    plt.close(fig)
    sources = [suv_source(case, model, mask, gt, pet, affine, images['pet'])
               for model, mask in [('96', p96), ('128', p128)]]
    return sources, {'case_id': case, 'small_missed_128': len(small_missed),
                     'slices': slice_labels, 'figure': target.name}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation96', required=True, type=Path)
    parser.add_argument('--evaluation128', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--per-category', type=int, default=2)
    args = parser.parse_args(argv)
    if args.per_category < 1:
        parser.error('Positive per-category count required')
    a, inputs_a = load_evaluation(args.evaluation96)
    b, inputs_b = load_evaluation(args.evaluation128)
    if a.keys() != b.keys():
        raise ValueError('Evaluation case IDs differ')
    for key in a:
        if any(a[key][field] != b[key][field] for field in ('patient', 'GT_SUVmax', 'GT_TMTV_ml', 'gt_lesions')):
            raise ValueError('Reference case metrics differ')
    output = new_report(args.output, [args.data_root, args.evaluation96, args.evaluation128])
    record = provenance(status='running', evaluation96_sha256=file_hash(args.evaluation96 / 'metrics.json'),
                        evaluation128_sha256=file_hash(args.evaluation128 / 'metrics.json'))
    atomic_json(output / 'manifest.json', record)
    selected = choose_examples(a, b, args.per_category)
    sources, examples = [], []
    for i, (case, reasons) in enumerate(selected.items(), 1):
        print(f'Review {i}/{len(selected)}: {case}; {reasons}', flush=True)
        for field in ('pet', 'reference'):
            if inputs_a[case][field]['sha256'] != inputs_b[case][field]['sha256']:
                raise ValueError('Reference/PET input hashes differ')
        paths = {'ct': args.data_root / case / 'CTres.nii.gz',
                 'pet': Path(inputs_a[case]['pet']['path']), 'gt': Path(inputs_a[case]['reference']['path']),
                 'p96': Path(inputs_a[case]['prediction']['path']), 'p128': Path(inputs_b[case]['prediction']['path'])}
        for key, path in paths.items():
            expected = (inputs_a if key in ('pet', 'gt', 'p96') else inputs_b)[case]
            if key != 'ct' and file_hash(path) != expected[{'pet':'pet', 'gt':'reference', 'p96':'prediction', 'p128':'prediction'}[key]]['sha256']:
                raise ValueError('Input changed since evaluation')
        src, example = review_case(case, paths, '; '.join(reasons), a[case], b[case], output / f'case{i:02d}.png')
        sources.extend(src); examples.append({**example, 'reasons': reasons})
        atomic_json(output / 'examples.json', examples)
    write_csv(output / 'suvmax_sources.csv', sources)
    summary = {}
    for size, rows in [('96', a), ('128', b)]:
        errors = np.array([r['SUVmax_absolute_error'] for r in rows.values()])
        summary[size] = {'examinations': len(errors), 'mean_absolute_error': float(errors.mean()),
            'median_absolute_error': float(np.median(errors)), 'largest_error_share_of_total': float(errors.max()/errors.sum()),
            'reference_exact_within_1e-6': sum(abs(r['Pred_SUVmax']-r['GT_SUVmax'])<1e-6 for r in rows.values()),
            'mean_error_excluding_largest_descriptive_only': float((errors.sum()-errors.max())/(len(errors)-1))}
    atomic_json(output / 'suvmax_summary.json', summary)
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete', 'examples': len(examples)})
    print(f'Complete: {output}. Review is exploratory; anatomical interpretation requires visual checking.', flush=True)


if __name__ == '__main__':
    main()
