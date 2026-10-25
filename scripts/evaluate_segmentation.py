"""Evaluate existing native-grid masks; never performs inference or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis_common import load_splits, new_report, provenance, write_csv
from scripts.training_runtime import atomic_json, file_hash
from scripts.training_data import fold_cases


def run_entries(run, splits, data_root, fold):
    cases, manifest = load_splits(splits)
    record = json.loads((run / 'run.json').read_text())
    if record['config']['split_id'] != manifest['identity'] or record['config']['scientific']['smoke']:
        raise ValueError('Run/split identity differs or this is a smoke diagnostic')
    entries = []
    for f in (range(5) if fold == 'all' else [int(fold)]):
        folder = run / f'fold_{f}' / 'predictions'
        expected = fold_cases(cases, manifest, f)['held_out']
        complete = json.loads((folder.parent / 'complete.json').read_text())
        if (complete['config'] != record['config'] or complete['split_id'] != manifest['identity']
                or len(complete['held_out_cases']) != len(expected)
                or set(complete['held_out_cases']) != {c.id for c in expected}):
            raise ValueError(f'Fold {f}: incompatible completion record')
        actual = {p.parent.relative_to(folder).as_posix() for p in folder.rglob('SEG_pred.nii.gz')}
        if actual != {c.id for c in expected}:
            raise ValueError(f'Fold {f}: missing or extra prediction case identities')
        for case in expected:
            entries.append({'case_id': case.id, 'patient': case.patient, 'fold': f,
                'reference': str(data_root / case.id / 'SEG.nii.gz'),
                'pet': str(data_root / case.id / 'SUV.nii.gz'),
                'prediction': str(folder / case.id / 'SEG_pred.nii.gz')})
    return entries


def validate_entries(entries, base):
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must be a nonempty JSON list')
    seen, predictions = set(), set()
    result = []
    for entry in entries:
        if not isinstance(entry.get('case_id'), str) or not isinstance(entry.get('patient'), str):
            raise ValueError('Each row requires case_id and patient strings')
        if entry['case_id'] in seen or not entry['case_id'].startswith(entry['patient'] + '/'):
            raise ValueError('Duplicate case or mismatched patient/case identity')
        seen.add(entry['case_id'])
        row = {**entry}
        for key in ('reference', 'pet', 'prediction'):
            row[key] = (base / entry[key]).resolve()
            if not row[key].is_file():
                raise ValueError(f'Missing {key}: {row[key]}')
        if row['prediction'] in predictions:
            raise ValueError('One prediction cannot be assigned to multiple examinations')
        predictions.add(row['prediction'])
        result.append(row)
    return result


def read_case(entry):
    import nibabel as nib
    import numpy as np
    images = {k: nib.load(entry[k]) for k in ('reference', 'pet', 'prediction')}
    pet = images['pet']
    if len(pet.shape) != 3 or pet.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('PET must be a 3D grid with millimetre units')
    for name, image in images.items():
        if (image.shape != pet.shape or not np.isfinite(image.affine).all()
                or not np.allclose(image.affine, pet.affine, atol=1e-4, rtol=0)
                or image.header.get_xyzt_units()[0] not in ('unknown', 'mm')):
            raise ValueError(f'{entry["case_id"]}: {name} does not share the native PET grid')
    voxel_ml = float(abs(np.linalg.det(pet.affine[:3, :3])) / 1000)
    if voxel_ml <= 0:
        raise ValueError('Singular affine')
    return (*(images[k].get_fdata(dtype=np.float32, caching='unchanged')
              for k in ('reference', 'prediction', 'pet')), voxel_ml)


def evaluate(entries, output, input_provenance):
    from scripts.segmentation_metrics import metrics, summarize, CONNECTIVITY, IOU_THRESHOLD
    output = new_report(output, [row[k] for row in entries for k in ('reference', 'pet', 'prediction')])
    record = provenance(**input_provenance, connectivity=CONNECTIVITY, iou_threshold=IOU_THRESHOLD,
                        status='running', cases=[])
    atomic_json(output / 'manifest.json', record)
    rows, lesions = [], []
    for n, entry in enumerate(entries, 1):
        print(f'Evaluation {n}/{len(entries)}: {entry["case_id"]}', flush=True)
        row, components = metrics(*read_case(entry))
        rows.append({'case_id': entry['case_id'], 'patient': entry['patient'], **row})
        lesions.extend({'case_id': entry['case_id'], 'patient': entry['patient'], **item} for item in components)
        record['cases'].append({'case_id': entry['case_id'], 'patient': entry['patient'],
            **{key: {'path': str(entry[key]), 'sha256': file_hash(entry[key])}
               for key in ('reference', 'pet', 'prediction')}})
        atomic_json(output / 'manifest.json', record)
        print(f'  Dice={row["DSC"]:.4f}; FPV={row["FPV_ml"]:.2f} mL; FNV={row["FNV_ml"]:.2f} mL', flush=True)
    write_csv(output / 'per_case.csv', rows)
    if lesions:
        write_csv(output / 'reference_lesions.csv', lesions)
    else:
        (output / 'reference_lesions.csv').write_text('case_id,patient,lesion,volume_ml,size_bin,found_C1,found_C2,found_C3\n')
    atomic_json(output / 'metrics.json', rows)
    atomic_json(output / 'summary.json', summarize(rows))
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete'})
    print(f'Complete: {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='JSON list: case_id, patient, reference, pet, prediction')
    source.add_argument('--run', type=Path, help='Existing trainer run with fold_N/predictions')
    parser.add_argument('--splits', type=Path, help='Required with --run; audited patient splits')
    parser.add_argument('--data-root', type=Path, help='Required with --run')
    parser.add_argument('--fold', choices=['all', '0', '1', '2', '3', '4'], default='all')
    parser.add_argument('--output', required=True, type=Path, help='New analysis directory; never overwritten')
    args = parser.parse_args(argv)
    if args.run:
        if not args.splits or not args.data_root:
            parser.error('--run requires --splits and --data-root')
        entries = run_entries(args.run.resolve(), args.splits, args.data_root.resolve(), args.fold)
        source_info = {'splits_sha256': file_hash(args.splits), 'run': str(args.run.resolve()),
                       'fold': args.fold, 'role': 'held_out'}
        base = Path.cwd()
    else:
        entries = json.loads(args.manifest.read_text())
        source_info = {'input_manifest_sha256': file_hash(args.manifest),
                       'role': 'explicit manifest; patient routing asserted by caller'}
        base = args.manifest.resolve().parent
    entries = validate_entries(entries, base)
    evaluate(entries, args.output, source_info)


if __name__ == '__main__':
    main()
