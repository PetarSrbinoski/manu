"""Export a saved checkpoint's inner-validation masks without training it."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings
from scripts.analysis_common import load_splits, new_report, provenance, verify_case
from scripts.training_data import build_transforms, export_prediction, fold_cases, prepare_case, predict_volume
from scripts.training_runtime import atomic_json, build_model, file_hash, implementation_hash, compatible_config
from scripts.training_options import scientific_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--splits', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=int, help='Optional bounded number of validation examinations')
    parser.add_argument('--device', default=settings.DEVICE)
    parser.add_argument('--sw-batch-size', type=int, default=settings.PATCH_COMPARISON_SW_BATCH_SIZE)
    parser.add_argument('--execute', action='store_true', help='Load trusted checkpoint and run inference')
    args = parser.parse_args(argv)
    if args.sw_batch_size < 1 or (args.cases is not None and args.cases < 1):
        parser.error('Positive window batch/case count required')
    # Preview does not unpickle checkpoint files.
    if not args.execute:
        print(f'Preview: {args.checkpoint} -> inner-validation masks in {args.output}; add --execute.')
        return
    import torch
    torch.set_num_threads(settings.CPU_THREADS)
    cases, manifest = load_splits(args.splits)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    current = {**config, 'script_sha256': implementation_hash(),
               'scientific': {**config['scientific'], 'patch_sampling': config['scientific'].get('patch_sampling', 'voxel')}}
    if checkpoint['split_id'] != manifest['identity'] or not compatible_config(config, current):
        raise ValueError('Checkpoint split/source identity differs from supplied manifest/current trainer')
    recipe = scientific_recipe()
    if any(config['scientific'].get(k) != v for k, v in recipe.items()):
        raise ValueError('Current scientific recipe differs from checkpoint; use its original trainer/configuration')
    selected = fold_cases(cases, manifest, int(config['fold']))['validation']
    if args.cases is not None:
        selected = selected[:args.cases]
    device = torch.device(args.device)
    output = new_report(args.output, [args.data_root, args.checkpoint, args.splits])
    record = provenance(checkpoint_sha256=file_hash(args.checkpoint), split_id=manifest['identity'],
                        fold=config['fold'], role='validation', selected_epoch=checkpoint['epoch'], status='running')
    atomic_json(output / 'manifest.json', record)
    model = build_model().to(device)
    model.load_state_dict(checkpoint['model'])
    transforms = build_transforms(labels=False)
    entries = []
    for index, case in enumerate(selected, 1):
        print(f'Inner validation {index}/{len(selected)}: {case.id}', flush=True)
        verify_case(case, args.data_root)
        data = prepare_case(case, args.data_root, transforms)
        logits = predict_volume(model, data['image'], device, config['scientific']['patch'], args.sw_batch_size)
        target = output / 'predictions' / case.id / 'SEG_pred.nii.gz'
        export_prediction(logits, data, transforms, args.data_root / case.id / 'SUV.nii.gz', target)
        entries.append({'case_id': case.id, 'patient': case.patient,
                        'reference': str((args.data_root / case.id / 'SEG.nii.gz').resolve()),
                        'pet': str((args.data_root / case.id / 'SUV.nii.gz').resolve()),
                        'prediction': str(target.resolve())})
        del logits, data
    atomic_json(output / 'evaluation_inputs.json', entries)
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete', 'examinations': len(entries)})


if __name__ == '__main__':
    main()
