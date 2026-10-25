"""Preview or launch matched 128³ fold-0 sampling pilots, pausing before test inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings


def pilot_commands(root, splits, strategy, smoke, resume):
    strategies = ['voxel', 'lesion-balanced'] if strategy == 'both' else [strategy]
    base = [sys.executable, '-u', str(settings.PROJECT_ROOT / 'scripts/train.py')]
    for sampling in strategies:
        output = root / (f'smoke-{sampling}' if smoke else sampling)
        command = base + (['--smoke'] if smoke else ['--fold', str(settings.SAMPLING_PILOT_FOLD)])
        command += ['--patch', str(settings.SAMPLING_PILOT_PATCH), '--sampling', sampling,
                    '--splits', str(splits), '--output', str(output),
                    '--epochs', str(1 if smoke else settings.EPOCHS),
                    '--sw-batch-size', str(settings.PATCH_COMPARISON_SW_BATCH_SIZE)]
        if not smoke:
            command += ['--stop-after-epoch', str(settings.SAMPLING_PILOT_STOP_EPOCH)]
        if resume:
            command += ['--resume']
        yield command


def comparison(reference, balanced):
    old = reference['pooled_detection']['C1']['size_strata']['small_lt1ml']['recall']
    new = balanced['pooled_detection']['C1']['size_strata']['small_lt1ml']['recall']
    old_fp = reference['per_exam']['FPV_ml']['mean']
    new_fp = balanced['per_exam']['FPV_ml']['mean']
    dice_delta = balanced['per_exam']['DSC']['mean'] - reference['per_exam']['DSC']['mean']
    gain = new - old if old is not None and new is not None else None
    checks = {'small_recall_gain': gain is not None and gain >= settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
              'mean_FPV_guard': new_fp <= old_fp * settings.SAMPLING_PILOT_MAX_FPV_RATIO,
              'mean_Dice_guard': dice_delta >= -settings.SAMPLING_PILOT_MAX_DICE_DROP}
    return {'small_recall_delta': gain, 'mean_FPV_reference_ml': old_fp, 'mean_FPV_balanced_ml': new_fp,
            'mean_Dice_delta': dice_delta, 'checks': checks, 'advance_to_further_validation': all(checks.values()),
            'criteria': {'minimum_small_recall_gain': settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
                         'maximum_mean_FPV_ratio': settings.SAMPLING_PILOT_MAX_FPV_RATIO,
                         'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP},
            'interpretation': 'Exploratory inner-validation screen, not clinical thresholds or final held-out evidence.'}


def freeze_protocol(root, splits, require_existing=False):
    from scripts.training_runtime import file_hash, implementation_hash
    protocol = {'fold': settings.SAMPLING_PILOT_FOLD, 'patch': settings.SAMPLING_PILOT_PATCH,
                'inspection_epoch': settings.SAMPLING_PILOT_STOP_EPOCH, 'schedule_epochs': settings.EPOCHS,
                'splits_sha256': file_hash(splits), 'trainer_sha256': implementation_hash(),
                'config_sha256': file_hash(Path(settings.__file__)),
                'metric_source_sha256': {name: file_hash(settings.PROJECT_ROOT / 'scripts' / name)
                                         for name in ('segmentation_metrics.py', 'evaluate_segmentation.py')},
                'commands': list(pilot_commands(root, splits, 'both', False, False)),
                'criteria': {'minimum_small_recall_gain': settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
                             'maximum_mean_FPV_ratio': settings.SAMPLING_PILOT_MAX_FPV_RATIO,
                             'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP},
                'primary_endpoint': 'Pooled native-grid recall of reference components <1 mL, C1 any overlap, inner validation only',
                'selection': f'Best inner-validation Dice checkpoint within the first {settings.SAMPLING_PILOT_STOP_EPOCH} epochs; same policy for both arms',
                'interpretation': 'Research screening criteria, not clinical thresholds or final evidence'}
    path = root / 'pilot_protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Pilot protocol/source/settings changed; preserve this experiment and choose a fresh root')
    else:
        if require_existing:
            raise ValueError('Missing pre-training pilot protocol; cannot create one retroactively during evaluation')
        if any((root / sampling / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'last.pt').exists()
               for sampling in ('voxel', 'lesion-balanced')):
            raise ValueError('Existing pilot checkpoints have no frozen protocol; use a fresh root')
        root.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(protocol, stream, indent=2)
    return protocol


def evaluate_pilot(root, splits, review):
    from scripts.analysis_common import new_report, provenance
    from scripts.training_runtime import atomic_json, file_hash
    configs = []
    for sampling in ('voxel', 'lesion-balanced'):
        checkpoint = root / sampling / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'best.pt'
        history = root / sampling / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'history.json'
        if not checkpoint.is_file() or not history.is_file():
            raise ValueError(f'Pilot checkpoint/history missing: {sampling}')
        if json.loads(history.read_text())[-1]['epoch'] != settings.SAMPLING_PILOT_STOP_EPOCH:
            raise ValueError('Both arms must reach the predefined inspection epoch')
        config = json.loads((root / sampling / 'run.json').read_text())['config']
        if config['scientific'].get('patch_sampling') != sampling:
            raise ValueError('Pilot arm has the wrong recorded sampler')
        normalized = {**config, 'scientific': {k: v for k, v in config['scientific'].items()
                                              if k not in ('patch_sampling', 'lesion_index')},
                      'versions': {k: v for k, v in config['versions'].items() if k != 'connected-components-3d'}}
        normalized.pop('lesion_index_cache_gib', None)
        configs.append(normalized)
    if configs[0] != configs[1]:
        raise ValueError('Pilot arms differ in settings/source beyond the intended sampling change')
    output = new_report(review, [settings.DATA_ROOT, root / 'voxel', root / 'lesion-balanced', splits])
    record = provenance(status='running', role='validation', splits_sha256=file_hash(splits),
                        stop_epoch=settings.SAMPLING_PILOT_STOP_EPOCH)
    atomic_json(output / 'manifest.json', record)
    summaries, case_rows = [], []
    for sampling in ('voxel', 'lesion-balanced'):
        checkpoint = root / sampling / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'best.pt'
        predictions = output / f'{sampling}-predictions'
        metrics = output / f'{sampling}-metrics'
        commands = [
            [sys.executable, '-u', str(settings.PROJECT_ROOT / 'scripts/predict_inner_validation.py'),
             '--checkpoint', str(checkpoint), '--splits', str(splits), '--data-root', str(settings.DATA_ROOT),
             '--output', str(predictions), '--sw-batch-size', str(settings.PATCH_COMPARISON_SW_BATCH_SIZE), '--execute'],
            [sys.executable, '-u', str(settings.PROJECT_ROOT / 'scripts/evaluate_segmentation.py'),
             '--manifest', str(predictions / 'evaluation_inputs.json'), '--output', str(metrics)],
        ]
        for command in commands:
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=settings.PROJECT_ROOT, check=True)
        summaries.append(json.loads((metrics / 'summary.json').read_text()))
        case_rows.append({r['case_id']: r for r in json.loads((metrics / 'metrics.json').read_text())})
    if case_rows[0].keys() != case_rows[1].keys() or any(
            case_rows[0][k][field] != case_rows[1][k][field]
            for k in case_rows[0] for field in ('patient', 'GT_TMTV_ml', 'GT_SUVmax', 'gt_lesions')):
        raise ValueError('Pilot evaluation cases/references differ')
    atomic_json(output / 'comparison.json', comparison(*summaries))
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete'})
    print(f'Inner-validation pilot comparison: {output / "comparison.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Execute both bounded pilots; default is preview')
    mode.add_argument('--smoke', action='store_true', help='Short execution check, not the learning pilot')
    mode.add_argument('--evaluate', action='store_true', help='Export inner-validation predictions and score both completed pilots')
    parser.add_argument('--strategy', choices=['both', 'voxel', 'lesion-balanced'], default='both')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--splits', type=Path, default=settings.PATCH_COMPARISON_OUTPUT_ROOT / 'splits.json')
    parser.add_argument('--output-root', type=Path, default=settings.SAMPLING_PILOT_OUTPUT_ROOT)
    parser.add_argument('--review-output', type=Path, help='New report directory for --evaluate; default: pilot root/inner-review')
    args = parser.parse_args(argv)
    if args.resume and args.smoke:
        parser.error('Use a fresh output root for repeated smoke checks')
    if args.evaluate and (args.resume or args.strategy != 'both'):
        parser.error('--evaluate compares both arms and does not accept --resume')
    splits = (settings.PROJECT_ROOT / args.splits).resolve()
    root = (settings.PROJECT_ROOT / args.output_root).resolve()
    if not splits.is_file():
        parser.error('An existing audited split manifest is required')
    if not 1 <= settings.SAMPLING_PILOT_STOP_EPOCH < min(settings.EPOCHS, settings.EARLY_STOPPING_MIN_EPOCHS):
        parser.error('Pilot inspection must precede the epoch budget and earliest automatic stop')
    for protected in (settings.DATA_ROOT.resolve(), settings.OUTPUT_ROOT.resolve(),
                      settings.PATCH_COMPARISON_OUTPUT_ROOT.resolve()):
        if root == protected or protected in root.parents or root in protected.parents:
            parser.error('Pilot outputs must be separate from source data and saved experiments')
    if args.evaluate:
        freeze_protocol(root, splits, require_existing=True)
        review = (settings.PROJECT_ROOT / args.review_output).resolve() if args.review_output else root / 'inner-review'
        evaluate_pilot(root, splits, review)
        return
    if args.run:
        freeze_protocol(root, splits)
    print('Execute requested.' if args.run or args.smoke else 'Preview only; add --smoke or --run.', flush=True)
    for command in pilot_commands(root, splits, args.strategy, args.smoke, args.resume):
        print(shlex.join(command), flush=True)
        if args.run or args.smoke:
            subprocess.run(command, cwd=settings.PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
