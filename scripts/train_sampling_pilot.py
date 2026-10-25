"""Preview or launch three matched 128³ fold-0 sampling/loss pilots."""
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

ARMS = {'voxel': ('voxel', 'dice'),
        'lesion-balanced': ('lesion-balanced', 'dice'),
        'lesion-balanced-dice-ce': ('lesion-balanced', 'dice-ce')}


def pilot_commands(root, splits, strategy, smoke, resume):
    strategies = list(ARMS) if strategy == 'all' else list(ARMS)[:2] if strategy == 'both' else [strategy]
    base = [sys.executable, '-u', str(settings.PROJECT_ROOT / 'scripts/train.py')]
    for arm in strategies:
        sampling, loss = ARMS[arm]
        output = root / (f'smoke-{arm}' if smoke else arm)
        command = base + (['--smoke'] if smoke else ['--fold', str(settings.SAMPLING_PILOT_FOLD)])
        command += ['--patch', str(settings.SAMPLING_PILOT_PATCH), '--sampling', sampling, '--loss', loss,
                    '--splits', str(splits), '--output', str(output),
                    '--epochs', str(1 if smoke else settings.EPOCHS),
                    '--sw-batch-size', str(settings.PATCH_COMPARISON_SW_BATCH_SIZE)]
        if loss == 'dice-ce':
            command += ['--dice-weight', str(settings.LOSS_DICE_WEIGHT), '--ce-weight', str(settings.LOSS_CE_WEIGHT)]
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
    checks = {'small_recall_gain': gain is not None and gain + 1e-12 >= settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
              'mean_FPV_guard': new_fp <= old_fp * settings.SAMPLING_PILOT_MAX_FPV_RATIO + 1e-12,
              'mean_Dice_guard': dice_delta + 1e-12 >= -settings.SAMPLING_PILOT_MAX_DICE_DROP}
    return {'small_recall_delta': gain, 'mean_FPV_reference_ml': old_fp, 'mean_FPV_candidate_ml': new_fp,
            'mean_Dice_delta': dice_delta, 'checks': checks, 'advance_to_further_validation': all(checks.values()),
            'criteria': {'minimum_small_recall_gain': settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
                         'maximum_mean_FPV_ratio': settings.SAMPLING_PILOT_MAX_FPV_RATIO,
                         'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP},
            'interpretation': 'Exploratory inner-validation screen, not clinical thresholds or final held-out evidence.'}


def loss_comparison(reference, candidate):
    result = comparison(reference, candidate)
    old_fp, new_fp = result['mean_FPV_reference_ml'], result['mean_FPV_candidate_ml']
    reduction = (old_fp - new_fp) / old_fp if old_fp > 0 else None
    delta = result['small_recall_delta']
    checks = {'mean_FPV_reduction': reduction is not None and reduction + 1e-12 >= settings.LOSS_PILOT_MIN_FPV_REDUCTION,
              'small_recall_guard': delta is not None and delta + 1e-12 >= -settings.LOSS_PILOT_MAX_SMALL_RECALL_DROP,
              'mean_Dice_guard': result['mean_Dice_delta'] + 1e-12 >= -settings.SAMPLING_PILOT_MAX_DICE_DROP}
    return {**result, 'mean_FPV_fraction_reduction': reduction, 'checks': checks,
            'advance_to_further_validation': all(checks.values()),
            'criteria': {'minimum_mean_FPV_fraction_reduction': settings.LOSS_PILOT_MIN_FPV_REDUCTION,
                         'maximum_small_recall_drop': settings.LOSS_PILOT_MAX_SMALL_RECALL_DROP,
                         'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP}}


def freeze_protocol(root, splits, require_existing=False):
    from scripts.training_runtime import file_hash, implementation_hash
    protocol = {'fold': settings.SAMPLING_PILOT_FOLD, 'patch': settings.SAMPLING_PILOT_PATCH,
                'inspection_epoch': settings.SAMPLING_PILOT_STOP_EPOCH, 'schedule_epochs': settings.EPOCHS,
                'splits_sha256': file_hash(splits), 'trainer_sha256': implementation_hash(),
                'config_sha256': file_hash(Path(settings.__file__)),
                'pilot_launcher_sha256': file_hash(Path(__file__)),
                'metric_source_sha256': {name: file_hash(settings.PROJECT_ROOT / 'scripts' / name)
                                         for name in ('segmentation_metrics.py', 'evaluate_segmentation.py')},
                'arms': {name: {'sampling': spec[0], 'loss': spec[1]} for name, spec in ARMS.items()},
                'commands': list(pilot_commands(root, splits, 'all', False, False)),
                'criteria': {'minimum_small_recall_gain': settings.SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN,
                             'maximum_mean_FPV_ratio': settings.SAMPLING_PILOT_MAX_FPV_RATIO,
                             'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP},
                'loss_comparison_criteria': {'minimum_mean_FPV_fraction_reduction': settings.LOSS_PILOT_MIN_FPV_REDUCTION,
                                             'maximum_small_recall_drop': settings.LOSS_PILOT_MAX_SMALL_RECALL_DROP,
                                             'maximum_mean_Dice_drop': settings.SAMPLING_PILOT_MAX_DICE_DROP},
                'primary_endpoint': 'Pooled native-grid recall of reference components <1 mL, C1 any overlap, inner validation only',
                'loss_primary_endpoint': 'C vs B: mean native-grid unmatched-component FP volume, inner validation only',
                'selection': f'Best inner-validation Dice checkpoint within the first {settings.SAMPLING_PILOT_STOP_EPOCH} epochs; same policy for all arms',
                'interpretation': 'Research screening criteria, not clinical thresholds or final evidence'}
    path = root / 'pilot_protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Pilot protocol/source/settings changed; preserve this experiment and choose a fresh root')
    else:
        if require_existing:
            raise ValueError('Missing pre-training pilot protocol; cannot create one retroactively during evaluation')
        if any((root / arm / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'last.pt').exists() for arm in ARMS):
            raise ValueError('Existing pilot checkpoints have no frozen protocol; use a fresh root')
        root.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(protocol, stream, indent=2)
    return protocol


def validate_arm_configs(configs):
    for arm, (sampling, loss) in ARMS.items():
        science = configs[arm]['scientific']
        expected_loss = 'softmax-Dice+CE' if loss == 'dice-ce' else 'softmax-Dice'
        if science.get('patch_sampling') != sampling or science.get('loss') != expected_loss:
            raise ValueError('Pilot arm has the wrong recorded sampler/loss')
        if loss == 'dice-ce' and science.get('loss_weights') != {'dice': settings.LOSS_DICE_WEIGHT, 'cross_entropy': settings.LOSS_CE_WEIGHT}:
            raise ValueError('Combined loss weights differ from frozen configuration')

    def normalize(config, *, sampling=False, loss=False):
        ignore = ({'patch_sampling', 'lesion_index'} if sampling else set()) | ({'loss', 'loss_weights'} if loss else set())
        result = {**config, 'scientific': {k: v for k, v in config['scientific'].items() if k not in ignore}}
        if sampling:
            result['versions'] = {k: v for k, v in config['versions'].items() if k != 'connected-components-3d'}
            result.pop('lesion_index_cache_gib', None)
        return result

    if normalize(configs['voxel'], sampling=True) != normalize(configs['lesion-balanced'], sampling=True):
        raise ValueError('A/B differ in settings/source beyond the intended sampling change')
    if normalize(configs['lesion-balanced'], loss=True) != normalize(configs['lesion-balanced-dice-ce'], loss=True):
        raise ValueError('B/C differ in settings/source beyond the intended loss change')


def evaluate_pilot(root, splits, review):
    from scripts.analysis_common import new_report, provenance
    from scripts.training_runtime import atomic_json, file_hash
    configs = {}
    for arm in ARMS:
        checkpoint = root / arm / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'best.pt'
        history = root / arm / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'history.json'
        if not checkpoint.is_file() or not history.is_file():
            raise ValueError(f'Pilot checkpoint/history missing: {arm}')
        if json.loads(history.read_text())[-1]['epoch'] != settings.SAMPLING_PILOT_STOP_EPOCH:
            raise ValueError('All three arms must reach the predefined inspection epoch')
        configs[arm] = json.loads((root / arm / 'run.json').read_text())['config']
    validate_arm_configs(configs)
    output = new_report(review, [settings.DATA_ROOT, *(root / arm for arm in ARMS), splits])
    record = provenance(status='running', role='validation', splits_sha256=file_hash(splits),
                        stop_epoch=settings.SAMPLING_PILOT_STOP_EPOCH)
    atomic_json(output / 'manifest.json', record)
    summaries, case_rows = {}, {}
    for arm in ARMS:
        checkpoint = root / arm / f'fold_{settings.SAMPLING_PILOT_FOLD}' / 'best.pt'
        predictions = output / f'{arm}-predictions'
        metrics = output / f'{arm}-metrics'
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
        summaries[arm] = json.loads((metrics / 'summary.json').read_text())
        case_rows[arm] = {r['case_id']: r for r in json.loads((metrics / 'metrics.json').read_text())}
    reference = case_rows['voxel']
    for arm in ARMS:
        rows = case_rows[arm]
        if reference.keys() != rows.keys() or any(reference[k][field] != rows[k][field]
                for k in reference for field in ('patient', 'GT_TMTV_ml', 'GT_SUVmax', 'gt_lesions')):
            raise ValueError('Pilot evaluation cases/references differ')
    a, b, c = (summaries[arm] for arm in ARMS)
    atomic_json(output / 'comparison.json', {
        'sampling_B_vs_A': comparison(a, b), 'loss_C_vs_B': loss_comparison(b, c),
        'combined_C_vs_A_descriptive': comparison(a, c),
        'interpretation': 'B/A isolates sampling; C/B isolates loss; C/A combines both changes. No automatic winner selection.'})
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete'})
    print(f'Inner-validation pilot comparison: {output / "comparison.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Execute three bounded pilots; default is preview')
    mode.add_argument('--smoke', action='store_true', help='Short execution check, not the learning pilot')
    mode.add_argument('--evaluate', action='store_true', help='Export inner-validation predictions and score all three completed pilots')
    parser.add_argument('--arm', '--strategy', dest='strategy', choices=['all', 'both', *ARMS], default='all',
                        help='Default all three; both retains the original A/B-only selection')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--splits', type=Path, default=settings.PATCH_COMPARISON_OUTPUT_ROOT / 'splits.json')
    parser.add_argument('--output-root', type=Path, default=settings.SAMPLING_PILOT_OUTPUT_ROOT)
    parser.add_argument('--review-output', type=Path, help='New report directory for --evaluate; default: pilot root/inner-review')
    args = parser.parse_args(argv)
    if args.resume and args.smoke:
        parser.error('Use a fresh output root for repeated smoke checks')
    if args.evaluate and (args.resume or args.strategy != 'all'):
        parser.error('--evaluate compares all three arms and does not accept --resume')
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
