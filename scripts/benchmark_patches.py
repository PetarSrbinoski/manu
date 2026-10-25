"""Bounded patch benchmark; preview by default, --execute uses the selected GPU.

Each patch/precision runs in a fresh child process. The fallback mode injects an
AMP-only invalid output, exercising the existing buffer/RNG restore and float32
retry instead of merely estimating fallback memory from a float32 forward pass.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings
from scripts.analysis_common import (case_record, load_splits, new_report, provenance,
                                     verify_case, write_csv)
from scripts.training_data import fold_cases
from scripts.training_options import scientific_recipe
from scripts.training_runtime import atomic_json, file_hash


def largest_case(cases, root):
    import nibabel as nib
    import math
    return max(cases, key=lambda c: (math.prod(nib.load(root / c.id / 'SUV.nii.gz').shape), c.id))


def fallback_hook(_module, _inputs, output):
    import torch
    return output * float('inf') if torch.is_autocast_enabled('cuda') else output


def measured(action, device):
    import resource
    import torch
    from scripts.training_runtime import gpu_peak, reset_peak
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    reset_peak(device)
    start = time.perf_counter()
    result = action()
    torch.cuda.synchronize(device)
    return result, {'seconds': time.perf_counter() - start, **gpu_peak(device),
                    'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def worker(job):
    import torch
    from monai.losses import DiceLoss
    from scripts.training_data import Case, TrainingDataset, build_transforms, prepare_case, predict_volume
    from scripts.training_runtime import build_model, seed_epoch, training_loss, hardware_info

    device = torch.device(job['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('CUDA required; no CPU fallback for GPU fit claims')
    torch.cuda.set_device(device)
    torch.set_num_threads(job['threads'])
    seed_epoch(job['seed'])
    model = build_model().to(device)
    if job['precision'] == 'fallback':
        model.register_forward_hook(fallback_hook)
    train_case = Case(**job['training_case'])
    valid_case = Case(**job['validation_case'])
    root = Path(job['data_root'])
    for case in (train_case, valid_case):
        verify_case(case, root)
    dataset = TrainingDataset([train_case], root, job['patch'])
    dataset.transform.set_random_state(seed=job['seed'])
    prepared = time.perf_counter()
    batch = dataset[0]
    inputs = batch['image'].as_tensor().unsqueeze(0).to(device)
    labels = batch['GT'].as_tensor().unsqueeze(0).to(device)
    del batch, dataset
    print(f'Training patch prepared in {time.perf_counter() - prepared:.1f}s', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=settings.LOSS_INCLUDE_BACKGROUND)

    def update():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, retried = training_loss(model, inputs, labels, loss_function, amp_enabled=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        return {'loss': float(loss.detach()), 'float32_retry': bool(retried),
                'optimizer_update': scaler.get_scale() >= before}

    rows = []
    # Cold first update includes optimizer-state allocation; steady updates are separate.
    for step in range(job['steps'] + 1):
        value, timing = measured(update, device)
        rows.append({'phase': 'training', 'warmup': step == 0, 'step': step,
                     'sw_batch_size': None, 'status': 'ok', **value, **timing})
        atomic_json(Path(job['result']), {'status': 'running', 'rows': rows, 'hardware': hardware_info()})
        print(f'Training {step}/{job["steps"]}: {timing}', flush=True)
    # Trainer keeps AdamW state and gradients during validation; preserve both.
    del inputs, labels
    prepared = time.perf_counter()
    data = prepare_case(valid_case, root, build_transforms(concatenate=False))
    data['image'] = torch.cat((data['CT'], data['PT']), dim=0)
    preprocessing_seconds = time.perf_counter() - prepared
    print(f'Largest inner-validation volume ready in {preprocessing_seconds:.1f}s', flush=True)
    for sw in job['sw_batches']:
        try:
            logits, timing = measured(lambda: predict_volume(model, data['image'], device, job['patch'], sw), device)
            del logits
            rows.append({'phase': 'validation', 'warmup': False, 'step': None, 'sw_batch_size': sw,
                         'status': 'ok', 'loss': None, 'float32_retry': job['precision'] == 'fallback',
                         'optimizer_update': None, **timing})
        except torch.OutOfMemoryError as error:
            gc.collect()
            torch.cuda.empty_cache()
            rows.append({'phase': 'validation', 'sw_batch_size': sw, 'status': 'out_of_memory', 'error': str(error)})
        atomic_json(Path(job['result']), {'status': 'running', 'rows': rows, 'hardware': hardware_info(),
                                          'validation_preprocessing_seconds': preprocessing_seconds})
    atomic_json(Path(job['result']), {'status': 'complete', 'rows': rows, 'hardware': hardware_info(),
                                      'validation_preprocessing_seconds': preprocessing_seconds})


def recommend(jobs):
    """Only recommend settings that passed training and both inference precisions."""
    result = {}
    for patch in sorted({job['patch'] for job in jobs}):
        variants = {j['precision']: j for j in jobs if j['patch'] == patch}
        eligible = []
        for sw in variants.get('amp', {}).get('sw_batches', []):
            good = True
            amp_time = None
            for precision in ('amp', 'fallback'):
                job = variants.get(precision)
                if not job or not Path(job['result']).exists():
                    good = False
                    break
                data = json.loads(Path(job['result']).read_text())
                train = [r for r in data.get('rows', []) if r['phase'] == 'training']
                valid = [r for r in data.get('rows', []) if r['phase'] == 'validation' and r['sw_batch_size'] == sw]
                if (data['status'] != 'complete' or len(train) != job['steps'] + 1
                        or not all(r.get('optimizer_update') for r in train)
                        or not valid or valid[0]['status'] != 'ok'
                        or (precision == 'fallback' and not all(r.get('float32_retry') for r in train))):
                    good = False
                if precision == 'amp' and valid and valid[0]['status'] == 'ok':
                    amp_time = valid[0]['seconds']
            if good:
                eligible.append((amp_time, sw))
        result[str(patch)] = {'recommended_sw_batch_size': min(eligible)[1] if eligible else None,
            'interpretation': 'Measured fit for this bounded sample only; no segmentation-quality evidence.',
            'eligible_sw_batches': [sw for _, sw in sorted(eligible)]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--patches', nargs='+', type=int, default=[settings.PATCH_SIZE, *settings.PATCH_COMPARISON_SIZES])
    parser.add_argument('--sw-batches', nargs='+', type=int, required=False, default=[settings.PATCH_COMPARISON_SW_BATCH_SIZE])
    parser.add_argument('--steps', type=int, default=settings.SMOKE_STEPS)
    parser.add_argument('--threads', type=int, default=settings.CPU_THREADS)
    parser.add_argument('--device', default=settings.DEVICE)
    parser.add_argument('--execute', action='store_true', help='Run bounded GPU workers; otherwise preview only')
    args = parser.parse_args(argv)
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            worker(job)
        except Exception as error:
            import torch
            previous = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else {}
            atomic_json(Path(job['result']), {**previous, 'status': 'out_of_memory' if isinstance(error, torch.OutOfMemoryError) else 'failed',
                                              'error': f'{type(error).__name__}: {error}'})
            raise
        return
    if not args.splits or not args.data_root or not args.output:
        parser.error('--splits, --data-root and --output required')
    if settings.BATCH_SIZE != 1 or settings.GRADIENT_ACCUMULATION != 1:
        parser.error('This bounded benchmark measures training batch 1 / accumulation 1 only')
    if (args.steps < 1 or args.threads < 1 or any(p < 64 or p % 32 for p in args.patches)
            or any(b < 1 for b in args.sw_batches) or len(set(args.patches)) != len(args.patches)
            or len(set(args.sw_batches)) != len(args.sw_batches)):
        parser.error('Positive bounds, unique batches/stride-compatible patches >=64 required')
    cases, manifest = load_splits(args.splits)
    roles = fold_cases(cases, manifest, args.fold)
    training = largest_case(roles['train'], args.data_root)
    validation = largest_case(roles['validation'], args.data_root)
    print(json.dumps({'training_case': training.id, 'largest_inner_validation_case': validation.id,
                      'patches': args.patches, 'sw_batches': args.sw_batches, 'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    output = new_report(args.output, [args.data_root, args.splits])
    record = provenance(split_id=manifest['identity'], splits_sha256=file_hash(args.splits),
                        scientific_recipe=scientific_recipe(), fold=args.fold, status='running', jobs=[])
    atomic_json(output / 'manifest.json', record)
    jobs = []
    for patch in args.patches:
        for precision in ('amp', 'fallback'):
            name = f'patch{patch}-{precision}'
            job = {'patch': patch, 'precision': precision, 'steps': args.steps, 'sw_batches': args.sw_batches,
                   'device': args.device, 'threads': args.threads, 'seed': settings.SEED,
                   'training_case': case_record(training), 'validation_case': case_record(validation),
                   'data_root': str(args.data_root.resolve()), 'result': str((output / f'{name}-results.json').resolve())}
            jobs.append(job)
            spec = output / f'{name}-job.json'
            atomic_json(spec, job)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(spec)]
            print(shlex.join(command), flush=True)
            with (output / f'{name}.log').open('x') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=settings.PROJECT_ROOT).returncode
            record['jobs'].append({'name': name, 'exit_code': code, 'spec': spec.name})
            atomic_json(output / 'manifest.json', record)
            if code == 130 or code < 0:
                raise RuntimeError(f'Worker interrupted ({code}); stopping benchmark')
    recommendations = recommend(jobs)
    atomic_json(output / 'recommendations.json', recommendations)
    flat = []
    for job in jobs:
        result = Path(job['result'])
        data = json.loads(result.read_text()) if result.exists() else {'status': 'failed', 'error': 'No worker result'}
        for row in data.get('rows', [{}]):
            flat.append({'patch': job['patch'], 'precision': job['precision'], 'worker_status': data['status'],
                         **row, 'worker_error': data.get('error')})
    fields = sorted({key for row in flat for key in row})
    write_csv(output / 'measurements.csv', [{key: row.get(key) for key in fields} for row in flat])
    atomic_json(output / 'manifest.json', {**record, 'status': 'complete_with_failures' if any(j['exit_code'] for j in record['jobs']) else 'complete'})
    print(json.dumps(recommendations, indent=2), flush=True)


if __name__ == '__main__':
    main()
