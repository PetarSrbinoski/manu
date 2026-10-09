"""Patient-level autoPET v2 adaptation of Microsoft's lymphoma baseline.

Model and transform recipe adapted from Microsoft Corporation, MIT licensed.
See README for the inspected source and deviations. Defaults live in config.py.
Run: python scripts/train.py --fold 0 --resume
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

# Make project imports available when this file is launched from any directory.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as settings
from scripts.preprocessing_cache import PreprocessingCache
# Expose training helpers for analysis scripts and regression tests.
from scripts.training_data import (
    FILES, Case, TrainingDataset, binary_prediction, build_transforms, discover_cases,
    export_prediction, fold_cases, foreground_dice, load_or_create_splits,
    predict_held_out, predict_volume, prepare_case, summarize_folds,
    training_transforms, verify_completed_fold,
)
from scripts.training_options import parse_args, scientific_recipe, resolved_config, stopping_policy
from scripts.training_runtime import (
    AMP_FORWARD_PREDECESSOR, AMP_OVERFLOW_PREDECESSOR, PILOT_PREDECESSOR,
    PILOT_STOPPING_POLICY, REFACTOR_PREDECESSOR, SCRIPTS_PREDECESSOR, PATHS_PREDECESSOR, FORMAT_VERSION, LOGGER, UPSTREAM,
    atomic_json, atomic_torch_save, build_model, compatible_config,
    early_stopping_status, elapsed_time, environment_info, file_hash, finite_tensors,
    gpu_peak, hardware_info, identity, implementation_hash, inference_forward,
    log_progress, random_state, reset_peak, restore_checkpoint, run_logging,
    run_on_rank_zero, save_checkpoint, seed_epoch, set_random_state, training_loss,
    save_run_record,
)


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
            LOGGER.info('Fold %d already completed; verified outputs and skipped training', fold)
            return
        roles = fold_cases(cases, manifest, fold)
        training, validation = roles['train'], roles['validation']
        LOGGER.info('Fold %d | train=%d exams | inner validation=%d exams | held-out=%d exams',
                    fold, len(training), len(validation), len(roles['held_out']))
        if args.smoke:
            training, validation = training[:max(2, world)], validation[:1]
            # smoke_steps attempted optimizer updates; AMP may skip overflows.
            count = args.smoke_steps * args.batch_size * args.accumulate * world
            training = [training[i % len(training)] for i in range(count)]
            LOGGER.info('Smoke limits: %d distinct training exams, %d attempted updates/epoch, %d validation exam; no held-out export',
                        len({c.id for c in training}), args.smoke_steps, len(validation))
        LOGGER.info('Building residual U-Net on %s | patch=%d | per-GPU batch=%d | accumulation=%d',
                    device, args.patch, args.batch_size, args.accumulate)
        seed_epoch(args.seed + fold)
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=settings.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=settings.MIN_LEARNING_RATE)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=settings.LOSS_INCLUDE_BACKGROUND)
        start_epoch, best, history = 0, -1.0, []
        selected_checkpoint = None
        if args.resume and (output / 'last.pt').exists():
            LOGGER.info('Loading resumable checkpoint: %s', output / 'last.pt')
            state = restore_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler,
                                       checkpoint_config, manifest['identity'], rank)
            start_epoch, best, history = state['epoch'], state['best'], state['history']
            selected_checkpoint = state['selected_checkpoint']
            # last.pt is the authoritative committed epoch. Repair a best.pt write
            # interrupted before last.pt publication, even if replay differs.
            if rank == 0 and selected_checkpoint is not None:
                atomic_torch_save(output / 'best.pt', selected_checkpoint)
            LOGGER.info('Resumed fold %d after epoch %d; next epoch=%d | best inner Dice=%s',
                        fold, start_epoch, start_epoch + 1, f'{best:.4f}' if best >= 0 else 'not evaluated yet')
        wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
        disk_cache = PreprocessingCache(args.disk_cache, args.disk_cache_gib, args.min_free_disk_gib) if args.disk_cache else None
        dataset = TrainingDataset(training, args.data_root, args.patch, args.cache_gib, disk_cache)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed + fold, drop_last=False)
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.workers, pin_memory=False, generator=generator,
                            persistent_workers=False)
        valid_transforms = build_transforms(concatenate=False)
        end_epoch = min(args.epochs, args.stop_after_epoch or args.epochs)
        policy = stopping_policy(args)
        stopped_early = run_on_rank_zero(lambda: early_stopping_status(history, **policy)['should_stop'], rank, world)
        LOGGER.info('Early stopping: patience=%d validation checks | minimum epochs=%d | minimum Dice improvement=%g | 0 patience disables',
                    policy['patience'], policy['min_epochs'], policy['min_delta'])
        for epoch in range(start_epoch, start_epoch if stopped_early else end_epoch):
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
            optimizer_updates, amp_skipped_updates = 0, 0
            amp_forward_retries = 0
            data_wait_seconds, training_step_seconds = 0.0, 0.0
            phase = f'Fold {fold} epoch {epoch + 1}/{args.epochs} training batches'
            LOGGER.info('Fold %d | epoch %d/%d | %d batches/rank | lr=%.6g',
                        fold, epoch + 1, args.epochs, len(loader), scheduler.get_last_lr()[0])
            optimizer.zero_grad(set_to_none=True)
            batches = iter(loader)
            last_report = started
            for step in range(len(loader)):
                report = step == 0 or (step + 1) % args.log_every == 0 or step + 1 == len(loader)
                if report or time.perf_counter() - last_report >= 30:
                    LOGGER.info('Fold %d epoch %d | waiting for training batch %d/%d (CPU loading/preprocessing)',
                                fold, epoch + 1, step + 1, len(loader))
                load_started = time.perf_counter()
                batch = next(batches)
                load_seconds = time.perf_counter() - load_started
                data_wait_seconds += load_seconds
                step_started = time.perf_counter()
                inputs, labels = batch['image'].to(device), batch['GT'].to(device)
                group_start = (step // args.accumulate) * args.accumulate
                # Normalize partial final accumulation groups by actual examples.
                group_examples = min(args.accumulate * args.batch_size,
                                     len(sampler) - group_start * args.batch_size)
                synchronize = (step + 1) % args.accumulate == 0 or step + 1 == len(loader)
                context = wrapped.no_sync() if world > 1 and not synchronize else nullcontext()
                with context:
                    loss, retried = training_loss(wrapped, inputs, labels, loss_function,
                                                 amp_enabled=device.type == 'cuda')
                    if retried:
                        amp_forward_retries += 1
                        LOGGER.warning('Float32 forward recovery succeeded | fold=%d epoch=%d batch=%d/%d | retries this epoch=%d',
                                       fold, epoch + 1, step + 1, len(loader), amp_forward_retries)
                    scaler.scale(loss * len(inputs) / group_examples).backward()
                if synchronize:
                    scaler.unscale_(optimizer)
                    if not scaler.is_enabled() and any(
                            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('Nonfinite gradients; no checkpoint written')
                    previous_scale = scaler.get_scale()
                    # GradScaler skips nonfinite gradients and backs off its scale.
                    # Raising before step/update prevents this normal AMP recovery.
                    scaler.step(optimizer)
                    scaler.update()
                    current_scale = scaler.get_scale()
                    if not math.isfinite(current_scale) or current_scale <= 0:
                        raise RuntimeError('Invalid AMP gradient scale; no checkpoint written')
                    if current_scale < previous_scale:
                        amp_skipped_updates += 1
                        LOGGER.warning('AMP overflow | fold=%d epoch=%d batch=%d/%d | optimizer update skipped; '
                                       'gradient scale %g -> %g | skipped this epoch=%d',
                                       fold, epoch + 1, step + 1, len(loader), previous_scale,
                                       current_scale, amp_skipped_updates)
                    else:
                        optimizer_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item() * len(inputs)
                examples += len(inputs)
                training_step_seconds += time.perf_counter() - step_started
                if report or time.perf_counter() - last_report >= 30:
                    allocated = torch.cuda.memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
                    log_progress(phase, step + 1, len(loader), started,
                                 f'loss={total_loss / examples:.5f} | batch wait={load_seconds:.1f}s | '
                                 f'GPU allocated={allocated:.0f} MiB | optimizer updates={optimizer_updates} | '
                                 f'AMP skipped={amp_skipped_updates} | float32 retries={amp_forward_retries}')
                    last_report = time.perf_counter()
                del inputs, labels, loss, batch
            del batches  # Release nonpersistent loader workers before full-volume validation.
            if optimizer_updates == 0:
                raise RuntimeError('All optimizer updates were skipped; no checkpoint written')
            scheduler.step()
            train_memory = gpu_peak(device)
            train_seconds = time.perf_counter() - started
            LOGGER.info('Training epoch finished | loss=%.5f | elapsed=%s | peak GPU allocated/reserved=%.0f/%.0f MiB',
                        total_loss / examples, elapsed_time(train_seconds),
                        train_memory['allocated_bytes'] / 1024**2, train_memory['reserved_bytes'] / 1024**2)
            LOGGER.info('Training timing | data wait=%.2fs | training steps=%.2fs | waiting for data=%.1f%%',
                        data_wait_seconds, training_step_seconds, 100 * data_wait_seconds / train_seconds)
            LOGGER.info('Optimizer updates=%d | AMP skipped updates=%d | float32 retries=%d | gradient scale=%g',
                        optimizer_updates, amp_skipped_updates, amp_forward_retries, scaler.get_scale())
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
                LOGGER.info('Fold %d epoch %d | full-volume inner validation: %d examinations',
                            fold, epoch + 1, len(validation))
                scores = []
                for case_index, case in enumerate(validation, 1):
                    LOGGER.info('Validation %d/%d: %s | loading and resampling on CPU',
                                case_index, len(validation), case.id)
                    data = prepare_case(case, args.data_root, valid_transforms, disk_cache)
                    data['image'] = torch.cat((data['CT'], data['PT']), dim=0)
                    logits = predict_volume(model, data['image'], device, args.patch, args.sw_batch_size)
                    scores.append(foreground_dice(binary_prediction(logits), data['GT'][0]))
                    log_progress('Inner validation examinations', case_index, len(validation), validation_start,
                                 f'foreground Dice={scores[-1]:.4f}')
                    del data, logits
                score = sum(scores) / len(scores)
                LOGGER.info('Inner validation finished | mean foreground Dice=%.4f', score)
            elif rank != 0:
                LOGGER.info('Waiting for rank zero validation/checkpoint selection')
            else:
                LOGGER.info('Validation not scheduled this epoch (interval=%d; final epoch always validated)', args.val_interval)
            validation_memory = gpu_peak(device)
            validation_seconds = time.perf_counter() - validation_start
            loss_key = 'loss_including_background' if settings.LOSS_INCLUDE_BACKGROUND else 'loss_excluding_background'
            row = {'epoch': epoch + 1, loss_key: total_loss / examples,
                   'inner_foreground_dice_2mm': score, 'next_lr': scheduler.get_last_lr()[0],
                   'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'data_wait_seconds': data_wait_seconds, 'training_step_seconds': training_step_seconds,
                   'optimizer_updates': optimizer_updates, 'amp_skipped_updates': amp_skipped_updates,
                   'amp_forward_retries': amp_forward_retries,
                   'gradient_scale': scaler.get_scale(),
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
                row['early_stopping'] = early_stopping_status(history, **policy)
                if score is not None and policy['patience']:
                    LOGGER.info('Early stopping monitor | checks without meaningful improvement=%d/%d | last improvement epoch=%d | minimum epoch=%d',
                                row['early_stopping']['bad_checks'], policy['patience'],
                                row['early_stopping']['last_improvement_epoch'], policy['min_epochs'])
                if score is not None and score > best:
                    LOGGER.info('New best inner foreground Dice: %.4f (previous=%s) | saving %s',
                                score, f'{best:.4f}' if best >= 0 else 'none', output / 'best.pt')
                    best = score
                    selected_checkpoint = {'model': {k: v.detach().cpu().clone()
                                                     for k, v in model.state_dict().items()},
                                           'epoch': epoch + 1, 'score': score, 'config': checkpoint_config,
                                           'split_id': manifest['identity']}
                    atomic_torch_save(output / 'best.pt', selected_checkpoint)
                LOGGER.info('Saving epoch %d checkpoint and history: %s', epoch + 1, output / 'last.pt')
                save_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler,
                                epoch + 1, best, history, checkpoint_config, manifest['identity'], states, selected_checkpoint)
                atomic_json(output / 'history.json', history)
                LOGGER.info('Epoch %d committed | validation=%s | validation peak GPU allocated/reserved=%.0f/%.0f MiB | process peak RAM=%.2f GiB',
                            epoch + 1, f'{score:.4f}' if score is not None else 'not scheduled',
                            validation_memory['allocated_bytes'] / 1024**2,
                            validation_memory['reserved_bytes'] / 1024**2, row['process_peak_rss_bytes'] / 1024**3)
            stopped_early = run_on_rank_zero(lambda: early_stopping_status(history, **policy)['should_stop'], rank, world)
            if stopped_early:
                LOGGER.info('Early stopping reached after epoch %d; retaining the best inner-validation checkpoint', epoch + 1)
                break
        if end_epoch < args.epochs:
            LOGGER.info('Stopped at requested epoch boundary %d/%d; rerun with --resume to continue', end_epoch, args.epochs)
            return  # Explicit epoch-boundary interruption for resume verification.
        if rank == 0:
            LOGGER.info('Training finished for fold %d (%s); loading selected checkpoint: %s',
                        fold, 'early stopping' if stopped_early else 'epoch budget', output / 'best.pt')
            selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
            if not compatible_config(selected['config'], checkpoint_config) or selected['split_id'] != manifest['identity']:
                raise ValueError('incompatible selected checkpoint (including fold identity)')
            model.load_state_dict(selected['model'])
            LOGGER.info('Selected epoch %d | inner foreground Dice=%.4f', selected['epoch'], selected['score'])
            reset_peak(device)
            inference_start = time.perf_counter()
            results = [] if args.smoke else predict_held_out(
                model, cases, manifest, fold, args.data_root, output, device, args.patch, args.sw_batch_size)
            atomic_json(output / 'complete.json', {'config': config, 'split_id': manifest['identity'],
                        'best_epoch': selected['epoch'], 'inner_foreground_dice_2mm': selected['score'],
                        'trained_epochs': history[-1]['epoch'], 'stopped_early': stopped_early,
                        'held_out_cases': [r['case'] for r in results],
                        'inference_seconds': time.perf_counter() - inference_start,
                        'inference_gpu_peak': gpu_peak(device), 'smoke': args.smoke})
            LOGGER.info('Fold %d complete | %d held-out predictions | outputs: %s', fold, len(results), output)
        if world > 1:
            dist.barrier()
    finally:
        if lock is not None:
            lock.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    if args is None:
        return
    with run_logging(args):
        run(args)


def run(args) -> None:
    try:
        import torch
        import monai
        import nibabel
        import numpy
    except ImportError as error:
        raise SystemExit(f'Missing dependency: {error}. Follow README setup; no packages installed automatically.')
    import torch.distributed as dist
    LOGGER.info('Dependencies loaded | torch=%s | MONAI=%s', torch.__version__, monai.__version__)
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
        LOGGER.info('Initializing distributed training | rank=%d/%d | local device=%s', rank, world, device)
        # Rank-zero full-volume validation can take longer than NCCL's default timeout.
        dist.init_process_group('nccl', timeout=timedelta(hours=24))
    try:
        manifest_path = args.splits or args.output / 'splits.json'
        if args.splits and not manifest_path.exists() and not args.prepare:
            raise ValueError('Shared --splits must exist; run --prepare first')
        cases = None
        manifest = None
        if rank == 0:
            LOGGER.info('Auditing examination files; full-cohort reads can take several minutes before training starts')
            cases = discover_cases(args.data_root)
            if len(cases) != args.expected_exams or len({c.patient for c in cases}) != args.expected_patients:
                raise ValueError('Unexpected cohort size; inspect data or explicitly configure expected counts')
            manifest = load_or_create_splits(cases, manifest_path, args.seed)
            LOGGER.info('Audit complete: %d examinations / %d patients | split identity=%s',
                        len(cases), len({c.patient for c in cases}), manifest['identity'])
        else:
            LOGGER.info('Waiting for rank zero to audit data and publish patient splits')
        if world > 1:
            payload: list[Any] = [cases, manifest]
            dist.broadcast_object_list(payload, src=0)
            cases, manifest = payload
        assert cases is not None and manifest is not None
        if args.prepare:
            LOGGER.info('Preparation complete; no training requested | splits: %s', manifest_path)
            return
        hardware, environment = hardware_info(), environment_info()
        config, resources = resolved_config(args, manifest['identity'], environment, device, world)
        available = hardware['available_ram_bytes']
        if available and args.cache_gib * world * 1024**3 > available / 2:
            raise ValueError('Cache budget exceeds half available RAM; leave room for transforms and volumes')
        if rank == 0:
            record = {'config': config, 'resources': resources, 'hardware': hardware, 'environment': environment,
                      'data_root': str(args.data_root.resolve()), 'split_path': str(manifest_path.resolve()),
                      'command': sys.argv}
            LOGGER.info('Resolved configuration:\n%s', json.dumps(record, indent=2))
            LOGGER.info('Device=%s | GPU count used=%d | effective batch=%d | workers/rank=%d | cache/rank=%.2f GiB',
                        device, world if device.type == 'cuda' else 0, config['scientific']['effective_batch_size'], args.workers, args.cache_gib)
            save_run_record(args.output, record, args.resume)
        if world > 1:
            dist.barrier()
        folds = range(5) if args.fold == 'all' else [int(args.fold or 0)]
        for fold in folds:
            LOGGER.info('Starting fold %d | epoch budget=%d | output=%s', fold, args.epochs, args.output / f'fold_{fold}')
            train_fold(args, cases, manifest, fold, config, device, rank, world)
        if rank == 0 and args.fold == 'all' and not args.smoke and not args.stop_after_epoch:
            summarize_folds(args.output, cases, manifest, config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
