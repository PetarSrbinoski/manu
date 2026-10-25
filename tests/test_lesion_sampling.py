"""Sampling distribution, paired geometry and strict legacy resume contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.data import MetaTensor

import config
from scripts.lesion_sampling import LesionBalancedCropd, LesionIndexCache, lesion_index
from scripts.training_data import training_transforms
from scripts.training_options import parse_args, resolved_config
from scripts.training_runtime import SAMPLING_PREDECESSOR, compatible_config, implementation_hash


def test_components_are_uniform_even_when_voxel_counts_differ():
    label = np.zeros((1, 64, 64, 64), np.float32)
    label[0, 15, 15, 15] = 1
    label[0, 35:45, 35:45, 35:45] = 1
    index = lesion_index(label)
    assert index.count == 2 and index.offsets.tolist() == [0, 1, 1001]
    crop = LesionBalancedCropd(keys=['GT', 'PT'], label_key='GT', spatial_size=(16,) * 3,
        pos=2, neg=1, num_samples=1, image_key='PT', image_threshold=0,
        fg_indices_key='GT_fg_indices', bg_indices_key='GT_bg_indices')
    crop.set_random_state(seed=9)
    # A recording crop isolates distribution from MONAI boundary corrections.
    selected = []
    def capture(data):
        selected.append(int(data['GT_fg_indices'][0]))
        assert '_lesion_index' not in data
        return [data]
    crop._crop = capture
    data = {'GT': label, 'PT': np.ones_like(label), '_lesion_index': index}
    for _ in range(2000):
        crop(data)
    tiny = np.ravel_multi_index((15, 15, 15), label.shape[1:])
    share = sum(i == tiny for i in selected) / len(selected)
    assert .45 < share < .55  # Reference voxel sampling would give ~0.1%.


def test_actual_monai_background_ratio_and_positive_targets():
    label = np.zeros((1, 64, 64, 64), np.float32)
    label[0, 16:20, 16:20, 16:20] = 1
    label[0, 44:48, 44:48, 44:48] = 1
    pet = np.ones_like(label)
    crop = LesionBalancedCropd(keys=['GT', 'PT'], label_key='GT', spatial_size=(8,) * 3,
        pos=2, neg=1, num_samples=1, image_key='PT', image_threshold=0,
        fg_indices_key='GT_fg_indices', bg_indices_key='GT_bg_indices')
    index = lesion_index(label)
    crop.set_random_state(seed=41)
    positive = 0
    for _ in range(400):
        sample = crop({'GT': label, 'PT': pet, '_lesion_index': index})[0]
        centre = tuple(crop.cropper.centers[0])
        positive += int(label[(0, *centre)] > 0)
        assert sample['GT'].shape == (1, 8, 8, 8)
    assert .59 < positive / 400 < .74


@pytest.mark.parametrize('cached_indices', [False, True])
def test_actual_monai_keeps_uniform_components_with_and_without_cached_indices(cached_indices):
    from monai.transforms import FgBgToIndicesd
    label = np.zeros((1, 64, 64, 64), np.float32)
    label[0, 16, 16, 16] = 1
    label[0, 35:45, 35:45, 35:45] = 1
    data = {'GT': label, 'PT': np.ones_like(label), '_lesion_index': lesion_index(label)}
    if cached_indices:
        data = FgBgToIndicesd(keys='GT', image_key='PT', image_threshold=0)(data)
    crop = LesionBalancedCropd(keys=['GT', 'PT'], label_key='GT', spatial_size=(8,) * 3,
        pos=1, neg=0, num_samples=1, image_key='PT', image_threshold=0,
        fg_indices_key='GT_fg_indices', bg_indices_key='GT_bg_indices')
    crop.set_random_state(seed=17)
    tiny = 0
    for _ in range(400):
        crop(data)
        centre = tuple(crop.cropper.centers[0])
        tiny += centre == (16, 16, 16)
        assert label[(0, *centre)] > 0
    assert .4 < tiny / 400 < .6


def test_determinism_paired_channels_and_index_cache_limits(monkeypatch):
    torch.set_num_threads(1)
    gt = torch.zeros((1, 80, 80, 80))
    gt[:, 8:12, 8:12, 8:12] = 1
    data = {k: MetaTensor(v) for k, v in [('GT', gt), ('CT', gt * 3), ('PT', gt * 7)]}
    monkeypatch.setattr(config, 'AFFINE_PROBABILITY', 0)
    transform = training_transforms(64, 'lesion-balanced')
    index = lesion_index(gt)
    transform.set_random_state(seed=5)
    first = transform({**data, '_lesion_index': index})[0]
    transform.set_random_state(seed=5)
    second = transform({**data, '_lesion_index': index})[0]
    torch.testing.assert_close(first['image'], second['image'])
    torch.testing.assert_close(first['image'][0], first['GT'][0] * 3)
    torch.testing.assert_close(first['image'][1], first['GT'][0] * 7)
    assert not any(k.startswith('_lesion') for k in first)
    cache = LesionIndexCache(index.nbytes)
    assert cache.get('first', gt) is cache.get('first', gt)
    cache.get('second', gt)
    assert list(cache.entries) == ['second'] and cache.bytes <= cache.limit
    uncached = LesionIndexCache(0)
    uncached.get('first', gt)
    assert not uncached.entries


def test_sampling_is_recorded_and_legacy_resume_only_allows_voxel():
    env = {'packages': {'torch': 'test'}, 'script_sha256': implementation_hash()}
    default, _ = resolved_config(parse_args(['--fold', '0']), 'split', env, torch.device('cpu'), 1)
    balanced, _ = resolved_config(parse_args(['--fold', '0', '--sampling', 'lesion-balanced']), 'split', env, torch.device('cpu'), 1)
    assert default['scientific']['patch_sampling'] == 'voxel'
    assert balanced['scientific']['patch_sampling'] == 'lesion-balanced'
    assert 'connected-components-3d' in balanced['versions']
    old = deepcopy(default)
    old['scientific'].pop('patch_sampling')
    old['script_sha256'] = SAMPLING_PREDECESSOR
    assert compatible_config(old, default)
    assert not compatible_config(old, balanced)
    changed = deepcopy(default)
    changed['scientific']['patch'] = 128
    assert not compatible_config(old, changed)
    old['script_sha256'] = 'unknown'
    assert not compatible_config(old, default)


def test_empty_reference_rejected_and_pilot_is_preview_only(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='positive'):
        lesion_index(np.zeros((4, 4, 4)))
    from scripts import train_sampling_pilot as pilot
    splits = tmp_path / 'splits.json'
    splits.write_text('{}')
    def unexpected(*args, **kwargs):
        raise AssertionError('Preview must not launch training')
    monkeypatch.setattr(pilot.subprocess, 'run', unexpected)
    pilot.main(['--splits', str(splits), '--output-root', str(tmp_path / 'pilot')])
    commands = list(pilot.pilot_commands(tmp_path / 'pilot', splits, 'both', False, False))
    assert len(commands) == 2
    for command in commands:
        args = parse_args(command[3:])
        assert args.fold == '0' and args.patch == 128
        assert args.epochs == 500 and args.stop_after_epoch == 50
        assert args.sw_batch_size == 1
    assert commands[0] != commands[1]


@pytest.mark.parametrize('sampling,loss_name', [('voxel', 'dice'), ('lesion-balanced', 'dice'),
                                             ('lesion-balanced', 'dice-ce')])
def test_training_epoch_resume_across_cleanup_matches_uninterrupted(tmp_path, monkeypatch, sampling, loss_name):
    from test_training import make_data
    from scripts import train
    from scripts.training_runtime import CLEANUP_PREDECESSOR
    root = tmp_path / 'data'
    make_data(root)
    monkeypatch.setattr(train, 'build_model', lambda: torch.nn.Conv3d(2, 2, 1))
    base = ['--smoke', '--epochs', '2', '--smoke-steps', '2', '--sampling', sampling,
            '--loss', loss_name,
            '--device', 'cpu', '--threads', '1', '--no-disk-cache', '--data-root', str(root),
            '--expected-exams', '10', '--expected-patients', '10']
    train.main(base + ['--output', str(tmp_path / 'continuous')])
    train.main(base + ['--output', str(tmp_path / 'resumed'), '--stop-after-epoch', '1'])
    # Exercise the exact pre-cleanup checkpoint/run migration on temporary data.
    output = tmp_path / 'resumed'
    for path in (output / 'fold_0').glob('*.pt'):
        state = torch.load(path, weights_only=False)
        state['config']['script_sha256'] = CLEANUP_PREDECESSOR
        torch.save(state, path)
    record = json.loads((output / 'run.json').read_text())
    record['config']['script_sha256'] = CLEANUP_PREDECESSOR
    (output / 'run.json').write_text(json.dumps(record))
    train.main(base + ['--output', str(tmp_path / 'resumed'), '--resume'])
    migration = json.loads((output / 'cleanup-resume-migration.json').read_text())
    assert migration['previous_run'] == record
    left = torch.load(tmp_path / 'continuous/fold_0/last.pt', weights_only=False)
    right = torch.load(tmp_path / 'resumed/fold_0/last.pt', weights_only=False)
    assert left['epoch'] == right['epoch'] == 2
    for key in left['model']:
        torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
    key = 'dice_ce_loss_including_background' if loss_name == 'dice-ce' else 'loss_including_background'
    assert left['history'][-1][key] == right['history'][-1][key]


def test_reference_voxel_transform_preserves_original_golden_outputs():
    fixture = json.loads((Path(__file__).parent / 'fixtures/voxel_sampler_golden.json').read_text())
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    gt = np.zeros((1, 80, 80, 80), np.float32)
    gt[:, 10:14, 10:14, 10:14] = 1
    gt[:, 50:60, 50:60, 50:60] = 1
    values = {'CT': rng.random(gt.shape, dtype=np.float32),
              'PT': rng.random(gt.shape, dtype=np.float32) * 10, 'GT': gt}
    for item in fixture['cases']:
        data = {k: MetaTensor(v.copy(), affine=torch.diag(torch.tensor([2., 2., 2., 1.]))) for k, v in values.items()}
        transform = training_transforms(64, 'voxel')
        transform.set_random_state(seed=item['seed'])
        result = transform(data)[0]
        for key in ('image', 'GT'):
            assert hashlib.sha256(result[key].cpu().numpy().tobytes()).hexdigest() == item[key]


def test_pilot_screen_requires_recall_and_both_error_guards():
    from scripts.train_sampling_pilot import comparison
    def summary(recall, fp, dice):
        return {'pooled_detection': {'C1': {'size_strata': {'small_lt1ml': {'recall': recall}}}},
                'per_exam': {'FPV_ml': {'mean': fp}, 'DSC': {'mean': dice}}}
    control = summary(.2, 100, .5)
    assert comparison(control, summary(.26, 105, .50))['advance_to_further_validation']
    assert not comparison(control, summary(.24, 100, .50))['advance_to_further_validation']
    assert not comparison(control, summary(.3, 120, .50))['advance_to_further_validation']
    assert not comparison(control, summary(.3, 100, .45))['advance_to_further_validation']
    assert not comparison(summary(None, 100, .5), summary(.3, 100, .5))['advance_to_further_validation']


def test_pilot_protocol_is_frozen_before_training_and_rejects_changed_criteria(tmp_path, monkeypatch):
    from scripts.train_sampling_pilot import freeze_protocol
    splits = tmp_path / 'splits.json'
    splits.write_text('{}')
    root = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='Missing pre-training'):
        freeze_protocol(root, splits, require_existing=True)
    smoke = root / 'smoke-voxel/fold_0/last.pt'
    smoke.parent.mkdir(parents=True)
    smoke.touch()  # Smoke checkpoints must not prevent the subsequent full pilot.
    first = freeze_protocol(root, splits)
    assert freeze_protocol(root, splits, require_existing=True) == first
    monkeypatch.setattr(config, 'SAMPLING_PILOT_MIN_SMALL_RECALL_GAIN', .01)
    with pytest.raises(ValueError, match='changed'):
        freeze_protocol(root, splits)


def test_pilot_evaluation_rejects_confounded_arms_before_inference(tmp_path, monkeypatch):
    from scripts import train_sampling_pilot as pilot
    splits = tmp_path / 'splits.json'
    splits.write_text('{}')
    root = tmp_path / 'pilot'
    for arm, (sampling, loss) in pilot.ARMS.items():
        folder = root / arm / 'fold_0'
        folder.mkdir(parents=True)
        (folder / 'best.pt').touch()
        (folder / 'history.json').write_text(json.dumps([{'epoch': 50}]))
        config = {'scientific': {'patch_sampling': sampling, 'patch': 96 if arm == 'lesion-balanced' else 128,
                                  'loss': 'softmax-Dice+CE' if loss == 'dice-ce' else 'softmax-Dice'},
                  'versions': {'torch': 'test'}, 'script_sha256': 'test'}
        if loss == 'dice-ce':
            config['scientific']['loss_weights'] = {'dice': 1., 'cross_entropy': 1.}
        (folder.parent / 'run.json').write_text(json.dumps({'config': config}))
    def unexpected(*args, **kwargs):
        raise AssertionError('Confounded arms must not launch inference')
    monkeypatch.setattr(pilot.subprocess, 'run', unexpected)
    with pytest.raises(ValueError, match='beyond the intended'):
        pilot.evaluate_pilot(root, splits, root / 'review')
    assert not (root / 'review').exists()
