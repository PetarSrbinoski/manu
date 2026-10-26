"""Combined loss numerics, recovery, strict resume and three-arm pilot contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from monai.losses import DiceLoss

import config
from scripts.training_options import parse_args, resolved_config
from scripts.training_runtime import (build_loss, training_loss, implementation_hash,
                                      compatible_config, LOSS_PREDECESSOR, CLEANUP_PREDECESSOR)
from scripts import train_sampling_pilot as pilot


def test_combined_loss_matches_weighted_components_and_has_finite_gradients():
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
    labels = torch.randint(0, 2, (2, 1, 4, 4, 4)).float()
    args = parse_args(['--fold', '0', '--loss', 'dice-ce', '--dice-weight', '.7', '--ce-weight', '1.3'])
    actual = build_loss(args)(logits, labels)
    expected = (.7 * DiceLoss(to_onehot_y=True, softmax=True, include_background=True)(logits, labels)
                + 1.3 * torch.nn.functional.cross_entropy(logits, labels[:, 0].long()))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_original_dice_loss_is_unchanged():
    logits = torch.randn(1, 2, 4, 4, 4)
    labels = torch.zeros(1, 1, 4, 4, 4)
    actual = build_loss(parse_args(['--fold', '0']))(logits, labels)
    expected = DiceLoss(to_onehot_y=True, softmax=True, include_background=config.LOSS_INCLUDE_BACKGROUND)(logits, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('flag,value', [('--dice-weight', '0'), ('--ce-weight', '-1'),
                                      ('--ce-weight', 'nan'), ('--dice-weight', 'inf')])
def test_invalid_combined_weights_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse_args(['--fold', '0', '--loss', 'dice-ce', flag, value])


def test_loss_metadata_and_weights_prevent_incompatible_resume():
    env = {'packages': {'torch': 'test'}, 'script_sha256': implementation_hash()}
    def resolved(extra):
        return resolved_config(parse_args(['--fold', '0', '--sampling', 'lesion-balanced', *extra]),
                               'split', env, torch.device('cpu'), 1)[0]
    dice = resolved([])
    combined = resolved(['--loss', 'dice-ce'])
    changed = resolved(['--loss', 'dice-ce', '--ce-weight', '2'])
    assert combined['scientific']['loss_weights'] == {'dice': 1., 'cross_entropy': 1.}
    assert not compatible_config(dice, combined)
    assert not compatible_config(combined, changed)
    old = deepcopy(dice)
    old['script_sha256'] = LOSS_PREDECESSOR
    assert compatible_config(old, dice)
    assert not compatible_config(old, combined)


@pytest.mark.parametrize('sampling,loss', [('voxel', 'dice'), ('lesion-balanced', 'dice'),
                                        ('lesion-balanced', 'dice-ce')])
def test_cleanup_resume_only_changes_the_exact_source_identity(sampling, loss):
    env = {'packages': {'torch': 'test'}, 'script_sha256': implementation_hash()}
    args = parse_args(['--fold', '0', '--sampling', sampling, '--loss', loss])
    current = resolved_config(args, 'split', env, torch.device('cpu'), 1)[0]
    saved = deepcopy(current)
    saved['script_sha256'] = CLEANUP_PREDECESSOR
    assert compatible_config(saved, current)
    for section, field, value in [('scientific', 'lr', 1.), ('scientific', 'patch', 256),
                                  ('scientific', 'loss', 'other'), ('versions', 'torch', 'other'),
                                  (None, 'threads', 1), (None, 'split_id', 'other'),
                                  (None, 'script_sha256', 'unknown')]:
        changed = deepcopy(saved)
        (changed[section] if section else changed)[field] = value
        assert not compatible_config(changed, current)
    missing_sampling = deepcopy(saved)
    del missing_sampling['scientific']['patch_sampling']
    assert not compatible_config(missing_sampling, current)


@pytest.mark.parametrize('change', [None, 'trainer', 'launcher', 'config', 'metrics', 'criteria', 'commands'])
def test_cleanup_protocol_migration_preserves_original_and_rejects_other_changes(tmp_path, change):
    splits = tmp_path / 'splits.json'
    splits.write_text('{}')
    root = tmp_path / 'pilot'
    current = pilot.freeze_protocol(root, splits)
    saved = deepcopy(current)
    saved['trainer_sha256'] = CLEANUP_PREDECESSOR
    saved['pilot_launcher_sha256'] = pilot.CLEANUP_PILOT_PREDECESSOR
    if change in ('trainer', 'launcher', 'config'):
        field = {'trainer': 'trainer_sha256', 'launcher': 'pilot_launcher_sha256', 'config': 'config_sha256'}[change]
        saved[field] = 'unknown'
    elif change == 'metrics':
        saved['metric_source_sha256']['segmentation_metrics.py'] = 'unknown'
    elif change == 'criteria':
        saved['criteria']['minimum_small_recall_gain'] = .9
    elif change == 'commands':
        saved['commands'][0].append('--changed')
    path = root / 'pilot_protocol.json'
    original = json.dumps(saved)
    path.write_text(original)
    if change:
        with pytest.raises(ValueError, match='changed'):
            pilot.freeze_protocol(root, splits, require_existing=True)
        assert not (root / 'cleanup-protocol-migration.json').exists()
    else:
        assert pilot.freeze_protocol(root, splits, require_existing=True) == saved
        migration = json.loads((root / 'cleanup-protocol-migration.json').read_text())
        assert migration['original_protocol'] == saved
        assert migration['current_protocol'] == current
    assert path.read_text() == original


def test_combined_loss_float32_retry_restores_buffers_and_gradients():
    class Overflow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm3d(2)
            self.conv = torch.nn.Conv3d(2, 2, 1)
        def forward(self, inputs):
            result = self.conv(self.bn(inputs))
            if torch.is_autocast_enabled(inputs.device.type):
                self.bn.running_mean.fill_(float('nan'))
                return result * float('nan')
            return result
    class CheckedLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = build_loss(parse_args(['--fold', '0', '--loss', 'dice-ce']))
        def forward(self, logits, labels):
            assert logits.dtype == torch.float32
            assert not torch.is_autocast_enabled(logits.device.type)
            return self.loss(logits, labels)
    torch.manual_seed(6)
    model = Overflow().train()
    reference = deepcopy(model)
    inputs = torch.randn(1, 2, 4, 4, 4)
    labels = torch.zeros(1, 1, 4, 4, 4)
    expected = CheckedLoss()(reference(inputs), labels)
    expected.backward()
    actual, retried = training_loss(model, inputs, labels, CheckedLoss(), amp_enabled=True)
    actual.backward()
    assert retried and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    for left, right in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_default_pilot_generates_three_separate_matched_arms(tmp_path):
    commands = list(pilot.pilot_commands(tmp_path, tmp_path / 'splits.json', 'all', False, False))
    parsed = [parse_args(command[3:]) for command in commands]
    assert [(a.sampling, a.loss) for a in parsed] == [('voxel', 'dice'), ('lesion-balanced', 'dice'), ('lesion-balanced', 'dice-ce')]
    assert len({a.output for a in parsed}) == 3
    assert all(a.fold == '0' and a.patch == 128 and a.epochs == 500 and a.stop_after_epoch == 50 for a in parsed)
    assert all(a.sw_batch_size == 1 for a in parsed)


def test_arm_validation_only_ignores_the_intended_comparison_factor():
    env = {'packages': {'torch': 'test'}, 'script_sha256': implementation_hash()}
    configs = {}
    for arm, (sampling, loss) in pilot.ARMS.items():
        args = parse_args(['--fold', '0', '--patch', '128', '--sampling', sampling, '--loss', loss])
        configs[arm] = resolved_config(args, 'split', env, torch.device('cpu'), 1)[0]
    pilot.validate_arm_configs(configs)
    changed = deepcopy(configs)
    changed['lesion-balanced-dice-ce']['scientific']['lr'] *= 2
    with pytest.raises(ValueError, match='beyond the intended loss'):
        pilot.validate_arm_configs(changed)
    changed = deepcopy(configs)
    changed['lesion-balanced-dice-ce']['scientific']['loss_weights']['cross_entropy'] = 2
    with pytest.raises(ValueError, match='weights differ'):
        pilot.validate_arm_configs(changed)


def test_loss_screen_requires_fp_reduction_and_recall_dice_guards():
    def summary(recall, fp, dice):
        return {'pooled_detection': {'C1': {'size_strata': {'small_lt1ml': {'recall': recall}}}},
                'per_exam': {'FPV_ml': {'mean': fp}, 'DSC': {'mean': dice}}}
    control = summary(.3, 100, .5)
    assert pilot.loss_comparison(control, summary(.3, 85, .5))['advance_to_further_validation']
    assert not pilot.loss_comparison(control, summary(.3, 95, .5))['advance_to_further_validation']
    assert not pilot.loss_comparison(control, summary(.25, 85, .5))['advance_to_further_validation']
    assert not pilot.loss_comparison(control, summary(.3, 85, .45))['advance_to_further_validation']
    assert pilot.loss_comparison(control, summary(.28, 90, .48))['advance_to_further_validation']
    assert pilot.comparison(summary(.2, 100, .5), summary(.25, 110, .48))['advance_to_further_validation']


@pytest.mark.parametrize('artifact_problem', [None, 'incomplete', 'duplicate', 'reference'])
def test_three_arm_evaluation_routes_and_scores_each_comparison(tmp_path, monkeypatch, artifact_problem):
    env = {'packages': {'torch': 'test'}, 'script_sha256': implementation_hash()}
    for arm, (sampling, loss) in pilot.ARMS.items():
        folder = tmp_path / 'pilot' / arm / 'fold_0'
        folder.mkdir(parents=True)
        (folder / 'best.pt').touch()
        (folder / 'history.json').write_text(json.dumps([{'epoch': 50}]))
        args = parse_args(['--fold', '0', '--patch', '128', '--sampling', sampling, '--loss', loss])
        cfg = resolved_config(args, 'split', env, torch.device('cpu'), 1)[0]
        (folder.parent / 'run.json').write_text(json.dumps({'config': cfg}))
    splits = tmp_path / 'splits.json'
    splits.write_text('{}')
    launched = []
    def child(command, **kwargs):
        launched.append(command)
        output = Path(command[command.index('--output') + 1])
        output.mkdir()
        if command[2].endswith('predict_inner_validation.py'):
            assert '--execute' in command
            (output / 'evaluation_inputs.json').write_text('[]')
        else:
            arm = output.name.removesuffix('-metrics')
            recall, fp, dice = {'voxel': (.2, 100, .5), 'lesion-balanced': (.3, 100, .52),
                                'lesion-balanced-dice-ce': (.3, 85, .53)}[arm]
            summary = {'pooled_detection': {'C1': {'size_strata': {'small_lt1ml': {'recall': recall}}}},
                       'per_exam': {'FPV_ml': {'mean': fp}, 'DSC': {'mean': dice}}}
            (output / 'summary.json').write_text(json.dumps(summary))
            row = {'case_id': 'p/e', 'patient': 'p', 'GT_TMTV_ml': 1., 'GT_SUVmax': 4., 'gt_lesions': 10}
            rows = [row, row] if arm == 'lesion-balanced' and artifact_problem == 'duplicate' else [row]
            (output / 'metrics.json').write_text(json.dumps(rows))
            source = {'case_id': 'p/e', 'patient': 'p',
                      'pet': {'sha256': 'a' * 64}, 'reference': {'sha256': 'b' * 64}}
            status = 'complete'
            if arm == 'lesion-balanced':
                if artifact_problem == 'incomplete':
                    status = 'running'
                elif artifact_problem == 'reference':
                    source['reference']['sha256'] = 'c' * 64
            (output / 'manifest.json').write_text(json.dumps({'status': status, 'cases': [source]}))
    monkeypatch.setattr(pilot.subprocess, 'run', child)
    if artifact_problem:
        with pytest.raises(ValueError):
            pilot.evaluate_pilot(tmp_path / 'pilot', splits, tmp_path / 'review')
        assert not (tmp_path / 'review/comparison.json').exists()
        assert json.loads((tmp_path / 'review/manifest.json').read_text())['status'] == 'running'
        return
    pilot.evaluate_pilot(tmp_path / 'pilot', splits, tmp_path / 'review')
    result = json.loads((tmp_path / 'review/comparison.json').read_text())
    assert len(launched) == 6
    assert result['sampling_B_vs_A']['advance_to_further_validation']
    assert result['loss_C_vs_B']['advance_to_further_validation']
    assert result['loss_C_vs_B']['small_recall_delta'] == 0
    assert all('scripts/train.py' not in command[2] for command in launched)
