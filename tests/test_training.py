"""CLI-to-checkpoint integration with the actual U-Net on small synthetic volumes."""
import json
import pytest
np = pytest.importorskip('numpy')
nib = pytest.importorskip('nibabel')
torch = pytest.importorskip('torch')
pytest.importorskip('monai')
from scripts.train import main


def make_data(root):
    for i in range(10):
        folder = root / f'p{i}' / 'visit'
        folder.mkdir(parents=True)
        mask = np.zeros((12, 12, 12), dtype=np.float32)
        mask[3:9, 3:9, 3:9] = 1
        for name, data in zip(('CTres', 'SUV', 'SEG'),
                              (np.full(mask.shape, 100, dtype=np.float32), mask * 7 + (i + 1) / 100, mask)):
            image = nib.Nifti1Image(data, np.diag([2., 2., 2., 1.]))
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / f'{name}.nii.gz')


@pytest.mark.parametrize('disk_cache', [False, True])
def test_smoke_can_stop_at_epoch_boundary_and_resume_without_overwrite(tmp_path, capsys, disk_cache):
    root, output = tmp_path / 'data', tmp_path / 'run'
    make_data(root)
    args = ['--smoke', '--device', 'cpu', '--threads', '2', '--epochs', '2', '--smoke-steps', '1',
            '--val-interval', '1', '--data-root', str(root), '--output', str(output),
            '--expected-exams', '10', '--expected-patients', '10']
    args += (['--disk-cache', str(tmp_path / 'cache'), '--disk-cache-gib', '.01', '--min-free-disk-gib', '0']
             if disk_cache else ['--no-disk-cache'])
    main(args + ['--stop-after-epoch', '1'])
    console = capsys.readouterr().out
    logs = list((output / 'logs').glob('*.log'))
    assert len(logs) == 1
    log = logs[0].read_text()
    for event in ('Data audit', 'Training patch ready', 'training batches | 1/1',
                  'Sliding-window inference', 'Inference windows | 1/1',
                  'mean foreground Dice=', 'Epoch 1 committed', 'Stopped at requested epoch boundary'):
        assert event in console and event in log
    folder = output / 'fold_0'
    assert (folder / 'last.pt').exists()
    assert not (folder / 'complete.json').exists()
    # Simulate best.pt being published from an epoch that never committed.
    selected = torch.load(folder / 'best.pt', weights_only=False)
    selected['epoch'] = 999
    torch.save(selected, folder / 'best.pt')
    main(args + ['--resume', '--stop-after-epoch', '1'])
    repaired = torch.load(folder / 'best.pt', weights_only=False)
    assert repaired['epoch'] == 1
    assert repaired['config']['fold'] == 0
    with pytest.raises(ValueError, match='exists'):
        main(args)
    main(args + ['--resume', '--log-every', '1'])  # logging frequency is not a scientific setting
    history = json.loads((folder / 'history.json').read_text())
    assert [r['epoch'] for r in history] == [1, 2]
    assert all(np.isfinite(r['loss_including_background']) for r in history)
    assert all(r['data_wait_seconds'] >= 0 and r['training_step_seconds'] > 0 for r in history)
    if disk_cache:
        assert list((tmp_path / 'cache').glob('*/entry.json'))
    assert (folder / 'complete.json').exists()
    assert sorted(p.name for p in folder.glob('*.pt')) == ['best.pt', 'last.pt']
    assert not (folder / 'predictions').exists()  # smoke never exposes outer outcomes
    main(args + ['--resume'])  # completed fold is safely skipped
    with pytest.raises(ValueError, match='incompatible'):
        main(args + ['--resume', '--patch', '96'])
    all_logs = [path.read_text() for path in (output / 'logs').glob('*.log')]
    assert len(all_logs) == 6  # each launch gets its own log, including rejected launches
    assert any('Run failed' in log and 'Traceback' in log and 'incompatible' in log for log in all_logs)
    assert any('Resumed fold 0 after epoch 1' in log for log in all_logs)
    assert any('already completed; verified outputs and skipped training' in log for log in all_logs)


@pytest.mark.parametrize('amp', [True, False])
@pytest.mark.parametrize('accumulate', [1, 2])
def test_gradient_overflow_recovers_only_with_grad_scaler(tmp_path, monkeypatch, capsys, amp, accumulate):
    from scripts import train
    root, output = tmp_path / 'data', tmp_path / 'run'
    make_data(root)
    model = torch.nn.Conv3d(2, 2, kernel_size=1)
    calls = 0

    def overflow_first_backward(gradient):
        nonlocal calls
        calls += 1
        return torch.full_like(gradient, float('inf')) if calls == 1 else gradient

    model.weight.register_hook(overflow_first_backward)
    monkeypatch.setattr(train, 'build_model', lambda: model)
    if amp:
        # Exercise real GradScaler overflow detection on CPU so CI needs no GPU.
        scaler_type = torch.amp.GradScaler
        monkeypatch.setattr(torch.amp, 'GradScaler', lambda *a, **kw: scaler_type('cpu', enabled=True))
    args = ['--smoke', '--device', 'cpu', '--threads', '2', '--smoke-steps', '2',
            '--accumulate', str(accumulate),
            '--data-root', str(root), '--output', str(output), '--no-disk-cache',
            '--expected-exams', '10', '--expected-patients', '10']
    if not amp:
        with pytest.raises(RuntimeError, match='Nonfinite gradients'):
            main(args)
        assert not (output / 'fold_0' / 'last.pt').exists()
        return
    main(args)
    state = torch.load(output / 'fold_0' / 'last.pt', weights_only=False)
    assert state['history'][0]['optimizer_updates'] == 1
    assert state['history'][0]['amp_skipped_updates'] == 1
    assert state['scaler']['scale'] == 32768
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert all(int(v['step']) == 1 for v in state['optimizer']['state'].values())
    assert 'AMP overflow' in capsys.readouterr().out


@pytest.mark.parametrize('upgrade', ['gradient', 'forward', 'pilot', 'refactor', 'scripts', 'paths'])
def test_known_source_migrations_preserve_checkpoints_and_reject_other_changes(tmp_path, monkeypatch, capsys, upgrade):
    from scripts import train
    root, output = tmp_path / 'data', tmp_path / 'run'
    make_data(root)
    monkeypatch.setattr(train, 'build_model', lambda: torch.nn.Conv3d(2, 2, kernel_size=1))
    # Keep the original selected model so final export must accept its old identity.
    monkeypatch.setattr(train, 'foreground_dice', lambda *args: .5)
    args = ['--smoke', '--device', 'cpu', '--threads', '2', '--epochs', '2', '--smoke-steps', '1',
            '--val-interval', '1', '--data-root', str(root), '--output', str(output),
            '--no-disk-cache', '--expected-exams', '10', '--expected-patients', '10']
    main(args + ['--stop-after-epoch', '1'])
    path = output / 'run.json'
    record = json.loads(path.read_text())
    predecessor = {'gradient': train.AMP_OVERFLOW_PREDECESSOR, 'forward': train.AMP_FORWARD_PREDECESSOR,
                   'pilot': train.PILOT_PREDECESSOR, 'refactor': train.REFACTOR_PREDECESSOR, 'scripts': train.SCRIPTS_PREDECESSOR, 'paths': train.PATHS_PREDECESSOR}[upgrade]
    previous = record['config'] = {**record['config'], 'script_sha256': predecessor}
    if upgrade == 'pilot':
        del previous['scientific']['early_stopping']
    record['environment']['script_sha256'] = predecessor
    path.write_text(json.dumps(record))
    for name in ('best.pt', 'last.pt'):
        checkpoint = output / 'fold_0' / name
        state = torch.load(checkpoint, weights_only=False)
        state['config']['script_sha256'] = predecessor
        if upgrade == 'pilot':
            state['config']['scientific'].pop('early_stopping', None)
        if state.get('selected_checkpoint'):
            state['selected_checkpoint']['config']['script_sha256'] = predecessor
            if upgrade == 'pilot':
                state['selected_checkpoint']['config']['scientific'].pop('early_stopping', None)
        torch.save(state, checkpoint)
    main(args + ['--resume'])
    migration_name = {'gradient': 'amp-resume-migration.json', 'forward': 'amp-forward-resume-migration.json',
                      'pilot': 'pilot-resume-migration.json', 'refactor': 'refactor-resume-migration.json', 'scripts': 'scripts-resume-migration.json', 'paths': 'paths-resume-migration.json'}[upgrade]
    migration = json.loads((output / migration_name).read_text())
    assert migration['previous_run']['config'] == previous
    current = json.loads(path.read_text())['config']
    assert current['script_sha256'] == train.implementation_hash()
    expected = {**previous, 'script_sha256': current['script_sha256']}
    if upgrade == 'pilot':
        expected['scientific'] = {**expected['scientific'], 'early_stopping': train.PILOT_STOPPING_POLICY}
    assert expected == current
    state = torch.load(output / 'fold_0' / 'last.pt', weights_only=False)
    assert state['epoch'] == 2
    assert state['selected_checkpoint']['epoch'] == 1
    message = {'pilot': 'Applying verified pilot upgrade', 'refactor': 'Applying verified training-script refactor',
               'scripts': 'Applying verified scripts-directory move', 'paths': 'Applying verified project-path fix'}.get(
        upgrade, 'Applying verified AMP-overflow resume fix')
    assert message in capsys.readouterr().out
    if upgrade in ('refactor', 'scripts', 'paths'):
        missing_policy = {**previous, 'scientific': {k: v for k, v in previous['scientific'].items() if k != 'early_stopping'}}
        assert not train.compatible_config(missing_policy, current)
    assert not train.compatible_config({**previous, 'script_sha256': 'unknown-source'}, current)
    assert not train.compatible_config(previous, {**current, 'threads': 3})
    assert not train.compatible_config(previous, {**current, 'scientific': {**current['scientific'], 'lr': .5}})
    assert not train.compatible_config(previous, {**current, 'split_id': 'different'})
    assert not train.compatible_config({**previous, 'fold': 0}, {**current, 'fold': 1})
    assert not train.compatible_config(previous, {**current, 'scientific': {**current['scientific'],
                                          'early_stopping': {**train.PILOT_STOPPING_POLICY, 'patience': 1}}})


def test_forward_overflow_recovery_is_checkpointed_and_logged(tmp_path, monkeypatch, capsys):
    from scripts import train
    root, output = tmp_path / 'data', tmp_path / 'run'
    make_data(root)

    class OverflowModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm3d(2)
            self.conv = torch.nn.Conv3d(2, 2, 1)
            self.injected = False

        def forward(self, inputs):
            result = self.conv(self.norm(inputs))
            if self.training and torch.is_autocast_enabled('cpu') and not self.injected:
                self.injected = True
                self.norm.running_var.fill_(float('inf'))
                return result * float('nan')
            return result

    monkeypatch.setattr(train, 'build_model', OverflowModel)
    actual_training_loss = train.training_loss
    # CPU autocast exercises the same retry path without requiring a CI GPU.
    monkeypatch.setattr(train, 'training_loss', lambda model, inputs, labels, loss_function, amp_enabled:
                        actual_training_loss(model, inputs, labels, loss_function, amp_enabled=True))
    main(['--smoke', '--device', 'cpu', '--threads', '2', '--smoke-steps', '2',
          '--data-root', str(root), '--output', str(output), '--no-disk-cache',
          '--expected-exams', '10', '--expected-patients', '10'])
    state = torch.load(output / 'fold_0' / 'last.pt', weights_only=False)
    assert state['history'][0]['amp_forward_retries'] == 1
    assert state['history'][0]['optimizer_updates'] == 2
    assert state['history'][0]['amp_skipped_updates'] == 0
    assert all(torch.isfinite(t).all() for t in state['model'].values())
    assert int(state['model']['norm.num_batches_tracked']) == 2
    assert 'Float32 forward recovery succeeded' in capsys.readouterr().out
