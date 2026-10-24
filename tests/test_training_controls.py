"""Exact binary decisions and validation-only, resumable stopping behavior."""
import json
import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
from scripts import train
from scripts.training_data import binary_prediction, foreground_dice
from scripts.training_runtime import early_stopping_status


def test_binary_prediction_matches_argmax_including_ties_and_noncontiguous_input():
    logits = torch.randn(2, 12, 14, 16).transpose(1, 3)
    logits[:, 0] = 3  # Equal class scores must select background.
    assert torch.equal(binary_prediction(logits), logits.argmax(0).bool())
    with pytest.raises(ValueError, match='two'):
        binary_prediction(torch.zeros(3, 2, 2, 2))
    expected = logits.argmax(0).numpy()
    for pred, target in [(expected, expected), (torch.from_numpy(expected), torch.from_numpy(expected))]:
        assert foreground_dice(pred, target) == 1
        assert foreground_dice(pred * 0, target * 0) == 1
        assert foreground_dice(pred * 0, target) == 0


def test_stopping_counts_only_validation_and_accumulates_small_improvements():
    history = []
    policy = dict(patience=2, min_delta=.01, min_epochs=6)
    for epoch, score in enumerate([None, .5, None, .506, None, .512, None, .51], 1):
        history.append(dict(epoch=epoch, inner_foreground_dice_2mm=score))
        assert not early_stopping_status(history, **policy)['should_stop']
    assert early_stopping_status(history, **policy)['bad_checks'] == 1
    history.append(dict(epoch=9, inner_foreground_dice_2mm=None))
    assert early_stopping_status(history, **policy)['bad_checks'] == 1
    history.append(dict(epoch=10, inner_foreground_dice_2mm=.49))
    assert early_stopping_status(history, **policy)['should_stop']
    assert not early_stopping_status(history, patience=0, min_delta=.01, min_epochs=6)['should_stop']
    assert not early_stopping_status(history, patience=2, min_delta=.01, min_epochs=12)['should_stop']


def test_early_stop_resume_exports_best_and_survives_interrupted_export(tmp_path, monkeypatch):
    from test_training import make_data
    root, output = tmp_path / 'data', tmp_path / 'run'
    make_data(root)
    monkeypatch.setattr(train, 'build_model', lambda: torch.nn.Conv3d(2, 2, 1))
    monkeypatch.setattr(train, 'foreground_dice', lambda *args: .5)
    args = ['--fold', '0', '--device', 'cpu', '--threads', '2', '--epochs', '12',
            '--val-interval', '2', '--early-stopping-patience', '2',
            '--early-stopping-min-epochs', '6', '--early-stopping-min-delta', '.001',
            '--data-root', str(root), '--output', str(output), '--no-disk-cache',
            '--expected-exams', '10', '--expected-patients', '10']
    train.main(args + ['--stop-after-epoch', '4'])
    assert not (output / 'fold_0/complete.json').exists()
    export = train.predict_held_out
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupted export')
    monkeypatch.setattr(train, 'predict_held_out', interrupted)
    with pytest.raises(RuntimeError, match='interrupted export'):
        train.main(args + ['--resume'])
    folder = output / 'fold_0'
    last = torch.load(folder / 'last.pt', weights_only=False)
    assert last['epoch'] == 6
    assert last['history'][-1]['early_stopping']['bad_checks'] == 2
    assert last['selected_checkpoint']['epoch'] == 2
    monkeypatch.setattr(train, 'predict_held_out', export)
    def unexpected(*args, **kwargs):
        raise AssertionError('Completed early stop must not train again')
    monkeypatch.setattr(train, 'training_loss', unexpected)
    train.main(args + ['--resume'])
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['stopped_early'] and complete['trained_epochs'] == 6
    assert complete['best_epoch'] == 2
    assert len(list((folder / 'predictions').glob('*/*/SEG_pred.nii.gz'))) == 2
    train.main(args + ['--resume'])  # Verified completed folds are skipped.
