import pytest
import config as config
from scripts.training_options import parse_args, scientific_recipe


def test_config_defaults_and_explicit_cli_override(monkeypatch):
    monkeypatch.setattr(config, 'PATCH_SIZE', 96)
    monkeypatch.setattr(config, 'EPOCHS', 120)
    monkeypatch.setattr(config, 'CPU_THREADS', 15)
    monkeypatch.setattr(config, 'LEARNING_RATE', .0001)
    args = parse_args(['--fold', 'all'])
    assert (args.patch, args.epochs, args.threads, args.lr) == (96, 120, 15, .0001)
    args = parse_args(['--fold', '0', '--patch', '64', '--epochs', '20', '--threads', '4'])
    assert (args.patch, args.epochs, args.threads) == (64, 20, 4)
    assert parse_args([]) is None
    assert parse_args(['--smoke', '--output', 'runs/test']).epochs == 1
    monkeypatch.setattr(config, 'WEIGHT_DECAY', .01)
    assert scientific_recipe()['weight_decay'] == .01
    monkeypatch.setattr(config, 'INFERENCE_OVERLAP', 1)
    with pytest.raises(SystemExit):
        parse_args(['--fold', '0'])


@pytest.mark.parametrize('flag,value', [('--early-stopping-patience', '-1'),
    ('--early-stopping-min-epochs', '0'), ('--early-stopping-min-delta', 'nan'),
    ('--early-stopping-min-delta', '-.1'), ('--early-stopping-min-delta', '1')])
def test_invalid_stopping_policy_is_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse_args(['--fold', '0', flag, value])


@pytest.mark.parametrize('location', ['project', 'scripts', 'elsewhere'])
def test_paths_are_project_relative_from_any_launch_directory(tmp_path, monkeypatch, location):
    root = config.PROJECT_ROOT
    monkeypatch.chdir({'project': root, 'scripts': root / 'scripts', 'elsewhere': tmp_path}[location])
    args = parse_args(['--fold', '0'])
    assert args.data_root == root / 'data/autopet-v2-lymphoma'
    assert args.output == root / 'runs/full-training'
    assert args.disk_cache == root / 'runs/preprocessed-cache'
    assert args.splits is None

    args = parse_args(['--fold', '0', '--data-root', 'data/alternate', '--output', 'runs/check',
                       '--splits', 'runs/shared.json', '--disk-cache', str(tmp_path / 'cache')])
    assert args.data_root == root / 'data/alternate'
    assert args.output == root / 'runs/check'
    assert args.splits == root / 'runs/shared.json'
    assert args.disk_cache == tmp_path / 'cache'
    assert parse_args(['--fold', '0', '--no-disk-cache']).disk_cache is None
    assert parse_args(['--fold', '0', '--output', str(tmp_path / 'run')]).output == tmp_path / 'run'

    # Alternate spellings cannot bypass the smoke-output protection.
    for output in ['runs/full-training', './runs/../runs/full-training', str(config.OUTPUT_ROOT)]:
        with pytest.raises(SystemExit):
            parse_args(['--smoke', '--output', output])
