"""CLI-to-checkpoint integration with the actual U-Net on small synthetic volumes."""
import json
import pytest
np = pytest.importorskip('numpy')
nib = pytest.importorskip('nibabel')
torch = pytest.importorskip('torch')
pytest.importorskip('monai')
from train import main


def test_smoke_can_stop_at_epoch_boundary_and_resume_without_overwrite(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'run'
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
    args = ['--smoke', '--device', 'cpu', '--threads', '2', '--epochs', '2', '--smoke-steps', '1',
            '--val-interval', '1', '--data-root', str(root), '--output', str(output),
            '--expected-exams', '10', '--expected-patients', '10']
    main(args + ['--stop-after-epoch', '1'])
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
    main(args + ['--resume'])
    history = json.loads((folder / 'history.json').read_text())
    assert [r['epoch'] for r in history] == [1, 2]
    assert all(np.isfinite(r['loss_including_background']) for r in history)
    assert (folder / 'complete.json').exists()
    assert sorted(p.name for p in folder.glob('*.pt')) == ['best.pt', 'last.pt']
    assert not (folder / 'predictions').exists()  # smoke never exposes outer outcomes
    main(args + ['--resume'])  # completed fold is safely skipped
    with pytest.raises(ValueError, match='incompatible'):
        main(args + ['--resume', '--patch', '96'])
