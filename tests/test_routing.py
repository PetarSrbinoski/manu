"""Prediction routing through real transforms/export and a deterministic tiny model."""
import pytest
np = pytest.importorskip('numpy')
nib = pytest.importorskip('nibabel')
torch = pytest.importorskip('torch')
pytest.importorskip('monai')
from train import Case, load_or_create_splits, fold_cases, predict_held_out, summarize_folds, atomic_json


def test_all_folds_export_each_exam_once_and_never_training_patients(tmp_path):
    root, output = tmp_path / 'data', tmp_path / 'run'
    cases = [Case(f'p{i}', 'v1') for i in range(10)] + [Case('p0', 'v2')]
    for index, case in enumerate(cases):
        folder = root / case.id
        folder.mkdir(parents=True)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        for name, array in zip(('CTres', 'SUV', 'SEG'), (np.ones_like(mask) * 100, mask * 7, mask)):
            image = nib.Nifti1Image(array, np.diag([2., 2., 2., 1.]))
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / f'{name}.nii.gz')
    manifest = load_or_create_splits(cases, output / 'splits.json')

    class SUVModel(torch.nn.Module):
        def forward(self, x):
            return torch.cat((3.5 - x[:, 1:2], x[:, 1:2] - 3.5), dim=1)

    model = SUVModel()
    for fold in range(5):
        folder = output / f'fold_{fold}'
        results = predict_held_out(model, cases, manifest, fold, root, folder, torch.device('cpu'), 64, 1)
        roles = fold_cases(cases, manifest, fold)
        assert {r['case'] for r in results} == {c.id for c in roles['held_out']}
        assert all(r['foreground_dice_native'] == 1 for r in results)
        atomic_json(folder / 'complete.json', {'config': {}, 'split_id': manifest['identity']})
    summarize_folds(output, cases, manifest, {})
    assert len(list(output.glob('fold_*/predictions/*/*/SEG_pred.nii.gz'))) == 11
    next(output.glob('fold_*/predictions/*/*/SEG_pred.nii.gz')).unlink()
    with pytest.raises(ValueError, match='Missing or extra'):
        summarize_folds(output, cases, manifest, {})
