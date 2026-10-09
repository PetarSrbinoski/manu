"""Synthetic geometry and channel-order checks, not real-data validation."""
import pytest
np = pytest.importorskip('numpy')

nib = pytest.importorskip('nibabel')
torch = pytest.importorskip('torch')
pytest.importorskip('monai')
from scripts.train import Case, build_transforms, predict_volume, export_prediction


def make_case(tmp_path, lesion_outside=False):
    case = Case('p', 'visit')
    folder = tmp_path / case.id
    folder.mkdir(parents=True)
    shape = (24, 28, 20)
    affine = np.array([[-2, 0, 0, 50], [0, 3, 0, -30], [0, 0, 2, 10], [0, 0, 0, 1.]])
    ct = np.full(shape, -1000, dtype=np.float32)
    ct[3:20, 4:24, 2:18] = 100
    mask = np.zeros(shape, dtype=np.uint8)
    mask[8:14, 10:18, 6:14] = 1
    if lesion_outside:
        mask[0, 0, 0] = 1
    pet = mask.astype(np.float32) * 7
    for filename, array in zip(('CTres.nii.gz', 'SUV.nii.gz', 'SEG.nii.gz'), (ct, pet, mask)):
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, folder / filename)
    return case, mask, affine


def test_prediction_uses_input_history_and_restores_original_geometry(tmp_path):
    case, mask, affine = make_case(tmp_path)
    transforms = build_transforms(labels=False)
    data = transforms(case.inputs(tmp_path, labels=False))
    assert data['CT'].min() >= 0 and data['CT'].max() <= 1
    assert data['PT'].max() == 7  # quantitative SUV is not normalized
    assert data['image'].shape[0] == 2

    class SUVModel(torch.nn.Module):
        def forward(self, x):
            return torch.cat((3.5 - x[:, 1:2], x[:, 1:2] - 3.5), dim=1)

    logits = predict_volume(SUVModel(), data['image'], torch.device('cpu'), 64, 1)
    assert logits.device.type == 'cpu'
    target = tmp_path / 'prediction.nii.gz'
    export_prediction(logits, data, transforms, tmp_path / case.id / 'SUV.nii.gz', target)
    result = nib.load(target)
    assert result.shape == mask.shape
    np.testing.assert_allclose(result.affine, affine)
    binary = result.get_fdata()
    assert set(np.unique(binary)) <= {0, 1}
    assert (binary * mask).sum() / mask.sum() > .95
    assert binary[0, 0, 0] == 0


def test_full_field_of_view_preserves_lesions_outside_ct_foreground(tmp_path):
    case, mask, _ = make_case(tmp_path, lesion_outside=True)
    transforms = build_transforms()
    data = transforms(case.inputs(tmp_path))
    # A corner lesion outside CT>0 must survive preprocessing and export.
    logits = torch.cat((1 - data['GT'], data['GT']), dim=0)
    target = tmp_path / 'prediction.nii.gz'
    export_prediction(logits, data, transforms, tmp_path / case.id / 'SUV.nii.gz', target)
    restored = nib.load(target).get_fdata()
    assert mask[0, 0, 0] == restored[0, 0, 0] == 1


def test_data_audit_rejects_duplicate_images_and_misaligned_masks(tmp_path):
    from shutil import copytree
    from scripts.train import discover_cases
    case, _, _ = make_case(tmp_path)
    assert len(discover_cases(tmp_path)) == 1
    duplicate = tmp_path / 'another-patient' / 'visit'
    copytree(tmp_path / case.id, duplicate)
    with pytest.raises(ValueError, match='duplicate'):
        discover_cases(tmp_path)
    # Remove the duplicate by changing its quantitative PET values.
    path = duplicate / 'SUV.nii.gz'
    pet = nib.load(path)
    changed = nib.Nifti1Image(pet.get_fdata() + 1, pet.affine, pet.header)
    nib.save(changed, path)
    label_path = duplicate / 'SEG.nii.gz'
    label = nib.load(label_path)
    affine = label.affine.copy()
    affine[0, 3] += 10
    nib.save(nib.Nifti1Image(label.get_fdata(), affine, label.header), label_path)
    with pytest.raises(ValueError, match='Geometry'):
        discover_cases(tmp_path)
