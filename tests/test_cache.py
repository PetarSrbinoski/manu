"""Cache must preserve random patches, geometry, data identity and storage limits."""
import pytest
np = pytest.importorskip('numpy')
nib = pytest.importorskip('nibabel')
torch = pytest.importorskip('torch')
pytest.importorskip('monai')
from scripts.training_data import TrainingDataset, build_transforms, discover_cases, export_prediction, prepare_case
from scripts.preprocessing_cache import PreprocessingCache


def make_case(root, name='p', offset=0):
    folder = root / name / 'visit'
    folder.mkdir(parents=True)
    rng = np.random.default_rng(4)
    mask = np.zeros((20, 22, 24), dtype=np.float32)
    mask[5:15, 5:15, 5:15] = 1
    pet = rng.random(mask.shape, dtype=np.float32) + mask * 7 + offset
    affine = np.diag([-2., 3., 2., 1.])
    affine[0, 3] = 42
    for name, array in zip(('CTres', 'SUV', 'SEG'), (pet * 40, pet, mask)):
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, folder / f'{name}.nii.gz')
    return folder


def test_disk_cache_matches_uncached_patches_and_restores_geometry_across_instances(tmp_path):
    root = tmp_path / 'data'
    make_case(root)
    case = discover_cases(root)[0]
    cache = PreprocessingCache(tmp_path / 'cache', max_gib=.01, min_free_gib=0)
    baseline = TrainingDataset([case], root, 64)
    cached = TrainingDataset([case], root, 64, disk_cache=cache)
    for seed in (3, 9, 17):
        baseline.transform.set_random_state(seed=seed)
        cached.transform.set_random_state(seed=seed)
        left, right = baseline[0], cached[0]
        torch.testing.assert_close(left['image'].as_tensor(), right['image'].as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(left['GT'].as_tensor(), right['GT'].as_tensor(), rtol=0, atol=0)
    assert len(list(cache.directory.glob('*/entry.json'))) == 1
    # New transform instances must be able to invert persisted metadata.
    transforms = build_transforms(concatenate=False)
    data = prepare_case(case, root, transforms, PreprocessingCache(cache.directory, .01, 0))
    logits = torch.cat((1 - data['GT'].float(), data['GT'].float()), dim=0)
    target = tmp_path / 'prediction.nii.gz'
    export_prediction(logits, data, transforms, root / case.id / 'SUV.nii.gz', target)
    original = nib.load(root / case.id / 'SEG.nii.gz')
    actual = nib.load(target)
    np.testing.assert_allclose(actual.affine, original.affine)
    assert actual.shape == original.shape
    np.testing.assert_array_equal(actual.get_fdata(), original.get_fdata())
    # No sample may mutate persisted arrays.
    again = prepare_case(case, root, build_transforms(concatenate=False), cache)
    torch.testing.assert_close(again['PT'], data['PT'])


def test_cache_invalidates_changed_sources_and_enforces_budget(tmp_path):
    root = tmp_path / 'data'
    folder = make_case(root)
    first = discover_cases(root)[0]
    cache = PreprocessingCache(tmp_path / 'cache', .01, 0)
    old = prepare_case(first, root, build_transforms(concatenate=False), cache)
    path = folder / 'SUV.nii.gz'
    image = nib.load(path)
    nib.save(nib.Nifti1Image(image.get_fdata() + 2, image.affine, image.header), path)
    new = prepare_case(discover_cases(root)[0], root, build_transforms(concatenate=False), cache)
    torch.testing.assert_close(new['PT'].as_tensor(), old['PT'].as_tensor() + 2, rtol=1e-5, atol=1e-5)
    assert len(list(cache.directory.glob('*/entry.json'))) == 2
    tiny = PreprocessingCache(tmp_path / 'tiny-cache', .000001, 0)
    data = prepare_case(discover_cases(root)[0], root, build_transforms(concatenate=False), tiny)
    assert data['PT'].numel() > 0
    assert not list(tiny.directory.glob('*/entry.json'))


def test_cache_evicts_old_entries_and_keeps_active_maps_readable(tmp_path):
    root = tmp_path / 'data'
    make_case(root, 'a')
    make_case(root, 'b', offset=2)
    first, second = discover_cases(root)
    cache = PreprocessingCache(tmp_path / 'cache', .01, 0)
    old = prepare_case(first, root, build_transforms(concatenate=False), cache)
    entry = next(cache.directory.glob('*/entry.json')).parent
    entry_bytes = sum(p.stat().st_size for p in entry.iterdir())
    # Enough for either entry alone, never both.
    bounded = PreprocessingCache(cache.directory, entry_bytes * 1.3 / 1024**3, 0)
    new = prepare_case(second, root, build_transforms(concatenate=False), bounded)
    assert not entry.exists()
    assert len(list(cache.directory.glob('*/entry.json'))) == 1
    torch.testing.assert_close(new['PT'].as_tensor(), old['PT'].as_tensor() + 2, rtol=1e-5, atol=1e-5)
    # A free-space reserve larger than the filesystem forces an uncached fallback.
    no_space = PreprocessingCache(tmp_path / 'no-space', .01, 1e12)
    data = prepare_case(first, root, build_transforms(concatenate=False), no_space)
    assert data['PT'].numel() > 0
    assert not list(no_space.directory.glob('*/entry.json'))
