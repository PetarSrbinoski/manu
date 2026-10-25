"""Shared synthetic cohort fixture for training/resume integration tests."""
import nibabel as nib
import numpy as np


def make_data(root):
    for index in range(10):
        folder = root / f'patient{index:02d}' / 'visit'
        folder.mkdir(parents=True)
        rng = np.random.default_rng(index)
        label = np.zeros((18, 20, 22), dtype=np.float32)
        label[4:8, 5:9, 6:10] = 1
        label[12:14, 13:15, 14:16] = 1
        pet = rng.random(label.shape, dtype=np.float32) + label * 5 + index / 10
        ct = pet * 30
        for name, array in [('CTres', ct), ('SUV', pet), ('SEG', label)]:
            image = nib.Nifti1Image(array, np.diag([2., 2., 2., 1.]))
            image.header.set_xyzt_units('mm')
            nib.save(image, folder / f'{name}.nii.gz')
