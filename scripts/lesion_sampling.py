"""Uniform reference-component foreground centres with the original background policy."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from monai.transforms import Randomizable, RandCropByPosNegLabeld


@dataclass(frozen=True)
class LesionIndex:
    voxels: np.ndarray
    offsets: np.ndarray

    @property
    def count(self):
        return len(self.offsets) - 1

    @property
    def nbytes(self):
        return self.voxels.nbytes + self.offsets.nbytes


def lesion_index(label):
    import cc3d
    array = np.asarray(label)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError('Expected a finite 3D reference label')
    labelled, count = cc3d.connected_components(np.ascontiguousarray(array > 0, dtype=np.uint8),
                                               connectivity=18, return_N=True)
    if not count:
        raise ValueError('Lesion-balanced sampling requires a positive reference')
    flat = labelled.ravel(order='C')
    fg = np.flatnonzero(flat)
    groups = flat[fg]
    ordered = fg[np.argsort(groups, kind='stable')]
    if flat.size < 2**31:
        ordered = ordered.astype(np.int32)
    sizes = np.bincount(groups, minlength=count + 1)[1:]
    return LesionIndex(ordered, np.r_[0, np.cumsum(sizes)])


class LesionIndexCache:
    """Byte-bounded, per-loader-process derived indices; never stores whole volumes."""
    def __init__(self, limit_bytes):
        self.limit = int(limit_bytes)
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key, label):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        index = lesion_index(label)
        if index.nbytes <= self.limit:
            while self.entries and self.bytes + index.nbytes > self.limit:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old.nbytes
            self.entries[key] = index
            self.bytes += index.nbytes
        return index


class LesionBalancedCropd(Randomizable):
    """Pick a component uniformly, then one of its voxels uniformly.

    The original MONAI crop still chooses foreground/background at the configured
    2:1 ratio, corrects centres at boundaries, and performs paired spatial crops.
    Negative candidates remain GT=0 and PET above the existing threshold.
    """
    def __init__(self, **kwargs):
        self._crop = RandCropByPosNegLabeld(**kwargs)
        self._fg_key, self._bg_key = kwargs.get('fg_indices_key'), kwargs.get('bg_indices_key')
        if not self._fg_key or not self._bg_key or kwargs.get('num_samples', 1) != 1:
            raise ValueError('Explicit foreground/background index keys and one sample required')
        self._label_key, self._image_key = kwargs['label_key'], kwargs.get('image_key')
        self._threshold = kwargs.get('image_threshold', 0)
        self.foreground_candidate = None

    @property
    def cropper(self):
        return self._crop.cropper

    def set_random_state(self, seed=None, state=None):
        super().set_random_state(seed=seed, state=state)
        self._crop.set_random_state(seed=int(self.R.randint(0, 2**32 - 1)))
        return self

    def __call__(self, data):
        d = dict(data)
        index = d.pop('_lesion_index', None)
        if index is None:
            index = lesion_index(d['GT'])
        component = int(self.R.randint(index.count))
        begin, end = index.offsets[component:component + 2]
        centre = int(index.voxels[int(self.R.randint(int(begin), int(end)))])
        self.foreground_candidate = centre
        # One foreground candidate changes only the positive distribution.
        # MONAI regenerates BOTH lists if either one is absent. Supply background
        # explicitly so an uncached volume cannot discard the selected component.
        if d.get(self._bg_key) is None:
            from monai.transforms.utils import map_binary_to_indices
            _, background = map_binary_to_indices(d[self._label_key], d.get(self._image_key), self._threshold)
            d[self._bg_key] = background
        d[self._fg_key] = np.array([centre], dtype=np.int64)
        return self._crop(d)
