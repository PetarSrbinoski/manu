"""Shared loading and display helpers for the visual analysis (read-only).

Geometry: all images were verified to share the SUV grid (verify_geometry.py), orientation LAS for every
image: array index i (x) runs patient right -> left, j (y) runs posterior -> anterior, k (z) runs
inferior -> superior. SimpleITK arrays are (k, j, i). Display convention (radiological):
  axial    arr[k]          -> flip rows so anterior is at the top; patient right on the viewer's left
  coronal  arr[:, j, :]    -> flip rows so superior is at the top; patient right on the viewer's left
  MIP      max over j      -> coronal maximum-intensity projection, same orientation as coronal
"""
from pathlib import Path
import json
import sys

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'evaluation' / 'a_vs_b_full_metrics'))
import evaluate_a_vs_b as ev  # noqa: E402  (unmodified; file locations only)

MANIFEST = PROJECT / 'experiments' / 'autopet3_subset10' / 'manifest.json'
C_PRED = PROJECT / 'experiments' / 'autopet3_subset10' / 'predictions'
EXPECTED_DIRECTION = np.diag([1.0, -1.0, 1.0]).ravel()

# Error-class colours (categorical slots validated all-pairs: aqua, orange, violet); reference = outline.
TP, FP, FN = '#1baf7a', '#eb6834', '#6250d6'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
REF_LINE = '#f5f5f2'   # reference outline on dark CT
REF_LINE_PET = '#0b0b0b'   # reference outline on inverted PET
MODEL_NAMES = {'A': 'A: Microsoft 10 ep', 'B': 'B: testTraining', 'C': 'C: autoPET III*'}


def subset_cases():
    ids = sorted(c['case_id'] for c in json.loads(MANIFEST.read_text())['cases'])
    by_id = {c['case_id']: c for c in ev.load_cases()}
    return [by_id[i] for i in ids]


def load(case):
    paths = {'CT': case['ct'], 'SUV': case['pet'], 'GT': case['gt'], 'A': case['pred']['A'],
             'B': case['pred']['B'], 'C': C_PRED / f"{case['case_id']}.nii.gz"}
    images = {k: sitk.ReadImage(str(p)) for k, p in paths.items()}
    ref = images['SUV']
    for name, img in images.items():   # re-check at render time: full geometry, not only the shape
        if (img.GetSize() != ref.GetSize() or not np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-3)
                or not np.allclose(img.GetOrigin(), ref.GetOrigin(), atol=1e-2)
                or not np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-3)
                or not np.allclose(img.GetDirection(), EXPECTED_DIRECTION, atol=1e-3)):
            raise ValueError(f'{name} of {case["case_id"]} is not on the expected LAS SUV grid')
    arrays = {k: sitk.GetArrayFromImage(v) for k, v in images.items()}
    for k in ('GT', 'A', 'B', 'C'):
        arrays[k] = arrays[k] > 0
    return ref, arrays


def axial(vol, k):
    return vol[k][::-1, :]


def coronal(vol, j):
    return vol[:, j, :][::-1, :]


def mip(vol):
    return vol.max(axis=1)[::-1, :]


def error_rgba(gt, pred):
    """Voxel-level TP/FP/FN as an RGBA overlay."""
    from matplotlib.colors import to_rgba
    out = np.zeros(gt.shape + (4,))
    for mask, colour in ((gt & pred, TP), (pred & ~gt, FP), (gt & ~pred, FN)):
        out[mask] = to_rgba(colour, 0.85)
    return out


def ct_window(ct):
    return np.clip((ct + 160) / 400, 0, 1)    # soft-tissue window, W 400 / L 40 HU


def suv_display(suv, vmax=10):
    return np.clip(suv / vmax, 0, 1)
