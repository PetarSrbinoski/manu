"""Geometry check for the 10 subset examinations: CT, SUV, reference, A, B, C must share one voxel grid.

Compares size, spacing, origin and the full direction matrix of every image with the SUV image
(not only the array shape), and reports the patient orientation implied by the direction matrix.
Read-only. Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/visual_analysis_and_report/verify_geometry.py
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'evaluation' / 'a_vs_b_full_metrics'))
import evaluate_a_vs_b as ev  # noqa: E402  (unmodified; used only for its file locations)

MANIFEST = PROJECT / 'experiments' / 'autopet3_subset10' / 'manifest.json'
C_PRED = PROJECT / 'experiments' / 'autopet3_subset10' / 'predictions'


def sources(case):
    return {'CT': case['ct'], 'SUV': case['pet'], 'GT': case['gt'], 'A': case['pred']['A'],
            'B': case['pred']['B'], 'C': C_PRED / f"{case['case_id']}.nii.gz"}


def main():
    ids = sorted(c['case_id'] for c in json.loads(MANIFEST.read_text())['cases'])
    by_id = {c['case_id']: c for c in ev.load_cases()}
    rows, ok_all = [], True
    reader = sitk.ImageFileReader()
    for i in ids:
        ref = None
        for name, path in sources(by_id[i]).items():
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
            info = {'size': reader.GetSize(), 'spacing': np.array(reader.GetSpacing()),
                    'origin': np.array(reader.GetOrigin()), 'direction': np.array(reader.GetDirection())}
            if name == 'SUV':
                ref = info
            rows.append((i, name, info))
        for case_id, name, info in [r for r in rows if r[0] == i]:
            same = (info['size'] == ref['size'] and np.allclose(info['spacing'], ref['spacing'], atol=1e-3)
                    and np.allclose(info['origin'], ref['origin'], atol=1e-2)
                    and np.allclose(info['direction'], ref['direction'], atol=1e-3))
            ok_all &= same
            info['same_grid_as_SUV'] = same
    with (HERE / 'geometry_verification.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case_id', 'image', 'size_xyz', 'spacing_xyz_mm', 'origin_xyz_mm', 'direction', 'orientation', 'same_grid_as_SUV'])
        for case_id, name, info in rows:
            d = info['direction'].reshape(3, 3)
            orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(info['direction'].tolist())
            w.writerow([case_id, name, info['size'], np.round(info['spacing'], 4).tolist(), np.round(info['origin'], 2).tolist(),
                        np.round(d, 4).ravel().tolist(), orient, info['same_grid_as_SUV']])
    orients = {sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(info['direction'].tolist()) for _, _, info in rows}
    print(f'{len(rows)} images, all on the SUV grid: {ok_all}; orientations: {orients}')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
