"""Component inventory for the 10 subset examinations: where are the false positives and false negatives?

For each examination and model (A, B, C), predicted lesions (18-connected, as in the evaluation) that share
no voxel with the reference are false-positive components; reference lesions that share no voxel with the
prediction are false-negative components. For every such component the script records volume, centroid
(voxel and mm), cranio-caudal position, SUVmax/SUVmean and mean CT HU. Candidate slices for the figures are
also stored. Nothing anatomical is inferred here; anatomy is assessed visually from the rendered images.

Read-only. Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/visual_analysis_and_report/analyze_components.py
"""
import json
from pathlib import Path
import sys

import cc3d
import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'evaluation' / 'a_vs_b_full_metrics'))
import evaluate_a_vs_b as ev  # noqa: E402

MANIFEST = PROJECT / 'experiments' / 'autopet3_subset10' / 'manifest.json'
C_PRED = PROJECT / 'experiments' / 'autopet3_subset10' / 'predictions'
CONNECTIVITY = ev.CONNECTIVITY


def load(case):
    paths = {'CT': case['ct'], 'SUV': case['pet'], 'GT': case['gt'], 'A': case['pred']['A'],
             'B': case['pred']['B'], 'C': C_PRED / f"{case['case_id']}.nii.gz"}
    images = {k: sitk.ReadImage(str(p)) for k, p in paths.items()}
    arrays = {k: sitk.GetArrayFromImage(v) for k, v in images.items()}          # (z, y, x) = (k, j, i)
    for k in ('GT', 'A', 'B', 'C'):
        arrays[k] = (arrays[k] > 0).astype(np.uint8)
    return images['SUV'], arrays


def components(mask, other, suv, ct, image, kind, model, case_id):
    labels, n = cc3d.connected_components(mask, connectivity=CONNECTIVITY, return_N=True)
    if n == 0:
        return []
    hit = np.bincount(labels[other > 0].ravel(), minlength=n + 1) > 0
    voxel_ml = float(np.prod(image.GetSpacing())) / 1000
    nz = labels.shape[0]
    stats = cc3d.statistics(labels)
    rows = []
    for label in np.flatnonzero(~hit[1:]) + 1:
        box = stats['bounding_boxes'][label]                    # slices in (k, j, i)
        inside = labels[box] == label
        values_suv, values_ct = suv[box][inside], ct[box][inside]
        k, j, i = stats['centroids'][label]
        x, y, z = image.TransformContinuousIndexToPhysicalPoint((float(i), float(j), float(k)))
        rows.append({'case_id': case_id, 'model': model, 'kind': kind, 'component': int(label),
                     'volume_ml': int(inside.sum()) * voxel_ml, 'voxels': int(inside.sum()),
                     'centroid_k': round(k, 1), 'centroid_j': round(j, 1), 'centroid_i': round(i, 1),
                     'x_mm': round(x, 1), 'y_mm': round(y, 1), 'z_mm': round(z, 1),
                     'cm_below_top_of_scan': round((nz - 1 - k) * image.GetSpacing()[2] / 10, 1),
                     'k_min': box[0].start, 'k_max': box[0].stop - 1,
                     'SUVmax': float(values_suv.max()), 'SUVmean': float(values_suv.mean()),
                     'CT_mean_HU': float(values_ct.mean())})
    return rows


def main():
    ids = sorted(c['case_id'] for c in json.loads(MANIFEST.read_text())['cases'])
    by_id = {c['case_id']: c for c in ev.load_cases()}
    all_rows, slices = [], {}
    for case_id in ids:
        image, a = load(by_id[case_id])
        suv, ct, gt = a['SUV'], a['CT'], a['GT']
        for model in ('A', 'B', 'C'):
            all_rows += components(a[model], gt, suv, ct, image, 'FP', model, case_id)
            all_rows += components(gt, a[model], suv, ct, image, 'FN', model, case_id)
        fp_union = np.zeros_like(gt)
        for model in ('A', 'B'):
            labels = cc3d.connected_components(a[model], connectivity=CONNECTIVITY)
            touched = np.unique(labels[gt > 0])
            fp_union |= ((labels > 0) & ~np.isin(labels, touched)).astype(np.uint8)
        slices[case_id] = {
            'axial_max_gt': int(gt.sum(axis=(1, 2)).argmax()),
            'axial_max_fp_AB': int(fp_union.sum(axis=(1, 2)).argmax()),
            'coronal_max_gt': int(gt.sum(axis=(0, 2)).argmax()),
            'n_slices': int(gt.shape[0]),
        }
        print(case_id[:16], slices[case_id], flush=True)
    table = pd.DataFrame(all_rows)
    table.to_csv(HERE / 'error_components.csv', index=False)
    (HERE / 'candidate_slices.json').write_text(json.dumps(slices, indent=2))
    summary = (table.groupby(['case_id', 'model', 'kind'])
               .agg(components=('volume_ml', 'size'), volume_ml=('volume_ml', 'sum'),
                    largest_ml=('volume_ml', 'max')).reset_index())
    summary['case_id'] = summary.case_id.str[6:16]
    print(summary.to_string())


if __name__ == '__main__':
    main()
