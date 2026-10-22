"""Supervisor-requested metrics for the 145 out-of-fold predictions of experiments A and B.

A = experiments/unet_10ep (Microsoft lymphoma-segmentation-dnn, 10 epochs, seed-42 patient folds)
B = /home/eva25/manu-runs/full-training (testTraining, budget 500 epochs, early stopping, seed-20261006 folds)

Both experiments are scored with the same code and the same definitions: those of Microsoft's
lymphoma-segmentation-dnn/metrics/metrics.py (commit 8112924), connectivity 18, IoU threshold 0.5,
arrays read with Microsoft's get_3darray_from_niftipath. Detection and FPV/FNV are computed from a
component contingency table instead of Microsoft's per-component full-volume loops (which take hours
per examination for A's predictions); the greedy matching order and tie-breaking are replicated
exactly. --validate compares this implementation with Microsoft's original functions on a sample.

Read-only with respect to all inputs: predictions, labels and images are only opened for reading.
Run inside WSL:
    ~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py
    ~/venvs/lymphoma_seg/bin/python evaluation/a_vs_b_full_metrics/evaluate_a_vs_b.py --validate 8
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import SimpleITK as sitk
import cc3d

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS_REPO = PROJECT / 'lymphoma-segmentation-dnn'
sys.path.insert(0, str(MS_REPO))
from metrics import metrics as ms  # noqa: E402  (Microsoft's unmodified module)

DATASET = Path('/home/eva25/data/lymphoma-baseline')   # byte-identical to B's source files (audit 26)
A_PRED = PROJECT / 'experiments/unet_10ep/work/results/predictions'
B_RUN = Path('/home/eva25/manu-runs/full-training')
CONNECTIVITY = 18
IOU_THRESHOLD = 0.5


def load_cases():
    cases = json.loads((DATASET / 'cases.json').read_text(encoding='utf-8'))
    b_folds = json.loads((B_RUN / 'splits.json').read_text())['patient_folds']
    out = []
    for c in sorted(cases, key=lambda c: c['case_id']):
        exam = c['case_id'][len(c['patient']) + 1:]
        out.append({
            'case_id': c['case_id'], 'patient': c['patient'],
            'ct': DATASET / 'imagesTr' / f"{c['case_id']}_0000.nii.gz",
            'pet': DATASET / 'imagesTr' / f"{c['case_id']}_0001.nii.gz",
            'gt': DATASET / 'labelsTr' / f"{c['case_id']}.nii.gz",
            'folds': {'A': c['fold'], 'B': b_folds[c['patient']]},
            'pred': {'A': A_PRED / f"fold{c['fold']}/unet/unet_fold{c['fold']}_randcrop192/{c['case_id']}.nii.gz",
                     'B': B_RUN / f"fold_{b_folds[c['patient']]}/predictions/{c['patient']}/{exam}/SEG_pred.nii.gz"},
        })
    if len(out) != 145 or len({c['patient'] for c in out}) != 144:
        sys.exit('Expected 145 examinations / 144 patients')
    return out


def read_binary(path, reference):
    """Microsoft's reader; checks geometry and that the mask is strictly 0/1."""
    image = sitk.ReadImage(str(path))
    if (image.GetSize() != reference.GetSize()
            or not np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=1e-3)
            or not np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=1e-2)
            or not np.allclose(image.GetDirection(), reference.GetDirection(), atol=1e-3)):
        raise ValueError(f'geometry differs from PET: {path}')
    array = ms.get_3darray_from_niftipath(str(path))
    values = np.unique(array)
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError(f'mask is not binary 0/1 ({values[:5]}): {path}')
    return array.astype(np.uint8)


def detection(gt, pred, pet):
    """Exact equivalent of Microsoft's FPV, FNV and calculate_patient_level_tp_fp_fn (3 criteria)."""
    gl, ng = cc3d.connected_components(gt, connectivity=CONNECTIVITY, return_N=True)
    pl, npd = cc3d.connected_components(pred, connectivity=CONNECTIVITY, return_N=True)
    gsize = np.bincount(gl.ravel(), minlength=ng + 1).astype(np.int64)
    psize = np.bincount(pl.ravel(), minlength=npd + 1).astype(np.int64)
    both = (gl > 0) & (pl > 0)
    inter = np.bincount(pl[both].astype(np.int64) * (ng + 1) + gl[both],
                        minlength=(npd + 1) * (ng + 1)).reshape(npd + 1, ng + 1)[1:, 1:]  # [pred, gt]
    row = {'gt_lesions': int(ng), 'pred_lesions': int(npd), 'notes': ''}

    pred_hit = inter.sum(axis=1) > 0
    gt_hit = inter.sum(axis=0) > 0
    row['FPV_vox'] = int(psize[1:][~pred_hit].sum())
    row['FNV_vox'] = int(gsize[1:][~gt_hit].sum())

    # criterion1: any overlap
    row.update(TP_C1=int(pred_hit.sum()), FP_C1=int((~pred_hit).sum()), FN_C1=int((~gt_hit).sum()))

    # best-IoU GT lesion per predicted lesion; strict '>' from 0 keeps the first maximum (j ascending)
    union = psize[1:, None] + gsize[None, 1:] - inter
    iou = np.where(inter > 0, inter / np.maximum(union, 1), 0.0)
    best = iou.argmax(axis=1) if ng else np.zeros(npd, dtype=int)
    best_iou = iou[np.arange(npd), best] if ng else np.zeros(npd)
    match = [int(b) + 1 if v > 0 else None for b, v in zip(best, best_iou)]

    # criterion2: IoU >= threshold, greedy in predicted-label order, GT lesion removed once matched
    remaining, tp, fp = set(range(1, ng + 1)), 0, 0
    for i in range(npd):
        if best_iou[i] >= IOU_THRESHOLD:
            tp += 1
            if match[i] not in remaining:
                row['notes'] += f'C2:list.remove would raise for pred {i + 1};'
            remaining.discard(match[i])
        else:
            fp += 1
    row.update(TP_C2=tp, FP_C2=fp, FN_C2=len(remaining))

    # criterion3: predicted lesion covers the SUVmax voxel of its best-IoU GT lesion
    # (np.argmax over gt_lesion*pet in the (x,y,z) array's C order -> first voxel of the maximum)
    # C-ordered copies give logical C-order flat indices, matching np.argmax on Microsoft's transposed view
    glc = np.ascontiguousarray(gl).ravel()
    petc = np.ascontiguousarray(pet).ravel()
    plc = np.ascontiguousarray(pl).ravel()
    idx = np.flatnonzero(glc)
    labels, values = glc[idx], petc[idx]
    order = np.lexsort((idx, -values, labels))           # by label, then highest SUV, then first index
    first = np.ones(len(order), dtype=bool)
    first[1:] = labels[order][1:] != labels[order][:-1]
    suvmax_voxel = dict(zip(labels[order][first].tolist(), idx[order][first].tolist()))
    suvmax_value = dict(zip(labels[order][first].tolist(), values[order][first].tolist()))
    remaining, tp, fp = set(range(1, ng + 1)), 0, 0
    for i in range(npd):
        j = match[i]
        if j is None:
            voxel = 0                                     # argmax of an all-zero product
            hit = plc[voxel] == i + 1
            if hit:
                row['notes'] += f'C3:unmatched pred {i + 1} covers voxel 0 (Microsoft raises);'
        else:
            if suvmax_value[j] <= 0:
                row['notes'] += f'C3:GT lesion {j} has SUVmax<=0;'
            hit = plc[suvmax_voxel[j]] == i + 1
        if hit:
            tp += 1
            if j not in remaining:
                row['notes'] += f'C3:list.remove would raise for pred {i + 1};'
            remaining.discard(j)
        else:
            fp += 1
    row.update(TP_C3=tp, FP_C3=fp, FN_C3=len(remaining))
    return row


def measures(gt, pred, pet, spacing):
    voxel_ml = float(np.prod(spacing)) / 1000
    out = {}
    for name, mask in (('GT', gt), ('Pred', pred)):
        n = int(mask.sum())
        suv_sum = float(pet[mask == 1].sum(dtype=np.float64)) if n else 0.0
        out[f'{name}_TMTV_ml'] = n * voxel_ml                               # = calculate_patient_level_tmtv
        out[f'{name}_SUVmax'] = float(pet[mask == 1].max()) if n else 0.0   # = ..._suvmean_suvmax('SUVmax') for SUV >= 0
        out[f'{name}_SUVmean'] = suv_sum / n if n else 0.0
        out[f'{name}_TLG'] = voxel_ml * suv_sum                             # = sum_i MTV_i * SUVmean_i
    out['voxel_ml'] = voxel_ml
    return out


def evaluate(case, experiment):
    pet_image = sitk.ReadImage(str(case['pet']))
    pet = ms.get_3darray_from_niftipath(str(case['pet']))
    gt = read_binary(case['gt'], pet_image)
    pred = read_binary(case['pred'][experiment], pet_image)
    spacing = sitk.ReadImage(str(case['gt'])).GetSpacing()                 # as calculate_test_metrics.py
    row = {'experiment': experiment, 'case_id': case['case_id'], 'patient': case['patient'],
           'fold': case['folds'][experiment],
           'DSC': float(ms.calculate_patient_level_dice_score(gt, pred))}
    row.update(measures(gt, pred, pet, spacing))
    det = detection(gt, pred, pet)
    row['FPV_ml'] = det.pop('FPV_vox') * row['voxel_ml']
    row['FNV_ml'] = det.pop('FNV_vox') * row['voxel_ml']
    row.update(det)
    return row, (gt, pred, pet, spacing)


def validate(cases, n):
    """Compare with Microsoft's original functions on the n examinations with the fewest component pairs."""
    sizes = []
    for case in cases:
        pet_image = sitk.ReadImage(str(case['pet']))
        gt = read_binary(case['gt'], pet_image)
        for e in ('A', 'B'):
            pred = read_binary(case['pred'][e], pet_image)
            ng = cc3d.connected_components(gt, connectivity=18, return_N=True)[1]
            npd = cc3d.connected_components(pred, connectivity=18, return_N=True)[1]
            sizes.append((ng * npd, case['case_id'], e))
    sizes.sort()
    # smallest workloads plus a few moderate ones (more components, still tractable)
    chosen = sizes[:n - n // 2] + sizes[len(sizes) // 4: len(sizes) // 4 + n // 2]
    by_id = {c['case_id']: c for c in cases}
    rows = []
    for work, case_id, e in chosen:
        started = time.time()
        fast, (gt, pred, pet, spacing) = evaluate(by_id[case_id], e)
        ref = {'DSC': ms.calculate_patient_level_dice_score(gt, pred),
               'FPV_ml': ms.calculate_patient_level_false_positive_volume(gt, pred, spacing),
               'FNV_ml': ms.calculate_patient_level_false_negative_volume(gt, pred, spacing),
               'GT_TMTV_ml': ms.calculate_patient_level_tmtv(gt, spacing),
               'Pred_TMTV_ml': ms.calculate_patient_level_tmtv(pred, spacing),
               'GT_SUVmax': ms.calculate_patient_level_lesion_suvmean_suvmax(pet, gt, 'SUVmax'),
               'Pred_SUVmax': ms.calculate_patient_level_lesion_suvmean_suvmax(pet, pred, 'SUVmax'),
               'GT_SUVmean': ms.calculate_patient_level_lesion_suvmean_suvmax(pet, gt, 'SUVmean'),
               'Pred_SUVmean': ms.calculate_patient_level_lesion_suvmean_suvmax(pet, pred, 'SUVmean'),
               'GT_TLG': ms.calculate_patient_level_tlg(pet, gt, spacing),
               'Pred_TLG': ms.calculate_patient_level_tlg(pet, pred, spacing),
               'gt_lesions': ms.calculate_patient_level_lesion_count(gt),
               'pred_lesions': ms.calculate_patient_level_lesion_count(pred)}
        for k, crit, kw in (('C1', 'criterion1', {}), ('C2', 'criterion2', {'threshold': 0.5}),
                            ('C3', 'criterion3', {'ptarray': pet})):
            ref[f'TP_{k}'], ref[f'FP_{k}'], ref[f'FN_{k}'] = ms.calculate_patient_level_tp_fp_fn(gt, pred, crit, **kw)
        for key, value in ref.items():
            ok = np.isclose(float(fast[key]), float(value), rtol=1e-6, atol=1e-6)
            rows.append({'case_id': case_id, 'experiment': e, 'metric': key, 'microsoft': float(value),
                         'this_script': float(fast[key]), 'match': bool(ok)})
        bad = [r for r in rows[-len(ref):] if not r['match']]
        print(f'validate {e} {case_id[:16]} gt x pred components={work}: '
              f'{"OK" if not bad else "MISMATCH " + str([(r["metric"], r["microsoft"], r["this_script"]) for r in bad])} '
              f'({time.time() - started:.0f}s)', flush=True)
    path = HERE / 'validation_against_microsoft.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mismatches = sum(not r['match'] for r in rows)
    print(f'{len(rows)} comparisons on {len(chosen)} evaluations, mismatches: {mismatches} -> {path}')
    return mismatches == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--validate', type=int, metavar='N', help='compare with Microsoft functions on N evaluations, then exit')
    args = parser.parse_args()
    cases = load_cases()
    for case in cases:
        for path in (case['pet'], case['gt'], case['pred']['A'], case['pred']['B']):
            if not path.is_file():
                sys.exit(f'Missing input: {path}')
    if args.validate:
        sys.exit(0 if validate(cases, args.validate) else 1)

    out = HERE / 'per_exam_metrics.csv'
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    tmp = out.with_suffix('.partial.csv')
    started, writer, rows = time.time(), None, 0
    with tmp.open('w', newline='') as f:
        for n, case in enumerate(cases, 1):
            for e in ('A', 'B'):
                t = time.time()
                row, _ = evaluate(case, e)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                rows += 1
                print(f'{n:3d}/145 {e} {case["case_id"][:16]} DSC {row["DSC"]:.3f} FPV {row["FPV_ml"]:8.1f} '
                      f'FNV {row["FNV_ml"]:7.1f} lesions gt/pred {row["gt_lesions"]}/{row["pred_lesions"]} '
                      f'{row["notes"]} ({time.time() - t:.1f}s)', flush=True)
    tmp.replace(out)
    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': rows, 'seconds': round(time.time() - started),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'microsoft_metrics_sha256': hashlib.sha256((MS_REPO / 'metrics/metrics.py').read_bytes()).hexdigest(),
        'dataset': str(DATASET), 'A_predictions': str(A_PRED), 'B_run': str(B_RUN),
        'connectivity': CONNECTIVITY, 'iou_threshold': IOU_THRESHOLD,
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'SimpleITK': sitk.Version.VersionString()},
    }
    (HERE / 'compute_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote {out} ({rows} rows) in {manifest["seconds"]} s')


if __name__ == '__main__':
    main()
