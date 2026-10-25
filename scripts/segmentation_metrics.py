"""Native-grid metrics, Microsoft-compatible on ordinary positive cases.

Definitions inspected at Microsoft lymphoma-segmentation-dnn commit
81129243db3c868370a7664c9adc2ad52609484f, metrics/metrics.py (MIT).
This independent implementation explicitly defines empty and ambiguous cases.
"""
from __future__ import annotations

import numpy as np
import cc3d

CONNECTIVITY = 18
IOU_THRESHOLD = 0.5
SIZE_BINS = ('small_lt1ml', 'medium_1to10ml', 'large_ge10ml')


def size_bin(ml):
    return SIZE_BINS[0] if ml < 1 else SIZE_BINS[1] if ml < 10 else SIZE_BINS[2]


def binary(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError('Masks must be finite, binary, three-dimensional arrays')
    return mask.astype(bool)


def components(mask):
    # Microsoft reads x,y,z arrays in Fortran order after transposing SimpleITK.
    return cc3d.connected_components(np.asfortranarray(binary(mask), dtype=np.uint8),
                                     connectivity=CONNECTIVITY, return_N=True)


def divide(n, d):
    return float(n / d) if d else None


def metrics(gt, pred, pet, voxel_ml):
    gt, pred, pet = binary(gt), binary(pred), np.asarray(pet)
    if gt.shape != pred.shape or gt.shape != pet.shape:
        raise ValueError('Array shapes differ')
    if not np.isfinite(pet).all() or (pet < 0).any():
        raise ValueError('Expected finite, nonnegative quantitative SUV')
    if not np.isfinite(voxel_ml) or voxel_ml <= 0:
        raise ValueError('Expected positive finite voxel volume in mL')
    gl, ng = components(gt)
    pl, npred = components(pred)
    gs = np.bincount(gl.ravel(), minlength=ng + 1)
    ps = np.bincount(pl.ravel(), minlength=npred + 1)
    # Sparse intersections avoid an unbounded number-of-lesions squared table.
    overlap = gt & pred
    pairs, counts = np.unique(pl[overlap].astype(np.int64) * (ng + 1) + gl[overlap], return_counts=True)
    matches = [[] for _ in range(npred + 1)]
    gfound = set()
    for pair, count in zip(pairs.tolist(), counts.tolist()):
        p, g = divmod(pair, ng + 1)
        matches[p].append((g, count / (ps[p] + gs[g] - count)))
        gfound.add(g)
    # First maximum in logical x,y,z C order; never chooses background for SUV=0.
    indices = np.flatnonzero(gt.ravel(order='C'))
    gflat, pflat, suv = gl.ravel(order='C'), pl.ravel(order='C'), pet.ravel(order='C')
    order = np.lexsort((indices, -suv[indices], gflat[indices]))
    sorted_labels = gflat[indices[order]]
    first = np.r_[True, sorted_labels[1:] != sorted_labels[:-1]] if len(order) else np.array([], bool)
    hottest = dict(zip(sorted_labels[first].tolist(), indices[order][first].tolist()))
    detected = {'C1': gfound, 'C2': set(), 'C3': set()}
    tp = {'C1': sum(bool(m) for m in matches[1:]), 'C2': 0, 'C3': 0}
    notes = []
    for p in range(1, npred + 1):
        if not matches[p]:
            continue
        g, iou = min(matches[p], key=lambda item: (-item[1], item[0]))
        for criterion, hit in [('C2', iou >= IOU_THRESHOLD), ('C3', pflat[hottest[g]] == p)]:
            if hit and g not in detected[criterion]:
                detected[criterion].add(g)
                tp[criterion] += 1
            elif hit:
                notes.append(f'{criterion}:duplicate match to reference {g}; counted as FP')
    n_gt, n_pred = int(gt.sum()), int(pred.sum())
    row = {'DSC': 2 * int(overlap.sum()) / (n_gt + n_pred) if n_gt + n_pred else 1.0,
           'voxel_ml': float(voxel_ml), 'gt_lesions': int(ng), 'pred_lesions': int(npred),
           'FPV_ml': float(sum(ps[p] for p in range(1, npred + 1) if not matches[p]) * voxel_ml),
           'FNV_ml': float(sum(gs[g] for g in range(1, ng + 1) if g not in gfound) * voxel_ml),
           'voxel_FP_ml': float((pred & ~gt).sum() * voxel_ml),
           'voxel_FN_ml': float((gt & ~pred).sum() * voxel_ml), 'notes': '; '.join(notes)}
    for prefix, mask in [('GT', gt), ('Pred', pred)]:
        values = pet[mask]
        row[prefix + '_TMTV_ml'] = float(mask.sum() * voxel_ml)
        row[prefix + '_SUVmax'] = float(values.max()) if len(values) else 0.0
    for measure in ('TMTV_ml', 'SUVmax'):
        diff = row['Pred_' + measure] - row['GT_' + measure]
        row[measure + '_signed_error'] = diff
        row[measure + '_absolute_error'] = abs(diff)
        row[measure + '_relative_error'] = divide(diff, row['GT_' + measure])
    lesions = []
    for g in range(1, ng + 1):
        lesions.append({'lesion': g, 'volume_ml': float(gs[g] * voxel_ml),
                        'size_bin': size_bin(gs[g] * voxel_ml),
                        **{f'found_{c}': int(g in detected[c]) for c in detected}})
    for c in detected:
        row[f'TP_{c}'], row[f'FP_{c}'], row[f'FN_{c}'] = tp[c], npred - tp[c], ng - len(detected[c])
        row[f'recall_{c}'] = divide(len(detected[c]), ng)
        row[f'precision_{c}'] = divide(tp[c], npred)
        for group in SIZE_BINS:
            selected = [lesion for lesion in lesions if lesion['size_bin'] == group]
            row[f'{group}_reference'] = len(selected)
            row[f'{group}_found_{c}'] = sum(l[f'found_{c}'] for l in selected)
    return row, lesions


def summarize(rows):
    summary = {'examinations': len(rows), 'patients': len({r['patient'] for r in rows}),
               'per_exam': {}, 'pooled_detection': {}}
    for key in rows[0]:
        values = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] is not None]
        if values:
            summary['per_exam'][key] = {'n': len(values), 'mean': float(np.mean(values)),
                                        'median': float(np.median(values)), 'sum': float(sum(values))}
    for c in ('C1', 'C2', 'C3'):
        ng = sum(r['gt_lesions'] for r in rows)
        found = ng - sum(r[f'FN_{c}'] for r in rows)
        tp, fp = sum(r[f'TP_{c}'] for r in rows), sum(r[f'FP_{c}'] for r in rows)
        recall, precision = divide(found, ng), divide(tp, tp + fp)
        summary['pooled_detection'][c] = {'reference': ng, 'found': found, 'TP': tp, 'FP': fp,
            'recall': recall, 'precision': precision,
            'F1': (2 * recall * precision / (recall + precision) if recall + precision else 0.0)
                  if recall is not None and precision is not None else None,
            'size_strata': {group: {'reference': sum(r[f'{group}_reference'] for r in rows),
                'found': sum(r[f'{group}_found_{c}'] for r in rows),
                'recall': divide(sum(r[f'{group}_found_{c}'] for r in rows), sum(r[f'{group}_reference'] for r in rows))}
                for group in SIZE_BINS}}
    return summary
