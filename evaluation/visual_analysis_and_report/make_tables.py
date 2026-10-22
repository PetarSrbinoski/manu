"""Tables for the visual analysis: SUVmax source per prediction, and the selected-examples table.

1. suvmax_source.csv: for every subset examination and model, the voxel that defines the predicted SUVmax
   (maximum SUV inside the predicted mask), whether it lies inside the reference, in a false-positive
   component (touching no reference lesion), or in a predicted component that touches the reference,
   and, for reviewed false-positive components, the visually identified structure.
2. selected_examples_table.csv / .md: each selected example with its metrics and the observed errors.

Read-only. Run inside WSL from the project root:
    ~/venvs/lymphoma_seg/bin/python evaluation/visual_analysis_and_report/make_tables.py
"""
import json

import cc3d
import numpy as np
import pandas as pd

import viz_common as vc

OBSERVED = {
    'E1': 'A: false-positive volume almost entirely in the brain (1019 ml) plus the bladder (92 ml); 62 small lesions missed. '
          'B: little false-positive volume (brain, 130 ml) but under-segments the large left upper-abdominal disease and misses '
          '94 small lesions (54 ml). C: near-complete agreement.',
    'E2': 'All three models miss many small lesions (A 76, B 90, C 50 missed lesions). A and B add false positives at both kidneys '
          'and in the brain; B also in the bladder and under-segments the large left upper-abdominal lesion.',
    'E3': 'A: 1073 ml false positives: brain (731 ml), bladder with an unidentified tubular pelvic extension (264 ml), left renal '
          'hilum (20 ml). B: brain (167 ml) and bladder (23 ml); B under-segments the left cervical disease. C: no false positives.',
    'E4': 'A: false positives in bladder (415 ml), heart (350 ml), brain (336 ml) and both kidneys (228 ml). B: brain (337 ml), '
          'kidneys (130 ml), heart (49 ml); one 7.3 ml lesion missed entirely. The abdominal disease is found by all three.',
    'E5': 'B covers the main lesion in the left axillary region only partly (none of it on the shown slice; 2 of 6 lesions found; Dice 0.126); A finds it but adds bladder '
          '(278 ml), heart (205 ml) and brain (163 ml) false positives. C: Dice 0.977.',
    'E6': 'Very small burden (6.7 ml) in the right axillary region: A and B miss most of it on the shown slice; A marks the whole '
          'bladder (379 ml) and brain cortex (53 ml), B the brain (36 ml). Dice of A/B is near 0 because of false positives; C finds the nodes.',
    'E7': 'Small pelvic lesions (14 ml): A and B touch all 5 lesions (recall 1.0, FNV 0) but cover them only partly; B covers almost none of the lesion on the shown slice. Both mark the bladder '
          '(A 126 ml, B 202 ml); B adds kidneys, heart and brain, hence its larger false-positive volume (466 vs 141 ml).',
}


def suvmax_source(cases, review):
    rows = []
    for case in cases:
        image, a = vc.load(case)
        suv, gt = a['SUV'], a['GT']
        short = case['case_id'][6:16]
        for model in 'ABC':
            pred = a[model]
            labels = cc3d.connected_components(pred.astype(np.uint8), connectivity=vc.ev.CONNECTIVITY)
            masked = np.where(pred, suv, -np.inf)
            k, j, i = np.unravel_index(np.argmax(masked), suv.shape)
            label = labels[k, j, i]
            touches = bool(gt[labels == label].any())
            where = 'inside reference' if gt[k, j, i] else ('predicted lesion touching reference' if touches else 'false-positive component')
            structure = ''
            if where == 'false-positive component':
                hit = review[(review.exam == short) & (review.model == model) & (review.component == label)]
                structure = hit.structure.iloc[0] if len(hit) else 'not reviewed (< 20 ml or model C)'
            rows.append({'exam': short, 'model': model, 'Pred_SUVmax': float(suv[k, j, i]),
                         'GT_SUVmax': float(suv[gt].max()), 'voxel_kji': f'{k},{j},{i}', 'location': where, 'structure': structure})
        print(short, [(r['model'], r['location'][:12], r['structure']) for r in rows[-3:]], flush=True)
    return pd.DataFrame(rows)


def main():
    cases = vc.subset_cases()
    review = pd.read_csv(vc.HERE / 'fp_anatomy_review.csv')
    src = suvmax_source(cases, review)
    src.round(3).to_csv(vc.HERE / 'suvmax_source.csv', index=False)

    metrics = pd.read_csv(vc.PROJECT / 'evaluation' / 'autopet3_vs_a_b' / 'per_exam_metrics_10.csv')
    metrics['exam'] = metrics.case_id.str[6:16]
    examples = json.loads((vc.HERE / 'selected_examples.json').read_text())
    rows = []
    for ex in examples:
        short = ex['exam_id'][6:16]
        m = metrics[metrics.exam == short].set_index('experiment')
        row = {'label': ex['label'], 'exam_id': ex['exam_id'], 'selected_because': ex['reason'],
               'slices': '; '.join(f"{s['view']} {s['index0'] + 1}" for s in ex['slices']),
               'ref_TMTV_ml': round(m.loc['A', 'GT_TMTV_ml'], 1), 'ref_lesions': int(m.loc['A', 'gt_lesions']),
               'ref_SUVmax': round(m.loc['A', 'GT_SUVmax'], 2)}
        for e in 'ABC':
            s = src[(src.exam == short) & (src.model == e)].iloc[0]
            row.update({f'{e}_Dice': round(m.loc[e, 'DSC'], 3), f'{e}_FPV_ml': round(m.loc[e, 'FPV_ml'], 1),
                        f'{e}_FNV_ml': round(m.loc[e, 'FNV_ml'], 1), f'{e}_TMTV_abs_err_ml': round(m.loc[e, 'TMTV_abs_error_ml'], 1),
                        f'{e}_recall_C1': round(m.loc[e, 'recall_C1'], 2), f'{e}_FP_lesions_C1': int(m.loc[e, 'FP_C1']),
                        f'{e}_Pred_SUVmax': round(m.loc[e, 'Pred_SUVmax'], 2),
                        f'{e}_SUVmax_source': s.location + (f' ({s.structure})' if s.structure else '')})
        row['observed_errors'] = OBSERVED[ex['label']]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(vc.HERE / 'selected_examples_table.csv', index=False)

    lines = ['# Selected examples: metrics and observed errors', '',
             'Metrics from `evaluation/autopet3_vs_a_b/per_exam_metrics_10.csv` (unchanged). SUVmax source from `suvmax_source.csv`. '
             'C = autoPET III, trained on data that include these examinations (orientation only).', '',
             '| Ex. | Ref. TMTV ml / lesions | Dice A / B / C | FPV ml A / B / C | FNV ml A / B / C | Lesion recall C1 A / B / C | SUVmax ref → A / B / C |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['ref_TMTV_ml']} / {r['ref_lesions']} | {r['A_Dice']} / {r['B_Dice']} / {r['C_Dice']} | "
                     f"{r['A_FPV_ml']} / {r['B_FPV_ml']} / {r['C_FPV_ml']} | {r['A_FNV_ml']} / {r['B_FNV_ml']} / {r['C_FNV_ml']} | "
                     f"{r['A_recall_C1']} / {r['B_recall_C1']} / {r['C_recall_C1']} | "
                     f"{r['ref_SUVmax']} → {r['A_Pred_SUVmax']} / {r['B_Pred_SUVmax']} / {r['C_Pred_SUVmax']} |")
    lines += ['', '| Ex. | Selected because | Slices (1-based) | Predicted SUVmax comes from (A; B; C) | Observed errors |', '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['selected_because']} | {r['slices']} | {r['A_SUVmax_source']}; {r['B_SUVmax_source']}; "
                     f"{r['C_SUVmax_source']} | {r['observed_errors']} |")
    (vc.HERE / 'selected_examples_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(src.groupby(['model', 'location']).size().to_string())
    print(src[src.location == 'false-positive component'].groupby(['model', 'structure']).size().to_string())


if __name__ == '__main__':
    main()
