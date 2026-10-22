"""Records the visual anatomical assessment of every false-positive component >= 20 ml of A and B.

The assessment was made by reviewing the sheets in review/ (axial CT, axial SUV and coronal CT through each
component's centroid, plus coronal SUV MIPs). It is a visual reading of the images by the analyst, not an
organ segmentation, and not a radiologist's report. 'high' = the structure is unambiguous on CT and PET
(e.g. calvarium-enclosed brain, midline fluid-filled pelvic bladder, bilateral contrast-enhancing kidneys,
left-ventricular myocardium ring); 'medium' = location fits but the boundary or identity is less certain;
'unassigned' = no structure named. Writes fp_anatomy_review.csv and fp_anatomy_summary.csv.
Run (any Python with pandas) from this folder:  python fp_anatomy_review.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_ML = 20

# (exam short id, model, volume ml rounded to 0.1) -> (structure, confidence, image evidence)
BRAIN = ('brain', 'high', 'inside the calvarium on axial CT; cortical FDG pattern on SUV')
BLADDER = ('urinary bladder', 'high', 'midline pelvic fluid-density structure behind the pubic symphysis; intense uptake (urine)')
HEART = ('heart (left-ventricular myocardium)', 'high', 'ring-shaped uptake around the left-ventricular cavity in the mediastinum')
KID_L = ('left kidney', 'high', 'contrast-enhancing renal parenchyma, patient left, bilateral counterpart visible')
KID_R = ('right kidney', 'high', 'contrast-enhancing renal parenchyma, patient right, bilateral counterpart visible')
REVIEW = {
    ('0b57b247b6', 'A', 216.2): BLADDER, ('0b57b247b6', 'A', 53.2): HEART,
    ('0b57b247b6', 'B', 141.5): BRAIN, ('0b57b247b6', 'B', 112.1): BLADDER,
    ('1fe6e48293', 'A', 415.4): BLADDER, ('1fe6e48293', 'A', 350.0): HEART, ('1fe6e48293', 'A', 336.0): BRAIN,
    ('1fe6e48293', 'A', 134.7): KID_L, ('1fe6e48293', 'A', 93.4): KID_R,
    ('1fe6e48293', 'B', 336.9): BRAIN, ('1fe6e48293', 'B', 78.3): KID_R, ('1fe6e48293', 'B', 51.7): KID_L,
    ('1fe6e48293', 'B', 48.5): HEART,
    ('22d07bdc49', 'A', 126.3): BLADDER, ('22d07bdc49', 'B', 202.1): BLADDER, ('22d07bdc49', 'B', 60.1): KID_L,
    ('22d07bdc49', 'B', 49.3): ('heart (left-ventricular myocardium)', 'medium',
                                'left heart border on axial CT and coronal CT; partial ring only'),
    ('22d07bdc49', 'B', 33.6): KID_R, ('22d07bdc49', 'B', 31.5): BRAIN,
    ('22d07bdc49', 'B', 23.9): ('unassigned', 'unassigned',
                                'right lower abdomen, low CT density, moderate uptake; structure not identifiable on these views'),
    ('249dd35d0c', 'A', 731.4): BRAIN,
    ('249dd35d0c', 'A', 263.9): ('urinary bladder (+ unidentified pelvic extension)', 'high',
                                 'bladder as above; an elongated tubular extension along the pelvic wall is not identified'),
    ('249dd35d0c', 'A', 20.5): ('left renal hilum region', 'medium', 'medial to the left kidney at the hilum; exact structure uncertain'),
    ('249dd35d0c', 'B', 166.8): BRAIN, ('249dd35d0c', 'B', 22.7): BLADDER,
    ('30001118d0', 'A', 1018.6): BRAIN, ('30001118d0', 'A', 92.4): BLADDER, ('30001118d0', 'B', 129.8): BRAIN,
    ('5ef99b0b6b', 'A', 172.6): BLADDER,
    ('62405d117c', 'A', 278.5): BLADDER, ('62405d117c', 'A', 204.6): HEART, ('62405d117c', 'A', 162.9): BRAIN,
    ('62405d117c', 'B', 63.6): BLADDER,
    ('6a3477cd9a', 'A', 216.3): BRAIN,
    ('6a3477cd9a', 'A', 58.0): ('right kidney region (collecting system?)', 'medium',
                                'at the lower pole / hilum of the right kidney; intense focal uptake'),
    ('6a3477cd9a', 'B', 62.5): BRAIN,
    ('6a3477cd9a', 'B', 44.5): ('right kidney region (collecting system?)', 'medium',
                                'at the lower pole / hilum of the right kidney; intense focal uptake'),
    ('6a3477cd9a', 'B', 43.6): BLADDER,
    ('c4a686881e', 'A', 334.7): BRAIN, ('c4a686881e', 'A', 278.6): BLADDER, ('c4a686881e', 'A', 29.1): KID_L,
    ('c4a686881e', 'B', 86.7): BLADDER, ('c4a686881e', 'B', 52.5): BRAIN, ('c4a686881e', 'B', 33.3): KID_L,
    ('c4a686881e', 'B', 27.2): KID_R,
    ('f47e31ceb5', 'A', 378.5): BLADDER, ('f47e31ceb5', 'A', 52.7): BRAIN,
    ('f47e31ceb5', 'B', 35.5): ('brain (lateral cortex)', 'medium', 'right lateral cortex at the inner skull table'),
}


def group(structure):
    for key in ('brain', 'bladder', 'heart', 'kidney', 'renal'):
        if key in structure:
            return {'renal': 'kidney'}.get(key, key)
    return 'unassigned'


def main():
    table = pd.read_csv(HERE / 'error_components.csv')
    table['exam'] = table.case_id.str[6:16]
    fp = table[(table.kind == 'FP') & table.model.isin(['A', 'B'])].copy()
    large = fp[fp.volume_ml >= MIN_ML].copy()
    keys = list(zip(large.exam, large.model, large.volume_ml.round(1)))
    missing = [k for k in keys if k not in REVIEW]
    extra = [k for k in REVIEW if k not in keys]
    if missing or extra:
        raise SystemExit(f'review table out of sync: missing {missing}, extra {extra}')
    large['structure'], large['confidence'], large['evidence'] = zip(*[REVIEW[k] for k in keys])
    large['group'] = large.structure.map(group)
    cols = ['exam', 'model', 'component', 'volume_ml', 'cm_below_top_of_scan', 'centroid_k', 'centroid_j', 'centroid_i',
            'SUVmax', 'SUVmean', 'CT_mean_HU', 'structure', 'group', 'confidence', 'evidence']
    large[cols].round(2).to_csv(HERE / 'fp_anatomy_review.csv', index=False)

    rows = []
    for model in ('A', 'B'):
        total = fp[fp.model == model].volume_ml.sum()
        sub = large[large.model == model]
        for g in ('brain', 'bladder', 'heart', 'kidney', 'unassigned'):
            part = sub[sub.group == g]
            rows.append({'model': model, 'group': g, 'components': len(part), 'exams': part.exam.nunique(),
                         'volume_ml': part.volume_ml.sum(), 'share_of_all_FP_volume': part.volume_ml.sum() / total})
        rest = total - sub.volume_ml.sum()
        rows.append({'model': model, 'group': f'components < {MIN_ML} ml (not reviewed)',
                     'components': int((fp[fp.model == model].volume_ml < MIN_ML).sum()), 'exams': fp[fp.model == model].exam.nunique(),
                     'volume_ml': rest, 'share_of_all_FP_volume': rest / total})
        rows.append({'model': model, 'group': 'all FP components', 'components': int((fp.model == model).sum()),
                     'exams': fp[fp.model == model].exam.nunique(), 'volume_ml': total, 'share_of_all_FP_volume': 1.0})
    summary = pd.DataFrame(rows)
    summary.round(3).to_csv(HERE / 'fp_anatomy_summary.csv', index=False)
    print(summary.round(3).to_string(index=False))
    print(large.groupby(['model', 'group']).confidence.value_counts().to_string())


if __name__ == '__main__':
    main()
