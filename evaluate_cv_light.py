"""Lightweight cross-validation evaluation: Dice, TMTV and SUVmax per held-out examination.

Run inside WSL:
    ~/venvs/lymphoma_seg/bin/python evaluate_cv_light.py             # all 5 folds; fails if a fold has no finished inference
    ~/venvs/lymphoma_seg/bin/python evaluate_cv_light.py --folds 0   # only the named folds

Every examination is scored with the prediction of the fold in which it was held out, never another.
No connected components, no lesion matching: three whole-mask measures per examination.

Definitions
  DSC        2*|P & G| / (|P| + |G|) on binary masks (voxel > 0). NaN if both masks are empty.
  TMTV       number of positive voxels * voxel volume from the NIfTI spacing, in ml.
  SUVmax     maximum of the SUV image inside the mask.
  Empty prediction: Pred_TMTV_ml = 0, Pred_SUVmax = NaN, Pred_empty = True. SUVmax errors are then NaN
  and are left out of the SUVmax summary (the summary states how many).
  absolute error = |Pred - GT|; relative error = (Pred - GT) / GT (signed; NaN if GT is 0).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT / 'lymphoma-segmentation-dnn' / 'data_split' / 'train_filepaths.csv'
N_FOLDS = 5
N_CASES = 145


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-3)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-3))


def evaluate_case(pet_path, gt_path, pred_path):
    pet_image, gt_image, pred_image = (sitk.ReadImage(str(p)) for p in (pet_path, gt_path, pred_path))
    for name, image in [('ground truth', gt_image), ('prediction', pred_image)]:
        if not same_geometry(pet_image, image):
            sys.exit(f'Geometry of {name} differs from the SUV image for {Path(gt_path).name}: '
                     f'size {image.GetSize()} vs {pet_image.GetSize()}, spacing {image.GetSpacing()} vs {pet_image.GetSpacing()}, '
                     f'origin {image.GetOrigin()} vs {pet_image.GetOrigin()}')
    pet = sitk.GetArrayFromImage(pet_image)
    gt = sitk.GetArrayFromImage(gt_image) > 0
    pred = sitk.GetArrayFromImage(pred_image) > 0
    voxel_ml = float(np.prod(gt_image.GetSpacing())) / 1000
    n_gt, n_pred = int(gt.sum()), int(pred.sum())
    row = {
        'DSC': 2 * int((gt & pred).sum()) / (n_gt + n_pred) if n_gt + n_pred else np.nan,
        'GT_TMTV_ml': n_gt * voxel_ml,
        'Pred_TMTV_ml': n_pred * voxel_ml,
        'GT_SUVmax': float(pet[gt].max()) if n_gt else np.nan,
        'Pred_SUVmax': float(pet[pred].max()) if n_pred else np.nan,
        'Pred_empty': n_pred == 0,
    }
    for name in ('TMTV_ml', 'SUVmax'):
        difference = row[f'Pred_{name}'] - row[f'GT_{name}']
        short = name.replace('_ml', '')
        row[f'{short}_absolute_error' + ('_ml' if name == 'TMTV_ml' else '')] = abs(difference)
        row[f'{short}_relative_error'] = difference / row[f'GT_{name}'] if row[f'GT_{name}'] else np.nan
    return row


def summarize(table, folds, complete, seconds):
    def error_lines(label, unit, absolute, relative):
        valid = absolute.dropna()
        return [f'| {label} MAE{unit} | {valid.mean():.3f} |', f'| {label} median absolute error{unit} | {valid.median():.3f} |',
                f'| {label} median relative error (signed) | {relative.dropna().median():+.3f} |',
                f'| {label} examinations used | {len(valid)} of {len(absolute)} |']
    lines = ['# Lightweight cross-validation evaluation', '',
             f'Written {datetime.now():%Y-%m-%d %H:%M:%S}. Folds: {folds}. Examinations: {len(table)}. '
             f'Evaluation time: {seconds:.0f} s.', '',
             ('All 5 folds: every one of the 145 examinations appears exactly once.' if complete else
              f'**PARTIAL: only folds {folds}. This is not the cross-validation result.**'), '',
             f'Empty predictions: {int(table.Pred_empty.sum())}.', '',
             '| Measure | Value |', '|---|---|',
             f'| DSC mean | {table.DSC.mean():.4f} |', f'| DSC median | {table.DSC.median():.4f} |',
             f'| DSC std | {table.DSC.std():.4f} |', f'| DSC min | {table.DSC.min():.4f} |', f'| DSC max | {table.DSC.max():.4f} |',
             *error_lines('TMTV', ' (ml)', table.TMTV_absolute_error_ml, table.TMTV_relative_error),
             *error_lines('SUVmax', '', table.SUVmax_absolute_error, table.SUVmax_relative_error), '',
             '## Dice per fold', '', '| fold | examinations | mean | median | std | min | max |', '|---|---|---|---|---|---|---|']
    for fold, g in table.groupby('fold'):
        lines.append(f'| {fold} | {len(g)} | {g.DSC.mean():.4f} | {g.DSC.median():.4f} | {g.DSC.std():.4f} | {g.DSC.min():.4f} | {g.DSC.max():.4f} |')
    return '\n'.join(lines) + '\n'


def main(args):
    root = PROJECT / 'experiments' / args.experiment
    cases = {c['case_id']: c for c in json.loads((args.dataset_root / 'cases.json').read_text(encoding='utf-8'))}
    datalist = pd.read_csv(TRAIN_CSV)
    if len(datalist) != N_CASES or datalist.ImageID.duplicated().any() or set(datalist.ImageID) != set(cases):
        sys.exit(f'{TRAIN_CSV} does not list exactly the {N_CASES} cases of cases.json')
    if any(cases[i]['fold'] != f for i, f in zip(datalist.ImageID, datalist.FoldID)):
        sys.exit('FoldID in train_filepaths.csv differs from cases.json')

    missing = [f for f in args.folds if not (root / 'state' / f'fold{f}.inference.done').is_file()]
    if missing:
        done = [f for f in range(N_FOLDS) if (root / 'state' / f'fold{f}.inference.done').is_file()]
        sys.exit(f'Inference is not complete for fold(s) {missing}. Completed folds: {done}. '
                 f'To evaluate only those, pass --folds {" ".join(map(str, done))}')

    start, rows = time.time(), []
    for fold in args.folds:
        held_out = datalist[datalist.FoldID == fold].sort_values('ImageID')
        pred_dir = root / 'work' / 'results' / 'predictions' / f'fold{fold}' / 'unet' / f'unet_fold{fold}_randcrop192'
        found, expected = sorted(p.name for p in pred_dir.glob('*')), sorted(f'{i}.nii.gz' for i in held_out.ImageID)
        if found != expected:
            sys.exit(f'Fold {fold}: files in {pred_dir} are not exactly the predictions of its {len(expected)} held-out examinations')
        for case in held_out.itertuples():
            row = {'ImageID': case.ImageID, 'patient': cases[case.ImageID]['patient'], 'fold': fold}
            row.update(evaluate_case(case.PTPATH, case.GTPATH, pred_dir / f'{case.ImageID}.nii.gz'))
            rows.append(row)
            print(f"fold {fold}  {len(rows):3d}  {case.ImageID[:16]}  DSC {row['DSC']:.4f}  TMTV {row['GT_TMTV_ml']:8.1f} -> {row['Pred_TMTV_ml']:8.1f} ml  "
                  f"SUVmax {row['GT_SUVmax']:6.2f} -> {row['Pred_SUVmax']:6.2f}", flush=True)
    table = pd.DataFrame(rows)
    seconds = time.time() - start

    complete = sorted(args.folds) == list(range(N_FOLDS))
    if table.ImageID.duplicated().any():
        sys.exit('An examination was evaluated more than once')
    if any(cases[i]['fold'] != f for i, f in zip(table.ImageID, table.fold)):
        sys.exit('An examination was evaluated under a fold in which it was not held out')
    if len(table) != sum((datalist.FoldID == f).sum() for f in args.folds) or (complete and len(table) != N_CASES):
        sys.exit(f'Unexpected number of evaluated examinations: {len(table)}')

    name = 'all_folds' if complete else 'PARTIAL_folds_' + '_'.join(map(str, sorted(args.folds)))
    out = root / 'light_eval'
    out.mkdir(exist_ok=True)
    table.to_csv(out / f'light_metrics_{name}.csv', index=False)
    summary = summarize(table, sorted(args.folds), complete, seconds)
    (out / f'light_summary_{name}.md').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f'Written: {out / f"light_metrics_{name}.csv"}\n         {out / f"light_summary_{name}.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default='unet_10ep')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/eva25/data/lymphoma-baseline'))
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)), choices=range(N_FOLDS))
    arguments = parser.parse_args()
    if len(set(arguments.folds)) != len(arguments.folds):
        sys.exit('--folds lists a fold twice')
    main(arguments)
