"""Light evaluation of autoPET III predictions: Dice, TMTV and SUVmax per examination.

Run inside WSL:
    ~/venvs/lymphoma_seg/bin/python evaluate_autopet_light.py                      # the smoke-test folder
    ~/venvs/lymphoma_seg/bin/python evaluate_autopet_light.py --pred-dir <folder>

The measures are computed by evaluate_case from evaluate_cv_light.py, the same function that scored
the Microsoft baseline, so the numbers are directly comparable. See that file for the definitions.

Every <case_id>.nii.gz in the prediction folder is scored against labelsTr/<case_id>.nii.gz, with
SUV taken from imagesTr/<case_id>_0001.nii.gz. Nothing is resampled: if a prediction or a ground
truth does not lie on the grid of the SUV image, the script stops and reports the difference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

from evaluate_cv_light import evaluate_case, same_geometry

SMOKE_DIR = Path('/home/eva25/autopet3_runs/smoke')


def describe(name, image):
    array = sitk.GetArrayFromImage(image)
    values, counts = np.unique(array, return_counts=True)
    shown = dict(zip(values.tolist(), counts.tolist())) if len(values) <= 10 else f'{len(values)} distinct values'
    print(f'  {name:<11} size {image.GetSize()}  spacing {tuple(round(s, 4) for s in image.GetSpacing())}  '
          f'origin {tuple(round(o, 2) for o in image.GetOrigin())}  {array.dtype}')
    print(f'  {"":<11} direction {tuple(round(d, 3) for d in image.GetDirection())}')
    print(f'  {"":<11} values {shown}' if name != 'SUV' else f'  {"":<11} min {array.min():.3f}  max {array.max():.3f}')
    return values


def main(args):
    cases = {c['case_id']: c for c in json.loads((args.dataset_root / 'cases.json').read_text(encoding='utf-8'))}
    predictions = sorted(args.pred_dir.glob('*.nii.gz'))
    if not predictions:
        sys.exit(f'No predictions (*.nii.gz) in {args.pred_dir}')

    rows = []
    for pred_path in predictions:
        case_id = pred_path.name[:-len('.nii.gz')]
        if case_id not in cases:
            sys.exit(f'{pred_path.name} is not a case of {args.dataset_root / "cases.json"}')
        gt_path = args.dataset_root / 'labelsTr' / f'{case_id}.nii.gz'
        pet_path = args.dataset_root / 'imagesTr' / f'{case_id}_0001.nii.gz'

        print(f'\n{case_id}')
        print(f'  prediction:   {pred_path}\n  ground truth: {gt_path}\n  SUV:          {pet_path}')
        pet_image, gt_image, pred_image = (sitk.ReadImage(str(p)) for p in (pet_path, gt_path, pred_path))
        describe('SUV', pet_image)
        gt_values = describe('ground truth', gt_image)
        pred_values = describe('prediction', pred_image)
        for name, image, values in [('ground truth', gt_image, gt_values), ('prediction', pred_image, pred_values)]:
            print(f'  {name} on the SUV grid: {same_geometry(pet_image, image)}')
            if not set(values.tolist()) <= {0, 1}:
                sys.exit(f'{name} of {case_id} has values other than 0 and 1: {values.tolist()}')

        # evaluate_case repeats the geometry check and stops the script on a mismatch.
        row = {'ImageID': case_id, 'patient': cases[case_id]['patient'], 'baseline_fold': cases[case_id]['fold']}
        row.update(evaluate_case(pet_path, gt_path, pred_path))
        rows.append(row)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print()
    for row in rows:
        for key, value in row.items():
            print(f'  {key:<26} {value}')
    print(f'\nWritten: {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-dir', type=Path, default=SMOKE_DIR / 'output')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/eva25/data/lymphoma-baseline'))
    parser.add_argument('--out', type=Path, default=SMOKE_DIR / 'light_metrics_smoke.csv')
    main(parser.parse_args())
