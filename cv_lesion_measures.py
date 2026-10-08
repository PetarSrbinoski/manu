"""Patient-level lesion measures for one fold's held-out predictions.

Same measures and same functions as Microsoft's segmentation/generate_lesion_measures.py
(imported unmodified from lymphoma-segmentation-dnn/metrics/metrics.py), except Dmax.
Dmax is left out because calculate_patient_level_dissemination builds an N x N x 3 array
over all lesion voxels and needs far more memory than this machine has for many of our cases.
Predictions are matched to ground truth by case id, not by sorted position.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd
import SimpleITK as sitk


def main(args):
    sys.path.insert(0, str(args.ms_repo))
    from metrics.metrics import (
        get_3darray_from_niftipath,
        calculate_patient_level_dice_score,
        calculate_patient_level_lesion_suvmean_suvmax,
        calculate_patient_level_lesion_count,
        calculate_patient_level_tmtv,
        calculate_patient_level_tlg,
    )
    test = pd.read_csv(args.test_csv)
    rows = []
    for number, case in enumerate(test.itertuples(), start=1):
        predpath = args.pred_dir / f'{case.ImageID}.nii.gz'
        if not predpath.is_file():
            sys.exit(f'Missing prediction for {case.ImageID}: {predpath}')
        ptarray = get_3darray_from_niftipath(case.PTPATH)
        gtarray = get_3darray_from_niftipath(case.GTPATH)
        predarray = get_3darray_from_niftipath(str(predpath))
        if not ptarray.shape == gtarray.shape == predarray.shape:
            sys.exit(f'Shape mismatch for {case.ImageID}: PET {ptarray.shape}, GT {gtarray.shape}, prediction {predarray.shape}')
        spacing = sitk.ReadImage(case.GTPATH).GetSpacing()
        row = {
            'PatientID': case.ImageID,
            'DSC': calculate_patient_level_dice_score(gtarray, predarray),
            'SUVmean_orig': calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmean'),
            'SUVmean_pred': calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmean'),
            'SUVmax_orig': calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmax'),
            'SUVmax_pred': calculate_patient_level_lesion_suvmean_suvmax(ptarray, predarray, marker='SUVmax'),
            'LesionCount_orig': calculate_patient_level_lesion_count(gtarray),
            'LesionCount_pred': calculate_patient_level_lesion_count(predarray),
            'TMTV_orig': calculate_patient_level_tmtv(gtarray, spacing),
            'TMTV_pred': calculate_patient_level_tmtv(predarray, spacing),
            'TLG_orig': calculate_patient_level_tlg(ptarray, gtarray, spacing),
            'TLG_pred': calculate_patient_level_tlg(ptarray, predarray, spacing),
        }
        rows.append(row)
        print(f"{number}/{len(test)}: {case.ImageID}: SUVmax GT {row['SUVmax_orig']:.2f} / pred {row['SUVmax_pred']:.2f}; "
              f"TMTV GT {row['TMTV_orig']:.1f} ml / pred {row['TMTV_pred']:.1f} ml; "
              f"lesions GT {row['LesionCount_orig']} / pred {row['LesionCount_pred']}", flush=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms-repo', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--pred-dir', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    main(parser.parse_args())
