import argparse
import csv
import json
from pathlib import Path
import sys
#koi fajlovi se CT, koi PET, koja e maskata i vo koj fold pripagja sekoj case.
PROJECT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT / 'data/lymphoma-baseline'
PIPELINE = PROJECT / 'lymphoma-segmentation-dnn'
SPLIT_FOLDER = PIPELINE / 'data_split'
N_FOLDS = 5

#ova poso nivniot kod bara csv file, a jas imam json

def write_datalist(dataset):
    # the existing patient-level split is the only source of fold ids; nothing is reshuffled here
    folds = json.loads((dataset / 'splits_patient_level.json').read_text(encoding='utf-8'))
    cases = json.loads((dataset / 'cases.json').read_text(encoding='utf-8'))
    fold_of = {case_id: fold['fold'] for fold in folds for case_id in fold['validation']}
    if len(fold_of) != len(cases) or any(fold_of[c['case_id']] != c['fold'] for c in cases):
        raise ValueError('splits_patient_level.json and cases.json disagree')
    SPLIT_FOLDER.mkdir(exist_ok=True)
    with open(SPLIT_FOLDER / 'train_filepaths.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageID', 'FoldID', 'CTPATH', 'PTPATH', 'GTPATH'])
        for case in sorted(cases, key=lambda c: c['case_id']):
            case_id = case['case_id']
            paths = [dataset / 'imagesTr' / f'{case_id}_0000.nii.gz',
                     dataset / 'imagesTr' / f'{case_id}_0001.nii.gz',
                     dataset / 'labelsTr' / f'{case_id}.nii.gz']
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(path)
            writer.writerow([case_id, case['fold'], *paths])
    # the pipeline only skips its own case-level split when both files exist; there is no test set
    with open(SPLIT_FOLDER / 'test_filepaths.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['ImageID', 'CTPATH', 'PTPATH', 'GTPATH'])


def verify(fold, dataset):
    # read the split back through the function the trainer itself calls
    sys.path.insert(0, str(PIPELINE / 'segmentation'))
    from initialize_train import get_train_valid_data_in_dict_format
    cases = json.loads((dataset / 'cases.json').read_text(encoding='utf-8'))
    patient_of = {c['case_id']: c['patient'] for c in cases}
    case_of = lambda item: Path(item['GT']).name[:-len('.nii.gz')]
    train, valid = get_train_valid_data_in_dict_format(fold)
    for item in train + valid:
        case_id = case_of(item)
        expected = [f'{case_id}_0000.nii.gz', f'{case_id}_0001.nii.gz']
        if [Path(item['CT']).name, Path(item['PT']).name] != expected:
            raise ValueError(f'CT/PET/SEG mismatch for {case_id}')
    train_cases, valid_cases = [case_of(i) for i in train], [case_of(i) for i in valid]
    train_patients = {patient_of[c] for c in train_cases}
    valid_patients = {patient_of[c] for c in valid_cases}
    shared = train_patients & valid_patients
    print(f'Fold {fold}')
    print(f'  VALIDATION: {len(valid_patients)} unique patients, {len(valid_cases)} examinations')
    print(f'  TRAIN:      {len(train_patients)} unique patients, {len(train_cases)} examinations')
    print(f'  Patient intersection between train and validation: {len(shared)}')
    repeated = sorted({p for p in patient_of.values() if list(patient_of.values()).count(p) > 1})
    for patient in repeated:
        where = [('validation' if c in valid_cases else 'train') for c in patient_of if patient_of[c] == patient]
        print(f'  {patient}: {len(where)} examinations -> {where}')
    ok = (not shared and len(set(train_cases + valid_cases)) == len(cases) == len(train_cases) + len(valid_cases))
    if fold == 0:
        ok = ok and (len(valid_patients), len(valid_cases), len(train_patients), len(train_cases)) == (29, 30, 115, 115)
        ok = ok and sum(patient_of[c] == 'PETCT_15f4b7254f' for c in valid_cases) == 2
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET,
                        help='folder with imagesTr, labelsTr, cases.json and splits_patient_level.json')
    args = parser.parse_args()
    dataset = args.dataset_root.expanduser().resolve()
    print(f'Dataset root: {dataset}')
    if not args.verify_only:
        write_datalist(dataset)
    results = [verify(fold, dataset) for fold in range(N_FOLDS)]
    if not all(results):
        sys.exit('DATALIST CHECK FAILED - do not train')
    print('DATALIST OK')
