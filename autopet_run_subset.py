"""Official pretrained autoPET III model on a fixed random subset of the lymphoma examinations.

Run inside WSL from the project root:
    ~/venvs/autopet3/bin/python autopet_run_subset.py --prepare-only   # select, write manifest, link inputs; no inference
    ~/venvs/autopet3/bin/python autopet_run_subset.py                  # the above, then inference
    ~/venvs/autopet3/bin/python autopet_run_subset.py --status         # which predictions exist

Selection: the patients of cases.json are sorted, 10 are drawn with random.Random(42), and each drawn
patient contributes one examination (the first by name if the patient has several). The result is
written to experiments/autopet3_subset10/manifest.json; a later run must reproduce it or it stops.

Inference is the same call as autopet_smoke_one_case.sh: nnUNetv2_predict_from_modelfolder with
the official defaults (folds 0-4 ensembled, mirroring TTA, step size 0.5). --continue_prediction
makes nnU-Net skip examinations whose prediction already exists, so an interrupted run can simply
be started again. The CT/SUV files are linked, not copied.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
DATASET_ROOT = Path('/home/eva25/data/lymphoma-baseline')
MODEL_DIR = (PROJECT / 'checkpoints' / 'autopet3' / 'Dataset222_AutoPETIII_2024'
             / 'autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3')
PREDICT = Path.home() / 'venvs' / 'autopet3' / 'bin' / 'nnUNetv2_predict_from_modelfolder'

N_CASES = 10
SEED = 42
NAME = f'autopet3_subset{N_CASES}'
EXPERIMENT = PROJECT / 'experiments' / NAME
MANIFEST = EXPERIMENT / 'manifest.json'
PRED_DIR = EXPERIMENT / 'predictions'
# Links to the inputs live on the Linux file system; the Windows drive cannot hold them reliably.
INPUT_DIR = Path.home() / 'autopet3_runs' / NAME / 'input'


def select_cases():
    cases = json.loads((DATASET_ROOT / 'cases.json').read_text(encoding='utf-8'))
    patients = sorted({c['patient'] for c in cases})
    chosen = random.Random(SEED).sample(patients, N_CASES)
    selected = [min((c for c in cases if c['patient'] == p), key=lambda c: c['exam']) for p in chosen]
    return sorted(selected, key=lambda c: c['case_id'])


def write_or_check_manifest(selected):
    manifest = {
        'description': 'Examinations predicted with the official pretrained autoPET III checkpoint. '
                       'Orientation only: the checkpoint was trained on data that includes them.',
        'seed': SEED,
        'n_cases': N_CASES,
        'selection': 'sorted patients of cases.json, random.Random(seed).sample, first examination per patient',
        'dataset_root': str(DATASET_ROOT),
        'baseline_results': 'experiments/unet_10ep/light_eval/light_metrics_all_folds.csv, rows where ImageID == case_id',
        'cases': [{'case_id': c['case_id'], 'patient': c['patient'], 'exam': c['exam'], 'baseline_fold': c['fold']}
                  for c in selected],
    }
    if MANIFEST.is_file():
        if json.loads(MANIFEST.read_text(encoding='utf-8'))['cases'] != manifest['cases']:
            sys.exit(f'The selection differs from the one stored in {MANIFEST}. Nothing was changed.')
    else:
        EXPERIMENT.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=4) + '\n', encoding='utf-8')
    return manifest['cases']


def link_inputs(cases):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in INPUT_DIR.iterdir():
        if not old.is_symlink():
            sys.exit(f'{old} is not a link created by this script. Remove it by hand.')
        old.unlink()
    for case in cases:
        for channel in ('0000', '0001'):  # 0000 = CT, 0001 = PET (SUV)
            source = DATASET_ROOT / 'imagesTr' / f"{case['case_id']}_{channel}.nii.gz"
            if not source.is_file():
                sys.exit(f'Missing: {source}')
            (INPUT_DIR / source.name).symlink_to(source)


def is_complete(path):
    """A prediction counts only if the whole gzip stream can be read."""
    try:
        with gzip.open(path, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False


def status(cases):
    done = 0
    for case in cases:
        path = PRED_DIR / f"{case['case_id']}.nii.gz"
        state = 'missing' if not path.is_file() else 'done' if is_complete(path) else 'CORRUPT'
        if state == 'CORRUPT':
            print(f'           delete it and run again: {path}')
        done += state == 'done'
        print(f"  {state:<8} {case['case_id']}")
    print(f'{done} of {len(cases)} predictions complete in {PRED_DIR}')
    return done == len(cases)


def main(args):
    cases = write_or_check_manifest(select_cases())
    print(f'Manifest: {MANIFEST}')
    if args.status:
        sys.exit(0 if status(cases) else 1)

    link_inputs(cases)
    print(f'Inputs:   {INPUT_DIR} ({2 * len(cases)} links)')
    status(cases)
    if args.prepare_only:
        return

    # torch >= 2.6 refuses nnU-Net checkpoints in torch.load unless this is set (see answers/20).
    environment = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    command = [str(PREDICT), '-i', str(INPUT_DIR), '-o', str(PRED_DIR), '-m', str(MODEL_DIR),
               '-npp', '1', '-nps', '1', '--continue_prediction']
    print('Running:', ' '.join(f'"{part}"' if ' ' in part else part for part in command), flush=True)
    code = subprocess.run(command, env=environment).returncode
    print()
    complete = status(cases)
    if code != 0 or not complete:
        sys.exit(f'Not finished (nnU-Net exit code {code}). Run the same command again to continue.')
    print('All predictions are complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='select and link the cases, do not run inference')
    mode.add_argument('--status', action='store_true', help='only report which predictions exist')
    main(parser.parse_args())
