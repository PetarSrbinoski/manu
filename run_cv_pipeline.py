"""Sequential 5-fold train + held-out evaluation with the unmodified Microsoft pipeline.

Run inside WSL from anywhere:
    ~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py            # start or resume
    ~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --dry-run  # checks and plan only
    ~/venvs/lymphoma_seg/bin/python run_cv_pipeline.py --train-inference-only  # skip the metric stages

Per fold, in order: train -> inference -> metrics -> lesion_measures. A stage is skipped only when
its completion marker exists and its outputs still pass the checks. Any failure stops everything.
Nothing is ever deleted; leftovers of an interrupted stage are only moved aside on request.
"""

#automation wrapper shto avtomatski go vodi celiot 5 fold experiment
#koristejki go originalniot Microsoft training/inference kod,
#proveruva deka podatocite i rezultatite se validni,
#i zachuvuva logs i reproducibility information.
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
MS_REPO = PROJECT / 'lymphoma-segmentation-dnn'
MS_SPLIT = MS_REPO / 'data_split'
TRAIN_CSV = MS_SPLIT / 'train_filepaths.csv'

# the only training arguments passed; everything else stays at Microsoft's defaults
EPOCHS = 10
NUM_WORKERS = 6
# Microsoft defaults that the paths and the checkpoint logic depend on (not passed, only mirrored)
NETWORK = 'unet'
PATCH = 192
VAL_INTERVAL = 2

N_FOLDS = 5
STAGES = ['train', 'inference', 'metrics', 'lesion_measures']
TEST_COLUMNS = ['ImageID', 'CTPATH', 'PTPATH', 'GTPATH']


class Stop(Exception):
    pass


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.dry = args.dry_run
        self.stages = STAGES[:2] if args.train_inference_only else STAGES
        self.dataset = args.dataset_root.expanduser().resolve()
        self.root = PROJECT / 'experiments' / args.experiment
        self.work = self.root / 'work'  # working directory of every Microsoft script
        self.results = self.work / 'results'  # Microsoft's RESULTS_FOLDER ('results' relative to the working directory)
        self.python = Path(sys.executable)
        self.torchrun = self.python.parent / 'torchrun'
        self.child = None

    # ---------- logging ----------
    def log(self, message, tag='pipeline', extra_log=None):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{tag}] {message}"
        print(line, flush=True)
        if not self.dry:
            for path in filter(None, [self.root / 'pipeline.log', extra_log]):
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')

    def run(self, tag, command, stage_log):
        self.log('command: ' + ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in command), tag, stage_log)
        self.log(f'working directory: {self.work}', tag, stage_log)
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.child = subprocess.Popen([str(c) for c in command], cwd=self.work, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors='replace', start_new_session=True)
        for line in self.child.stdout:
            self.log(line.rstrip('\n'), tag, stage_log)
        code = self.child.wait()
        self.child = None
        self.log(f'exit code: {code}', tag, stage_log)
        return code

    def kill_child(self, *_):
        if self.child is not None:
            os.killpg(self.child.pid, signal.SIGTERM)
        raise Stop('interrupted by signal; the running stage was terminated and is NOT marked complete')

    # ---------- paths ----------
    def code(self, fold):
        return f'{NETWORK}_fold{fold}_randcrop{PATCH}'

    def out(self, kind, fold):
        return self.results / kind / f'fold{fold}' / NETWORK / self.code(fold)

    def marker(self, fold, stage):
        return self.root / 'state' / f'fold{fold}.{stage}.done'

    def stage_outputs(self, fold, stage):
        return {'train': [self.out('logs', fold), self.out('models', fold)],
                'inference': [self.out('predictions', fold)],
                'metrics': [self.out('test_metrics', fold)],
                'lesion_measures': [self.out('test_lesion_measures_no_dmax', fold)]}[stage]

    def commands(self, fold):
        seg = MS_REPO / 'segmentation'
        return {
            'train': [self.torchrun, '--standalone', '--nproc_per_node=1', seg / 'trainddp.py',
                      '--fold', fold, '--epochs', EPOCHS, '--num_workers', NUM_WORKERS],
            'inference': [self.python, seg / 'inference.py', '--fold', fold],
            'metrics': [self.python, seg / 'calculate_test_metrics.py', '--fold', fold],
            'lesion_measures': [self.python, PROJECT / 'cv_lesion_measures.py', '--ms-repo', MS_REPO,
                                '--test-csv', self.root / 'test_lists' / f'fold{fold}_test_filepaths.csv',
                                '--pred-dir', self.out('predictions', fold),
                                '--out-csv', self.out('test_lesion_measures_no_dmax', fold) / 'testlesionmeasures_no_dmax.csv'],
        }

    # ---------- datalist ----------
    def load_datalist(self):
        import pandas as pd
        self.df = pd.read_csv(TRAIN_CSV)
        self.cases = {c['case_id']: c for c in json.loads((self.dataset / 'cases.json').read_text(encoding='utf-8'))}

    def check_datalist(self):
        problems = []
        df, cases = self.df, self.cases
        if len(df) != 145 or df.ImageID.nunique() != 145 or set(df.ImageID) != set(cases):
            problems.append('train_filepaths.csv does not list exactly the 145 cases of cases.json')
            return problems
        if any(cases[i]['fold'] != f for i, f in zip(df.ImageID, df.FoldID)):
            problems.append('FoldID in train_filepaths.csv differs from cases.json')
        for row in df.itertuples():
            expected = [self.dataset / 'imagesTr' / f'{row.ImageID}_0000.nii.gz',
                        self.dataset / 'imagesTr' / f'{row.ImageID}_0001.nii.gz',
                        self.dataset / 'labelsTr' / f'{row.ImageID}.nii.gz']
            if [Path(row.CTPATH), Path(row.PTPATH), Path(row.GTPATH)] != expected:
                problems.append(f'paths of {row.ImageID} are not the expected files under {self.dataset}')
                break
            if not all(p.is_file() for p in expected):
                problems.append(f'missing file for {row.ImageID}')
                break
        patient = df.ImageID.map(lambda i: cases[i]['patient'])
        for fold in range(N_FOLDS):
            shared = set(patient[df.FoldID == fold]) & set(patient[df.FoldID != fold])
            if shared:
                problems.append(f'fold {fold}: patients in both training and held-out set: {sorted(shared)}')
        sizes = [(patient[df.FoldID == f].nunique(), int((df.FoldID == f).sum())) for f in range(N_FOLDS)]
        if sizes != [(29, 30), (29, 29), (29, 29), (29, 29), (28, 28)]:
            problems.append(f'unexpected held-out fold sizes (patients, exams): {sizes}')
        return problems

    def held_out(self, fold):
        return self.df[self.df.FoldID == fold][TEST_COLUMNS].sort_values('ImageID')

    def write_test_lists(self, fold):
        """Microsoft's inference and metric scripts evaluate whatever test_filepaths.csv lists.
        inference.py reads <repo>/data_split/, the metric scripts read ./../data_split/ of the working directory."""
        test = self.held_out(fold)
        for path in [MS_SPLIT / 'test_filepaths.csv', self.root / 'data_split' / 'test_filepaths.csv',
                     self.root / 'test_lists' / f'fold{fold}_test_filepaths.csv']:
            path.parent.mkdir(parents=True, exist_ok=True)
            test.to_csv(path, index=False)
        self.log(f'test list for fold {fold} written: {len(test)} held-out examinations', f'fold{fold}')

    # ---------- checks of finished stages ----------
    def check_train(self, fold):
        import pandas as pd
        logs, models = self.out('logs', fold), self.out('models', fold)
        trainlog, validlog = logs / 'trainlog_gpu0.csv', logs / 'validlog_gpu0.csv'
        if not trainlog.is_file() or not validlog.is_file():
            return 'training or validation log is missing'
        n_train, valid = len(pd.read_csv(trainlog)), pd.read_csv(validlog)
        if n_train != EPOCHS:
            return f'training log has {n_train} epochs, expected {EPOCHS}'
        if len(valid) != EPOCHS // VAL_INTERVAL or valid.Metric.isna().any():
            return f'validation log has {len(valid)} valid rows, expected {EPOCHS // VAL_INTERVAL}'
        expected = {f'model_ep={epoch:04d}.pth' for epoch in range(VAL_INTERVAL, EPOCHS + 1, VAL_INTERVAL)}
        found = {p.name for p in models.glob('*.pth')}
        if found != expected:
            return f'checkpoints found {sorted(found)}, expected {sorted(expected)}'
        return None

    def selected_epoch(self, fold):
        """The rule hard-coded in Microsoft's inference.py: best_epoch = 2*(argmax(validation Metric) + 1)."""
        import pandas as pd
        valid = pd.read_csv(self.out('logs', fold) / 'validlog_gpu0.csv')
        return 2 * (int(valid.Metric.values.argmax()) + 1), float(valid.Metric.max())

    def check_inference(self, fold):
        expected = sorted(f'{i}.nii.gz' for i in self.held_out(fold).ImageID)
        found = sorted(p.name for p in self.out('predictions', fold).glob('*'))
        if found != expected:
            missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
            return f'predictions do not match the held-out list: {len(missing)} missing {missing[:3]}, {len(extra)} unexpected {extra[:3]}'
        # Microsoft's metric scripts pair sorted(prediction paths) with sorted(ground-truth paths) by position
        gt_sorted = [Path(p).name for p in sorted(self.held_out(fold).GTPATH)]
        pred_sorted = [Path(p).name for p in sorted(str(p) for p in self.out('predictions', fold).glob('*.nii.gz'))]
        if gt_sorted != pred_sorted:
            return 'sorted prediction files and sorted ground-truth files are not in the same case order'
        return None

    def check_table(self, fold, path):
        import pandas as pd
        if not path.is_file():
            return f'{path.name} is missing'
        table = pd.read_csv(path)
        if sorted(table.PatientID) != sorted(self.held_out(fold).ImageID):
            return f'{path.name} does not contain exactly the held-out cases of fold {fold}'
        return None

    def check_stage(self, fold, stage):
        if stage == 'train':
            return self.check_train(fold)
        if stage == 'inference':
            return self.check_inference(fold)
        if stage == 'metrics':
            return self.check_table(fold, self.out('test_metrics', fold) / 'testmetrics.csv')
        return self.check_table(fold, self.out('test_lesion_measures_no_dmax', fold) / 'testlesionmeasures_no_dmax.csv')

    def prediction_overview(self, fold):
        """Voxel and connected-component counts per prediction (connectivity 18, as in Microsoft's metrics).
        Microsoft's lesion-detection code loops over predicted x true lesions, so these numbers predict its run time."""
        import cc3d
        import pandas as pd
        import SimpleITK as sitk
        rows = []
        for case in self.held_out(fold).itertuples():
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(self.out('predictions', fold) / f'{case.ImageID}.nii.gz')))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(case.GTPATH))
            if pred.shape != gt.shape:
                raise Stop(f'prediction and ground truth of {case.ImageID} have different shapes: {pred.shape} vs {gt.shape}')
            rows.append({'ImageID': case.ImageID, 'pred_voxels': int((pred > 0).sum()), 'gt_voxels': int((gt > 0).sum()),
                         'pred_components': int(cc3d.connected_components(pred > 0, connectivity=18, return_N=True)[1]),
                         'gt_components': int(cc3d.connected_components(gt > 0, connectivity=18, return_N=True)[1])})
        table = pd.DataFrame(rows)
        table.to_csv(self.root / 'state' / f'fold{fold}.prediction_overview.csv', index=False)
        self.log(f'predicted components per case: median {int(table.pred_components.median())}, max {int(table.pred_components.max())}; '
                 f'true lesions per case: median {int(table.gt_components.median())}, max {int(table.gt_components.max())}', f'fold{fold}')

    # ---------- preflight ----------
    def git(self, *arguments):
        # the repo was cloned by Windows git with CRLF checkout; without autocrlf WSL git reports every file as modified
        return subprocess.run(['git', '-C', str(MS_REPO), '-c', 'core.autocrlf=true', *arguments],
                              capture_output=True, text=True).stdout

    def running_processes(self):
        """Python processes (other than this one) that are executing one of the training / evaluation scripts."""
        scripts = {'trainddp.py', 'inference.py', 'calculate_test_metrics.py', 'generate_lesion_measures.py',
                   'cv_lesion_measures.py', 'run_cv_pipeline.py', 'torchrun'}
        found = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:
                continue
            if Path(argv[0]).name.startswith('python') and scripts & {Path(a).name for a in argv[1:]}:
                found.append(f'pid {entry.name}: ' + ' '.join(argv).strip())
        return found

    def preflight(self):
        problems = []
        if os.name != 'posix' or not str(PROJECT).startswith('/'):
            return ['this script must be started inside WSL (Linux), not from Windows']
        if not self.torchrun.is_file():
            problems.append(f'torchrun not found next to {self.python}; start the script with ~/venvs/lymphoma_seg/bin/python')
        for module, package in [('torch', 'torch'), ('monai', 'monai'), ('pandas', 'pandas'),
                                ('SimpleITK', 'SimpleITK (needed by inference.py and the metric scripts)'),
                                ('cc3d', 'connected-components-3d (needed by the metric scripts)')]:
            try:
                __import__(module)
            except ImportError:
                problems.append(f'Python package missing: {package}')
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append('PyTorch does not see a CUDA GPU')
        except ImportError:
            pass
        status = self.git('status', '--porcelain', '--untracked-files=no')
        if status.strip() or self.git('diff', 'HEAD').strip():
            problems.append('Microsoft repository has modified tracked files:\n' + status)
        if not TRAIN_CSV.is_file():
            problems.append(f'{TRAIN_CSV} is missing; run make_datalist.py first')
        else:
            try:
                self.load_datalist()
                problems += self.check_datalist()
            except ImportError:
                pass
        running = self.running_processes()
        if running:
            problems.append('another training / evaluation / pipeline process is running:\n  ' + '\n  '.join(running))
        return problems

    # ---------- manifest ----------
    def manifest(self):
        import monai
        import numpy
        import pandas
        import torch
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        versions = {'python': sys.version.split()[0], 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                    'monai': monai.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__}
        for module in ('SimpleITK', 'cc3d', 'itk', 'nibabel'):
            try:
                versions[module] = getattr(__import__(module), '__version__', 'unknown')
            except ImportError:
                versions[module] = 'not installed'
        return {
            'experiment': self.args.experiment,
            'microsoft_repo': {'path': str(MS_REPO), 'remote': self.git('remote', 'get-url', 'origin').strip(),
                               'commit': self.git('rev-parse', 'HEAD').strip(),
                               'status_tracked_files': self.git('status', '--porcelain', '--untracked-files=no'),
                               'diff_against_HEAD': self.git('diff', 'HEAD'),
                               'untracked': self.git('status', '--porcelain').splitlines()},
            'dataset': {'root': str(self.dataset), 'sha256_cases_json': sha(self.dataset / 'cases.json'),
                        'sha256_splits_patient_level_json': sha(self.dataset / 'splits_patient_level.json'),
                        'sha256_train_filepaths_csv': sha(TRAIN_CSV),
                        'held_out_examinations_per_fold': {str(f): int((self.df.FoldID == f).sum()) for f in range(N_FOLDS)}},
            'training_arguments': {'passed': {'--epochs': EPOCHS, '--num_workers': NUM_WORKERS},
                                   'microsoft_defaults_not_passed': {
                                       '--network-name': 'unet', '--input-patch-size': 192, '--train-bs': 1, '--cache-rate': 0.1,
                                       '--lr': 2e-4, '--wd': 1e-5, '--val-interval': 2, '--sw-bs': 2}},
            'commands': {f'fold{f}': {stage: ' '.join(str(c) for c in command) for stage, command in self.commands(f).items()}
                         for f in range(N_FOLDS)},
            'working_directory_of_all_commands': str(self.work),
            'software': versions,
            'gpu': torch.cuda.get_device_name(0),
        }

    def write_manifest(self):
        path, current = self.root / 'manifest.json', self.manifest()
        entry = {'started': datetime.now().isoformat(timespec='seconds'), 'folds_requested': self.args.folds,
                 'stages_requested': self.stages, 'software': current['software']}
        if path.is_file():
            stored = json.loads(path.read_text(encoding='utf-8'))
            for key in ('dataset', 'training_arguments', 'commands'):
                if stored[key] != current[key]:
                    raise Stop(f'manifest.json of this experiment has a different "{key}" than the current setup; '
                               f'use a new --experiment name instead of mixing runs')
            if stored['microsoft_repo']['commit'] != current['microsoft_repo']['commit']:
                raise Stop('Microsoft repository commit differs from the one recorded in manifest.json')
            stored['runs'].append(entry)
        else:
            stored = dict(current, created=entry['started'], runs=[entry])
        path.write_text(json.dumps(stored, indent=2), encoding='utf-8')

    # ---------- execution ----------
    def archive(self, fold, stage, tag):
        target = self.root / 'archived_partial' / f'{datetime.now():%Y%m%d_%H%M%S}_fold{fold}_{stage}'
        for path in self.stage_outputs(fold, stage):
            if path.exists():
                destination = target / path.relative_to(self.results)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
                self.log(f'moved leftover {path} -> {destination}', tag)

    def do_stage(self, fold, stage):
        tag = f'fold{fold}/{stage}'
        marker = self.marker(fold, stage)
        leftovers = [p for p in self.stage_outputs(fold, stage) if p.exists() and any(p.iterdir())]
        if marker.is_file():
            problem = self.check_stage(fold, stage)
            if problem:
                raise Stop(f'{tag}: completion marker exists but the outputs fail the check: {problem}')
            self.log('already complete, skipping', tag)
            return
        if leftovers:
            if fold not in self.args.archive_partial:
                raise Stop(f'{tag}: outputs exist without a completion marker (interrupted or foreign run): '
                           f'{[str(p) for p in leftovers]}. Nothing was touched. To move them aside and redo this stage, '
                           f'restart with --archive-partial {fold}')
            if self.dry:
                self.log('WOULD move leftover outputs aside and run', tag)
                return
            self.archive(fold, stage, tag)
        if self.dry:
            self.log('WOULD RUN: ' + ' '.join(str(c) for c in self.commands(fold)[stage]), tag)
            return
        if stage != 'train':
            self.write_test_lists(fold)
            for previous in STAGES[:STAGES.index(stage)]:
                problem = self.check_stage(fold, previous)
                if problem:
                    raise Stop(f'{tag}: earlier stage {previous} no longer passes its check: {problem}')
        if stage == 'inference':
            epoch, dice = self.selected_epoch(fold)
            self.log(f'Microsoft rule selects checkpoint model_ep={epoch:04d}.pth (validation mean DSC {dice:.4f})', tag)
        stage_log = self.root / 'logs' / f'fold{fold}_{stage}.log'
        stage_log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now()
        code = self.run(tag, self.commands(fold)[stage], stage_log)
        if code != 0:
            raise Stop(f'{tag}: command failed with exit code {code}; see {stage_log}')
        problem = self.check_stage(fold, stage)
        if problem:
            raise Stop(f'{tag}: command exited normally but the outputs fail the check: {problem}')
        if stage == 'inference':
            self.prediction_overview(fold)
        info = {'fold': fold, 'stage': stage, 'started': started.isoformat(timespec='seconds'),
                'finished': datetime.now().isoformat(timespec='seconds'),
                'command': [str(c) for c in self.commands(fold)[stage]]}
        if stage != 'train':
            info['selected_epoch'], info['validation_dsc_of_selected_epoch'] = self.selected_epoch(fold)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(info, indent=2), encoding='utf-8')
        self.log(f'COMPLETE in {(datetime.now() - started).total_seconds() / 60:.1f} min', tag)

    def summarize(self):
        import pandas as pd
        folder = self.root / 'summary'
        folder.mkdir(exist_ok=True)
        index, metrics, measures = [], [], []
        patient_of = {i: c['patient'] for i, c in self.cases.items()}
        for fold in range(N_FOLDS):
            done = {stage: self.marker(fold, stage).is_file() for stage in STAGES}
            row = {'fold': fold, 'held_out_patients': self.held_out(fold).ImageID.map(patient_of).nunique(),
                   'held_out_examinations': len(self.held_out(fold)), **{f'{stage}_complete': done[stage] for stage in STAGES}}
            if done['train']:
                row['selected_epoch'], row['validation_dsc_of_selected_epoch'] = self.selected_epoch(fold)
                row['checkpoint'] = str(self.out('models', fold) / f"model_ep={row['selected_epoch']:04d}.pth")
                row['training_logs'] = str(self.out('logs', fold))
            if done['inference']:
                row['predictions'] = str(self.out('predictions', fold))
            if done['metrics']:
                row['test_metrics'] = str(self.out('test_metrics', fold) / 'testmetrics.csv')
                metrics.append(pd.read_csv(row['test_metrics']).assign(fold=fold))
            if done['lesion_measures']:
                row['lesion_measures'] = str(self.out('test_lesion_measures_no_dmax', fold) / 'testlesionmeasures_no_dmax.csv')
                measures.append(pd.read_csv(row['lesion_measures']).assign(fold=fold))
            index.append(row)
        pd.DataFrame(index).to_csv(folder / 'index.csv', index=False)
        lines = [f'# {self.args.experiment}: cross-validation summary', '', f'Written {datetime.now():%Y-%m-%d %H:%M:%S}.', '',
                 f'Folds with all stages complete: {[r["fold"] for r in index if all(r[f"{s}_complete"] for s in STAGES)]}', '',
                 'Result locations per fold: `summary/index.csv`.', '']
        for name, tables in [('all_folds_testmetrics.csv', metrics), ('all_folds_lesion_measures_no_dmax.csv', measures)]:
            if not tables:
                continue
            table = pd.concat(tables, ignore_index=True)
            if table.PatientID.duplicated().any() or any(self.cases[i]['fold'] != f for i, f in zip(table.PatientID, table.fold)):
                raise Stop(f'{name}: a case appears twice or under a fold in which it was not held out')
            table.insert(1, 'patient', table.PatientID.map(patient_of))
            table.to_csv(folder / name, index=False)
            lines += [f'`summary/{name}`: {len(table)} held-out examinations from folds {sorted(table.fold.unique().tolist())}, '
                      f'each predicted only by the model of the fold in which it was held out.', '']
        if metrics:
            table = pd.concat(metrics, ignore_index=True)
            lines += ['## Per-examination segmentation metrics (Microsoft `calculate_test_metrics.py`), plain descriptive statistics', '',
                      '| fold | examinations | DSC mean | DSC median | FPV median (ml) | FNV median (ml) | TP/FP/FN criterion 1 | criterion 2 | criterion 3 |',
                      '|---|---|---|---|---|---|---|---|---|']
            groups = [(str(f), g) for f, g in table.groupby('fold')] + [('all', table)]
            for label, g in groups:
                counts = [f"{int(g[f'TP_C{c}'].sum())}/{int(g[f'FP_C{c}'].sum())}/{int(g[f'FN_C{c}'].sum())}" for c in (1, 2, 3)]
                lines.append(f"| {label} | {len(g)} | {g.DSC.mean():.4f} | {g.DSC.median():.4f} | {g.FPV.median():.2f} | "
                             f"{g.FNV.median():.2f} | {counts[0]} | {counts[1]} | {counts[2]} |")
            lines += ['', 'TP/FP/FN are lesion counts summed over examinations. The "all" row mixes models from different folds.', '']
        (folder / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
        self.log(f'summary written to {folder}')

    def main(self):
        if not self.dry:
            self.root.mkdir(parents=True, exist_ok=True)
        self.log(f'experiment "{self.args.experiment}" at {self.root}' + (' (DRY RUN: nothing is executed or written)' if self.dry else ''))
        problems = self.preflight()
        for problem in problems:
            self.log('PREFLIGHT PROBLEM: ' + problem)
        if problems and not self.dry:
            raise Stop(f'{len(problems)} preflight problem(s); nothing was started')
        if not hasattr(self, 'df'):
            raise Stop('datalist could not be loaded; cannot show the plan')
        lock = self.root / 'pipeline.lock'
        if not self.dry:
            if lock.is_file() and Path(f'/proc/{lock.read_text().strip()}').exists():
                raise Stop(f'another pipeline instance (pid {lock.read_text().strip()}) is using this experiment')
            lock.write_text(str(os.getpid()))
        try:
            if not self.dry:
                self.work.mkdir(exist_ok=True)
                self.write_manifest()
            self.log(f'Microsoft repo commit {self.git("rev-parse", "HEAD").strip()}, tracked files unmodified: '
                     f'{not self.git("status", "--porcelain", "--untracked-files=no").strip()}')
            for fold in self.args.folds:
                for stage in self.stages:
                    self.do_stage(fold, stage)
                if not self.dry:
                    self.summarize()
            if not self.dry:
                self.log(f'ALL REQUESTED FOLDS COMPLETE (stages: {", ".join(self.stages)})')
            elif problems:
                raise Stop(f'dry run finished, but {len(problems)} preflight problem(s) above would block a real start')
        finally:
            if not self.dry:
                (self.root / 'pipeline.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default=f'unet_{EPOCHS}ep', help='name of the folder under experiments/')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/eva25/data/lymphoma-baseline'))
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)), choices=range(N_FOLDS))
    parser.add_argument('--archive-partial', type=int, nargs='+', default=[], metavar='FOLD',
                        help='for these folds, move outputs of an interrupted stage to archived_partial/ and redo the stage')
    parser.add_argument('--train-inference-only', action='store_true',
                        help='run only train -> inference per fold; the metrics and lesion_measures stages are not touched')
    parser.add_argument('--dry-run', action='store_true', help='run all checks and print the plan; execute and write nothing')
    pipeline = Pipeline(parser.parse_args())
    signal.signal(signal.SIGTERM, pipeline.kill_child)
    signal.signal(signal.SIGINT, pipeline.kill_child)
    try:
        pipeline.main()
    except Stop as stop:
        pipeline.log(f'PIPELINE STOPPED: {stop}')
        sys.exit(1)
