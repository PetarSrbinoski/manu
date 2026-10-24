"""Read-only experiment inputs and isolated analysis report publication."""
from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from scripts.training_data import Case, FILES, fold_cases
from scripts.training_runtime import file_hash, identity


def load_splits(path: Path):
    manifest = json.loads(path.read_text())
    body = {k: v for k, v in manifest.items() if k != 'identity'}
    if identity(body) != manifest.get('identity'):
        raise ValueError('Split manifest identity does not match its contents')
    cases = [Case(c['patient'], c['exam'], tuple(c.get('hashes', ()))) for c in manifest['cases']]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError('Duplicate examination in split manifest')
    for case in cases:
        if any(not value or value in ('.', '..') or '/' in value or '\\' in value
               for value in (case.patient, case.exam)):
            raise ValueError('Unsafe patient/examination identifier')
    if len(manifest['folds']) != 5:
        raise ValueError('Expected five folds')
    patients = {c.patient for c in cases}
    if (set(manifest['patient_folds']) != patients
            or any(v not in range(5) for v in manifest['patient_folds'].values())):
        raise ValueError('Invalid patient-fold assignments')
    for fold in range(5):
        roles = fold_cases(cases, manifest, fold)
        expected = {c.patient for c in cases if manifest['patient_folds'][c.patient] == fold}
        if expected != {c.patient for c in roles['held_out']}:
            raise ValueError('Outer assignments disagree with role lists')
    return cases, manifest


def verify_case(case, root):
    if len(case.hashes) != 3:
        raise ValueError('Use an audited split manifest with three source hashes per examination')
    for name, expected in zip(FILES, case.hashes):
        if file_hash(root / case.id / name) != expected:
            raise ValueError(f'Source checksum differs: {case.id}/{name}')


def new_report(output: Path, protected=()):
    output = output.resolve()
    for source in protected:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Analysis output must be separate from input files/directories')
    output.mkdir(parents=True, exist_ok=False)
    return output


def provenance(**extra):
    names = ('numpy', 'nibabel', 'monai', 'torch', 'connected-components-3d')
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).parent
    return {'command': sys.argv, 'python': platform.python_version(), 'packages': packages,
            'source_sha256': {p.name: file_hash(p) for p in sorted(source.glob('*.py'))},
            'config_sha256': file_hash(source.parent / 'config.py'), **extra}


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_evaluation(folder):
    """Load a completed evaluation without silently collapsing case identities.

    Stored source hashes describe the inputs at evaluation time. Consumers that
    read those source images must additionally verify their current contents.
    """
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows = json.loads((folder / 'metrics.json').read_text())
    inputs = manifest['cases']
    if (manifest['status'] != 'complete' or not rows
            or len(rows) != len({r['case_id'] for r in rows})
            or len(inputs) != len({r['case_id'] for r in inputs})):
        raise ValueError('Evaluation must be complete with unique case identities')
    metrics = {r['case_id']: r for r in rows}
    sources = {r['case_id']: r for r in inputs}
    if metrics.keys() != sources.keys():
        raise ValueError('Evaluation metrics and manifest case IDs differ')
    for case, row in metrics.items():
        if (row['patient'] != sources[case]['patient']
                or not case.startswith(row['patient'] + '/')):
            raise ValueError('Evaluation patient/case identities differ')
    return metrics, sources


def match_evaluations(reference, candidate):
    """Require the same examinations, reference metrics and reference/PET inputs."""
    a, inputs_a = reference
    b, inputs_b = candidate
    if a.keys() != b.keys():
        raise ValueError('Evaluation case IDs differ')
    for case in a:
        if any(a[case][field] != b[case][field]
               for field in ('patient', 'GT_SUVmax', 'GT_TMTV_ml', 'gt_lesions')):
            raise ValueError('Reference case metrics differ')
        if any(inputs_a[case][field]['sha256'] != inputs_b[case][field]['sha256']
               for field in ('pet', 'reference')):
            raise ValueError('Reference/PET input hashes differ')
