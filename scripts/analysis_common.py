"""Read-only experiment inputs and isolated analysis report publication."""
from __future__ import annotations

import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from scripts.training_data import Case, FILES, fold_cases
from scripts.training_runtime import atomic_json, file_hash, identity


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


def case_record(case):
    return asdict(case)
