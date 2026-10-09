from pathlib import Path

import pytest

from scripts.train import Case, load_or_create_splits, fold_cases


def test_patient_separation_repeat_visits_and_immutable_manifest(tmp_path):
    cases = [Case(f'p{i}', 'visit1') for i in range(20)] + [Case('p0', 'visit2')]
    path = tmp_path / 'splits.json'
    manifest = load_or_create_splits(cases, path, seed=7)
    assert load_or_create_splits(cases, path, seed=7) == manifest
    outer = []
    for fold in range(5):
        roles = fold_cases(cases, manifest, fold)
        patients = [{c.patient for c in roles[r]} for r in ('train', 'validation', 'held_out')]
        assert all(patients)
        assert not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2])
        visits = [r for r, group in roles.items() for c in group if c.patient == 'p0']
        assert len(visits) == 2 and len(set(visits)) == 1
        outer.extend(c.id for c in roles['held_out'])
    assert sorted(outer) == sorted(c.id for c in cases)
    with pytest.raises(ValueError, match='manifest'):
        load_or_create_splits(cases, path, seed=8)
    with pytest.raises(ValueError, match='duplicate'):
        load_or_create_splits(cases + [cases[0]], tmp_path / 'duplicate.json', seed=7)
