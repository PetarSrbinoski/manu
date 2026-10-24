"""Review and pilot comparisons must use complete, matched evaluation inputs."""
from copy import deepcopy
import json

import pytest

from scripts.analysis_common import load_evaluation, match_evaluations


def evaluation(folder):
    folder.mkdir()
    row = {'case_id': 'p/e', 'patient': 'p', 'GT_TMTV_ml': 1.,
           'GT_SUVmax': 4., 'gt_lesions': 2, 'DSC': .5}
    source = {'case_id': 'p/e', 'patient': 'p',
              'reference': {'sha256': 'a' * 64}, 'pet': {'sha256': 'b' * 64},
              'prediction': {'sha256': 'c' * 64}}
    (folder / 'metrics.json').write_text(json.dumps([row]))
    (folder / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'cases': [source]}))
    return folder


def test_matching_evaluations_allow_different_predictions(tmp_path):
    a = load_evaluation(evaluation(tmp_path / 'a'))
    b = deepcopy(a)
    b[0]['p/e']['DSC'] = .7
    b[1]['p/e']['prediction']['sha256'] = 'd' * 64
    match_evaluations(a, b)


@pytest.mark.parametrize('problem', ['incomplete', 'empty', 'duplicate_metrics',
                                   'duplicate_sources', 'missing_source', 'patient'])
def test_invalid_evaluation_artifacts_are_rejected(tmp_path, problem):
    folder = evaluation(tmp_path / 'evaluation')
    rows = json.loads((folder / 'metrics.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    if problem == 'incomplete':
        manifest['status'] = 'running'
    elif problem == 'empty':
        rows = []
        manifest['cases'] = []
    elif problem == 'duplicate_metrics':
        rows *= 2
    elif problem == 'duplicate_sources':
        manifest['cases'] *= 2
    elif problem == 'missing_source':
        manifest['cases'] = []
    elif problem == 'patient':
        manifest['cases'][0]['patient'] = 'other'
    (folder / 'metrics.json').write_text(json.dumps(rows))
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_evaluation(folder)


@pytest.mark.parametrize('field', ['cases', 'patient', 'GT_SUVmax', 'GT_TMTV_ml',
                                 'gt_lesions', 'pet', 'reference'])
def test_comparisons_reject_changed_examinations_and_references(tmp_path, field):
    a = load_evaluation(evaluation(tmp_path / 'a'))
    b = deepcopy(a)
    if field == 'cases':
        b[0]['q/e'] = b[0].pop('p/e')
        b[1]['q/e'] = b[1].pop('p/e')
    elif field in ('pet', 'reference'):
        b[1]['p/e'][field]['sha256'] = 'd' * 64
    else:
        b[0]['p/e'][field] = 'other' if field == 'patient' else 99
    with pytest.raises(ValueError):
        match_evaluations(a, b)
