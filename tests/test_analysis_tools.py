"""CPU-only scientific contract tests. Run with unittest; pytest also discovers them."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from scripts.segmentation_metrics import metrics, components, size_bin, summarize
from scripts.evaluate_segmentation import read_case, validate_entries, evaluate, run_entries
from scripts.analysis_common import load_splits, new_report
from scripts.training_data import Case, load_or_create_splits


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((9, 9, 9), np.uint8)
        self.pred = self.gt.copy()
        self.pet = np.ones(self.gt.shape, np.float32) * 3

    def score(self):
        return metrics(self.gt, self.pred, self.pet, 0.024)

    def test_empty_conventions(self):
        r, lesions = self.score()
        self.assertEqual(r['DSC'], 1)
        self.assertIsNone(r['recall_C1'])
        self.assertIsNone(r['precision_C1'])
        self.assertEqual(lesions, [])
        self.gt[1, 1, 1] = 1
        r, _ = self.score()
        self.assertEqual((r['DSC'], r['FN_C1'], r['FNV_ml'], r['Pred_SUVmax']), (0, 1, .024, 0))
        self.pred[5, 5, 5] = 1
        self.gt[:] = 0
        r, _ = self.score()
        self.assertEqual((r['FP_C1'], r['FPV_ml']), (1, .024))
        self.assertIsNone(r['TMTV_ml_relative_error'])

    def test_known_volume_and_partial_component_errors(self):
        self.gt[1:3, 1:3, 1:3] = 1  # 8 voxels = .192 mL
        self.pred[1, 1, 1] = 1
        self.pred[6, 6, 6] = 1
        self.pet[1, 1, 1] = 9
        self.pet[6, 6, 6] = 12
        r, lesions = self.score()
        self.assertAlmostEqual(r['DSC'], .2)
        self.assertAlmostEqual(r['GT_TMTV_ml'], .192)
        self.assertEqual(r['FNV_ml'], 0)
        self.assertAlmostEqual(r['voxel_FN_ml'], .168)
        self.assertAlmostEqual(r['FPV_ml'], .024)
        self.assertEqual((r['GT_SUVmax'], r['Pred_SUVmax']), (9, 12))
        self.assertEqual((r['TP_C1'], r['FP_C1'], r['FN_C1']), (1, 1, 0))
        self.assertEqual((r['TP_C2'], r['FP_C2'], r['FN_C2']), (0, 2, 1))
        self.assertEqual(lesions[0]['size_bin'], 'small_lt1ml')

    def test_connectivity18_and_memory_order(self):
        self.gt[1, 1, 1] = self.gt[2, 2, 1] = 1  # edge joins
        self.gt[3, 3, 2] = 1  # corner does not
        lab, count = components(self.gt)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(lab, components(np.asfortranarray(self.gt))[0])

    def test_split_and_suvmax_tie(self):
        self.gt[1:6, 2, 2] = 1
        self.pred[1:3, 2, 2] = self.pred[4:6, 2, 2] = 1
        r, _ = self.score()
        self.assertEqual((r['TP_C1'], r['FN_C1']), (2, 0))
        self.assertEqual((r['TP_C3'], r['FP_C3']), (1, 1))  # first C-order maximum

    def test_merge_best_iou_tie(self):
        self.gt[1, 1, 1] = self.gt[1, 1, 3] = 1
        self.pred[1, 1, 1:4] = 1
        r, lesions = self.score()
        self.assertEqual((r['TP_C1'], r['FN_C1']), (1, 0))
        self.assertEqual((r['TP_C3'], r['FN_C3']), (1, 1))
        self.assertEqual([l['found_C3'] for l in lesions], [1, 0])

    def test_iou_half_and_zero_suv(self):
        self.gt[1:3, 1, 1] = 1
        self.pred[1, 1, 1] = 1
        self.pet[:] = 0
        r, _ = self.score()
        self.assertEqual((r['TP_C2'], r['TP_C3']), (1, 1))

    def test_invalid_values_rejected(self):
        for bad in (np.nan, 2):
            self.pred = self.pred.astype(float)
            self.pred[1, 1, 1] = bad
            with self.assertRaises(ValueError):
                self.score()
        self.pred[:] = 0
        self.pet[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            self.score()

    def test_size_boundaries_and_pooled_counts(self):
        self.assertEqual([size_bin(x) for x in (.999, 1, 9.999, 10)],
                         ['small_lt1ml', 'medium_1to10ml', 'medium_1to10ml', 'large_ge10ml'])
        self.gt[1, 1, 1] = 1
        r, _ = self.score()
        report = summarize([{'patient': 'p', **r}, {'patient': 'p', **r}])
        self.assertEqual(report['patients'], 1)
        self.assertEqual(report['pooled_detection']['C1']['reference'], 2)
        self.assertEqual(report['pooled_detection']['C1']['size_strata']['small_lt1ml']['recall'], 0)

    def test_matches_pinned_upstream_golden_values(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/microsoft_metrics_golden.json').read_text())
        comparisons = 0
        for case in fixture['cases']:
            seed = case['seed']
            rng = np.random.default_rng(seed)
            gt = np.asfortranarray((rng.random((6, 6, 6)) > .94).astype(np.uint8))
            pred = np.asfortranarray((rng.random((6, 6, 6)) > .94).astype(np.uint8))
            gt[0, 0, 0] = pred[0, 0, 0] = 0
            pet = rng.random((6, 6, 6)) * 20 + .01
            if seed < 4:
                pred = gt.copy(order='F')
            actual, _ = metrics(gt, pred, pet, .024)
            for key, expected in case['metrics'].items():
                with self.subTest(seed=seed, metric=key):
                    self.assertAlmostEqual(actual[key], expected, places=10)
                comparisons += 1
        self.assertEqual(comparisons, 192)


class InputTests(unittest.TestCase):
    def test_native_geometry_units_and_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([2., 3., 4., 1.])
            mask = np.zeros((5, 5, 5), np.float32)
            mask[2, 2, 2] = 1
            for name, arr in [('gt', mask), ('pred', mask), ('pet', mask * 7)]:
                image = nib.Nifti1Image(arr, affine)
                image.header.set_xyzt_units('mm')
                nib.save(image, root / f'{name}.nii.gz')
            entry = {'case_id': 'p/e', 'patient': 'p', 'reference': 'gt.nii.gz',
                     'prediction': 'pred.nii.gz', 'pet': 'pet.nii.gz'}
            entries = validate_entries([entry], root)
            self.assertAlmostEqual(read_case(entries[0])[-1], .024)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(entries, root / 'report', {'synthetic': True})
            report = json.loads((root / 'report/summary.json').read_text())
            self.assertEqual(report['per_exam']['DSC']['mean'], 1)
            self.assertTrue((root / 'report/reference_lesions.csv').is_file())
            with self.assertRaises(FileExistsError):
                evaluate(entries, root / 'report', {})
            shifted = affine.copy()
            shifted[0, 3] = 1
            nib.save(nib.Nifti1Image(mask, shifted), root / 'pred.nii.gz')
            with self.assertRaises(ValueError):
                read_case(entries[0])
            with self.assertRaises(ValueError):
                validate_entries([entry, entry], root)
            with self.assertRaises(ValueError):
                validate_entries([{**entry, 'patient': 'other'}], root)

    def test_split_identity_patient_routing_and_extra_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [Case(f'p{i}', 'e') for i in range(15)] + [Case('p0', 'visit2')]
            splits = root / 'splits.json'
            manifest = load_or_create_splits(cases, splits)
            recovered, _ = load_splits(splits)
            from scripts.audit_sampling import select_training
            selected = select_training(recovered, manifest, 0, 2, 9)
            self.assertTrue(all(c.patient in manifest['folds'][0]['train'] for c in selected))
            config = {'split_id': manifest['identity'], 'scientific': {'smoke': False}}
            (root / 'run/fold_0').mkdir(parents=True)
            (root / 'run/run.json').write_text(json.dumps({'config': config}))
            (root / 'run/fold_0/complete.json').write_text(json.dumps({
                'config': config, 'split_id': manifest['identity'],
                'held_out_cases': [c.id for c in recovered if c.patient in manifest['folds'][0]['held_out']]}))
            for c in recovered:
                if c.patient in manifest['folds'][0]['held_out']:
                    path = root / 'run/fold_0/predictions' / c.id / 'SEG_pred.nii.gz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            entries = run_entries(root / 'run', splits, root / 'data', '0')
            self.assertEqual(len(entries), len([c for c in recovered if c.patient in manifest['folds'][0]['held_out']]))
            extra = root / 'run/fold_0/predictions/unexpected/e/SEG_pred.nii.gz'
            extra.parent.mkdir(parents=True)
            extra.touch()
            with self.assertRaises(ValueError):
                run_entries(root / 'run', splits, root / 'data', '0')
            manifest['folds'][0]['train'].append('intruder')
            splits.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_splits(splits)

    def test_output_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                new_report(root / 'data/report', [root / 'data'])


class SamplingTests(unittest.TestCase):
    def test_actual_crop_and_augmented_component_coverage_reproducible(self):
        import torch
        from monai.data import MetaTensor
        from scripts.audit_sampling import audit_volume
        import config
        torch.set_num_threads(1)
        gt = torch.zeros((1, 80, 80, 80))
        gt[:, 10:12, 10:12, 10:12] = 1
        gt[:, 60:65, 60:65, 60:65] = 1
        affine = torch.diag(torch.tensor([2., 2., 2., 1.]))
        data = {k: MetaTensor(v, affine=affine) for k, v in
                [('GT', gt), ('PT', torch.ones_like(gt) * 8), ('CT', torch.zeros_like(gt))]}
        original = data['GT'].clone()
        with patch.object(config, 'AFFINE_PROBABILITY', 0):
            a = audit_volume(data, 64, 5, 42, 5)
            b = audit_volume(data, 64, 5, 42, 5)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 5)
        self.assertEqual(len(a[1]), 2)
        self.assertTrue(any(r['high_suv_background_voxels'] > 0 for r in a[0]))
        self.assertEqual([round(l['volume_ml'], 3) for l in a[1]], [.064, 1.0])
        torch.testing.assert_close(data['GT'], original)


class BenchmarkTests(unittest.TestCase):
    def test_preview_never_starts_a_worker_or_creates_output(self):
        from scripts import benchmark_patches as benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [Case(f'p{i}', 'e') for i in range(15)]
            splits = root / 'splits.json'
            load_or_create_splits(cases, splits)
            for case in cases:
                path = root / 'data' / case.id / 'SUV.nii.gz'
                path.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4)), path)
            with patch.object(benchmark.subprocess, 'run') as launch, contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(['--splits', str(splits), '--data-root', str(root / 'data'),
                                '--output', str(root / 'report'), '--steps', '1'])
                launch.assert_not_called()
            self.assertFalse((root / 'report').exists())

    def test_recommendation_requires_amp_and_actual_fallback_success(self):
        from scripts.benchmark_patches import recommend
        with tempfile.TemporaryDirectory() as directory:
            jobs = []
            for precision in ('amp', 'fallback'):
                result = Path(directory) / f'{precision}.json'
                rows = [{'phase': 'training', 'optimizer_update': True, 'float32_retry': precision == 'fallback'} for _ in range(2)]
                rows += [{'phase': 'validation', 'sw_batch_size': 1, 'status': 'ok', 'seconds': 3},
                         {'phase': 'validation', 'sw_batch_size': 2, 'status': 'out_of_memory', 'seconds': 1}]
                result.write_text(json.dumps({'status': 'complete', 'rows': rows}))
                jobs.append({'patch': 96, 'precision': precision, 'steps': 1, 'sw_batches': [1, 2], 'result': str(result)})
            self.assertEqual(recommend(jobs)['96']['recommended_sw_batch_size'], 1)
            result.write_text(json.dumps({'status': 'failed', 'rows': rows}))
            self.assertIsNone(recommend(jobs)['96']['recommended_sw_batch_size'])


if __name__ == '__main__':
    unittest.main()
