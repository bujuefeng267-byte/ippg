"""Synthetic time/frequency/missingness invariants and fixed-weight CPU/GPU check."""
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import run_trial as trial


class SamplingTests(unittest.TestCase):
    def test_global_clock_and_frequency_preserved_30_and_180(self):
        phases = []
        for fps in (30.00003000003, 180.001800018):
            start = 43
            original_time = (np.arange(round(12*fps))+start)/fps
            signal = np.sin(2*np.pi*1.4*original_time)[:, None]
            index, sampled, meta = trial.resample_segment(signal, start, fps)
            time = index/30
            central = (time > time[0]+1) & (time < time[-1]-1)
            basis = np.column_stack([np.sin(2*np.pi*1.4*time[central]), np.cos(2*np.pi*1.4*time[central])])
            coef = np.linalg.lstsq(basis, sampled[central, 0], rcond=None)[0]
            self.assertLess(abs(coef[0]-1), .008)
            self.assertLess(abs(coef[1]), .002)
            self.assertGreaterEqual(time[0], original_time[0]-1e-9)
            self.assertLessEqual(time[-1], original_time[-1]+1e-9)
            self.assertEqual(meta['antialias_applied'], fps > 100)
            phases.append(coef)
        np.testing.assert_allclose(phases[0], phases[1], atol=.008)

    def test_high_frequency_alias_does_not_become_84_bpm(self):
        fps = 180.
        t = np.arange(3600)/fps
        idx, out, _ = trial.resample_segment(np.sin(2*np.pi*28.6*t)[:, None], 0, fps)
        time = idx/30
        center = (time > 1) & (time < 19)
        basis = np.column_stack([np.sin(2*np.pi*1.4*time[center]), np.cos(2*np.pi*1.4*time[center])])
        amplitude = np.linalg.norm(np.linalg.lstsq(basis, out[center, 0], rcond=None)[0])
        self.assertLess(amplitude, .003)
        # Direct every-sixth-frame decimation would alias this tone at full amplitude.
        aliased = np.sin(2*np.pi*28.6*np.arange(600)/30)
        self.assertGreater(np.std(aliased), .7)

    def test_short_gap_geometry_only_bounded_no_extrapolation(self):
        n = 30
        trace = pd.DataFrame(dict(face_x0=np.full(n,.2), face_y0=.1, face_x1=.7, face_y1=.8, rgb_valid=True))
        trace.loc[[0, 5, 6, 7, 15, 16, 17, 18, 29], 'rgb_valid'] = False
        boxes, observed, filled = trial.prepare_boxes(trace, 30.)
        np.testing.assert_array_equal(np.flatnonzero(filled), [5,6,7])
        self.assertFalse(observed[5:8].any())
        self.assertTrue(np.isfinite(boxes[5:8]).all())
        self.assertTrue(np.isnan(boxes[[0,15,16,17,18,29]]).all())

    def test_segment_resampling_cannot_bridge_missing_time(self):
        a, _, _ = trial.resample_segment(np.ones((301,1)), 0, 30.)
        b, _, _ = trial.resample_segment(np.ones((301,1)), 600, 30.)
        self.assertLess(a[-1], b[0])
        self.assertEqual(a[-1], 300)
        self.assertEqual(b[0], 600)

    def test_high_fps_tiny_segment_is_explicitly_empty(self):
        i, x, m = trial.resample_segment(np.ones((2,1)), 0, 180.)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(x), 0)

    def test_crop_geometry_rectangular_image(self):
        self.assertEqual(trial.crop_bounds(np.array([.25,.25,.75,.75]), 400, 200), (50,25,350,175))
        with self.assertRaises(ValueError):
            trial.crop_bounds(np.array([2.,2.,3.,3.]),400,200)


class ReconstructionTests(unittest.TestCase):
    def test_overlap_preserves_same_intervals_and_covers_awkward_tail(self):
        for length in (181,182,269,270,271,359,362,1597):
            time = np.arange(length)/30
            crop = (10+np.sin(2*np.pi*1.4*time))[:,None]
            raw, meta = trial.overlap_infer(crop, lambda x:np.diff(x[:,0]))
            expected = np.diff((crop[:,0]-crop.mean())/crop.std())
            self.assertEqual(len(raw), length-1)
            self.assertTrue(np.isfinite(raw).all())
            self.assertEqual(meta['uncovered_intervals'], 0)
            self.assertEqual(meta['clip_starts'][-1], length-181)
            np.testing.assert_allclose(raw, expected, atol=2e-6)

    def test_no_fake_last_derivative_and_integral_timing(self):
        time = np.arange(600)/30
        signal = np.sin(2*np.pi*1.4*time)
        integrated, base = trial.reconstruct(np.diff(signal))
        np.testing.assert_allclose(integrated, signal-signal[0], atol=2e-14)
        self.assertTrue(np.isnan(base[:48]).all() and np.isnan(base[-48:]).all())
        self.assertTrue(np.isfinite(base[48:-48]).all())
        self.assertGreater(np.corrcoef(base[90:-90],signal[90:-90])[0,1],.99)

    def test_short_flat_and_missing_remain_absent(self):
        short, _ = trial.overlap_infer(np.arange(180)[:,None], lambda x:np.diff(x[:,0]))
        self.assertTrue(np.isnan(short).all())
        flat, record = trial.overlap_infer(np.ones((400,1)), lambda x:np.zeros(180))
        self.assertTrue(np.isnan(flat).all())
        self.assertEqual(record['status'],'flat_crop')
        integrated, wave = trial.reconstruct(np.r_[np.ones(179),np.nan])
        self.assertTrue(np.isnan(integrated).all() and np.isnan(wave).all())


class BackendTests(unittest.TestCase):
    def test_official_weight_strict_load_shape_and_cpu_gpu_agreement(self):
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _, cpu, meta = trial.load_model('cpu')
        block = np.random.default_rng(17).normal(size=(181,72,72,3)).astype(np.float32)
        reference = cpu(block)
        self.assertEqual(reference.shape, (180,))
        self.assertTrue(np.isfinite(reference).all())
        self.assertTrue(meta['weights_only'] and meta['strict'])
        if not torch.cuda.is_available():
            self.fail('Expected previously verified GPU to be available')
        _, gpu, _ = trial.load_model('cuda')
        prediction = gpu(block)
        np.testing.assert_allclose(prediction, reference, atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    trial.dump(trial.HERE/'test_receipt.json', dict(passed=result.wasSuccessful(),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        created_utc=datetime.now(timezone.utc).isoformat(),
        adapter_sha256=trial.sha(trial.HERE/'run_trial.py'), tests_sha256=trial.sha(__file__),
        checkpoint_sha256=trial.PARAMETERS['checkpoint_sha256'], reference_used=False,
        invariants=['global time/frequency consistency','180Hz anti-alias negative control',
            'bounded missing-face geometry','no bridging long gaps','complete overlap tails',
            'derivative integration timing','explicit missing signal','official strict checkpoint CPU/GPU']))
    raise SystemExit(0 if result.wasSuccessful() else 1)
