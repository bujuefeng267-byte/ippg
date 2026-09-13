"""Synthetic checks fixed before real inference; no real references read."""
import io
import json
import unittest

import numpy as np
import pandas as pd
from scipy.signal import detrend

from run_cpace_trial import (HERE, ROIS, code_hashes, dump, final_hr,
                             infer_trace, fill_short_gaps, gap_frame_limit)

OBSERVATIONS = {}


def synthetic_trace(seconds=35, fps=30., hz=1.2, common=False):
    t = np.arange(round(seconds*fps))/fps
    trace = pd.DataFrame({'time_s': t})
    for i, roi in enumerate(ROIS):
        mu = np.array([150., 95., 70.]) + i*np.array([4., -2., 3.])
        pulse = np.sin(2*np.pi*hz*t) + .2*np.sin(4*np.pi*hz*t)
        rgb = mu*(1 + .004*pulse[:,None]*(np.ones(3) if common else np.array([.3, 1., .1])))
        for ci, channel in enumerate('rgb'):
            trace[f'{roi}_{channel}'] = rgb[:,ci]
        trace[f'{roi}_valid'] = True
    return trace


def as_wave(trace, outputs, observed, interpolated, mode='default'):
    return pd.DataFrame({'time_s': trace.time_s, 'base': outputs[mode]['forehead'],
                         'observed': observed, 'interpolated': interpolated})


def blank(trace, a, b, rois=tuple(ROIS)):
    for roi in rois:
        trace.loc[a:b-1, [f'{roi}_{c}' for c in 'rgb']] = np.nan
        trace.loc[a:b-1, f'{roi}_valid'] = False


class CoreChecks(unittest.TestCase):
    def test_known_pulse_from_video_seed(self):
        trace = synthetic_trace()
        outputs, observed, interp, records = infer_trace(trace, 30.)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]['seed_bpm'], 72., delta=1)
        for mode in ('default', 'no_homodyne'):
            table = final_hr(as_wave(trace, outputs, observed, interp, mode), 30.)
            valid = table[table.accepted]
            self.assertGreater(len(valid), 10)
            self.assertLessEqual(float(np.max(abs(valid.spectral_peak_bpm-72))), 2)
            self.assertLessEqual(float(np.max(abs(valid.ridge_bpm-72))), 2)
            OBSERVATIONS[f'known_pulse_{mode}'] = {'seed_bpm': records[0]['seed_bpm'],
                'accepted_windows': len(valid), 'hr_range': [float(valid.ridge_bpm.min()), float(valid.ridge_bpm.max())]}

    def test_common_brightness_is_a_documented_limitation(self):
        trace = synthetic_trace(common=True, hz=1.6)
        outputs, observed, interp, records = infer_trace(trace, 30.)
        mu = trace[[f'forehead_{c}' for c in 'rgb']].mean().to_numpy()
        q = mu/np.linalg.norm(mu)
        remaining_common_direction = np.linalg.norm((np.eye(3)-np.outer(q,q)) @ np.ones(3))
        self.assertGreater(remaining_common_direction, .1)
        table = final_hr(as_wave(trace, outputs, observed, interp), 30.)
        # Record false-pulse susceptibility without changing original math.
        OBSERVATIONS['pure_common_brightness'] = {
            'true_cardiac_pulse_present': False, 'imposed_common_brightness_hz': 1.6,
            'normalized_common_direction_residual_norm': float(remaining_common_direction),
            'seed_bpm': records[0]['seed_bpm'],
            'default_output_std': float(np.nanstd(outputs['default']['forehead'])),
            'accepted_windows': int(table.accepted.sum()),
            'accepted_ridge_bpm': table.loc[table.accepted, 'ridge_bpm'].tolist(),
            'interpretation': 'Author core is not guaranteed to reject periodic common brightness; retain, do not repair for better results'}

    def test_nan_long_gap_no_bridge_and_no_cross_gap_hr(self):
        trace = synthetic_trace(seconds=40)
        blank(trace, 570, 600)
        outputs, observed, interp, records = infer_trace(trace, 30.)
        self.assertTrue(np.isnan(outputs['default']['forehead'][570:600]).all())
        self.assertFalse(interp[570:600].any())
        table = final_hr(as_wave(trace, outputs, observed, interp), 30.)
        touching = (table.time_s+5 > 19) & (table.time_s-5 < 20)
        self.assertFalse(table.loc[touching, 'accepted'].any())
        self.assertTrue(table.loc[touching, ['spectral_peak_bpm','ridge_bpm']].isna().all().all())
        self.assertEqual(sum(r['status']=='processed' for r in records), 2)

    def test_future_across_long_gap_does_not_change_prior_segment(self):
        trace = synthetic_trace(seconds=40)
        blank(trace, 570, 600)
        other = synthetic_trace(seconds=40, hz=1.8)
        changed = trace.copy()
        changed.iloc[600:] = other.iloc[600:]
        first = infer_trace(trace, 30.)[0]
        second = infer_trace(changed, 30.)[0]
        for mode in first:
            np.testing.assert_allclose(first[mode]['forehead'][:570], second[mode]['forehead'][:570], equal_nan=True, rtol=0, atol=0)

    def test_bounded_short_gap_preserves_provenance(self):
        trace = synthetic_trace()
        blank(trace, 300, 303, rois=('left_cheek',))
        outputs, observed, interp, _ = infer_trace(trace, 30.)
        self.assertTrue(interp[300:303].all())
        self.assertFalse(observed[300:303].any())
        self.assertTrue(np.isfinite(outputs['default']['forehead'][300:303]).all())
        self.assertEqual(gap_frame_limit(.1, 29.999), 2)

    def test_all_missing_and_missing_forehead_not_substituted(self):
        for selected in [tuple(ROIS), ('forehead',)]:
            trace = synthetic_trace()
            blank(trace, 0, len(trace), selected)
            outputs, observed, interp, _ = infer_trace(trace, 30.)
            self.assertTrue(np.isnan(outputs['default']['forehead']).all())
            self.assertFalse(observed.any() or interp.any())

    def test_flags_match_finite_mask(self):
        trace = synthetic_trace()
        outputs, observed, interp, _ = infer_trace(trace, 30.)
        for mode in outputs:
            finite = np.isfinite(outputs[mode]['forehead'])
            self.assertFalse((observed & ~finite).any())
            self.assertFalse((interp & ~finite).any())


if __name__ == '__main__':
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreChecks)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (HERE/'synthetic_selfchecks.log').write_text(log.getvalue(), encoding='utf-8')
    dump(HERE/'synthetic_selfchecks.json', {'successful': result.wasSuccessful(),
         'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
         'code_hashes': code_hashes(), 'observations': OBSERVATIONS,
         'real_video_reference_read': False})
    print(log.getvalue(), flush=True)
    print(json.dumps(OBSERVATIONS, indent=2), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
