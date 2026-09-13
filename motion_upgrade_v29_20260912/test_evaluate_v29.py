"""Synthetic tests of 6 s/10 s evaluation clocks, denominators and gap durations."""
import unittest
import numpy as np
import pandas as pd
from evaluate_v29 import (waveform_availability, hr_availability, match_centers,
    validate_hr, pool, comparison, core_module)


class EvaluationV29Tests(unittest.TestCase):
    def test_waveform_missing_time_is_partitioned_without_overlap(self):
        result = waveform_availability([False, False, True, True, False, False, True, True, True, False], 5.)
        self.assertEqual(result['first_finite_time_s'], .4)
        self.assertEqual(result['last_finite_time_s'], 1.6)
        self.assertAlmostEqual(result['missing_duration_s'], 1.)
        self.assertAlmostEqual(result['leading_missing_s'], .4)
        self.assertAlmostEqual(result['internal_missing_s'], .4)
        self.assertAlmostEqual(result['trailing_missing_s'], .2)
        self.assertEqual(result['longest_missing_s'], .4)
        self.assertEqual(result['missing_runs'], 3)

    def test_all_missing_is_not_double_counted_as_both_edges(self):
        result = waveform_availability(np.zeros(300, bool), 30.)
        self.assertIsNone(result['first_finite_time_s'])
        self.assertEqual(result['missing_duration_s'], 10.)
        self.assertEqual(result['longest_missing_s'], 10.)
        self.assertEqual(result['leading_missing_s'], 10.)
        self.assertEqual(result['trailing_missing_s'], 0.)
        self.assertEqual(result['internal_missing_s'], 0.)

    def test_actual_center_intersection_preserves_fractional_fps(self):
        for fps, frames in [(30.00003000003, 1555), (180.001800018, 10693)]:
            n6, n10 = round(6*fps), round(10*fps)
            c6 = (np.arange(0, frames-n6+1, round(fps))+n6/2)/fps
            c10 = (np.arange(0, frames-n10+1, round(fps))+n10/2)/fps
            matched = match_centers(c6, c10)
            self.assertEqual((matched >= 0).sum(), len(c10))
            self.assertEqual((matched < 0).sum(), 4)
            np.testing.assert_allclose(c6[matched >= 0], c10[matched[matched >= 0]], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(match_centers([3.01, 4.01], [3.02, 4.02]), [-1, -1])

    def test_hr_gap_duration_is_hop_count_not_window_length(self):
        fps = 30.00003000003
        starts = np.arange(5)*round(fps)
        width = round(6*fps)
        hr = pd.DataFrame(dict(time_s=(starts+width/2)/fps,
            window_start_s=starts/fps, window_end_s=(starts+width)/fps,
            accepted=[False, True, False, False, True], status=['gap','ok','gap','gap','ok']))
        result = hr_availability(hr, fps)
        self.assertEqual(result['first_accepted_center_s'], hr.time_s.iloc[1])
        self.assertEqual(result['first_accepted_window_end_s'], hr.window_end_s.iloc[1])
        self.assertAlmostEqual(result['rejected_update_duration_s'], 3*round(fps)/fps)
        self.assertAlmostEqual(result['longest_rejected_update_duration_s'], 2*round(fps)/fps)
        self.assertLess(result['longest_rejected_update_duration_s'], 2.)

    def test_common_new_lost_accuracy_uses_separate_denominators(self):
        new, old, ref = np.array([81., 90., np.nan]), np.array([80., np.nan, 86.]), np.array([80., 80., 80.])
        result, masks = comparison(new, np.isfinite(new), ref, old, np.isfinite(old))
        self.assertEqual(result['common_new']['MAE_bpm'], 1)
        self.assertEqual(result['common_V28']['MAE_bpm'], 0)
        self.assertEqual(result['newly_covered']['MAE_bpm'], 10)
        self.assertEqual(result['newly_covered']['Nwithin5'], 0)
        self.assertEqual(result['lost_V28']['MAE_bpm'], 6)
        self.assertEqual(result['new_output_windows'], 1)
        np.testing.assert_array_equal(masks[1], [False, True, False])

    def test_pool_is_weighted_by_valid_windows(self):
        result = pool([dict(Nplanned=10,Noutput=10,Nref=10,Nvalid=10,Nwithin5=10,MAE_bpm=2.,RMSE_bpm=2.),
                       dict(Nplanned=2,Noutput=1,Nref=2,Nvalid=1,Nwithin5=0,MAE_bpm=20.,RMSE_bpm=20.)])
        self.assertAlmostEqual(result['MAE_bpm'], 40/11)
        self.assertAlmostEqual(result['P5_valid_pct'], 1000/11)
        self.assertAlmostEqual(result['R5_all_reference_pct'], 1000/12)
        self.assertEqual(result['Nplanned'], 12)

    def test_accepted_hr_requires_full_saved_waveform_and_exact_plan(self):
        fps, frames = 30.00003000003, 240
        width = round(6*fps)
        starts = np.arange(0, frames-width+1, round(fps))
        hr = pd.DataFrame(dict(time_s=(starts+width/2)/fps, window_start_s=starts/fps,
            window_end_s=(starts+width)/fps, accepted=True, ridge_bpm=80.,
            evidence_candidates_json='[{"supported_bpm":[80.0]}]'))
        validate_hr(hr, frames, fps, 6, np.ones(frames, bool))
        bad = np.ones(frames, bool); bad[10] = False
        with self.assertRaises(AssertionError):
            validate_hr(hr, frames, fps, 6, bad)
        wrong = hr.copy(); wrong['window_end_s'] = wrong.window_start_s+6.
        with self.assertRaises(AssertionError):
            validate_hr(wrong, frames, fps, 6, np.ones(frames, bool))

    def test_reference_uses_actual_half_open_short_window_not_rounded_six(self):
        core = core_module()
        origin = np.int64(1700000000000000000)
        offsets = np.array([0, 1, 2, 3, 4, 5, 5.999995, 7, 8])*1e9
        raw = pd.DataFrame(dict(host_utc_ns=origin+offsets.astype(np.int64),
                               hr_bpm=[80,80,80,80,80,80,140,80,80]))
        end = round(6*30.00003000003)/30.00003000003
        actual = core.reference_windows(raw, origin, [0], [end], 0)
        rounded = core.reference_windows(raw, origin, [0], [6.], 0)
        self.assertEqual(actual.reference_bpm.iloc[0], 80.)
        self.assertGreater(rounded.reference_bpm.iloc[0], 80.)


if __name__ == '__main__':
    unittest.main()
