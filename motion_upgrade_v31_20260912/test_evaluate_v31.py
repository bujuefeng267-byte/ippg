"""Synthetic tests for V31 evaluation denominators and protection contracts."""
import unittest
import numpy as np
import pandas as pd

from evaluate_v31 import (protection_mask, verify_protection, window_changes,
    jump_metrics, promotion, TOL)
from evaluation_common_v31 import aggregate, compare


class EvaluationContracts(unittest.TestCase):
    def test_protection_union_includes_rejected_overlap_gaps_and_tail(self):
        finite = np.ones(16, bool); finite[0] = False
        actual = protection_mask(16, np.array([0, 3, 6]), 6,
                                 np.array([True, False, True]), finite)
        expected = np.zeros(16, bool); expected[0] = True; expected[3:9] = True; expected[12:] = True
        np.testing.assert_array_equal(actual, expected)

    def test_sample_protection_is_exact_and_rejects_changed_mask(self):
        fps, frames, starts = 10., 120, np.array([0, 10, 20])
        wave = pd.DataFrame(dict(time_s=np.arange(frames)/fps, base=np.arange(frames, dtype=float),
            covered=True, observed=True, interpolated=False, protected_sample=True))
        hr = pd.DataFrame(dict(time_s=(starts+50)/fps, window_start_s=starts/fps,
            window_end_s=(starts+100)/fps, accepted=True, ridge_bpm=80.))
        decisions = pd.DataFrame(dict(window_index=np.arange(3), time_s=hr.time_s,
            window_start_s=hr.window_start_s, window_end_s=hr.window_end_s,
            eligible=False, protected_window=True, modified_samples_in_window=0))
        result = verify_protection(wave, hr, wave.copy(), hr.copy(), decisions, fps, starts)
        self.assertEqual(result['protected_samples'], frames)
        changed = wave.copy(); changed.loc[10, 'base'] += 1e-12
        with self.assertRaises(AssertionError):
            verify_protection(changed, hr, wave, hr, decisions, fps, starts)
        changed = wave.copy(); changed.loc[10, 'observed'] = False
        with self.assertRaises(AssertionError):
            verify_protection(changed, hr, wave, hr, decisions, fps, starts)

    def test_original_correct_loss_and_new_large_from_missing(self):
        r = np.full(6, 80.)
        old = np.array([85., 80., 110., 100., np.nan, 81.])
        new = np.array([86., np.nan, 90., 85., 101., 80.])
        counts, flags = window_changes(new, np.isfinite(new), r, old, np.isfinite(old),
            np.array([5, 10, 15, 20, 25, 30.]), np.array([10, 15, 20, 25, 30, 35.]))
        self.assertEqual(counts['old_correct_to_wrong_or_missing_windows'], 2)
        self.assertEqual(counts['new_large_error_introduced_windows'], 1)
        self.assertEqual(counts['old_large_error_reduced_windows'], 2)
        self.assertEqual(counts['old_large_error_below_20_windows'], 2)
        self.assertEqual(counts['old_large_error_within_5_windows'], 1)
        self.assertEqual(counts['first20_center']['planned_windows'], 3)
        self.assertEqual(counts['first20_center']['V28_large_error_windows'], 1)
        self.assertTrue(flags['new_large_error_introduced'][4])

    def test_error_boundaries_and_invalid_reference(self):
        r = np.array([80., 80., np.nan, 0.])
        old = np.array([85., 100., 80., 80.]); new = np.array([85., 100., 110., 110.])
        counts, _ = window_changes(new, np.ones(4, bool), r, old, np.ones(4, bool),
            np.arange(4.), np.arange(4.)+5)
        self.assertEqual(counts['old_correct_windows'], 1)
        self.assertEqual(counts['old_large_error_windows'], 1)
        self.assertEqual(counts['new_large_error_introduced_windows'], 0)

    def test_jumps_do_not_bridge_missing_or_count_equal12(self):
        y = np.array([70., 82., np.nan, 120., 133., 150.])
        result = jump_metrics(y, np.isfinite(y))
        self.assertEqual(result, dict(adjacent_valid_pairs=3, large_jump_count=2, max_jump_bpm=17.))
        self.assertEqual(jump_metrics(np.array([np.nan]), np.array([False]))['max_jump_bpm'], 0.)

    def test_waveform_pool_uses_time_not_frames(self):
        common = dict(Nvalid=1, Nwithin5=1, MAE_bpm=1., RMSE_bpm=1., Nplanned=1, Nref=1, Noutput=1)
        rows = [dict(**common, fps=10., waveform_frames=100, waveform_finite_frames=100),
                dict(**common, fps=100., waveform_frames=1000, waveform_finite_frames=0)]
        pooled = aggregate(rows)
        self.assertEqual(pooled['waveform_time_coverage_pct'], 50.)
        self.assertAlmostEqual(pooled['waveform_frame_coverage_pct'], 100/11)

    def test_common_new_lost_denominators(self):
        y = np.array([82., 90., np.nan]); old = np.array([80., np.nan, 100.]); ref = np.full(3, 80.)
        result, _ = compare(y, np.isfinite(y), ref, old, np.isfinite(old))
        self.assertEqual([result[k] for k in ('common_output_windows', 'new_output_windows', 'lost_output_windows')], [1, 1, 1])
        self.assertEqual(result['newly_covered']['MAE_bpm'], 10.)
        self.assertEqual(result['lost_V28']['MAE_bpm'], 20.)

    def test_promotion_requires_real_gain_and_all_guards(self):
        base = dict(MAE_bpm=10., P5_valid_pct=50., R5_all_reference_pct=40.)
        row = dict(old_correct_to_wrong_or_missing_windows=0, new_large_error_introduced_windows=0,
            delta_MAE_bpm=0., delta_P5_pp=0., delta_R5_pp=0., delta_hr_coverage_pp=0.,
            delta_waveform_coverage_pp=0., jumps=dict(large_jump_count=1), V28_jumps=dict(large_jump_count=1))
        groups = dict(common_new=dict(MAE_bpm=9.), common_V28=dict(MAE_bpm=10.))
        unchanged = promotion(base, base, [row], groups)
        self.assertFalse(unchanged['substantive_gain'])
        improved = dict(base, MAE_bpm=9.)
        self.assertTrue(all(promotion(improved, base, [row], groups).values()))
        self.assertTrue(all(promotion(dict(base, R5_all_reference_pct=41.), base, [row], groups).values()))
        broken = dict(row, old_correct_to_wrong_or_missing_windows=1)
        self.assertFalse(all(promotion(improved, base, [broken], groups).values()))
        broken = dict(row, delta_MAE_bpm=2*TOL)
        self.assertFalse(promotion(improved, base, [broken], groups)['each_MAE'])


if __name__ == '__main__': unittest.main()
