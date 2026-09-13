"""Metric contract checks using synthetic values only."""
import unittest
import numpy as np
import pandas as pd
from evaluate_v28 import aggregate, bools, compare, core_module


class EvaluationContractTests(unittest.TestCase):
    def test_missing_output_remains_failure_in_all_reference_success(self):
        score = core_module().score(np.array([70., np.nan, 86.]), np.array([True, False, True]),
                                   np.array([70., 80., 80.]))
        self.assertEqual(score['Nvalid'], 2)
        self.assertEqual(score['Nwithin5'], 1)
        self.assertEqual(score['P5_valid_pct'], 50)
        self.assertAlmostEqual(score['R5_all_reference_pct'], 100/3)

    def test_pooled_is_window_weighted_and_wave_coverage_time_weighted(self):
        rows = [dict(Nplanned=10, Noutput=10, Nref=10, Nvalid=10, Nwithin5=10,
                     MAE_bpm=2., RMSE_bpm=2., waveform_frames=300,
                     waveform_finite_frames=300, fps=30.),
                dict(Nplanned=2, Noutput=1, Nref=2, Nvalid=1, Nwithin5=0,
                     MAE_bpm=20., RMSE_bpm=20., waveform_frames=1800,
                     waveform_finite_frames=0, fps=180.)]
        result = aggregate(rows)
        self.assertAlmostEqual(result['MAE_bpm'], 40/11)
        self.assertAlmostEqual(result['RMSE_bpm'], np.sqrt(440/11))
        self.assertAlmostEqual(result['P5_valid_pct'], 1000/11)
        self.assertAlmostEqual(result['R5_all_reference_pct'], 1000/12)
        self.assertAlmostEqual(result['waveform_time_coverage_pct'], 50)
        self.assertAlmostEqual(result['waveform_frame_coverage_pct'], 100/7)

    def test_common_window_comparison_does_not_use_new_or_lost(self):
        y, old, ref = np.array([81., 90., np.nan]), np.array([80., np.nan, 90.]), np.array([80., 80., 80.])
        result, masks = compare(y, np.isfinite(y), ref, old, np.isfinite(old))
        self.assertEqual(result['common_windows'], 1)
        self.assertEqual(result['common_MAE_bpm'], 1)
        self.assertEqual(result['common_comparator_MAE_bpm'], 0)
        self.assertEqual(result['new_MAE_bpm'], 10)
        self.assertEqual(result['lost_comparator_MAE_bpm'], 10)
        np.testing.assert_array_equal(masks[0], [True, False, False])

    def test_no_truthiness_conversion_of_missing_or_text_flags(self):
        np.testing.assert_array_equal(bools(pd.Series([True, False, 1, 0])), [True, False, True, False])
        for bad in [[1, np.nan], ['True', 'False'], [0, 2]]:
            with self.assertRaises(AssertionError):
                bools(pd.Series(bad))


if __name__ == '__main__':
    unittest.main()
