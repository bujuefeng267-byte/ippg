"""Synthetic contracts for frozen ten-second V30 metric definitions."""
import unittest
import numpy as np
import pandas as pd
from evaluate_v30 import (aggregate, compare, promotion, protocol_bindings, validate_signal_tables)


class EvaluationV30Tests(unittest.TestCase):
    def test_pool_uses_valid_windows_and_reference_denominators(self):
        rows = [dict(Nplanned=10, Noutput=10, Nref=10, Nvalid=10, Nwithin5=10,
                     MAE_bpm=2., RMSE_bpm=2., fps=30., waveform_frames=300, waveform_finite_frames=300),
                dict(Nplanned=2, Noutput=1, Nref=2, Nvalid=1, Nwithin5=0,
                     MAE_bpm=20., RMSE_bpm=20., fps=30., waveform_frames=300, waveform_finite_frames=150)]
        pooled = aggregate(rows)
        self.assertAlmostEqual(pooled['MAE_bpm'], 40/11)
        self.assertAlmostEqual(pooled['P5_valid_pct'], 1000/11)
        self.assertAlmostEqual(pooled['R5_all_reference_pct'], 1000/12)
        self.assertAlmostEqual(pooled['HR_coverage_pct'], 1100/12)

    def test_waveform_time_weight_avoids_overweighting_180fps(self):
        common = dict(Nplanned=1, Noutput=1, Nref=1, Nvalid=1, Nwithin5=1, MAE_bpm=1., RMSE_bpm=1.)
        pooled = aggregate([{**common, 'fps': 30., 'waveform_frames': 300, 'waveform_finite_frames': 300},
                            {**common, 'fps': 180., 'waveform_frames': 1800, 'waveform_finite_frames': 0}])
        self.assertEqual(pooled['waveform_time_coverage_pct'], 50.)
        self.assertAlmostEqual(pooled['waveform_frame_coverage_pct'], 100/7)

    def test_common_new_lost_have_separate_error_denominators(self):
        y, old, ref = np.array([81., 90., np.nan]), np.array([80., np.nan, 86.]), np.array([80., 80., 80.])
        result, masks = compare(y, np.isfinite(y), ref, old, np.isfinite(old))
        self.assertEqual(result['common_new']['MAE_bpm'], 1.)
        self.assertEqual(result['common_V28']['MAE_bpm'], 0.)
        self.assertEqual(result['newly_covered']['MAE_bpm'], 10.)
        self.assertEqual(result['lost_V28']['MAE_bpm'], 6.)
        self.assertEqual(result['newly_covered']['Nwithin5'], 0)
        np.testing.assert_array_equal(masks[1], [False, True, False])

    def test_missing_reference_is_not_scored_as_success(self):
        y, ref = np.array([80., 80.]), np.array([80., np.nan])
        result, _ = compare(y, np.array([True, True]), ref, np.array([80., np.nan]), np.array([True, False]))
        self.assertEqual(result['new_output_windows'], 1)
        self.assertEqual(result['newly_covered']['Nvalid'], 0)
        self.assertTrue(np.isnan(result['newly_covered']['MAE_bpm']))

    def test_eight_guards_require_each_case_and_common_improvement(self):
        pooled, old = dict(MAE_bpm=9., R5_all_reference_pct=50.), dict(MAE_bpm=10., R5_all_reference_pct=45.)
        row = dict(delta_MAE_bpm=0., delta_P5_pp=0., delta_R5_pp=0., delta_hr_coverage_pp=0., delta_waveform_coverage_pp=0.)
        groups = dict(common_new={'MAE_bpm': 9.}, common_V28={'MAE_bpm': 10.})
        checks = promotion(pooled, old, [row], groups)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(checks.values()))
        checks = promotion(pooled, old, [{**row, 'delta_MAE_bpm': 3.01}], groups)
        self.assertFalse(checks['each_MAE'])
        checks = promotion(pooled, old, [row], dict(common_new={'MAE_bpm': 10.}, common_V28={'MAE_bpm': 10.}))
        self.assertFalse(checks['common_MAE'])

    def test_nested_protocol_binds_absolute_sources_and_inputs(self):
        digest = 'a'*64
        self.assertEqual(protocol_bindings({'x': [{'source_hashes': {'/tmp/a.py': digest}}],
                                           'inputs': {'data1': {'/tmp/in.csv': digest}}}),
                         {'/tmp/a.py': digest, '/tmp/in.csv': digest})
        with self.assertRaises(AssertionError):
            protocol_bindings({'a': {'/tmp/a.py': digest}, 'b': {'/tmp/a.py': 'b'*64}})

    def test_saved_waveform_clock_and_nan_provenance_contract(self):
        fps, frames = 30.00003000003, 360
        width, hop = round(10*fps), round(fps)
        starts = np.arange(0, frames-width+1, hop)
        wave = pd.DataFrame(dict(time_s=np.arange(frames)/fps, base=np.sin(np.arange(frames)/5),
                                 covered=True, observed=True, interpolated=False))
        hr = pd.DataFrame(dict(time_s=(starts+width/2)/fps, window_start_s=starts/fps,
            window_end_s=(starts+width)/fps, accepted=True, ridge_bpm=84.))
        validate_signal_tables(wave, hr, frames, fps)
        bad = wave.copy(); bad.loc[10, ['base', 'covered', 'observed']] = [np.nan, False, False]
        with self.assertRaisesRegex(AssertionError, 'missing samples'):
            validate_signal_tables(bad, hr, frames, fps)
        wrong = hr.copy(); wrong.window_end_s = wrong.window_start_s+10
        with self.assertRaises(AssertionError): validate_signal_tables(wave, wrong, frames, fps)


if __name__ == '__main__': unittest.main()
