"""Denominator, fixed-window and source-separation contracts for V27 QA."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict

import numpy as np
import pandas as pd

import qa_v27 as qa


class EvaluationContracts(unittest.TestCase):
    def test_missing_predictions_are_not_correct_or_removed_from_r5(self):
        metric = qa.score([85, 100, np.nan, 90], [True, True, False, True],
                          [80, 80, 80, np.nan])
        self.assertEqual((metric['Nplanned'], metric['Noutput'], metric['Nref'], metric['Nvalid']), (4, 3, 3, 2))
        self.assertEqual(metric['Nwithin5'], 1)
        self.assertEqual(metric['P5_valid_pct'], 50)
        self.assertAlmostEqual(metric['R5_all_reference_pct'], 100/3)
        self.assertEqual(metric['hr_output_coverage_pct'], 75)
        self.assertEqual(metric['MAE_bpm'], 12.5)

    def test_five_bpm_boundary_is_inclusive_and_zero_reference_unavailable(self):
        metric = qa.score([75., 85., 85.000001, 0.], [1, 1, 1, 1], [80., 80., 80., 0.])
        self.assertEqual(metric['Nwithin5'], 2)
        self.assertEqual(metric['Nref'], 3)
        self.assertAlmostEqual(metric['P5_valid_pct'], 200/3)

    def test_no_outputs_means_no_mae_and_zero_r5(self):
        metric = qa.score([np.nan]*3, [False]*3, [80]*3)
        self.assertTrue(np.isnan(metric['MAE_bpm']))
        self.assertTrue(np.isnan(metric['P5_valid_pct']))
        self.assertEqual(metric['R5_all_reference_pct'], 0)

    def test_pool_is_window_weighted_not_clip_mean(self):
        first = qa.score([180.], [True], [80.])
        second = qa.score([80.]*99, [True]*99, [80.]*99)
        metric = qa.pool([first, second])
        self.assertEqual(metric['MAE_bpm'], 1)
        self.assertEqual(metric['RMSE_bpm'], 10)
        self.assertEqual(metric['P5_valid_pct'], 99)

    def test_original_reference_half_open_timing_and_integer_origin(self):
        origin = 1789115680595038500
        raw = pd.DataFrame(dict(host_utc_ns=origin+np.arange(21, dtype=np.int64)*1_000_000_000,
                                hr_bpm=80.+np.arange(21)))
        values, valid = qa.independent_reference(raw, origin, [0., 10.], [10., 20.])
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(values, [84.5, 94.5], rtol=0, atol=1e-12)
        broken = raw.drop(index=[2, 3])
        values, valid = qa.independent_reference(broken, origin, [0., 10.], [10., 20.])
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(valid[1])

    def test_original_eight_promotion_guards_are_separate(self):
        row = dict(common_windows=10, common_MAE_bpm=9, common_V25_MAE_bpm=10,
            delta_MAE_bpm=3., delta_P5_pp=-3., delta_R5_pp=-3.,
            delta_hr_coverage_pp=-3., delta_waveform_coverage_pp=-3.)
        baseline = dict(MAE_bpm=10, R5_all_reference_pct=30)
        pooled = dict(MAE_bpm=9, R5_all_reference_pct=35)
        checks = qa.promotion([row], pooled, baseline)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(checks.values()))
        self.assertFalse(qa.promotion([dict(row, delta_MAE_bpm=3.001)], pooled, baseline)['each_MAE'])
        self.assertFalse(qa.promotion([row], dict(pooled, MAE_bpm=10), baseline)['pooled_MAE'])

    def test_patch_main_hr_has_its_own_contract_not_fused_wave_coverage(self):
        fps = 30.
        starts = np.array([0, 30])
        table = pd.DataFrame(dict(time_s=[5., 6.], accepted=[True, False], ridge_bpm=[84., np.nan]))
        # This validates patch HR independent of a display waveform; its saved
        # patch contributions will be checked by the separate provenance audit.
        values, accepted = qa.assert_hr(table, starts, fps, 330)
        self.assertEqual(values[0], 84)
        self.assertFalse(accepted[1])
        with self.assertRaises(AssertionError):
            qa.assert_hr(table.assign(ridge_bpm=[np.nan, np.nan]), starts, fps, 330)

    def test_actual_high_fps_keeps_full_plan(self):
        fps, frames = 180.00180001800018, 10693
        starts = np.arange(0, frames-round(10*fps)+1, round(fps))
        self.assertEqual(len(starts), 50)
        self.assertLess((starts[-1]+round(10*fps))/fps, frames/fps)
        self.assertEqual(sum(qa.EXPECTED_WINDOWS.values()), 309)

    def test_independent_patch_replay_and_corruption_detection(self):
        from patch_hr_v27 import infer_patch_hr, DEFAULT_CONFIG
        from test_patch_hr_v27 import fixture
        frequencies={'a':('forehead',84.),'b':('left_cheek',84.),'c':('right_cheek',120.)}
        for mode in ('median','psd_cluster'):
            args=fixture(frequencies,seconds=12.)
            channels,regions,obs,fill,quality,sources,trace,fps=args
            signals=pd.DataFrame({'time_s':trace.time_s})
            for key,value in channels.items():signals[key]=value
            for key in regions:
                signals[f'{key}/observed']=obs[key];signals[f'{key}/interpolated']=fill[key]
                signals[f'{key}/quality']=quality[key];signals[f'{key}/tracking_source']=sources[key]
            meta=dict(patch_regions=regions,channel_order=list(channels))
            # The audit consumes the same serialized BVP as production.
            with TemporaryDirectory() as temp:
                folder=Path(temp);signals.to_csv(folder/'signals.csv',index=False)
                saved=pd.read_csv(folder/'signals.csv')
                canonical={key:saved[key].to_numpy(float) for key in channels}
                wave,hr,diag=infer_patch_hr(canonical,regions,obs,fill,quality,sources,trace,fps,mode=mode)
                wave.to_csv(folder/'waveform.csv',index=False)
                hr.to_csv(folder/'heart_rate.csv',index=False)
                diag.to_csv(folder/'patch_diagnostics.csv',index=False)
                result=qa.audit_patch_result(folder,saved,meta,trace,fps,asdict(DEFAULT_CONFIG))
                self.assertTrue(result['passed'])
                bad=hr.copy();bad.loc[0,'ridge_bpm']+=1
                bad.to_csv(folder/'heart_rate.csv',index=False)
                with self.assertRaises(AssertionError):
                    qa.audit_patch_result(folder,saved,meta,trace,fps,asdict(DEFAULT_CONFIG))


if __name__ == '__main__':
    unittest.main()
