"""Meaningful synthetic contracts; no human videos or reference labels read."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import analyze_video_v29 as cli
from quality_readout_v29 import estimate_evidence, SamplingConfig
from evidence_hr import estimate_evidence as original_estimate
from direct_guard_router_v29 import route_components as route6
from direct_guard_router_v28 import route_components as original_route

def colour_pulse(fps=30., seconds=30.):
    t = np.arange(round(seconds*fps))/fps
    trace = pd.DataFrame(dict(frame=np.arange(len(t)), time_s=t, rgb_valid=True,
        motion_x=0., motion_y=0., face_x0=.2, face_x1=.6, face_y0=.2, face_y1=.6,
        pixel_rgb_kind='anchored_paired_relative_colour', anchor_hz=.15, anchor_order=4))
    values = []
    for k, roi in enumerate(cli.ROIS):
        pulse = np.sin(2*np.pi*1.4*t+.02*k)
        rgb = np.column_stack([130*(1+.002*pulse), 100*(1+.008*pulse), 80*(1+.001*pulse)])
        values.append(rgb)
        for j, channel in enumerate('rgb'):
            trace[f'{roi}_{channel}'] = trace[f'baseline_{roi}_{channel}'] = rgb[:, j]
            trace[f'{roi}_pixel_log_delta_{channel}'] = np.r_[0., np.diff(np.log(rgb[:, j]))]
        trace[f'{roi}_valid'] = True
        trace[f'{roi}_quality'] = 1.
        trace[f'{roi}_pixel_source'] = 'tracked_ratio'
        trace[f'{roi}_pixel_tracks'] = 50
    merged = np.mean(values, axis=0)
    for j, channel in enumerate('rgb'):
        trace[channel] = trace[f'baseline_{channel}'] = merged[:, j]
    return trace


class ShortWindowContracts(unittest.TestCase):
    def test_profiles_have_only_one_ambiguity_difference(self):
        profiles = cli.PROFILE_CONFIGS
        self.assertEqual(json.loads(json.dumps(profiles)), profiles)
        strict, relaxed = json.loads(json.dumps(profiles['short6'])), json.loads(json.dumps(profiles['short6_relaxed']))
        self.assertEqual(strict['fusion_config']['ambiguity_ratio'], .8)
        self.assertEqual(relaxed['fusion_config']['ambiguity_ratio'], .9)
        relaxed['fusion_config']['ambiguity_ratio'] = .8
        self.assertEqual(strict, relaxed)
        self.assertEqual(strict['window_s'], 6.)
        self.assertEqual(strict['hr_sampling'], {'min_observed':.9,'max_interpolated':.1,'min_concentration':.12})
        self.assertEqual(strict['fusion_config']['min_rois'], 2)
        self.assertEqual(strict['max_gap_s'], .1)

    def test_six_seconds_accepts_fractional_fps_without_weakening_gates(self):
        for fps in (30.00003000003, 180.001800018):
            with self.subTest(fps=fps):
                t = np.arange(round(6*fps))/fps
                signal = np.sin(2*np.pi*1.5*t)
                trace = pd.DataFrame({'rgb_valid':np.ones(len(t), bool)})
                out = estimate_evidence(signal, trace, np.zeros(len(t), bool), fps, window_s=6)
                self.assertEqual(len(out), 1)
                self.assertTrue(out.accepted.iloc[0])
                self.assertLess(abs(out.ridge_bpm.iloc[0]-90), 3)
                trace.loc[:round(.15*len(t)), 'rgb_valid'] = False
                rejected = estimate_evidence(signal, trace, np.zeros(len(t), bool), fps, window_s=6)
                self.assertFalse(rejected.accepted.iloc[0])
                self.assertEqual(rejected.status.iloc[0], 'insufficient_observed_rgb')

    def test_10s_quality_readout_matches_original_exactly(self):
        fps = 30.00003000003
        t = np.arange(round(24*fps))/fps
        signal = np.sin(2*np.pi*1.5*t)+.2*np.sin(2*np.pi*2.8*t)
        signal[round(12*fps):round(13*fps)] = np.nan
        trace, filled = pd.DataFrame({'rgb_valid':np.isfinite(signal)}), np.zeros(len(t), bool)
        actual = estimate_evidence(signal, trace, filled, fps)
        expected = original_estimate(signal, trace, filled, fps)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_interpolation_limit_and_missingness_remain_strict(self):
        fps = 30.
        t = np.arange(180)/fps
        signal, observed = np.sin(2*np.pi*1.5*t), pd.DataFrame({'rgb_valid':True}, index=np.arange(len(t)))
        fill = np.zeros(len(t), bool)
        fill[:20] = True
        rejected = estimate_evidence(signal, observed, fill, fps, window_s=6)
        self.assertFalse(rejected.accepted.iloc[0])
        signal[90] = np.nan
        missing = estimate_evidence(signal, observed, np.zeros(len(t), bool), fps, window_s=6)
        self.assertEqual(missing.status.iloc[0], 'gap_or_filter_edge')
        self.assertTrue(missing.ridge_bpm.isna().all())

    def test_full_pipeline_six_second_plan_saved_wave_replay_and_labels(self):
        for profile in cli.PROFILE_CONFIGS:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory(prefix='v29_pulse_') as temp:
                trace, fps, out = colour_pulse(fps=30.00003000003, seconds=20), 30.00003000003, Path(temp)
                wave, hr, decisions = cli.run_pipeline(trace, fps, out, profile)
                fallback = pd.read_csv(out/'short_window_fallback/waveform.csv')
                np.testing.assert_array_equal(wave.covered, fallback.covered)
                np.testing.assert_allclose(wave.base, fallback.base, equal_nan=True, atol=0, rtol=0)
                self.assertEqual(len(hr), 15)
                self.assertEqual(len(decisions), 15)
                self.assertGreater(int(hr.accepted.sum()), 0)
                self.assertLess(abs(hr.loc[hr.accepted,'ridge_bpm'].median()-84), 3)
                np.testing.assert_allclose(hr.window_end_s-hr.window_start_s, round(6*fps)/fps, atol=1e-12)
                self.assertTrue(hr.loc[~hr.accepted,'ridge_bpm'].isna().all())
                self.assertTrue(hr.profile.eq(profile).all())
                np.testing.assert_array_equal(hr.strict_accepted, hr.accepted)
                self.assertTrue(hr.loc[hr.accepted,'quality_tier'].str.contains('relaxed' if profile.endswith('relaxed') else 'short6').all())
                replay = cli.read_saved_hr(out/'waveform.csv', trace, fps, profile,
                                           'saved_v29_short_window_direct_guard_waveform_offline')
                np.testing.assert_allclose(hr.ridge_bpm, replay.ridge_bpm, equal_nan=True, atol=0, rtol=0)
                strict10 = pd.read_csv(out/'heart_rate_10s.csv')
                old_reader = cli.v28.read_saved_hr(out/'waveform.csv', trace, fps, 'test')
                np.testing.assert_allclose(strict10.ridge_bpm, old_reader.ridge_bpm, equal_nan=True, atol=0, rtol=0)
                self.assertEqual(len(strict10), 11)
                self.assertLess(hr.loc[hr.accepted,'window_end_s'].min(), strict10.loc[strict10.accepted,'window_end_s'].min())

    def test_no_face_gap_is_not_filled_by_relaxed_mode(self):
        fps, trace = 30., colour_pulse(seconds=24)
        a, b = 12*30, 14*30
        for roi in cli.ROIS:
            for channel in 'rgb':
                trace.loc[a:b-1,[f'{roi}_{channel}',f'baseline_{roi}_{channel}',f'{roi}_pixel_log_delta_{channel}']] = np.nan
            trace.loc[a:b-1,f'{roi}_valid'] = False
            trace.loc[a:b-1,f'{roi}_quality'] = 0.
            trace.loc[a:b-1,f'{roi}_pixel_source'] = 'missing'
            trace.loc[a:b-1,f'{roi}_pixel_tracks'] = 0
        trace.loc[a:b-1,'rgb_valid'] = False
        trace.loc[a:b-1,['r','g','b','baseline_r','baseline_g','baseline_b']] = np.nan
        with tempfile.TemporaryDirectory(prefix='v29_gap_') as temp:
            wave, hr, _ = cli.run_pipeline(trace, fps, Path(temp), 'short6_relaxed')
            self.assertTrue(wave.base.iloc[a:b].isna().all())
            self.assertFalse(wave.covered.iloc[a:b].any())
            crossings = (hr.window_start_s < 14) & (hr.window_end_s > 12)
            self.assertFalse(hr.loc[crossings,'accepted'].any())

    def test_reference_and_identity_columns_are_ignored(self):
        fps, trace = 30., colour_pulse(seconds=14)
        with tempfile.TemporaryDirectory(prefix='v29_ref_ignore_') as temp:
            first = cli.run_pipeline(trace, fps, Path(temp)/'first', 'short6')
            trace['reference_bpm'], trace['case_id'] = 99999, 'irrelevant'
            second = cli.run_pipeline(trace, fps, Path(temp)/'second', 'short6')
            for a,b in zip(first,second):
                pd.testing.assert_frame_equal(a,b,check_exact=True)

    def test_six_second_direct_guard_keeps_v28_decision_conditions(self):
        for window in (6.,10.):
            fps, seconds = 30.00003000003, 12.
            n=round(seconds*fps); t=np.arange(n)/fps
            velocity = np.sqrt(2)*.5*np.sin(2*np.pi*3*t)
            trace=pd.DataFrame(dict(time_s=t,face_y0=.2,face_y1=.6,motion_x=velocity*.4/fps,motion_y=0.))
            old=pd.DataFrame(dict(time_s=t,base=np.sin(2*np.pi*3*t),covered=True,observed=True,interpolated=False))
            candidate=old.copy();candidate['base']=.7*np.sin(2*np.pi*110/60*t)
            starts=np.arange(0,n-round(window*fps)+1,round(fps));centers=(starts+round(window*fps)/2)/fps
            hr=pd.DataFrame(dict(time_s=centers,ridge_bpm=180.,accepted=True))
            proposals=pd.DataFrame(dict(time_s=centers,proposal_bpm=110.,proposal_supported=True,
                generated=True,contributing_rois=2,channels=json.dumps(['baseline/forehead/pos','tracked/left_cheek/chrom'])))
            actual,decisions=route6(old,hr,candidate,proposals,trace,fps,window_s=window)
            self.assertTrue(decisions.route_eligible.all())
            np.testing.assert_allclose(actual.base,candidate.base,atol=1e-15)
            if window==10.:
                expected,_=original_route(old,hr,candidate,proposals,trace,fps)
                pd.testing.assert_frame_equal(actual,expected,check_exact=True)
            else:
                hr['ridge_bpm']=90.
                retained,decisions=route6(old,hr,candidate,proposals,trace,fps,window_s=window)
                self.assertFalse(decisions.route_eligible.any())
                self.assertTrue(decisions.reason.eq('direct_motion_evidence_insufficient').all())
                np.testing.assert_array_equal(retained.base,old.base)


class CliCacheContracts(unittest.TestCase):
    def test_short_no_face_cli_and_cache_identity_and_content_checks(self):
        with tempfile.TemporaryDirectory(prefix='v29_cli_') as temp:
            root=Path(temp);video=root/'blank.avi';out=root/'fresh'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(96,96))
            self.assertTrue(writer.isOpened())
            for _ in range(60): writer.write(np.zeros((96,96,3),np.uint8))
            writer.release()
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(['--video',str(video),'--out',str(out),'--profile','short6'])
                cli.main(['--video',str(video),'--out',str(root/'cached'),'--profile','short6_relaxed','--trace-cache',str(out)])
            self.assertTrue(pd.read_csv(out/'waveform.csv').base.isna().all())
            for name in ('heart_rate.csv','heart_rate_10s.csv','routing_decisions.csv'):
                self.assertTrue(pd.read_csv(out/name).empty)
            with self.assertRaisesRegex(ValueError,'content SHA256'):
                cli.load_trace_cache(out,video,cli.frontend_identity(video),'0'*64)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                cli.load_trace_cache(out,video,{},cli.sha(video))
            (out/'frame_trace.csv').write_text((out/'frame_trace.csv').read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'cache SHA256'):
                cli.load_trace_cache(out,video,cli.frontend_identity(video),cli.sha(video))
            with self.assertRaises(FileExistsError):
                cli.main(['--video',str(video),'--out',str(out),'--profile','short6'])


if __name__=='__main__':
    unittest.main()

