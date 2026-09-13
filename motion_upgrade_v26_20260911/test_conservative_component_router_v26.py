"""Synthetic contracts only; no real case, evaluation, or reference is read."""
import json
import unittest

import numpy as np
import pandas as pd

from conservative_component_router_v26 import route_components


def inputs(fps=30, seconds=10, old_bpm=180, candidate_bpm=110, motion_bpm=180, speed=.5):
    n = round(seconds * fps)
    t = np.arange(n) / fps
    velocity = np.sqrt(2) * speed * np.sin(2 * np.pi * motion_bpm / 60 * t)
    trace = pd.DataFrame(dict(time_s=t, face_y0=np.full(n, .2), face_y1=np.full(n, .6),
                              motion_x=velocity * .4 / fps, motion_y=np.zeros(n)))
    old = pd.DataFrame(dict(time_s=t, base=2 * np.sin(2 * np.pi * old_bpm / 60 * t) + .3,
                            covered=np.ones(n, bool), observed=np.ones(n, bool),
                            interpolated=np.zeros(n, bool)))
    candidate = old.copy(deep=True)
    candidate['base'] = .7 * np.sin(2 * np.pi * candidate_bpm / 60 * t + .2) - .1
    starts = np.arange(0, max(0, n - round(10 * fps) + 1), round(fps))
    centers = (starts + round(10 * fps) / 2) / fps
    hr = pd.DataFrame(dict(time_s=centers, ridge_bpm=np.full(len(starts), old_bpm),
                           accepted=np.ones(len(starts), bool)))
    proposals = pd.DataFrame(dict(window_index=np.arange(len(starts)), time_s=centers,
        proposal_bpm=np.full(len(starts), candidate_bpm),
        proposal_supported=np.ones(len(starts), bool), generated=np.ones(len(starts), bool),
        contributing_rois=np.full(len(starts), 2),
        channels=[json.dumps(['baseline/forehead/pos', 'tracked/left_cheek/chrom'])] * len(starts)))
    return old, hr, candidate, proposals, trace, fps


def gap(wave, a, b):
    wave.loc[a:b - 1, 'base'] = np.nan
    wave.loc[a:b - 1, ['covered', 'observed', 'interpolated']] = False


class ConservativeComponentRouterTests(unittest.TestCase):
    def test_all_false_is_exact_old_waveform(self):
        args = inputs(seconds=12, speed=0)
        args[0].loc[10:15, 'interpolated'] = True
        args[0].loc[10:15, 'observed'] = False
        out, decisions = route_components(*args)
        for column in ('time_s', 'base', 'covered', 'observed', 'interpolated'):
            np.testing.assert_array_equal(out[column], args[0][column])
        self.assertFalse(decisions.route_eligible.any())
        self.assertTrue((out.candidate_weight == 0).all())

    def test_large_motion_180_to_110_routes_actual_saved_component(self):
        args = inputs()
        before = [x.copy(deep=True) if hasattr(x, 'copy') else x for x in args]
        out, decisions = route_components(*args)
        self.assertTrue(decisions.route_eligible.all())
        self.assertGreater(decisions.old_motion_risk.iloc[0], .79)
        self.assertGreater(decisions.motion_risk_decrease.iloc[0], .79)
        np.testing.assert_array_equal(out.base, args[2].base)
        self.assertTrue((out.candidate_weight == 1).all())
        self.assertTrue((out.source == 'component_waveform').all())
        for original, previous in zip(args[:-1], before[:-1]):
            pd.testing.assert_frame_equal(original, previous)

    def test_180_to_90_is_not_mistaken_for_motion_escape(self):
        args = inputs(candidate_bpm=90)
        out, decisions = route_components(*args)
        self.assertFalse(decisions.route_eligible.any())
        self.assertAlmostEqual(decisions.old_motion_risk.iloc[0], .8, delta=.002)
        self.assertAlmostEqual(decisions.candidate_motion_risk.iloc[0], .64, delta=.002)
        self.assertEqual(decisions.reason.iloc[0], 'insufficient_motion_risk_decrease')
        np.testing.assert_array_equal(out.base, args[0].base)

    def test_same_frequency_and_small_motion_do_not_route(self):
        for candidate_bpm, speed in ((180, .5), (110, .005)):
            with self.subTest(candidate_bpm=candidate_bpm, speed=speed):
                args = inputs(candidate_bpm=candidate_bpm, speed=speed)
                out, decisions = route_components(*args)
                self.assertFalse(decisions.route_eligible.any())
                np.testing.assert_array_equal(out.base, args[0].base)

    def test_old_gap_never_filled_even_with_finite_candidate(self):
        args = inputs(seconds=25)
        gap(args[0], 15 * 30, 16 * 30)
        out, decisions = route_components(*args)
        self.assertTrue(decisions.route_eligible.any())
        self.assertTrue((decisions.reason == 'old_waveform_gap').any())
        np.testing.assert_array_equal(np.isfinite(out.base), np.isfinite(args[0].base))
        self.assertTrue(out.base.iloc[450:480].isna().all())
        self.assertTrue((out.candidate_weight.iloc[450:480] == 0).all())
        self.assertFalse(out.covered.iloc[450:480].any())

    def test_any_candidate_gap_disqualifies_whole_window(self):
        args = inputs()
        gap(args[2], 100, 101)
        out, decisions = route_components(*args)
        self.assertFalse(decisions.route_eligible.any())
        self.assertEqual(decisions.reason.iloc[0], 'candidate_waveform_gap')
        np.testing.assert_array_equal(out.base, args[0].base)

    def test_full_window_hann_ratio_and_convex_samples(self):
        args = inputs(seconds=12)
        args[1]['accepted'] = [False, True, False]
        out, decisions = route_components(*args)
        np.testing.assert_array_equal(decisions.route_eligible, [False, True, False])
        taper = np.hanning(302)[1:-1]
        total = np.zeros(360)
        numerator = np.zeros(360)
        for a in (0, 30, 60):
            total[a:a + 300] += taper
        numerator[30:330] = taper
        expected_alpha = numerator / total
        np.testing.assert_array_equal(out.candidate_weight, expected_alpha)
        expected = args[0].base.to_numpy().copy()
        used = expected_alpha > 0
        expected[used] = ((1 - expected_alpha[used]) * expected[used] +
                          expected_alpha[used] * args[2].base.to_numpy()[used])
        np.testing.assert_array_equal(out.base, expected)
        self.assertLess(out.candidate_weight.max(), 1)
        self.assertLess(np.abs(np.diff(out.candidate_weight)).max(), .025)

    def test_sampling_masks_follow_actual_positive_weight_sources(self):
        args = inputs(seconds=12)
        args[1]['accepted'] = [False, True, False]
        args[2].loc[100, 'observed'] = False
        args[2].loc[100, 'interpolated'] = True
        args[0].loc[150, 'observed'] = False
        args[0].loc[150, 'interpolated'] = True
        out, _ = route_components(*args)
        self.assertFalse(out.observed.iloc[100])
        self.assertFalse(out.observed.iloc[150])
        self.assertTrue(out.interpolated.iloc[100])
        self.assertTrue(out.interpolated.iloc[150])
        self.assertTrue(out.observed.iloc[200])
        self.assertTrue(out.covered.all())

    def test_repeated_methods_cannot_replace_two_physical_rois(self):
        args = inputs()
        args[3]['channels'] = json.dumps(['baseline/forehead/pos', 'tracked/forehead/chrom'])
        out, decisions = route_components(*args)
        self.assertFalse(decisions.route_eligible.any())
        self.assertEqual(decisions.reason.iloc[0], 'candidate_requires_two_distinct_physical_rois')
        np.testing.assert_array_equal(out.base, args[0].base)

    def test_missing_motion_or_unsupported_proposal_keeps_old(self):
        for kind in ('motion', 'proposal', 'old_hr'):
            with self.subTest(kind=kind):
                args = inputs()
                if kind == 'motion':
                    args[4].drop(columns='motion_x', inplace=True)
                elif kind == 'proposal':
                    args[3]['proposal_supported'] = False
                else:
                    args[1]['accepted'] = False
                out, decisions = route_components(*args)
                self.assertFalse(decisions.route_eligible.any())
                np.testing.assert_array_equal(out.base, args[0].base)

    def test_same_physical_rule_at_30_and_180_fps(self):
        low, dl = route_components(*inputs(fps=30))
        high, dh = route_components(*inputs(fps=180))
        self.assertTrue(dl.route_eligible.all())
        self.assertTrue(dh.route_eligible.all())
        np.testing.assert_allclose(dl.old_motion_risk, dh.old_motion_risk, atol=.0003, rtol=0)
        self.assertEqual(low.candidate_weight.mean(), high.candidate_weight.mean())

    def test_reference_and_identity_columns_are_ignored(self):
        args = inputs(seconds=12)
        expected, decisions = route_components(*args)
        for table in args[:-1]:
            table['reference_bpm'] = 99999
            table['case_id'] = 'arbitrary_identity'
        actual, actual_decisions = route_components(*args)
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(actual_decisions, decisions)

    def test_mismatched_clocks_and_provenance_are_rejected(self):
        for kind in ('time', 'window', 'covered', 'observed'):
            with self.subTest(kind=kind):
                args = inputs()
                if kind == 'time':
                    args[2]['time_s'] += .01
                elif kind == 'window':
                    args[1]['time_s'] += 1
                elif kind == 'covered':
                    args[2].loc[0, 'covered'] = False
                else:
                    args[2]['observed'] = args[2]['observed'].astype(float)
                    args[2].loc[0, 'observed'] = np.nan
                with self.assertRaises(ValueError):
                    route_components(*args)


if __name__ == '__main__':
    unittest.main()
