"""Synthetic contracts only; no real case, evaluation, or reference is read."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from direct_guard_router_v28 import route_components
from conservative_component_router_v26 import route_components as route_v26
from motion_harmonic_evidence_v26 import motion_evidence_harmonics


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


class DirectGuardRouterTests(unittest.TestCase):
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

    def test_double_only_motion_retains_old_waveform(self):
        args = inputs(old_bpm=90, candidate_bpm=110, motion_bpm=180)
        legacy, old_decisions = route_v26(*args)
        output, decisions = route_components(*args)
        self.assertTrue(old_decisions.route_eligible.all())
        self.assertFalse(decisions.route_eligible.any())
        self.assertEqual(decisions.reason.iloc[0], 'direct_motion_evidence_insufficient')
        self.assertGreater(decisions.old_motion_risk.iloc[0], .5)
        self.assertLess(decisions.old_motion_risk_direct.iloc[0], .5)
        self.assertGreater(decisions.old_motion_risk_double.iloc[0], .79)
        for name in ('base', 'covered', 'observed', 'interpolated'):
            np.testing.assert_array_equal(output[name], args[0][name])
        self.assertFalse(np.array_equal(legacy.base, output.base))

    def test_direct_high_preserves_v26_mixture_exactly(self):
        args = inputs(seconds=12)
        args[1]['accepted'] = [False, True, False]
        args[2].loc[100, ['observed', 'interpolated']] = [False, True]
        legacy, old_decisions = route_v26(*args)
        output, decisions = route_components(*args)
        pd.testing.assert_frame_equal(output, legacy)
        self.assertEqual(output.base.to_numpy().tobytes(), legacy.base.to_numpy().tobytes())
        np.testing.assert_array_equal(decisions.route_eligible, old_decisions.route_eligible)
        self.assertGreater(decisions.old_motion_risk_direct.iloc[1], .79)
        self.assertLess(decisions.candidate_motion_risk_direct.iloc[1], .01)

    def test_direct_threshold_is_inclusive_and_no_new_threshold(self):
        import direct_guard_router_v28 as router
        self.assertEqual(router.MIN_OLD_MOTION_RISK, .50)
        args = inputs()
        def provider(direct):
            def evaluate(trace, a, b, fps, grid):
                evidence = motion_evidence_harmonics(trace, a, b, fps, grid)
                evidence['profile_direct'] = np.full(len(grid), direct)
                return evidence
            return evaluate
        for risk, expected in ((np.nextafter(.50, 0), False), (.50, True),
                               (np.nextafter(.50, 1), True), (np.nan, False)):
            with self.subTest(risk=risk):
                with patch('direct_guard_router_v28.motion_evidence_harmonics', provider(risk)):
                    output, decisions = route_components(*args)
                self.assertEqual(bool(decisions.route_eligible.iloc[0]), expected)
                if not expected:
                    self.assertEqual(decisions.reason.iloc[0], 'direct_motion_evidence_insufficient')
                    np.testing.assert_array_equal(output.base, args[0].base)

    def test_missing_direct_profile_does_not_enable_routing(self):
        args = inputs()
        def missing_direct(trace, a, b, fps, grid):
            evidence = motion_evidence_harmonics(trace, a, b, fps, grid)
            evidence.pop('profile_direct')
            return evidence
        with patch('direct_guard_router_v28.motion_evidence_harmonics', missing_direct):
            output, decisions = route_components(*args)
        self.assertFalse(decisions.route_eligible.any())
        self.assertEqual(decisions.reason.iloc[0], 'direct_motion_evidence_insufficient')
        np.testing.assert_array_equal(output.base, args[0].base)

    def test_eligibility_is_strict_subset_of_v26_across_motion_relations(self):
        decreased = False
        for old_bpm in (60, 90, 110, 150, 180):
            for candidate_bpm in (60, 110, 180):
                for motion_bpm in (90, 180):
                    for speed in (0, .05, .5):
                        args = inputs(old_bpm=old_bpm, candidate_bpm=candidate_bpm,
                                      motion_bpm=motion_bpm, speed=speed)
                        _, before = route_v26(*args)
                        _, after = route_components(*args)
                        self.assertFalse((after.route_eligible & ~before.route_eligible).any())
                        decreased |= bool((before.route_eligible & ~after.route_eligible).any())
        self.assertTrue(decreased)

    def test_direct_guard_window_tapers_and_source_masks(self):
        args = inputs(seconds=12)
        args[2].loc[100, ['observed', 'interpolated']] = [False, True]
        def middle_only(trace, a, b, fps, grid):
            evidence = motion_evidence_harmonics(trace, a, b, fps, grid)
            if a != round(fps):
                evidence['profile_direct'] = np.zeros(len(grid))
            return evidence
        with patch('direct_guard_router_v28.motion_evidence_harmonics', middle_only):
            output, decisions = route_components(*args)
        np.testing.assert_array_equal(decisions.route_eligible, [False, True, False])
        taper = np.hanning(302)[1:-1]
        total = np.zeros(360)
        for a in (0, 30, 60):
            total[a:a + 300] += taper
        numerator = np.zeros(360)
        numerator[30:330] = taper
        alpha = numerator / total
        np.testing.assert_array_equal(output.candidate_weight, alpha)
        np.testing.assert_array_equal(output.covered, args[0].covered)
        np.testing.assert_array_equal(np.isfinite(output.base), np.isfinite(args[0].base))
        self.assertTrue((output.source.iloc[:30] == 'old_waveform').all())
        self.assertTrue((output.source.iloc[30:330] == 'old_component_mixture').all())
        self.assertTrue((output.source.iloc[330:] == 'old_waveform').all())
        self.assertFalse(output.observed.iloc[100])
        self.assertTrue(output.interpolated.iloc[100])


if __name__ == '__main__':
    unittest.main()
