"""Synthetic counterexamples for routing; no recordings, labels or saved metrics.

These are engineering safeguards, not evidence of improved human HR accuracy.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from guarded_fusion import (GuardConfig, choose_branch, guarded_fuse,
                            tracking_eligibility, _pulse_support)
from test_fusion_v2 import make_case


ROIS = ('forehead', 'left_cheek', 'right_cheek')
EVIDENCE = GuardConfig(routing_mode='evidence')


def proposal(bpm=84., quality=.8, ambiguity=0.):
    return dict(ridge_bpm=bpm, quality_proxy=quality, motion_overlap=0.,
                runner_up_ratio=ambiguity)


def features(pulse=.8, risk=0., available=True):
    return dict(pulse_support=pulse, motion_risk=risk, motion_available=available)


def evidence(b=.8, t=.8, br=0., tr=0., available=True):
    return dict(baseline=features(b, br, available), tracked=features(t, tr, available))


def route(b, t, ev=None, b_generated=True, t_generated=True, tracking_ok=True):
    return choose_branch(b, t, b_generated, t_generated, tracking_ok, EVIDENCE, ev)


class GuardEvidenceChoices(unittest.TestCase):
    def test_default_legacy_rules_and_exact_score_unchanged(self):
        b, t = proposal(84., .6, .2), proposal(144., .99, .1)
        b['motion_overlap'], t['motion_overlap'] = .4, .8
        result = choose_branch(b, t, True, True, True)
        self.assertEqual(GuardConfig().routing_mode, 'legacy')
        self.assertEqual(result[:2], ('baseline', 'frequency_disagreement_fallback'))
        self.assertEqual(result[2], .6 * (1 - .25 * .4) * (1 - .5 * .2))
        self.assertEqual(result[3], .99 * (1 - .25 * .8) * (1 - .5 * .1))
        self.assertEqual(result, choose_branch(b, t, True, True, True, evidence=evidence(.1, 1.)))

    def test_tracking_can_win_large_disagreement_with_supported_clean_candidate(self):
        result = route(proposal(144.), proposal(84.), evidence(.75, .9, br=1.))
        self.assertEqual(result[:2], ('tracked', 'disagreement_resolved_tracked_evidence'))

    def test_symmetric_counterexample_preserves_supported_baseline(self):
        result = route(proposal(84.), proposal(144.), evidence(.9, .75, tr=1.))
        self.assertEqual(result[:2], ('baseline', 'disagreement_resolved_baseline_evidence'))

    def test_no_preference_for_lower_candidate_frequency(self):
        low_tracked = route(proposal(144.), proposal(84.), evidence(.75, .9, br=1.))
        high_tracked = route(proposal(84.), proposal(144.), evidence(.75, .9, br=1.))
        self.assertEqual(low_tracked, high_tracked)

    def test_large_disagreement_with_similar_evidence_is_unresolved(self):
        result = route(proposal(84.), proposal(144.), evidence(.85, .9))
        self.assertEqual(result[:2], ('none', 'unresolved_frequency_disagreement'))

    def test_both_weak_are_unresolved_without_filling_previous_hr(self):
        # API has no history or previous-HR argument.
        result = route(proposal(84.), proposal(85.), evidence(.1, .12))
        self.assertEqual(result[0], 'none')

    def test_motion_coinciding_with_strong_true_pulse_is_not_a_veto(self):
        # Deliberately put a synthetic true pulse at maximum motion risk.
        # Its strong pulse support can still exceed a clean but weak rival.
        result = route(proposal(144.), proposal(84.), evidence(.25, .95, tr=1.))
        self.assertEqual(result[0], 'tracked')
        self.assertGreater(result[3], EVIDENCE.min_evidence_score)
        single = route(proposal(144.), proposal(84.), evidence(.0, .95, tr=1.), b_generated=False)
        self.assertEqual(single[0], 'tracked')

    def test_motion_unknown_is_not_misread_as_evidence_of_stationarity(self):
        neutral = route(proposal(84.), proposal(144.), evidence(.8, .8, available=False))
        misleading_risk = route(proposal(84.), proposal(144.), evidence(.8, .8, br=1., available=False))
        self.assertEqual(neutral, misleading_risk)
        self.assertEqual(neutral[0], 'none')

    def test_frequency_ambiguity_can_prevent_supported_peak_winning(self):
        ev = evidence(.7, .9)
        self.assertEqual(route(proposal(84.), proposal(85.), ev)[0], 'tracked')
        self.assertEqual(route(proposal(84.), proposal(85., ambiguity=.75), ev)[0], 'baseline')

    def test_tracking_quorum_cannot_be_overridden_by_score(self):
        result = route(proposal(144.), proposal(84.), evidence(.4, 1.), tracking_ok=False)
        self.assertEqual(result[0], 'baseline')
        self.assertEqual(route(proposal(144.), proposal(84.), evidence(0., 1.),
                               b_generated=False, tracking_ok=False)[0], 'none')

    def test_nan_features_or_invalid_bpm_cannot_create_confident_output(self):
        result = route(proposal(np.nan), proposal(84.), evidence(1., np.nan))
        self.assertEqual(result[0], 'none')
        result = route(proposal(84.), proposal(np.inf), evidence(.8, 1.))
        self.assertEqual(result[0], 'baseline')

    def test_invalid_configuration_is_rejected(self):
        for kwargs in [dict(routing_mode='optimal_truth'), dict(min_evidence_score=0),
                       dict(max_motion_evidence_penalty=1), dict(disagreement_score_ratio=1),
                       dict(min_disagreement_score_margin=np.nan)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GuardConfig(**kwargs)


class GuardEvidenceWaveformIntegration(unittest.TestCase):
    def fixture(self, seconds=18):
        time, _, signals, trace = make_case(seconds=seconds, bpm=84)
        for roi in ROIS:
            trace[f'{roi}_pixel_source'] = 'tracked_ratio'
        return time, signals, trace

    def test_contributing_roi_count_not_increased_by_duplicate_methods_or_rows(self):
        rows = [dict(roi=roi, channel=roi+'_pos', waveform_weight=1.,
                     peak_concentration=.6, prominence_ratio=.8) for roi in ROIS[:2]]
        diag = pd.DataFrame(rows)
        support, count = _pulse_support(proposal(quality=.8), diag)
        correlated = pd.concat([diag, diag.assign(channel=diag.roi+'_chrom'), diag], ignore_index=True)
        self.assertEqual(_pulse_support(proposal(quality=.8), correlated), (support, count))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(support, .75)
        zero_weight = pd.DataFrame([dict(roi='right_cheek', channel='right_cheek_pos',
            waveform_weight=0., peak_concentration=1., prominence_ratio=1.)])
        self.assertEqual(_pulse_support(proposal(quality=.8), pd.concat([diag, zero_weight])), (support, count))

    def test_actual_contributing_rois_still_determine_tracking_eligibility(self):
        _, _, trace = self.fixture()
        trace['left_cheek_pixel_source'] = 'baseline_ratio_fallback'
        self.assertFalse(tracking_eligibility(trace, 0, 300, EVIDENCE,
                          contributing_rois=['forehead', 'left_cheek'])[0])
        self.assertTrue(tracking_eligibility(trace, 0, 300, EVIDENCE,
                         contributing_rois=['forehead', 'right_cheek'])[0])

    def test_equal_clean_disagreeing_sinusoids_yield_no_artificial_waveform(self):
        time, signals, trace = self.fixture()
        other = {name: np.sin(2*np.pi*2.4*time) for name in signals}
        table, wave, diag, routing, _ = guarded_fuse(signals, other, trace, 30., guard=EVIDENCE)
        self.assertTrue((routing.selected_branch == 'none').all())
        self.assertTrue((routing.selection_reason == 'unresolved_frequency_disagreement').all())
        self.assertTrue(np.isnan(wave).all())
        self.assertFalse(table.accepted.any())
        self.assertTrue(table.ridge_bpm.isna().all())
        self.assertTrue((diag.waveform_weight == 0.).all())
        self.assertTrue((table.status == 'guard_unresolved_frequency_disagreement').all())

    def test_same_frequency_motion_remains_soft_and_risk_is_not_multiplied_twice(self):
        _, signals, trace = self.fixture()
        def fixed_motion(trace, start, stop, fps, grid):
            return dict(profile=np.full(len(grid), .25), available=True, observed_fraction=1.,
                        speed_face_per_s=.2, strength=.5, reliability=.5, status='synthetic')
        with patch('motion_evidence.motion_evidence', side_effect=fixed_motion):
            table, wave, _, routing, _ = guarded_fuse(signals, signals, trace, 30., guard=EVIDENCE)
        self.assertTrue(table.accepted.all())
        self.assertTrue(np.isfinite(wave).any())
        np.testing.assert_allclose(routing.baseline_motion_risk, .25, rtol=0., atol=0.)
        expected = routing.baseline_pulse_support * (1-.5*routing.baseline_ambiguity) * (1-.45*.25)
        np.testing.assert_allclose(routing.baseline_score, expected, rtol=0., atol=1e-15)

    def test_extra_reference_columns_are_never_read_or_used(self):
        _, signals, trace = self.fixture()
        original = guarded_fuse(signals, signals, trace, 30., guard=EVIDENCE)
        contaminated = trace.assign(reference_bpm=199., ground_truth_hr=np.nan, polar_hr=42.)
        changed = guarded_fuse(signals, signals, contaminated, 30., guard=EVIDENCE)
        np.testing.assert_array_equal(original[1], changed[1])
        pd.testing.assert_frame_equal(original[3], changed[3])

    def test_short_clip_has_empty_evidence_routing_schema(self):
        _, signals, trace = self.fixture(seconds=3)
        table, wave, _, routing, _ = guarded_fuse(signals, signals, trace, 30., guard=EVIDENCE)
        self.assertEqual(len(table), 0)
        self.assertEqual(len(routing), 0)
        self.assertIn('motion_available', routing)
        self.assertTrue(np.isnan(wave).all())


if __name__ == '__main__':
    unittest.main()
