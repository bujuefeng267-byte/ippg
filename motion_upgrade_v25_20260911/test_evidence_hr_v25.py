"""Controlled counterexamples only: no video, human reference or fitted shift."""
import json
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from evidence_hr import (DEFAULT_CONFIG, EvidenceConfig, score_candidates,
                         evidence_path, estimate_evidence)
from legacy_motion import estimate as legacy_estimate
from waveform_hr import estimate_fused_waveform
from test_waveform_hr_v21 import fixture, HR_COLUMNS


FPS = 30.
GRID = np.arange(42., 211.)


def waves(seconds=30, tones=((84., 1.),)):
    time = np.arange(round(seconds*FPS))/FPS
    return time, sum(amplitude*np.sin(2*np.pi*bpm/60*time+.13*index)
                     for index, (bpm, amplitude) in enumerate(tones))


def estimate(wave, observed=None, interpolated=None, motion=None):
    if observed is None:
        observed = np.ones(len(wave), bool)
    if interpolated is None:
        interpolated = np.zeros(len(wave), bool)
    trace = pd.DataFrame({'rgb_valid': observed})
    return estimate_evidence(wave, trace, interpolated, FPS,
        motion_trace=trace if motion is not None else None,
        motion_provider=(lambda *args: motion) if motion is not None else None)


class EvidenceHRTests(unittest.TestCase):
    def test_clean_low_and_high_frequency_do_not_get_halved(self):
        for bpm in (72., 84., 168., 192.):
            with self.subTest(bpm=bpm):
                _, wave = waves(tones=((bpm, 1.),))
                result = estimate(wave)
                self.assertTrue(result.accepted.all())
                self.assertLess(np.max(abs(result.ridge_bpm-bpm)), 1.1)
                self.assertFalse(result.evidence_reacquired.any())

    def test_existing_fundamental_competes_with_stronger_second_harmonic(self):
        _, wave = waves(tones=((84., .5), (168., 1.)))
        result = estimate(wave)
        self.assertTrue(result.accepted.all())
        self.assertTrue((result.spectral_peak_bpm > 160).all())
        self.assertLess(np.max(abs(result.ridge_bpm-84)), 1.1)
        self.assertTrue((result.evidence_candidate_count >= 2).all())

    def test_no_spectral_peak_at_subharmonic_means_no_arithmetic_half(self):
        _, wave = waves(tones=((180., 1.),))
        candidates, emissions = score_candidates(wave[:300], FPS, GRID)
        self.assertFalse(any(abs(row['bpm']-90) < 6 for row in candidates))
        self.assertTrue(np.isneginf(emissions[np.abs(GRID-90) < 6]).all())
        self.assertLess(abs(estimate(wave).ridge_bpm.median()-180), 1.1)

    def test_motion_risk_can_demote_a_competing_actual_peak(self):
        _, wave = waves(tones=((84., .8), (156., 1.)))
        motion = dict(profile=np.exp(-.5*((GRID-156)/6)**2), available=True,
                      strength=.5, reliability=.5)
        result = estimate(wave, motion=motion)
        self.assertTrue(result.accepted.all())
        self.assertLess(np.max(abs(result.ridge_bpm-84)), 1.1)
        rows = json.loads(result.evidence_candidates_json.iloc[0])
        high = min(rows, key=lambda row: abs(row['bpm']-156))
        # Profile is already weighted by the provider: never multiply again.
        self.assertAlmostEqual(high['motion_overlap'], motion['profile'][int(high['bpm']-42)])

    def test_pulse_and_motion_same_frequency_does_not_force_rejection(self):
        _, wave = waves(tones=((96., 1.),))
        result = estimate(wave, motion=dict(profile=np.exp(-.5*((GRID-96)/6)**2),
            available=True, strength=1., reliability=1.))
        self.assertTrue(result.accepted.all())
        self.assertLess(np.max(abs(result.ridge_bpm-96)), 1.1)

    def test_unavailable_motion_is_neutral_and_explicitly_unknown(self):
        _, wave = waves()
        expected = estimate(wave)
        actual = estimate(wave, motion=dict(profile=np.zeros(len(GRID)),
            available=False, strength=np.nan, reliability=0.))
        pd.testing.assert_frame_equal(actual, expected)
        candidate = json.loads(actual.evidence_candidates_json.iloc[0])[0]
        self.assertFalse(candidate['motion_available'])
        self.assertIsNone(candidate['motion_strength'])

    def test_same_validity_and_coverage_as_legacy_on_corrupt_sampling(self):
        _, wave = waves(seconds=45)
        wave[600:660] = np.nan
        observed = np.ones(len(wave), bool)
        observed[180:250] = False
        interpolated = ~observed
        reference = legacy_estimate(wave, pd.DataFrame({'rgb_valid': observed}),
                                    interpolated, FPS)
        result = estimate(wave, observed, interpolated)
        np.testing.assert_array_equal(result.accepted, reference.accepted)
        np.testing.assert_array_equal(result.status, reference.status)
        self.assertTrue(result.loc[~result.accepted, 'ridge_bpm'].isna().all())
        self.assertTrue((result.evidence_tracker_state == 'acquired').sum() >= 2)

    def test_flat_and_central_impulse_windows_retain_legacy_rejection(self):
        flat = estimate(np.zeros(600))
        self.assertFalse(flat.accepted.any())
        wave = np.r_[np.zeros(360), 1., np.zeros(359)]
        result = estimate(wave)
        legacy = legacy_estimate(wave, pd.DataFrame({'rgb_valid':np.ones(len(wave), bool)}),
                                 np.zeros(len(wave), bool), FPS)
        np.testing.assert_array_equal(result.accepted, legacy.accepted)
        central = result.time_s.between(8., 16.)
        self.assertTrue(result.loc[central, 'status'].eq('diffuse_spectrum').all())
        self.assertFalse(result.loc[central, 'accepted'].any())
        self.assertTrue(result.loc[~result.accepted, 'ridge_bpm'].isna().all())

    def test_dp_reacquires_supported_jump_without_a_fictitious_ramp(self):
        emissions = np.full((12, len(GRID)), -np.inf)
        emissions[:6, GRID == 84] = 1.
        emissions[6:, GRID == 168] = 1.
        path, reacquired = evidence_path(emissions, GRID, 1.)
        np.testing.assert_array_equal(GRID[path], np.r_[np.full(6,84.),np.full(6,168.)])
        np.testing.assert_array_equal(np.flatnonzero(reacquired), [6])

    def test_dp_does_not_chase_a_one_window_competitor(self):
        emissions = np.full((15, len(GRID)), -np.inf)
        emissions[:, GRID == 84] = 1.
        emissions[:, GRID == 150] = 0.
        emissions[7, GRID == 150] = 2.
        path, reacquired = evidence_path(emissions, GRID, 1.)
        np.testing.assert_array_equal(GRID[path], np.full(15, 84.))
        self.assertFalse(reacquired.any())

    def test_actual_waveform_frequency_change_is_reacquired(self):
        time, wave = waves(seconds=45)
        wave[time >= 20] = np.sin(2*np.pi*156/60*time[time >= 20])
        result = estimate(wave)
        self.assertTrue(result.accepted.all())
        self.assertLess(np.max(abs(result.loc[result.time_s < 15, 'ridge_bpm']-84)), 1.1)
        self.assertLess(np.max(abs(result.loc[result.time_s > 27, 'ridge_bpm']-156)), 1.1)
        self.assertTrue(result.evidence_reacquired.any())
        for row in result.itertuples():
            self.assertTrue(any(row.ridge_bpm in c['supported_bpm']
                                for c in json.loads(row.evidence_candidates_json)))

    def test_smooth_chirp_keeps_continuity_without_reacquisition(self):
        time, _ = waves(seconds=40)
        wave = np.sin(2*np.pi*(80/60*time+.5*(.5/60)*time**2))
        result = estimate(wave)
        self.assertTrue(result.accepted.all())
        expected = 80+.5*result.time_s
        self.assertLess(np.max(abs(result.ridge_bpm-expected)), 2.1)
        self.assertFalse(result.evidence_reacquired.any())

    def test_saved_waveform_roundtrip_and_support_audit(self):
        _, wave = waves(tones=((84., .5), (168., 1.)))
        result = estimate(wave)
        restored = pd.read_csv(StringIO(pd.DataFrame({'base': wave}).to_csv(index=False)))
        replay = estimate(restored.base.to_numpy())
        np.testing.assert_allclose(replay.ridge_bpm, result.ridge_bpm, rtol=0, atol=1e-9)
        for row in result.itertuples():
            candidates = json.loads(row.evidence_candidates_json)
            self.assertTrue(any(row.ridge_bpm in candidate['supported_bpm'] for candidate in candidates))
            self.assertGreaterEqual(row.evidence_selected_relative_power, DEFAULT_CONFIG.min_relative_power)
            self.assertLessEqual(abs(row.ridge_bpm-row.evidence_selected_peak_bpm), 3.)

    def test_wrapper_legacy_unchanged_evidence_ignores_wrong_proposals_and_reference(self):
        case = fixture(bpm=84., proposal_bpm=177.)
        legacy, mask = estimate_fused_waveform(*case, fps=FPS)
        explicit, explicit_mask = estimate_fused_waveform(*case, fps=FPS, hr_mode='legacy')
        pd.testing.assert_frame_equal(legacy, explicit)
        pd.testing.assert_frame_equal(mask, explicit_mask)
        evidence, evidence_mask = estimate_fused_waveform(*case, fps=FPS, hr_mode='evidence')
        for frame in case[1:]:
            frame['reference_bpm'] = 200.
            frame['ground_truth_bpm'] = 45.
        case[3]['ridge_bpm'] = 48.
        repeated, _ = estimate_fused_waveform(*case, fps=FPS, hr_mode='evidence')
        pd.testing.assert_frame_equal(evidence[HR_COLUMNS], repeated[HR_COLUMNS])
        pd.testing.assert_frame_equal(mask, evidence_mask)
        self.assertLess(np.max(abs(evidence.ridge_bpm-84)), 1.1)
        self.assertTrue(evidence.loc[~evidence.accepted, 'ridge_bpm'].isna().all())

    def test_empty_and_short_input_have_complete_typed_schema(self):
        for seconds in (0, 3):
            case = fixture(seconds=seconds)
            table, _ = estimate_fused_waveform(*case, fps=FPS, hr_mode='evidence')
            self.assertEqual(len(table), 0)
            self.assertEqual(table.accepted.dtype, np.dtype(bool))
            self.assertEqual(table.ridge_bpm.dtype, np.dtype(float))
            self.assertIn('evidence_candidates_json', table)

    def test_invalid_config_and_motion_fail_explicitly(self):
        with self.assertRaises(ValueError):
            EvidenceConfig(min_relative_power=2.)
        _, wave = waves()
        with self.assertRaises(ValueError):
            score_candidates(wave[:300], FPS, GRID, dict(profile=np.ones(2),
                available=True, strength=1., reliability=1.))


if __name__ == '__main__':
    unittest.main()
