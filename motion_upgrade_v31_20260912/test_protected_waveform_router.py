"""Synthetic invariants only; no videos, real predictions or reference files."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from evidence_hr import estimate_evidence, evidence_path
from protected_waveform_router import (ProtectionError, constrained_evidence_path,
    read_saved_protected_hr, route_protected_waveform)


def neutral_motion(trace, start, stop, fps, grid):
    return {'available': False}


def inputs(seconds=35, fps=30):
    t = np.arange(round(seconds*fps))/fps
    signal = np.sin(2*np.pi*1.5*t) + .25*np.sin(2*np.pi*3*t+.2)
    trace = pd.DataFrame({'time_s': t, 'rgb_valid': np.ones(len(t), bool)})
    old = pd.DataFrame(dict(time_s=t, base=signal, covered=True, observed=True,
                            interpolated=False, source='v28_original_source'))
    candidate = old.copy(deep=True)
    candidate['base'] = .7*np.sin(2*np.pi*1.8*t+.3)
    hr = estimate_evidence(signal, trace, np.zeros(len(t), bool), fps,
                           motion_trace=trace, motion_provider=neutral_motion)
    return old, hr, candidate, trace, fps


def saved_readout(wave, trace, fps, hr, eligible):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'waveform.csv'
        wave.to_csv(path, index=False)
        return read_saved_protected_hr(path, trace, fps, hr, eligible,
                                       motion_provider=neutral_motion)


class ProtectedRouterTests(unittest.TestCase):
    def test_isolated_eligible_window_has_no_overlap_leakage(self):
        old, hr, candidate, trace, fps = inputs(seconds=20)
        eligible = np.zeros(len(hr), bool)
        eligible[5] = True
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps, eligible)
        self.assertTrue(decisions.eligible.iloc[5])
        self.assertFalse(decisions.effective_changed_window.any())
        self.assertTrue(wave.protected_sample.all())
        for name in old.columns:
            np.testing.assert_array_equal(wave[name], old[name])
        self.assertEqual(wave.base.to_numpy().tobytes(), old.base.to_numpy().tobytes())

    def test_long_run_has_tapered_core_and_exact_protected_windows(self):
        old, hr, candidate, trace, fps = inputs()
        eligible = np.zeros(len(hr), bool)
        eligible[2:23] = True
        snapshots = [item.copy(deep=True) for item in (old, hr, candidate, trace)]
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps, eligible)
        # Last preceding protected window is [1, 11); next is [23, 33).
        expected = np.zeros(len(wave))
        expected[11*fps:23*fps] = np.hanning(12*fps+2)[1:-1]
        np.testing.assert_allclose(wave.v31_candidate_weight, expected, atol=1e-15, rtol=0)
        self.assertLess(expected[11*fps], .001)
        self.assertLess(expected[23*fps-1], .001)
        self.assertTrue(decisions.effective_changed_window.any())
        for i in np.flatnonzero(~eligible):
            a, b = i*fps, (i+10)*fps
            for name in old.columns:
                np.testing.assert_array_equal(wave[name].iloc[a:b], old[name].iloc[a:b])
            self.assertEqual(wave.base.iloc[a:b].to_numpy().tobytes(),
                             old.base.iloc[a:b].to_numpy().tobytes())
        self.assertFalse(decisions.loc[decisions.protected_window,
                                       'effective_changed_window'].any())
        for item, before in zip((old, hr, candidate, trace), snapshots):
            pd.testing.assert_frame_equal(item, before)

    def test_ten_interior_eligible_windows_leave_only_one_second_core(self):
        old, hr, candidate, trace, fps = inputs(seconds=23)
        eligible = np.zeros(len(hr), bool)
        eligible[2:12] = True
        wave, _ = route_protected_waveform(old, hr, candidate, trace, fps, eligible)
        self.assertEqual(int((wave.v31_candidate_weight > 0).sum()), fps)
        np.testing.assert_array_equal(np.flatnonzero(wave.v31_candidate_weight > 0),
                                      np.arange(11*fps, 12*fps))

    def test_gaps_and_contributor_masks_are_preserved_truthfully(self):
        old, hr, candidate, trace, fps = inputs(seconds=30)
        # A genuine missing prefix belongs to protected old rejected windows.
        old.loc[:14, 'base'] = np.nan
        old.loc[:14, ['covered', 'observed', 'interpolated']] = False
        hr = estimate_evidence(old.base.to_numpy(), pd.DataFrame({'rgb_valid': old.observed}),
            old.interpolated.to_numpy(), fps, motion_trace=trace, motion_provider=neutral_motion)
        old.loc[450, ['observed', 'interpolated']] = [False, True]
        candidate.loc[451, ['observed', 'interpolated']] = [False, True]
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps,
                                                   np.ones(len(hr), bool))
        self.assertFalse(decisions.eligible.iloc[0])
        np.testing.assert_array_equal(wave.covered, old.covered)
        np.testing.assert_array_equal(np.isfinite(wave.base), np.isfinite(old.base))
        self.assertEqual(wave.base.iloc[:15].to_numpy().tobytes(),
                         old.base.iloc[:15].to_numpy().tobytes())
        self.assertGreater(wave.v31_candidate_weight.iloc[450], 0)
        self.assertGreater(wave.v31_candidate_weight.iloc[451], 0)
        self.assertFalse(wave.observed.iloc[450])
        self.assertFalse(wave.observed.iloc[451])
        self.assertTrue(wave.interpolated.iloc[450])
        self.assertTrue(wave.interpolated.iloc[451])
        self.assertEqual(wave.source.iloc[0], old.source.iloc[0])
        self.assertEqual(wave.source.iloc[451], 'v28_cdf_measured_mixture')

    def test_candidate_gap_demotes_whole_window_before_union_protection(self):
        old, hr, candidate, trace, fps = inputs(seconds=20)
        candidate.loc[10*fps, 'base'] = np.nan
        candidate.loc[10*fps, ['covered', 'observed', 'interpolated']] = False
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps,
                                                   np.ones(len(hr), bool))
        self.assertTrue((decisions.reason == 'candidate_window_gap').any())
        self.assertFalse(decisions.loc[decisions.protected_window,
                                       'effective_changed_window'].any())
        self.assertEqual(wave.base.iloc[10*fps], old.base.iloc[10*fps])
        np.testing.assert_array_equal(wave.covered, old.covered)

    def test_identical_candidate_preserves_exact_numeric_bits(self):
        old, hr, _, trace, fps = inputs(seconds=12)
        wave, decisions = route_protected_waveform(old, hr, old.copy(), trace, fps,
                                                   np.ones(len(hr), bool))
        self.assertEqual(wave.base.to_numpy().tobytes(), old.base.to_numpy().tobytes())
        self.assertFalse(decisions.effective_changed_window.any())

    def test_constraints_prevent_future_candidate_scores_moving_anchors(self):
        grid = np.array([60., 120.])
        emissions = np.array([[0., 10.], [0., 20.], [0., 10.]])
        unconstrained, _ = evidence_path(emissions, grid, 1.)
        np.testing.assert_array_equal(grid[unconstrained], [120, 120, 120])
        path, reacquired = constrained_evidence_path(emissions, grid, 1., [60, np.nan, 60])
        np.testing.assert_array_equal(grid[path], [60, 120, 60])
        np.testing.assert_array_equal(reacquired, [False, True, True])
        changed_scores = emissions.copy()
        changed_scores[1, 1] = 200000
        altered, _ = constrained_evidence_path(changed_scores, grid, 1., [60, np.nan, 60])
        np.testing.assert_array_equal(grid[altered][[0, 2]], [60, 60])

    def test_anchor_cannot_create_energy_or_offgrid_frequency(self):
        for anchor in (60., 91.):
            with self.subTest(anchor=anchor), self.assertRaises(ProtectionError):
                constrained_evidence_path([[-np.inf, 1]], [60., 120.], 1., [anchor])

    def test_saved_noop_readout_preserves_hr_and_verifies_real_candidates(self):
        old, hr, candidate, trace, fps = inputs(seconds=20)
        eligible = np.zeros(len(hr), bool)
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps, eligible)
        final = saved_readout(wave, trace, fps, hr, decisions.eligible.to_numpy())
        np.testing.assert_array_equal(final.accepted, hr.accepted)
        np.testing.assert_array_equal(final.ridge_bpm, hr.ridge_bpm)
        self.assertTrue(final.protected_anchor_verified.all())
        for _, row in final.iterrows():
            candidates = json.loads(row.evidence_candidates_json)
            self.assertTrue(any(row.ridge_bpm in item['supported_bpm'] for item in candidates))

    def test_real_saved_wave_rejects_forged_unsupported_anchor(self):
        old, hr, _, trace, fps = inputs(seconds=10)
        hr.loc[0, 'ridge_bpm'] = 205.
        with self.assertRaises(ProtectionError):
            saved_readout(old, trace, fps, hr, [False])

    def test_saved_changed_wave_keeps_noneligible_hr_exact(self):
        old, hr, candidate, trace, fps = inputs(seconds=70)
        eligible = np.zeros(len(hr), bool)
        eligible[2:58] = True
        wave, decisions = route_protected_waveform(old, hr, candidate, trace, fps, eligible)
        final = saved_readout(wave, trace, fps, hr, decisions.eligible.to_numpy())
        np.testing.assert_array_equal(final.ridge_bpm[~eligible], hr.ridge_bpm[~eligible])
        self.assertTrue(final.protected_anchor_verified[~eligible].all())
        self.assertTrue((final.ridge_bpm[eligible] != hr.ridge_bpm[eligible]).any())

    def test_old_reject_stays_missing_and_new_gate_failure_is_explicit(self):
        old, hr, _, trace, fps = inputs(seconds=12)
        hr.loc[0, ['accepted', 'ridge_bpm', 'status']] = [False, np.nan, 'original_reject']
        good = saved_readout(old, trace, fps, hr, np.ones(len(hr), bool))
        self.assertFalse(good.accepted.iloc[0])
        self.assertTrue(np.isnan(good.ridge_bpm.iloc[0]))
        self.assertEqual(good.status.iloc[0], 'original_reject')
        flat = old.copy()
        flat['base'] = 0.
        failed = saved_readout(flat, trace, fps, hr, np.ones(len(hr), bool))
        self.assertFalse(failed.accepted.any())
        self.assertTrue((failed.readout_failure.iloc[1:] == 'flat_signal').all())

    def test_invalid_masks_clocks_and_eligibility_are_rejected(self):
        for kind in ('clock', 'mask', 'flags'):
            old, hr, candidate, trace, fps = inputs(seconds=10)
            eligible = [True]
            if kind == 'clock':
                candidate['time_s'] += .01
            elif kind == 'mask':
                candidate.loc[0, 'covered'] = False
            else:
                eligible = ['False']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                route_protected_waveform(old, hr, candidate, trace, fps, eligible)


if __name__ == '__main__':
    unittest.main()
