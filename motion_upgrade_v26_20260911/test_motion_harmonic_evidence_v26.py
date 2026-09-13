"""Synthetic physical-motion controls; no human video or reference is loaded."""
import unittest

import numpy as np
import pandas as pd

from motion_evidence import motion_evidence
from motion_harmonic_evidence_v26 import (
    motion_evidence_harmonics, DIRECT_WEIGHT, DOUBLE_WEIGHT, HALF_WEIGHT,
)


def trace_with_velocity(fps, seconds=10, motion_bpm=180, rms_speed=0.5):
    n = int(round(seconds * fps))
    t = np.arange(n) / fps
    velocity = np.sqrt(2.0) * rms_speed * np.sin(2 * np.pi * motion_bpm / 60 * t)
    # Frozen trace contract: displacement / IMAGE height per frame, not velocity.
    return pd.DataFrame({
        'time_s': t, 'face_y0': np.full(n, .2), 'face_y1': np.full(n, .6),
        'motion_x': velocity * .4 / fps, 'motion_y': np.zeros(n),
    })


class MotionHarmonicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.arange(42., 211.)

    def evidence(self, trace, fps, grid=None):
        return motion_evidence_harmonics(trace, 0, len(trace), fps,
                                         self.grid if grid is None else grid)

    def at(self, evidence, bpm, key='profile'):
        return evidence[key][int(bpm - self.grid[0])]

    def test_fixed_weights_and_exact_soft_combination(self):
        self.assertEqual((DIRECT_WEIGHT, DOUBLE_WEIGHT, HALF_WEIGHT), (1., .8, .5))
        e = self.evidence(trace_with_velocity(30), 30)
        expected = np.maximum.reduce([e['profile_direct'], .8 * e['profile_double'],
                                      .5 * e['profile_half']])
        np.testing.assert_array_equal(e['profile'], expected)
        self.assertLessEqual(e['profile'].max(), e['strength'] * e['reliability'] + 1e-15)

    def test_tiny_true_same_frequency_micro_motion_remains_tiny(self):
        e = self.evidence(trace_with_velocity(30, motion_bpm=90, rms_speed=.005), 30)
        self.assertTrue(e['available'])
        self.assertAlmostEqual(e['speed_face_per_s'], .005, places=12)
        self.assertLess(self.at(e, 90), .0005)
        self.assertLess(e['profile'].max(), .0005)
        # A colocated pulse at 90 BPM may be real. This function returns only
        # the bounded cue, and has no pulse waveform or rejection interface.
        self.assertNotIn('accepted', e)

    def test_large_double_frequency_motion_reaches_fundamental(self):
        e = self.evidence(trace_with_velocity(30, motion_bpm=180, rms_speed=.5), 30)
        self.assertGreater(self.at(e, 90), .63)
        self.assertLess(self.at(e, 90), .65)
        self.assertLess(self.at(e, 90, 'profile_direct'), .001)
        self.assertEqual(e['dominant_relation'][90 - 42], 'double')

    def test_motion_above_hr_search_max_is_not_lost(self):
        e = self.evidence(trace_with_velocity(30, motion_bpm=300, rms_speed=.5), 30)
        self.assertEqual(e['extended_grid_bpm'][0], 21.)
        self.assertEqual(e['extended_grid_bpm'][-1], 420.)
        self.assertGreater(self.at(e, 150, 'profile_double'), .79)
        self.assertGreater(self.at(e, 150), .63)
        self.assertLess(self.at(e, 150, 'profile_direct'), .001)

    def test_half_frequency_relation_has_fixed_weaker_weight(self):
        e = self.evidence(trace_with_velocity(30, motion_bpm=75, rms_speed=.5), 30)
        self.assertAlmostEqual(self.at(e, 150), .4, delta=.002)
        self.assertEqual(e['dominant_relation'][150 - 42], 'half')

    def test_observed_no_motion_does_not_penalize_real_high_hr(self):
        trace = trace_with_velocity(30, motion_bpm=200, rms_speed=0.)
        # Adding a real pulse diagnostic cannot affect a geometric-only cue.
        trace['synthetic_true_pulse'] = np.sin(2 * np.pi * 200 / 60 * trace.time_s)
        e = self.evidence(trace, 30)
        self.assertTrue(e['available'])
        self.assertEqual(e['status'], 'observed_flat_motion')
        self.assertEqual(e['strength'], 0.)
        np.testing.assert_array_equal(e['profile'], np.zeros(len(self.grid)))
        self.assertTrue((e['dominant_relation'] == 'none').all())

    def test_same_physical_motion_at_30_and_180_fps(self):
        low = self.evidence(trace_with_velocity(30, motion_bpm=300), 30)
        high = self.evidence(trace_with_velocity(180, motion_bpm=300), 180)
        self.assertAlmostEqual(low['speed_face_per_s'], high['speed_face_per_s'], places=12)
        self.assertAlmostEqual(low['strength'], high['strength'], places=12)
        np.testing.assert_allclose(low['profile'], high['profile'], atol=.0003, rtol=0.)
        self.assertGreater(self.at(low, 150), .63)
        self.assertGreater(self.at(high, 150), .63)

    def test_missing_motion_is_unknown_not_observed_low_motion(self):
        trace = trace_with_velocity(30).drop(columns=['motion_x'])
        e = self.evidence(trace, 30)
        self.assertFalse(e['available'])
        self.assertTrue(np.isnan(e['strength']))
        self.assertTrue(np.isnan(e['speed_face_per_s']))
        self.assertEqual(e['reliability'], 0.)
        np.testing.assert_array_equal(e['profile'], np.zeros(len(self.grid)))
        self.assertTrue((e['dominant_relation'] == 'unknown').all())

    def test_fragmented_valid_motion_is_not_interpolated(self):
        trace = trace_with_velocity(30)
        trace.loc[np.arange(len(trace)) % 60 == 59, 'motion_x'] = np.nan
        e = self.evidence(trace, 30)
        self.assertGreater(e['observed_fraction'], .9)
        self.assertFalse(e['available'])
        self.assertEqual(e['status'], 'insufficient_contiguous_motion_support')
        self.assertEqual(e['spectral_coverage'], 0.)

    def test_partial_contiguous_support_keeps_absolute_scale(self):
        trace = trace_with_velocity(30)
        trace.loc[180:, 'motion_x'] = np.nan
        legacy = motion_evidence(trace, 0, len(trace), 30, self.grid)
        e = self.evidence(trace, 30)
        for key in ('available', 'observed_fraction', 'speed_face_per_s', 'strength',
                    'reliability', 'spectral_coverage', 'status'):
            self.assertEqual(e[key], legacy[key])
        self.assertAlmostEqual(e['reliability'], .6)
        self.assertAlmostEqual(self.at(e, 90), .8 * .8 * .6, delta=.002)

    def test_no_false_double_support_at_or_above_nyquist(self):
        fps = 10
        trace = trace_with_velocity(fps, motion_bpm=180)
        grid = np.array([90., 149., 150., 200.])
        e = self.evidence(trace, fps, grid)
        np.testing.assert_array_equal(e['frequency_support_double'], [True, True, False, False])
        np.testing.assert_array_equal(e['profile_double'][2:], [0., 0.])
        self.assertLess(e['extended_grid_bpm'][-1], fps * 30)
        self.assertTrue(e['frequency_support_direct'].all())

    def test_bad_window_or_grid_is_rejected(self):
        trace = trace_with_velocity(30)
        for grid in ([], [0, 90], [90, 90], [91, 90], [90, np.nan], [900]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                self.evidence(trace, 30, grid)
        with self.assertRaises(ValueError):
            motion_evidence_harmonics(trace, 0, len(trace) + 1, 30, self.grid)

    def test_trace_is_unchanged_and_reference_columns_are_ignored(self):
        trace = trace_with_velocity(30)
        original = trace.copy(deep=True)
        first = self.evidence(trace, 30)
        pd.testing.assert_frame_equal(trace, original)
        trace['reference_bpm'] = np.linspace(50, 190, len(trace))
        trace['video_id'] = 'arbitrary_not_a_method_input'
        second = self.evidence(trace, 30)
        for key in ('profile', 'profile_direct', 'profile_double', 'profile_half'):
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == '__main__':
    unittest.main()
