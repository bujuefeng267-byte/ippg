"""Synthetic numerical colour/motion examples only; no human references read."""
import unittest

import numpy as np
import pandas as pd

from signal_candidates_v26 import (METHODS, ROI_NAMES, DEFAULT_CONFIG,
    SignalCandidateConfig, project_log_window, make_candidate_signals)
from motion_fusion import FusionConfig, fuse_windows


FPS = 30.


def fixture(seconds=24., pulse_hz=1.4, illumination=None, coloured_motion=False):
    n = round(seconds*FPS)
    time = np.arange(n)/FPS
    pulse = np.sin(2*np.pi*pulse_hz*time) + .15*np.sin(4*np.pi*pulse_hz*time+.3)
    trace = pd.DataFrame(dict(frame=np.arange(n), time_s=time))
    for index, roi in enumerate(ROI_NAMES):
        colour = np.log([120.+index*3, 100.+index*2, 85.+index])
        logged = colour + pulse[:, None]*np.array([.0006, .002, .00025])
        if illumination is not None:
            logged += illumination(time)[:, None]
        if coloured_motion:
            logged += (.08*np.sin(2*np.pi*2.6*time+.4))[:, None]*np.array([1., .5, 0.])
        rgb = np.exp(logged)
        for j, c in enumerate('rgb'):
            trace[f'{roi}_{c}'] = rgb[:, j]
            trace[f'baseline_{roi}_{c}'] = rgb[:, j]
        trace[f'{roi}_valid'] = True
        trace[f'{roi}_quality'] = 1.
        trace[f'{roi}_pixel_source'] = 'tracked_ratio'
    trace['motion_x'] = 20*np.sin(2*np.pi*pulse_hz*time)
    trace['motion_y'] = 0.
    trace['face_y0'], trace['face_y1'] = 0., 100.
    return trace, pulse


def amplitude(time, signal, frequency):
    design = np.column_stack([np.sin(2*np.pi*frequency*time),
                              np.cos(2*np.pi*frequency*time), np.ones(len(time))])
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    return float(np.hypot(coefficients[0], coefficients[1]))


class LogSignalCandidateTests(unittest.TestCase):
    def test_common_multiplicative_illumination_cancels_before_frequency_selection(self):
        time = np.arange(300)/FPS
        clean = np.exp(np.log([120., 100., 85.])+np.sin(2*np.pi*1.4*time)[:,None]*[.0006,.002,.00025])
        changed = clean*np.exp(.8*np.sin(2*np.pi*1.4*time)+.6*np.sin(2*np.pi*2.6*time))[:,None]
        for method in METHODS:
            expected, _ = project_log_window(clean, method)
            actual, _ = project_log_window(changed, method)
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
            self.assertGreater(np.std(actual), .0002)

    def test_complete_filtered_waveform_preserves_same_frequency_colour_pulse(self):
        clean, _ = fixture()
        damaged, _ = fixture(illumination=lambda t: .7*np.sin(2*np.pi*1.4*t)+.35*np.sin(2*np.pi*2.6*t))
        expected, _, _ = make_candidate_signals(clean, FPS)
        actual, _, _ = make_candidate_signals(damaged, FPS)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=3e-10, equal_nan=True)
            self.assertGreater(np.nanstd(actual[key]), .8)

    def test_coloured_motion_example_reduces_motion_to_pulse_ratio(self):
        trace, _ = fixture(seconds=30, coloured_motion=True)
        signals, _, _ = make_candidate_signals(trace, FPS)
        time = trace.time_s.to_numpy()
        region = (time >= 5) & (time < 25)
        raw_green = np.log(trace.forehead_g.to_numpy())
        original_ratio = amplitude(time[region], raw_green[region], 2.6)/amplitude(time[region], raw_green[region], 1.4)
        for method in METHODS:
            result = signals[f'forehead_{method}']
            ratio = amplitude(time[region], result[region], 2.6)/amplitude(time[region], result[region], 1.4)
            self.assertLess(ratio, original_ratio*.25)

    def test_collinear_chromatic_input_is_not_self_cancelled_to_zero(self):
        time = np.arange(48)/FPS
        # Pulse/motion along this same colour direction cannot be separated.
        rgb = np.exp(np.log([120.,100.,85.])+np.sin(2*np.pi*1.4*time)[:,None]*[.01,.005,0.])
        for method in METHODS:
            output, diagnostic = project_log_window(rgb, method)
            self.assertEqual(diagnostic['status'], 'colour_degeneracy_measured_component_fallback')
            self.assertGreater(np.std(output), .001)

    def test_achromatic_or_flat_input_is_explicitly_unidentifiable_not_fabricated(self):
        time = np.arange(48)/FPS
        rgb = np.exp(np.log([120.,100.,85.])+.2*np.sin(2*np.pi*1.4*time)[:,None])
        for method in METHODS:
            output, diagnostic = project_log_window(rgb, method)
            self.assertEqual(diagnostic['status'], 'achromatic_or_flat_unidentifiable')
            np.testing.assert_array_equal(output, np.zeros(len(output)))

    def test_pure_achromatic_candidate_fails_existing_flat_signal_gate(self):
        trace, _ = fixture()
        for roi in ROI_NAMES:
            for c, level in zip('rgb', [120.,100.,85.]):
                trace[f'{roi}_{c}'] = level*np.exp(.2*np.sin(2*np.pi*1.4*trace.time_s))
        signals, _, _ = make_candidate_signals(trace, FPS)
        proposals, wave, diagnostics = fuse_windows(signals, trace, FPS,
                                                   config=FusionConfig(methods=METHODS))
        self.assertFalse(proposals.accepted.any())
        self.assertTrue(np.isnan(wave).all())
        self.assertTrue(diagnostics.channel_status.eq('flat_signal').any())

    def test_same_frequency_coloured_nuisance_is_retained_as_unresolved_colour(self):
        time = np.arange(48)/FPS
        # Two latent processes at exactly the same frequency/phase collapse to
        # one observed colour direction. No claim of separating them is valid.
        pulse = np.sin(2*np.pi*1.4*time)
        mixed_colour = np.array([.0006,.002,.00025])+np.array([.01,.005,0.])
        rgb = np.exp(np.log([120.,100.,85.])+pulse[:,None]*mixed_colour)
        for method in METHODS:
            output, diagnostic = project_log_window(rgb, method)
            self.assertEqual(diagnostic['status'], 'colour_degeneracy_measured_component_fallback')
            self.assertLess(diagnostic['colour_covariance_rank_ratio'], DEFAULT_CONFIG.min_identifiable_rank_ratio)
            self.assertGreater(np.std(output), .001)

    def test_coefficients_bounded_even_with_degenerate_second_component(self):
        time = np.arange(48)/FPS
        rgb = np.exp(np.log([120.,100.,85.])+np.sin(2*np.pi*1.4*time)[:,None]*[.01,.02,0.])
        for method in METHODS:
            output, diagnostic = project_log_window(rgb, method)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreaterEqual(diagnostic['alpha'], DEFAULT_CONFIG.min_alpha)
            self.assertLessEqual(diagnostic['alpha'], DEFAULT_CONFIG.max_alpha)
            self.assertEqual(diagnostic['max_signed_coefficient_l1'], 1.)

    def test_gap_limit_and_provenance_preserved_without_expanding_long_gaps(self):
        trace, _ = fixture(seconds=30)
        for roi in ROI_NAMES:
            trace.loc[300:302, f'{roi}_valid'] = False
            trace.loc[600:603, f'{roi}_valid'] = False
        signals, wave, _ = make_candidate_signals(trace, FPS, gap_s=.1)
        for roi in ROI_NAMES:
            self.assertTrue(wave.loc[300:302, f'{roi}_interpolated'].all())
            self.assertFalse(wave.loc[600:603, f'{roi}_interpolated'].any())
            self.assertFalse(wave.loc[300:302, f'{roi}_observed'].any())
            self.assertFalse(wave.loc[600:603, f'{roi}_observed'].any())
            for method in METHODS:
                output = signals[f'{roi}_{method}']
                self.assertTrue(np.isfinite(output[300:303]).all())
                self.assertTrue(np.isnan(output[552:652]).all())
                self.assertTrue(np.isnan(output[:48]).all())
                self.assertTrue(np.isnan(output[-48:]).all())

    def test_postgap_changes_cannot_rewrite_pregap_waveform(self):
        trace, _ = fixture(seconds=30)
        for roi in ROI_NAMES:
            trace.loc[450:459, f'{roi}_valid'] = False
        expected, _, _ = make_candidate_signals(trace, FPS)
        for roi in ROI_NAMES:
            trace.loc[460:, f'{roi}_g'] *= 1+.2*np.sin(np.arange(len(trace)-460))
        actual, _, _ = make_candidate_signals(trace, FPS)
        for key in expected:
            np.testing.assert_array_equal(actual[key][:450], expected[key][:450])

    def test_branches_not_mixed_and_inputs_never_mutated(self):
        trace, _ = fixture()
        saved = trace.copy(deep=True)
        active, _, _ = make_candidate_signals(trace, FPS)
        baseline, _, _ = make_candidate_signals(trace, FPS, rgb_source='baseline')
        for key in active:
            np.testing.assert_array_equal(active[key], baseline[key])
        pd.testing.assert_frame_equal(trace, saved)
        trace['forehead_g'] *= 1+.05*np.sin(2*np.pi*2.1*trace.time_s)
        same_baseline, _, _ = make_candidate_signals(trace, FPS, rgb_source='baseline')
        for key in baseline:
            np.testing.assert_array_equal(same_baseline[key], baseline[key])

    def test_missing_roi_does_not_borrow_another_roi_or_create_quorum(self):
        trace, _ = fixture()
        trace['left_cheek_valid'] = False
        trace['right_cheek_valid'] = False
        signals, wave, _ = make_candidate_signals(trace, FPS)
        for method in METHODS:
            self.assertTrue(np.isnan(signals[f'left_cheek_{method}']).all())
            self.assertTrue(np.isnan(signals[f'right_cheek_{method}']).all())
            self.assertTrue(np.isfinite(signals[f'forehead_{method}'][48:-48]).all())
        self.assertFalse(wave.left_cheek_observed.any())
        self.assertEqual(wave.attrs['downstream_minimum_rois'], 2)
        proposals, fused, _ = fuse_windows(signals, trace, FPS,
                                           config=FusionConfig(methods=METHODS))
        self.assertFalse(proposals.accepted.any())
        self.assertTrue(np.isnan(fused).all())

    def test_reference_like_fields_and_motion_frequency_are_not_used(self):
        trace, _ = fixture()
        original, _, _ = make_candidate_signals(trace, FPS)
        trace['reference_bpm'], trace['ground_truth_bpm'] = 180., 45.
        trace['motion_x'], trace['motion_y'] = 1e6, np.nan
        actual, _, _ = make_candidate_signals(trace, FPS)
        for key in original:
            np.testing.assert_array_equal(actual[key], original[key])

    def test_short_empty_invalid_and_nonadjacent_inputs(self):
        for seconds in (0., 3.):
            trace, _ = fixture(seconds=seconds)
            signals, wave, diagnostic = make_candidate_signals(trace, FPS)
            self.assertEqual(set(signals), {f'{roi}_{method}' for roi in ROI_NAMES for method in METHODS})
            self.assertEqual(len(wave), len(trace))
            self.assertTrue(diagnostic.empty)
            self.assertTrue(all(np.isnan(x).all() for x in signals.values()))
        with self.assertRaises(ValueError):
            SignalCandidateConfig(max_alpha=0.)
        trace, _ = fixture()
        trace.loc[100:, 'frame'] += 1
        with self.assertRaises(ValueError):
            make_candidate_signals(trace, FPS)
        with self.assertRaises(ValueError):
            project_log_window(np.zeros((20,3)), 'logpos')


if __name__ == '__main__':
    unittest.main()
