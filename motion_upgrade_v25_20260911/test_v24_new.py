"""Independent synthetic regressions for causal anchoring and branch routing.

No physiological labels or real-video metrics enter these tests. The synthetic
84 bpm input is a controlled numerical colour modulation, not human evidence.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_motion_v2 as cli
from anchored_reconstruction import anchored_reconstruction, baseline_view
from guarded_fusion import choose_branch, tracking_eligibility, guarded_fuse
from motion_fusion import fuse_windows
from waveform_hr import estimate_fused_waveform
from test_fusion_v2 import make_case


ROIS = ('forehead', 'left_cheek', 'right_cheek')
FPS = 30.


def controlled_trace(n=600, bias=None, baseline_motion=False):
    time = np.arange(n)/FPS
    pulse = np.sin(2*np.pi*1.4*time)
    truth = np.log([120., 100., 90.])+pulse[:, None]*[.0006, .002, .00025]
    delta = np.diff(truth, axis=0, prepend=truth[:1])
    if bias is not None:
        delta[1:] += bias
    baseline = truth.copy()
    if baseline_motion:
        baseline += np.sin(2*np.pi*2.6*time)[:, None]*[.02, .08, -.01]
    virtual = truth[0]+np.cumsum(delta, axis=0)
    records = dict(frame=np.arange(n), time_s=time, rgb_valid=True, source='mesh',
                   face_detected=True, motion_x=0., motion_y=0., pixel_mode='tracking_screened')
    for c, j in zip('rgb', range(3)):
        records[c] = np.exp(virtual[:, j])
        records[f'baseline_{c}'] = np.exp(baseline[:, j])
    for roi in ROIS:
        records[f'{roi}_valid'] = np.ones(n, bool)
        records[f'{roi}_quality'] = np.ones(n)
        records[f'{roi}_pixel_source'] = ['baseline_reset']+['tracked_ratio']*(n-1)
        records[f'{roi}_pixel_tracks'] = np.r_[0, np.full(n-1, 100)]
        records[f'{roi}_pixel_reset'] = np.r_[True, np.zeros(n-1, bool)]
        for c, j in zip('rgb', range(3)):
            records[f'{roi}_{c}'] = np.exp(virtual[:, j])
            records[f'baseline_{roi}_{c}'] = np.exp(baseline[:, j])
            records[f'{roi}_pixel_log_delta_{c}'] = np.r_[np.nan, delta[1:, j]]
    return pd.DataFrame(records), truth


def region_rgb(trace, roi='forehead'):
    return trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy(float)


class CausalAnchorTests(unittest.TestCase):
    def test_common_pulse_is_preserved_and_input_audit_is_unchanged(self):
        trace, truth = controlled_trace()
        saved = trace.copy(deep=True)
        anchored = anchored_reconstruction(trace, FPS)
        for roi in ROIS:
            np.testing.assert_allclose(np.log(region_rgb(anchored, roi)), truth, rtol=0, atol=2e-12)
            for c in 'rgb':
                np.testing.assert_array_equal(anchored[f'unanchored_{roi}_{c}'], trace[f'{roi}_{c}'])
        for c in ('frame', 'time_s', 'source', 'face_detected', 'rgb_valid'):
            pd.testing.assert_series_equal(anchored[c], trace[c])
        pd.testing.assert_frame_equal(trace, saved)
        original = baseline_view(anchored)
        for roi in ROIS:
            np.testing.assert_array_equal(region_rgb(original, roi), trace[[f'baseline_{roi}_{c}' for c in 'rgb']])

    def test_constant_tiny_increment_bias_does_not_grow_without_bound(self):
        trace, truth = controlled_trace(n=3600, bias=[.00003, .00015, -.00002])
        anchored = anchored_reconstruction(trace, FPS)
        error = np.log(region_rgb(anchored))-truth
        self.assertLess(np.max(np.abs(error[:, 1])), .014)
        # Unbounded integration would exceed 0.53 log units by 120 seconds.
        self.assertGreater(np.log(region_rgb(trace)[-1, 1])-truth[-1, 1], .53)
        self.assertLess(abs(error[-1, 1]-error[1799, 1]), 1e-7)
        self.assertGreater(error[-1, 1], 0.)

    def test_fourth_order_anchor_does_not_leak_strong_2p6hz_baseline_motion(self):
        trace, truth = controlled_trace(n=1800, baseline_motion=True)
        anchored = anchored_reconstruction(trace, FPS)
        time = trace.time_s.to_numpy()[600:]
        design = np.column_stack([np.sin(2*np.pi*2.6*time), np.cos(2*np.pi*2.6*time), np.ones(len(time))])
        error = np.log(region_rgb(anchored))[600:]-truth[600:]
        beta = np.linalg.lstsq(design, error[:, 1], rcond=None)[0]
        self.assertLess(np.hypot(beta[0], beta[1]), 2e-6)
        observed = np.log(region_rgb(anchored))[600:, 1]
        self.assertGreater(np.corrcoef(observed, truth[600:, 1])[0, 1], .999)

    def test_causal_prefix_does_not_change_when_future_frames_are_appended(self):
        trace, _ = controlled_trace(n=900, bias=[0., .00015, 0.])
        short = anchored_reconstruction(trace.iloc[:400].copy(), FPS)
        full = anchored_reconstruction(trace, FPS).iloc[:400]
        pd.testing.assert_frame_equal(short, full)

    def test_missing_region_and_nonadjacent_frames_reset_without_zero_filling(self):
        trace, _ = controlled_trace(n=200, bias=[0., .00015, 0.])
        uninterrupted = anchored_reconstruction(trace, FPS)
        damaged = trace.copy(deep=True)
        damaged.loc[70, 'left_cheek_valid'] = False
        for c in 'rgb':
            damaged.loc[70, f'baseline_left_cheek_{c}'] = np.nan
            damaged.loc[70, f'left_cheek_{c}'] = np.nan
        output = anchored_reconstruction(damaged, FPS)
        self.assertTrue(np.isnan(region_rgb(output, 'left_cheek')[70]).all())
        self.assertEqual(output.left_cheek_pixel_source.iloc[70], 'missing')
        self.assertEqual(output.left_cheek_pixel_source.iloc[71], 'baseline_reset')
        np.testing.assert_array_equal(region_rgb(output, 'left_cheek')[71],
                                      damaged[[f'baseline_left_cheek_{c}' for c in 'rgb']].iloc[71])
        np.testing.assert_array_equal(region_rgb(output, 'forehead'), region_rgb(uninterrupted, 'forehead'))
        damaged.loc[120:, 'frame'] += 3
        skipped = anchored_reconstruction(damaged, FPS)
        for roi in ROIS:
            self.assertEqual(skipped[f'{roi}_pixel_source'].iloc[120], 'baseline_reset')
            np.testing.assert_array_equal(region_rgb(skipped, roi)[120],
                                          damaged[[f'baseline_{roi}_{c}' for c in 'rgb']].iloc[120])

    def test_fallback_source_does_not_reuse_an_untrusted_tracked_increment(self):
        trace, truth = controlled_trace(n=300, bias=[.01, .02, -.01])
        for roi in ROIS:
            trace.loc[1:, f'{roi}_pixel_source'] = 'baseline_ratio_fallback'
        anchored = anchored_reconstruction(trace, FPS)
        for roi in ROIS:
            np.testing.assert_allclose(np.log(region_rgb(anchored, roi)), truth, rtol=0, atol=2e-12)
            self.assertTrue((anchored[f'{roi}_pixel_source'].iloc[1:] == 'baseline_ratio_fallback').all())


class GuardedFusionTests(unittest.TestCase):
    def row(self, bpm=84., score=.6):
        return dict(ridge_bpm=bpm, quality_proxy=score, motion_overlap=0., runner_up_ratio=0.)

    def test_branch_disagreement_preserves_baseline_despite_larger_score(self):
        choice = choose_branch(self.row(), self.row(144., .99), True, True, True)
        self.assertEqual(choice[0], 'baseline')
        self.assertEqual(choice[1], 'frequency_disagreement_fallback')
        self.assertEqual(choose_branch(self.row(), self.row(85., .95), True, True, True)[0], 'tracked')
        self.assertEqual(choose_branch(self.row(), self.row(85., .95), True, True, False)[0], 'baseline')
        self.assertEqual(choose_branch(self.row(), self.row(85., .95), False, True, True)[0], 'tracked')
        self.assertEqual(choose_branch(self.row(), self.row(85., .95), False, True, False)[0], 'none')

    def test_tracking_quorum_counts_only_rois_that_really_contributed(self):
        trace, _ = controlled_trace(n=300)
        trace['left_cheek_pixel_source'] = 'baseline_ratio_fallback'
        self.assertTrue(tracking_eligibility(trace, 0, 300)[0])
        self.assertFalse(tracking_eligibility(trace, 0, 300, contributing_rois=['forehead', 'left_cheek'])[0])
        self.assertTrue(tracking_eligibility(trace, 0, 300, contributing_rois=['forehead', 'right_cheek'])[0])
        trace['right_cheek_pixel_source'] = 'numerical_reset'
        self.assertFalse(tracking_eligibility(trace, 0, 300)[0])

    def fixture(self):
        time, pulse, signals, trace = make_case(seconds=24, bpm=84)
        for roi in ROIS:
            trace[f'{roi}_pixel_source'] = 'tracked_ratio'
        return time, pulse, signals, trace

    def test_actual_routing_keeps_original_waveform_on_frequency_disagreement(self):
        time, _, signals, trace = self.fixture()
        different = {name: np.sin(2*np.pi*2.4*time) for name in signals}
        baseline_table, baseline_wave, _ = fuse_windows(signals, trace, FPS)
        table, wave, diagnostics, routing, _ = guarded_fuse(signals, different, trace, FPS)
        self.assertTrue((routing.selected_branch == 'baseline').all())
        self.assertTrue((routing.selection_reason == 'frequency_disagreement_fallback').all())
        np.testing.assert_allclose(wave, baseline_wave, rtol=0, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(table.accepted, baseline_table.accepted)
        self.assertTrue((diagnostics.loc[diagnostics.branch == 'tracked', 'waveform_weight'] == 0).all())
        self.assertTrue((table.waveform_roi_count <= 3).all())

    def test_qualified_tracking_can_fill_baseline_gap_and_hr_uses_saved_wave(self):
        _, _, signals, trace = self.fixture()
        absent = {name: np.full_like(value, np.nan) for name, value in signals.items()}
        table, wave, diagnostics, routing, _ = guarded_fuse(absent, signals, trace, FPS)
        self.assertTrue((routing.selected_branch == 'tracked').all())
        self.assertTrue((routing.selection_reason == 'baseline_unavailable').all())
        roi_wave = pd.DataFrame({'time_s': trace.time_s})
        for roi in ROIS:
            roi_wave[f'{roi}_interpolated'] = False
        table['waveform_generated'] = table.accepted & (table.waveform_roi_count >= 2)
        hr, provenance = estimate_fused_waveform(wave, trace, roi_wave, table, diagnostics, FPS)
        self.assertTrue(hr.accepted.all())
        self.assertLess(np.max(np.abs(hr.ridge_bpm-84)), 1.1)
        np.testing.assert_array_equal(np.isfinite(wave), provenance.covered)
        self.assertTrue((diagnostics.loc[diagnostics.branch == 'baseline', 'waveform_weight'] == 0).all())

    def test_no_branch_produces_nan_waveform_not_zero_or_held_hr(self):
        _, _, signals, trace = self.fixture()
        absent = {name: np.full_like(value, np.nan) for name, value in signals.items()}
        table, wave, diagnostics, routing, _ = guarded_fuse(absent, absent, trace, FPS)
        self.assertTrue((routing.selected_branch == 'none').all())
        self.assertFalse(table.accepted.any())
        self.assertTrue(np.isnan(wave).all())
        self.assertTrue(table.ridge_bpm.isna().all())
        self.assertTrue((diagnostics.waveform_weight == 0).all())


class V24CLIBoundaryTests(unittest.TestCase):
    def test_guarded_short_clip_emits_complete_empty_result_schema(self):
        trace, _ = controlled_trace(n=90)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/'controlled.avi'
            video.write_bytes(b'identity fixture; decoding is mocked')
            output = root/'result'
            args = ['analyze_motion_v2.py', str(video), '--output', str(output),
                    '--algorithm-mode', 'guarded_fusion', '--pixel-mode', 'tracking_screened']
            with patch.object(cli, 'extract', return_value=(trace, FPS)), \
                 patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['branch_selection_counts'], {})
            self.assertEqual(len(pd.read_csv(output/'branch_routing.csv')), 0)
            for variant in ('pos', 'chrom', 'fusion'):
                hr = pd.read_csv(output/f'{variant}_heart_rate.csv')
                self.assertEqual(len(hr), 0)
                self.assertIn('accepted', hr)


if __name__ == '__main__':
    unittest.main()
