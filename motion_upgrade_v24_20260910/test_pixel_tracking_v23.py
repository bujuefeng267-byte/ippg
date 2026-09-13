"""Independent semantic checks for paired-pixel log-colour reconstruction.

Only controlled arrays and the newly implemented project module are used.
Injected colour modulations are known numerical inputs, not physiological data.
"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import pixel_tracking as pixels
import baseline_frontend as baseline
from test_frontend_v2 import landmarks_for_box
from pixel_tracking import PixelConfig, PixelTracker, paired_log_change


class PairedColourChangeTests(unittest.TestCase):
    def setUp(self):
        self.config = PixelConfig()
        rng = np.random.default_rng(71023)
        self.previous = rng.uniform(70.0, 150.0, size=(40, 3))

    def call(self, previous, current, screened=True):
        delta, kept, info = paired_log_change(
            previous, current, screened=screened, config=self.config)
        self.assertEqual(np.asarray(kept).shape, (len(previous),))
        self.assertEqual(np.asarray(kept).dtype, np.dtype(bool))
        self.assertEqual(info['n_input'], len(previous))
        self.assertEqual(info['n_kept'], np.count_nonzero(kept))
        self.assertLessEqual(info['n_kept'], info['n_finite'])
        self.assertLessEqual(info['n_finite'], info['n_input'])
        return delta, kept, info

    def test_identical_heterogeneous_pixels_have_exactly_zero_change(self):
        for screened in (False, True):
            with self.subTest(screened=screened):
                delta, kept, info = self.call(self.previous, self.previous.copy(), screened)
                np.testing.assert_allclose(delta, np.zeros(3), rtol=0, atol=1e-14)
                self.assertTrue(kept.all())
                self.assertEqual(info['screen_rejected'], 0)

    def test_common_small_rgb_modulation_survives_screening(self):
        # Distinct RGB changes catch channel reversal and over-aggressive
        # rejection of the regional common component.
        for known in ([.00012, -.00023, .00008], [.04, .04, .04], [-.02, .03, -.01]):
            known = np.asarray(known)
            current = self.previous * np.exp(known)
            for screened in (False, True):
                with self.subTest(known=known.tolist(), screened=screened):
                    delta, kept, info = self.call(self.previous, current, screened)
                    np.testing.assert_allclose(delta, known, rtol=0, atol=1e-13)
                    self.assertTrue(kept.all())
                    self.assertEqual(info['screen_rejected'], 0)

    def test_local_outliers_are_removed_without_removing_common_modulation(self):
        common = np.array([.0002, -.0004, .0001])
        current = self.previous * np.exp(common)
        current[:8] *= np.exp([.15, -.12, .10])
        screened, kept, info = self.call(self.previous, current, True)
        raw, _, raw_info = self.call(self.previous, current, False)
        self.assertFalse(kept[:8].any())
        self.assertTrue(kept[8:].all())
        self.assertEqual(info['screen_rejected'], 8)
        np.testing.assert_allclose(screened, common, rtol=0, atol=1e-13)
        self.assertGreater(np.linalg.norm(raw - common), .005)
        self.assertEqual(raw_info['screen_rejected'], 0)

    def test_absolute_colour_distribution_does_not_create_a_change(self):
        # Surviving groups have very different static colours.  Each pair is
        # still compared with its own previous samples, so no false jump is
        # allowed when a different group supplies a subsequent pair.
        common = np.array([.0001, .0003, -.0002])
        groups = [np.tile([55., 75., 95.], (20, 1)),
                  np.tile([155., 175., 195.], (20, 1))]
        for previous in groups:
            for screened in (False, True):
                delta, _, _ = self.call(previous, previous * np.exp(common), screened)
                np.testing.assert_allclose(delta, common, rtol=0, atol=1e-13)

    def test_invalid_pairs_are_excluded_and_never_zero_filled(self):
        previous = self.previous.copy()
        current = previous * np.exp([.001, -.002, .003])
        previous[0, 0] = np.nan
        previous[1, 1] = np.inf
        previous[2, 2] = 0.
        current[3, 0] = np.nan
        current[4, 1] = np.inf
        current[5, 2] = 0.
        current[6, 0] = 255.
        previous[7, 1] = -1.
        for screened in (False, True):
            delta, kept, info = self.call(previous, current, screened)
            self.assertFalse(kept[:8].any())
            self.assertTrue(kept[8:].all())
            self.assertEqual(info['n_finite'], 32)
            np.testing.assert_allclose(delta, [.001, -.002, .003], rtol=0, atol=1e-13)

    def test_minimum_11_12_pair_boundary(self):
        for count in (0, 1, 11, 12):
            for screened in (False, True):
                with self.subTest(count=count, screened=screened):
                    previous = self.previous[:count]
                    delta, kept, _ = self.call(previous, previous * np.exp(.001), screened)
                    if count < 12:
                        self.assertIsNone(delta)
                    else:
                        np.testing.assert_allclose(delta, [.001] * 3, rtol=0, atol=1e-13)

    def test_minimum_pair_count_is_rechecked_after_screening(self):
        previous = self.previous[:14]
        current = previous * np.exp([.001, -.002, .003])
        current[:3] *= np.exp([.12, -.10, .08])
        delta, kept, info = self.call(previous, current, True)
        self.assertIsNone(delta)
        self.assertEqual(np.count_nonzero(kept), 11)
        self.assertEqual(info['screen_rejected'], 3)

    def test_reordering_pairs_cannot_change_result(self):
        common = np.array([-.0005, .001, .0003])
        current = self.previous * np.exp(common)
        current[:5] *= np.exp([.1, -.1, .1])
        order = np.random.default_rng(7).permutation(len(current))
        for screened in (False, True):
            first, kept, info = self.call(self.previous, current, screened)
            second, kept_order, info_order = self.call(self.previous[order], current[order], screened)
            np.testing.assert_allclose(first, second, rtol=0, atol=1e-13)
            np.testing.assert_array_equal(kept[order], kept_order)
            self.assertEqual(info['screen_rejected'], info_order['screen_rejected'])

    def test_input_arrays_are_not_mutated(self):
        previous = self.previous.copy()
        current = previous * np.exp(.001)
        previous[0, 1] = np.nan
        current[1, 2] = np.inf
        previous_saved = previous.copy()
        current_saved = current.copy()
        self.call(previous, current, True)
        np.testing.assert_array_equal(previous, previous_saved)
        np.testing.assert_array_equal(current, current_saved)

    def test_color_change_is_computed_in_float_without_uint8_wrap(self):
        previous = np.tile(np.array([110, 150, 210], dtype=np.uint8), (40, 1))
        current = np.tile(np.array([109, 153, 208], dtype=np.uint8), (40, 1))
        known = np.log(current[0].astype(float) / previous[0].astype(float))
        for screened in (False, True):
            delta, kept, _ = self.call(previous, current, screened)
            np.testing.assert_allclose(delta, known, rtol=0, atol=1e-13)
            self.assertTrue(kept.all())


def baseline_record(image, index, box=None):
    h, w = image.shape[:2]
    if box is None:
        box = (0, 0, w, h)
    row = baseline.sample_rois(image, landmarks_for_box(box, (h, w)))
    row.update(frame=index, time_s=index / 30., source='mesh', face_detected=True,
               flow_available=False, bridge_age_frames=0, motion_x=0., motion_y=0.,
               face_x0=box[0]/w, face_y0=box[1]/h,
               face_x1=box[2]/w, face_y1=box[3]/h)
    return row


class PixelTrackerIntegrationTests(unittest.TestCase):
    def test_real_lk_translation_keeps_common_pulse_and_region_identity(self):
        # A known translating colour texture tests actual OpenCV forward/back
        # tracking and patch correspondence. No camera, detector or HR label.
        rng = np.random.default_rng(73023)
        texture = cv2.GaussianBlur(rng.normal(0, 18, (240, 300)).astype(np.float32), (3, 3), .65)
        original = np.stack([120 + texture, 105 + texture, 90 + texture], axis=2)
        n_frames = 20
        modulation = np.sin(2*np.pi*np.arange(n_frames)/10)
        known = modulation[:, None] * np.array([.004, .012, -.003])[None, :]
        for mode in ('tracking_only', 'tracking_screened'):
            with self.subTest(mode=mode):
                tracker = PixelTracker(mode=mode)
                rows = []
                for index in range(n_frames):
                    shift = index % 4
                    image = (np.roll(original, shift, axis=1) * np.exp(known[index])).astype(np.float32)
                    record = baseline_record(image, index, (shift, 0, 300+shift, 240))
                    rows.append(tracker.update(image, record, index))
                for name in baseline.REGION_NAMES:
                    accepted = [r[f'{name}_pixel_source'] == 'tracked_ratio' for r in rows[1:]]
                    self.assertGreaterEqual(sum(accepted), 17)
                    values = np.array([[r[f'{name}_{c}'] for c in 'rgb'] for r in rows])
                    observed = np.log(values / values[0])
                    # Bound accumulated correspondence noise relative to the
                    # known 1.2% green modulation; don't assert exact LK pixels.
                    np.testing.assert_allclose(observed, known, rtol=0, atol=.0018)
                    self.assertGreater(np.corrcoef(observed[:, 1], modulation)[0, 1], .99)
                self.assertEqual([r['frame'] for r in rows], list(range(n_frames)))
                np.testing.assert_allclose([r['time_s'] for r in rows], np.arange(n_frames)/30.)

    def test_lk_failure_has_explicit_fallback_and_gap_has_explicit_reset(self):
        tracker = PixelTracker()
        image = np.full((200, 200, 3), [100, 120, 140], dtype=np.uint8)
        first = baseline_record(image, 0)
        row0 = tracker.update(image, first, 0)
        changed = np.full_like(image, [101, 118, 142])
        row1 = baseline_record(changed, 1)
        with patch.object(pixels.cv2, 'calcOpticalFlowPyrLK', return_value=(None, None, None)):
            out1 = tracker.update(changed, row1, 1)
        for name in baseline.REGION_NAMES:
            self.assertEqual(row0[f'{name}_pixel_source'], 'baseline_reset')
            self.assertEqual(out1[f'{name}_pixel_source'], 'baseline_ratio_fallback')
            self.assertEqual(out1[f'{name}_pixel_tracks'], 0)
            self.assertFalse(out1[f'{name}_pixel_reset'])
            np.testing.assert_allclose([out1[f'{name}_{c}'] for c in 'rgb'], [101, 118, 142])
        missing = baseline_record(np.zeros_like(image), 2)
        out2 = tracker.update(np.zeros_like(image), missing, 2)
        out3 = tracker.update(image, baseline_record(image, 3), 3)
        out8 = tracker.update(changed, baseline_record(changed, 8), 8)
        for name in baseline.REGION_NAMES:
            self.assertEqual(out2[f'{name}_pixel_source'], 'missing')
            self.assertFalse(out2[f'{name}_valid'])
            self.assertTrue(np.isnan(out2[f'{name}_g']))
            for recovered, expected in ((out3, [100, 120, 140]), (out8, [101, 118, 142])):
                self.assertEqual(recovered[f'{name}_pixel_source'], 'baseline_reset')
                self.assertTrue(recovered[f'{name}_pixel_reset'])
                np.testing.assert_array_equal([recovered[f'{name}_{c}'] for c in 'rgb'], expected)

    def test_fb_rejects_backward_failure_large_error_and_forward_failure(self):
        tracker = PixelTracker()
        tracker.previous_gray = np.zeros((100, 100), np.uint8)
        points = np.array([[20, 20], [30, 30], [40, 40], [50, 50]], np.float32)
        forward = points + [1, 0]
        # Only first 3 forward statuses succeed; backward status fails for
        # point 1, and point 2 exceeds the geometric tolerance.
        reverse = points[:3].copy()
        reverse[2] += [2, 0]
        with patch.object(pixels.cv2, 'calcOpticalFlowPyrLK', side_effect=[
                (forward.astype(np.float32), np.array([[1], [1], [1], [0]], np.uint8), None),
                (reverse, np.array([[1], [0], [1]], np.uint8), None)]):
            _, good, error = tracker._flow(np.zeros((100, 100), np.uint8), points)
        np.testing.assert_array_equal(good, [True, False, False, False])
        self.assertEqual(error[0], 0.)
        self.assertGreater(error[2], tracker.config.fb_max_px)

    def test_patch_sampling_rejects_entire_patch_outside_image(self):
        image = np.full((30, 40, 3), [70., 100., 150.], np.float32)
        # These centres would wrap/clip if only the centre or int cast were checked.
        points = [[10, 10], [-1, 10], [2, 10], [38, 10], [10, 28], [np.nan, 10], [np.inf, 10]]
        sampled = pixels.sample_patches(image, points, 3)
        np.testing.assert_array_equal(sampled[0], [70., 100., 150.])
        self.assertTrue(np.isnan(sampled[1:]).all())

    def test_single_region_loss_does_not_reset_other_regions(self):
        tracker = PixelTracker()
        image = np.full((200, 200, 3), [100, 120, 140], np.uint8)
        tracker.update(image, baseline_record(image, 0), 0)
        masked = image.copy()
        masked[92:140, 32:80] = 0
        with patch.object(pixels.cv2, 'calcOpticalFlowPyrLK', return_value=(None, None, None)):
            output = tracker.update(masked, baseline_record(masked, 1), 1)
        self.assertEqual(output['left_cheek_pixel_source'], 'missing')
        self.assertTrue(np.isnan(output['left_cheek_g']))
        for name in ('forehead', 'right_cheek'):
            self.assertTrue(output[f'{name}_valid'])
            self.assertEqual(output[f'{name}_pixel_source'], 'baseline_ratio_fallback')
            self.assertFalse(output[f'{name}_pixel_reset'])
            np.testing.assert_array_equal([output[f'{name}_{c}'] for c in 'rgb'], [100, 120, 140])

    def test_replay_keeps_baseline_time_detection_and_rgb_audit(self):
        image = np.full((200, 200, 3), [90, 120, 170], np.uint8)
        records = [baseline_record(image, i) for i in range(3)]
        records[1].update(source='flow_tracked', face_detected=False, bridge_age_frames=1)
        frame_trace = pd.DataFrame(records)
        frame_trace.attrs['test_marker'] = 'immutable_baseline'
        saved = frame_trace.copy(deep=True)
        frames = [image[:, :, ::-1].copy() for _ in records]

        class Capture:
            index = 0
            released = False
            def isOpened(self): return True
            def get(self, key): return 30.
            def read(self):
                if self.index == len(frames): return False, None
                frame = frames[self.index]
                self.index += 1
                return True, frame
            def release(self): self.released = True

        cap = Capture()
        with patch.object(pixels.cv2, 'VideoCapture', return_value=cap), \
             patch.object(pixels.cv2, 'calcOpticalFlowPyrLK', return_value=(None, None, None)):
            result = pixels.apply_pixel_tracking('controlled_frames', frame_trace, 30., 'tracking_screened')
        self.assertTrue(cap.released)
        self.assertEqual(len(result), len(frame_trace))
        for col in ('frame', 'time_s', 'source', 'face_detected', 'bridge_age_frames',
                    'rgb_valid', 'face_x0', 'face_y0', 'face_x1', 'face_y1'):
            pd.testing.assert_series_equal(result[col], frame_trace[col])
        for name in baseline.REGION_NAMES:
            for channel in 'rgb':
                np.testing.assert_array_equal(result[f'baseline_{name}_{channel}'], frame_trace[f'{name}_{channel}'])
                np.testing.assert_allclose(result[f'{name}_{channel}'], frame_trace[f'{name}_{channel}'])
        pd.testing.assert_frame_equal(frame_trace, saved)
        self.assertEqual(result.attrs['test_marker'], 'immutable_baseline')
        with patch.object(pixels.cv2, 'VideoCapture') as opening:
            disabled = pixels.apply_pixel_tracking('unused', frame_trace, 30., 'disabled')
            opening.assert_not_called()
        pd.testing.assert_frame_equal(disabled, frame_trace)


if __name__ == '__main__':
    unittest.main()
