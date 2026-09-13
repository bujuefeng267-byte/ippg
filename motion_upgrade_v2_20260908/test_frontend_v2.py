"""Array and controlled-sequence checks for independent sampling, no video rerun."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import legacy_motion as legacy
import motion_frontend as frontend


def landmarks_for_box(box, shape=(200, 200)):
    x0, y0, x1, y1 = box
    height, width = shape
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
    return legacy.mesh_object(np.repeat(corners, 25, axis=0), width, height)


def painted_frame():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    colors = {"forehead": [80, 100, 120], "left_cheek": [120, 140, 160], "right_cheek": [160, 180, 200]}
    for name, (x0, y0, x1, y1) in frontend.REGION_BOXES.items():
        image[int(y0*200):int(y1*200), int(x0*200):int(x1*200)] = colors[name]
    return image, landmarks_for_box((0, 0, 200, 200)), colors


class SamplingTests(unittest.TestCase):
    def test_independent_regions_and_unchanged_merged_rgb(self):
        image, landmarks, colors = painted_frame()
        sampled = frontend.sample_rois(image, landmarks)
        for name, color in colors.items():
            np.testing.assert_array_equal([sampled[f"{name}_{c}"] for c in "rgb"], color)
            self.assertTrue(sampled[f"{name}_valid"])
            self.assertEqual(sampled[f"{name}_quality"], 1)
        np.testing.assert_array_equal([sampled[c] for c in "rgb"], legacy.roi_mean_rgb(image, landmarks))

    def test_local_occlusion_does_not_contaminate_other_regions(self):
        image, landmarks, colors = painted_frame()
        image[92:140, 32:80] = 0
        sampled = frontend.sample_rois(image, landmarks)
        self.assertFalse(sampled["left_cheek_valid"])
        self.assertTrue(np.isnan(sampled["left_cheek_r"]))
        self.assertEqual(sampled["left_cheek_quality"], 0)
        self.assertEqual(sampled["roi_valid_regions"], 2)
        self.assertEqual(sampled["roi_status"], "sampled_partial_regions")
        np.testing.assert_array_equal([sampled[f"right_cheek_{c}"] for c in "rgb"], colors["right_cheek"])
        np.testing.assert_array_equal([sampled[c] for c in "rgb"], legacy.roi_mean_rgb(image, landmarks))

    def test_pixel_thresholds_are_strict_and_minimum_is_25(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        landmarks = landmarks_for_box((0, 0, 200, 200))
        patch_pixels = image[24:36, 64:70]
        patch_pixels[:] = 20
        patch_pixels.reshape(-1, 3)[:24] = [21, 100, 244]
        # A sliced view need not be contiguous; address pixels explicitly.
        image[24:28, 64:70] = [21, 100, 244]
        image[28, 64] = [245, 100, 100]
        before = frontend.sample_rois(image, landmarks)
        self.assertEqual(before["forehead_valid_pixels"], 24)
        self.assertFalse(before["forehead_valid"])
        image[28, 64] = [21, 100, 244]
        after = frontend.sample_rois(image, landmarks)
        self.assertTrue(after["forehead_valid"])
        self.assertEqual(after["forehead_valid_pixels"], 25)
        self.assertAlmostEqual(after["forehead_quality"], 25 / (72 * 36))

    def test_black_white_and_nonfinite_pixels_never_become_samples(self):
        landmarks = landmarks_for_box((0, 0, 200, 200))
        for fill in (0.0, 255.0, np.nan, np.inf):
            with self.subTest(fill=fill):
                sampled = frontend.sample_rois(np.full((200, 200, 3), fill), landmarks)
                self.assertFalse(sampled["rgb_valid"])
                for name in frontend.REGION_NAMES:
                    self.assertFalse(sampled[f"{name}_valid"])
                    self.assertEqual(sampled[f"{name}_quality"], 0)
                    self.assertTrue(np.isnan(sampled[f"{name}_r"]))

    def test_missing_small_and_invalid_face_geometry_do_not_fabricate_rgb(self):
        image = np.full((200, 200, 3), 100, dtype=np.uint8)
        examples = [(None, "no_face_or_track"),
                    (landmarks_for_box((50, 50, 80, 80)), "face_too_small"),
                    (landmarks_for_box((np.nan, 0, 100, 100)), "invalid_face_geometry")]
        for landmarks, reason in examples:
            sampled = frontend.sample_rois(image, landmarks)
            self.assertEqual(sampled["roi_status"], reason)
            self.assertFalse(sampled["rgb_valid"])
            self.assertTrue(np.isnan(sampled["r"]))

    def test_outside_geometry_is_not_clamped_into_a_border_strip(self):
        image = np.full((200, 200, 3), 100, dtype=np.uint8)
        sampled = frontend.sample_rois(image, landmarks_for_box((300, -50, 600, 250)))
        self.assertFalse(sampled["rgb_valid"])
        for name in frontend.REGION_NAMES:
            self.assertEqual(sampled[f"{name}_status"], "outside_image")
            self.assertEqual(sampled[f"{name}_pixel_count"], 0)
            self.assertTrue(np.isnan(sampled[f"{name}_r"]))

    def test_partial_frame_intersection_reports_visible_fraction(self):
        image = np.full((200, 200, 3), 100, dtype=np.uint8)
        landmarks = landmarks_for_box((-80, 0, 120, 200))
        sampled = frontend.sample_rois(image, landmarks)
        self.assertTrue(sampled["forehead_valid"])
        self.assertGreater(sampled["forehead_visible_fraction"], 0)
        self.assertLess(sampled["forehead_visible_fraction"], 1)
        self.assertAlmostEqual(sampled["forehead_quality"], sampled["forehead_visible_fraction"])
        self.assertFalse(sampled["left_cheek_valid"])
        self.assertTrue(sampled["right_cheek_valid"])
        np.testing.assert_array_equal([sampled[c] for c in "rgb"], legacy.roi_mean_rgb(image, landmarks))

    def test_sampling_quality_does_not_penalize_valid_uniform_patch_brightness(self):
        landmarks = landmarks_for_box((0, 0, 200, 200))
        dark = frontend.sample_rois(np.full((200, 200, 3), 40, dtype=np.uint8), landmarks)
        light = frontend.sample_rois(np.full((200, 200, 3), 200, dtype=np.uint8), landmarks)
        for name in frontend.REGION_NAMES:
            self.assertEqual(dark[f"{name}_quality"], light[f"{name}_quality"])
            self.assertAlmostEqual(dark[f"{name}_laplacian_var"], 0)
            self.assertLess(dark[f"{name}_brightness"], light[f"{name}_brightness"])


class FrozenSequenceTests(unittest.TestCase):
    def run_frontend(self, use_legacy=False, max_seconds=None):
        rgb, landmarks, _ = painted_frame()
        frames = [rgb[:, :, ::-1].copy() for _ in range(10)]
        detections = [True] + [False] * 8 + [True]

        class Capture:
            index = 0
            def isOpened(self): return True
            def get(self, key): return 30.0
            def read(self):
                if self.index >= len(frames): return False, None
                frame = frames[self.index]
                self.index += 1
                return True, frame
            def release(self): pass

        capture = Capture()
        settings = []

        class Mesh:
            def __init__(self, **kwargs):
                self.static = kwargs["static_image_mode"]
                settings.append(kwargs)
            def process(self, frame):
                idx = capture.index - 1
                found = idx == 8 if self.static else detections[idx]
                return SimpleNamespace(multi_face_landmarks=[landmarks] if found else None)
            def close(self): pass

        def flow(previous, gray, points):
            return None if points is None else points.copy()

        with patch.object(legacy.cv2, "VideoCapture", return_value=capture), \
             patch.object(legacy.mp.solutions.face_mesh, "FaceMesh", Mesh), \
             patch.object(legacy, "flow_affine", side_effect=flow):
            result = (legacy.extract("controlled_sequence", "robust", max_seconds=max_seconds)
                      if use_legacy else frontend.extract("controlled_sequence", max_seconds=max_seconds))
        return result, settings

    def test_legacy_columns_and_detection_bridge_decisions_match(self):
        (old, old_fps), old_settings = self.run_frontend(use_legacy=True)
        (new, new_fps), new_settings = self.run_frontend()
        self.assertEqual(old_fps, new_fps)
        self.assertEqual(old_settings, new_settings)
        pd.testing.assert_frame_equal(old, new[frontend.LEGACY_COLUMNS])
        self.assertEqual(new.source.tolist(), ["mesh"] + ["flow_tracked"]*6 + ["missing", "redetected", "mesh"])
        self.assertEqual(new.loc[7, "roi_status"], "bridge_limit_reached")
        for name in frontend.REGION_NAMES:
            self.assertFalse(new.loc[7, f"{name}_valid"])
            self.assertTrue(np.isnan(new.loc[7, f"{name}_r"]))
            self.assertEqual(new.loc[7, f"{name}_quality"], 0)

    def test_max_seconds_keeps_legacy_rounding_and_time_axis(self):
        (new, fps), _ = self.run_frontend(max_seconds=.1)
        (old, _), _ = self.run_frontend(use_legacy=True, max_seconds=.1)
        self.assertEqual(len(new), 3)
        np.testing.assert_allclose(new.time_s, np.arange(3)/fps)
        pd.testing.assert_frame_equal(old, new[frontend.LEGACY_COLUMNS])


if __name__ == "__main__":
    unittest.main()
