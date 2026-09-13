"""Synthetic photometry/geometry contracts; no real video/reference is loaded."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd

import stable_patch_frontend_v27 as frontend
from stable_patch_frontend_v27 import (PATCH_IDS, DEFAULT_CONFIG, StableAnchors,
    StablePatchSampler, choose_geometry, extract_patches, sample_polygon, seed_patch)


def landmarks():
    x, y = np.meshgrid(np.linspace(80, 560, 21), np.linspace(60, 580, 21))
    return np.column_stack([x.ravel(), y.ravel()])


def image():
    return np.random.default_rng(271).uniform(100, 150, (640, 640, 3)).astype(np.float32)


def identity_flow(gray, points):
    return points.copy(), np.ones(len(points), bool), np.zeros(len(points))


def no_flow(gray, points):
    return np.full_like(points, np.nan), np.zeros(len(points), bool), np.full(len(points), np.nan)


def rows_by_id(rows):
    return {row['patch_id']: row for row in rows}


class AnchorAndPhotometryTests(unittest.TestCase):
    def test_collinear_landmarks_do_not_establish_unstable_patch_ids(self):
        anchors = StableAnchors()
        line = np.linspace(80, 560, 30)
        self.assertFalse(anchors.initialize(np.column_stack([line, line]), 0))
        self.assertIsNone(anchors.anchors)
        self.assertTrue(anchors.initialize(landmarks(), 1))
        self.assertEqual(anchors.initial_frame, 1)

    def test_exactly_twelve_fixed_anchors_follow_translation(self):
        anchors = StableAnchors()
        points = landmarks()
        self.assertTrue(anchors.initialize(points, 7))
        initial = json.dumps(anchors.metadata(), sort_keys=True)
        original = anchors.project(points)
        moved = anchors.project(points+[17., -9.])
        self.assertEqual(tuple(anchors.anchors), PATCH_IDS)
        for ident in PATCH_IDS:
            np.testing.assert_allclose(moved[ident][0], original[ident][0]+[17., -9.], atol=1e-10, rtol=0)
        self.assertTrue(anchors.initialize(points*1.1, 99))
        self.assertEqual(json.dumps(anchors.metadata(), sort_keys=True), initial)
        self.assertEqual(anchors.initial_frame, 7)

    def test_local_coordinates_follow_rotation_and_scale(self):
        points = landmarks()
        anchors = StableAnchors()
        anchors.initialize(points, 0)
        angle = .12
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        moved_points = points@rotation.T*1.1+[5, -30]
        before, after = anchors.project(points), anchors.project(moved_points)
        for ident in PATCH_IDS:
            np.testing.assert_allclose(after[ident][0], before[ident][0]@rotation.T*1.1+[5, -30], atol=1e-9, rtol=0)

    def test_anchors_require_real_detection_and_survive_missing(self):
        sampler = StablePatchSampler(30)
        frame = image()
        first = sampler.update(frame, landmarks(), 0, 'flow_tracked', 1)
        self.assertTrue(all(not row['valid'] for row in first))
        self.assertIsNone(sampler.anchors.anchors)
        sampler.update(frame, landmarks(), 1, 'redetected')
        frozen = json.dumps(sampler.anchors.metadata(), sort_keys=True)
        missing = sampler.update(frame, None, 2, 'missing')
        self.assertTrue(all(not row['valid'] and row['pixel_source'] == 'missing' for row in missing))
        self.assertTrue(all(np.isnan(row['raw_r']) and np.isnan(row['tracked_r']) for row in missing))
        restored = sampler.update(frame, landmarks(), 3, 'mesh')
        self.assertTrue(all(row['pixel_source'] == 'baseline_reset' and row['pixel_reset'] for row in restored))
        self.assertEqual(json.dumps(sampler.anchors.metadata(), sort_keys=True), frozen)

    def test_one_patch_dark_occlusion_does_not_remove_other_histories(self):
        sampler = StablePatchSampler(30)
        original = image()
        sampler.update(original, landmarks(), 0, 'mesh')
        quads = sampler.anchors.project(landmarks())
        occluded = original.copy()
        target = 'forehead_0'
        cv2.fillConvexPoly(occluded, np.rint(quads[target][0]*256).astype(np.int32), (0, 0, 0), shift=8)
        with patch.object(sampler.flow, '_flow', side_effect=identity_flow):
            rows = rows_by_id(sampler.update(occluded, landmarks(), 1, 'mesh'))
            restored = rows_by_id(sampler.update(original, landmarks(), 2, 'mesh'))
        self.assertFalse(rows[target]['valid'])
        self.assertTrue(np.isnan(rows[target]['tracked_g']))
        self.assertEqual(restored[target]['pixel_source'], 'baseline_reset')
        for ident in PATCH_IDS:
            if ident != target:
                self.assertTrue(rows[ident]['valid'])
                self.assertFalse(rows[ident]['pixel_reset'])
                self.assertEqual(restored[ident]['pixel_source'], 'tracked_ratio')

    def test_shared_colour_pulse_is_not_rejected_as_local_outlier(self):
        sampler = StablePatchSampler(30)
        original = image()
        delta = np.array([.010, -.008, .006])
        first = rows_by_id(sampler.update(original, landmarks(), 0, 'mesh'))
        with patch.object(sampler.flow, '_flow', side_effect=identity_flow):
            rows = sampler.update(original*np.exp(delta), landmarks(), 1, 'mesh')
        for row in rows:
            self.assertEqual(row['pixel_source'], 'tracked_ratio')
            self.assertGreaterEqual(row['pixel_tracks'], 12)
            self.assertEqual(row['pixel_screen_rejected'], 0)
            np.testing.assert_allclose([row[f'paired_log_delta_{c}'] for c in 'rgb'], delta, atol=2e-7, rtol=0)
            previous = first[row['patch_id']]
            np.testing.assert_allclose([row[f'tracked_{c}'] for c in 'rgb'],
                                      [previous[f'tracked_{c}']*np.exp(delta[j]) for j, c in enumerate('rgb')],
                                      atol=3e-5, rtol=0)

    def test_colours_remain_independent_not_averaged_into_three_regions(self):
        points = landmarks()
        anchors = StableAnchors()
        anchors.initialize(points, 0)
        frame = np.full((640, 640, 3), 100, np.float32)
        for index, (ident, projected) in enumerate(anchors.project(points).items()):
            color = (80+index*5, 100+index*4, 120+index*3)
            cv2.fillConvexPoly(frame, np.rint(projected[0]*256).astype(np.int32), color, shift=8)
        sampler = StablePatchSampler(30)
        rows = sampler.update(frame, points, 0, 'mesh')
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({round(row['raw_r'], 3) for row in rows}), 12)
        self.assertEqual(len({round(row['tracked_r'], 3) for row in rows}), 12)
        self.assertTrue(all(row['valid_pixels'] > 25 for row in rows))

    def test_failed_FB_uses_same_patch_raw_ratio_and_reports_missing_pair(self):
        sampler = StablePatchSampler(30)
        original = image()
        sampler.update(original, landmarks(), 0, 'mesh')
        with patch.object(sampler.flow, '_flow', side_effect=no_flow):
            rows = sampler.update(original*1.01, landmarks(), 1, 'mesh')
        for row in rows:
            self.assertEqual(row['pixel_source'], 'baseline_ratio_fallback')
            self.assertEqual(row['pixel_tracks'], 0)
            self.assertTrue(np.isnan(row['paired_log_delta_r']))
            self.assertAlmostEqual(row['pixel_log_delta_r'], row['raw_log_delta_r'], places=12)
            self.assertAlmostEqual(row['tracked_r'], row['raw_r'], places=10)
        jumped = sampler.update(original, landmarks(), 3, 'mesh')
        self.assertTrue(all(row['pixel_reset'] for row in jumped))
        self.assertTrue(all(np.isnan(row['pixel_log_delta_r']) for row in jumped))

    def test_screening_is_confined_to_affected_patch_and_retains_other_points(self):
        sampler = StablePatchSampler(30)
        original = np.full((640, 640, 3), 120, np.float32)
        sampler.update(original, landmarks(), 0, 'mesh')
        quad = sampler.anchors.project(landmarks())['forehead_0'][0]
        point = seed_patch(quad)[12]
        current = original.copy()
        x, y = np.rint(point).astype(int)
        current[y-3:y+4, x-3:x+4] = 200
        with patch.object(sampler.flow, '_flow', side_effect=identity_flow):
            rows = rows_by_id(sampler.update(current, landmarks(), 1, 'mesh'))
        self.assertGreater(rows['forehead_0']['pixel_screen_rejected'], 0)
        self.assertEqual(rows['forehead_0']['pixel_source'], 'tracked_ratio')
        self.assertAlmostEqual(rows['forehead_0']['tracked_r'], 120, places=5)
        self.assertTrue(all(rows[ident]['pixel_screen_rejected'] == 0 for ident in PATCH_IDS if ident != 'forehead_0'))

    def test_real_LK_translation_keeps_photometry(self):
        original = image().astype(np.uint8)
        transform = np.array([[1, 0, 3], [0, 1, 2]], np.float32)
        moved = cv2.warpAffine(original, transform, (640, 640), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)
        sampler = StablePatchSampler(30)
        sampler.update(original, landmarks(), 0, 'mesh')
        rows = sampler.update(moved, landmarks()+[3, 2], 1, 'mesh')
        self.assertTrue(all(row['valid'] for row in rows))
        self.assertGreaterEqual(sum(row['pixel_source'] == 'tracked_ratio' for row in rows), 10)
        for row in rows:
            if row['pixel_source'] == 'tracked_ratio':
                self.assertLess(abs(row['pixel_log_delta_g']), .003)
                self.assertLessEqual(row['pixel_fb_error_median'], DEFAULT_CONFIG.fb_max_px)

    def test_visibility_quality_uses_real_image_intersection(self):
        frame = np.full((100, 100, 3), 120, np.uint8)
        partial = np.array([[-10, 10], [10, 10], [10, 30], [-10, 30]], float)
        result = sample_polygon(frame, partial)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['quality'], .5, places=6)
        self.assertAlmostEqual(result['raw_g'], 120, places=12)
        outside = sample_polygon(frame, partial-[200, 0])
        self.assertFalse(outside['valid'])
        self.assertTrue(np.isnan(outside['raw_r']))

    def test_30_and_180_FPS_keep_complete_clock_and_bridge_duration(self):
        frame = image()
        for fps in (30, 180):
            with self.subTest(fps=fps):
                sampler = StablePatchSampler(fps)
                rows = []
                with patch.object(sampler.flow, '_flow', side_effect=identity_flow):
                    for i in range(3):
                        rows.extend(sampler.update(frame, landmarks(), i, 'mesh'))
                table = pd.DataFrame(rows)
                self.assertEqual(len(table), 36)
                for _, group in table.groupby('patch_id'):
                    np.testing.assert_array_equal(group.frame, [0, 1, 2])
                    np.testing.assert_allclose(group.time_s, np.arange(3)/fps, atol=0, rtol=0)
                age, bridges = 0, 0
                for _ in range(round(.2*fps)+2):
                    points, age, bridged = choose_geometry(None, landmarks(), age, fps)
                    bridges += int(bridged)
                self.assertEqual(bridges, round(.2*fps))
                self.assertIsNone(points)
                _, age, bridged = choose_geometry(landmarks(), None, age, fps)
                self.assertEqual(age, 0)
                self.assertFalse(bridged)

    def test_missing_values_are_not_zero_or_temporally_filled(self):
        sampler = StablePatchSampler(30)
        first = sampler.update(image(), None, 0, 'missing')
        self.assertEqual(len(first), 12)
        for row in first:
            self.assertFalse(row['valid'])
            self.assertEqual(row['quality'], 0)
            self.assertTrue(np.isnan(row['raw_r']))
            self.assertTrue(np.isnan(row['tracked_r']))
            self.assertTrue(np.isnan(row['pixel_log_delta_r']))


class ExtractContractTests(unittest.TestCase):
    def test_fresh_detection_writes_bound_complete_outputs_and_rejects_overwrite(self):
        points = landmarks()
        face = SimpleNamespace(landmark=[SimpleNamespace(x=x/640, y=y/640) for x, y in points])
        detected = SimpleNamespace(multi_face_landmarks=[face])
        missing = SimpleNamespace(multi_face_landmarks=None)
        for fps in (30., 180.):
            with self.subTest(fps=fps), TemporaryDirectory() as temporary:
                root = Path(temporary)
                video = root/'synthetic_decoder_source.bin'
                video.write_bytes(b'bounded synthetic test source; not a real face video')
                frames = [image().astype(np.uint8) for _ in range(3)]
                cap = Mock()
                cap.isOpened.return_value = True
                cap.get.side_effect = lambda key: fps if key == cv2.CAP_PROP_FPS else 3
                cap.read.side_effect = [(True, x[..., ::-1].copy()) for x in frames]+[(False, None)]
                mesh, fallback = Mock(), Mock()
                mesh.process.side_effect = [detected, missing, detected]
                fallback.process.return_value = missing
                out = root/'output'
                with patch.object(frontend.cv2, 'VideoCapture', return_value=cap), \
                     patch.object(frontend.legacy.mp.solutions.face_mesh, 'FaceMesh', side_effect=[mesh, fallback]) as constructor, \
                     patch.object(frontend.legacy, 'flow_affine', return_value=None):
                    metadata = extract_patches(video, out)
                self.assertEqual(constructor.call_count, 2)
                self.assertFalse(constructor.call_args_list[0].kwargs['static_image_mode'])
                self.assertTrue(constructor.call_args_list[1].kwargs['static_image_mode'])
                self.assertEqual(metadata['frames'], 3)
                self.assertFalse(metadata['reference_used'])
                self.assertFalse(metadata['RGB_aggregation_across_patches'])
                trace = pd.read_csv(out/'patch_trace.csv')
                self.assertEqual(len(trace), 36)
                self.assertEqual(set(trace.patch_id), set(PATCH_IDS))
                self.assertFalse(trace.loc[trace.frame == 1, 'valid'].any())
                self.assertTrue(trace.loc[trace.frame == 1, ['raw_r', 'tracked_r']].isna().all().all())
                self.assertTrue((trace.loc[trace.frame == 2, 'pixel_source'] == 'baseline_reset').all())
                geometry = pd.read_csv(out/'frame_trace.csv')
                np.testing.assert_allclose(geometry.time_s, np.arange(3)/fps, atol=1e-16, rtol=0)
                self.assertTrue(np.isnan(np.load(out/'landmark_trace.npz')['points_px'][1]).all())
                for name, expected in metadata['output_hashes'].items():
                    self.assertEqual(frontend.digest(out/name), expected)
                self.assertEqual(metadata['video_sha256'], frontend.digest(video))
                with self.assertRaises(FileExistsError):
                    extract_patches(video, out)


if __name__ == '__main__':
    unittest.main()
