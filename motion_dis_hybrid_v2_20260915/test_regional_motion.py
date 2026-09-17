"""Synthetic measurement/provenance checks; no real data or physiology used."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from regional_motion import (DEFAULT_CONFIG, PARENT_REGIONS, REGIONS,
                             RegionalMotionTracker, extract_regional_trace,
                             regional_bounds, seed_grid, sha256, validate_geometry_cache)


def texture(width=640, height=360):
    rng = np.random.default_rng(31415)
    gray = rng.uniform(60, 185, (height, width)).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), .7)
    return np.stack([gray + 15, gray, gray - 10], axis=-1).astype(np.uint8)


def record(index=0, fps=30.):
    return dict(frame=index, time_s=index/fps, face_x0=.08, face_y0=.02,
                face_x1=.92, face_y1=.98,
                **{f'{region}_valid': True for region in PARENT_REGIONS},
                motion_x=987654., motion_y=-987654.)


def translate(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


class RegionalMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)

    def test_signed_motion_and_original_height_scaling(self):
        normalized = []
        for width, height, dx, dy in [(640, 360, 3, -2), (1280, 720, 6, -4)]:
            image = texture(width, height)
            tracker = RegionalMotionTracker()
            first = tracker.update(image, record(), 0)
            self.assertTrue(all(not first[f'{name}_valid'] for name in REGIONS))
            self.assertTrue(all(np.isnan(first[f'{name}_r']) for name in REGIONS))
            second = tracker.update(translate(image, dx, dy), record(1), 1)
            values = []
            for name in REGIONS:
                self.assertTrue(second[f'{name}_valid'], name)
                self.assertGreaterEqual(second[f'{name}_tracks'], 12)
                self.assertLessEqual(second[f'{name}_seeds'], 49)
                self.assertLessEqual(second[f'{name}_fb_error'], .5)
                got = [second[f'{name}_dx'], second[f'{name}_dy']]
                np.testing.assert_allclose(got, [dx/height, dy/height], atol=.12/height, rtol=0)
                values.append(got)
            normalized.append(np.mean(values, axis=0))
        np.testing.assert_allclose(normalized[0], normalized[1], atol=.0003, rtol=0)

    def test_4k_small_face_has_original_patch_support_before_lk_scaling(self):
        # A 250x250 face in a 4K frame has ample original 7x7 patch support.
        # V1 incorrectly applied 4-pixel margins and 3-pixel grid spacing after
        # reducing to 960px, leaving fewer than the unchanged 12-track minimum.
        width, height = 3840, 2160
        small_width, small_height = 960, 540
        image = cv2.resize(texture(small_width, small_height), (width, height),
                           interpolation=cv2.INTER_LINEAR)
        previous = record()
        previous.update(face_x0=1700/width, face_y0=800/height,
                        face_x1=1950/width, face_y1=1050/height)
        current = dict(previous, frame=1, time_s=1/30)
        dx, dy = 4, -4
        # Move the cached face with the texture so a legitimate translated
        # boundary patch is not rejected by a stationary synthetic face box.
        current.update(face_x0=previous['face_x0'] + dx/width,
                       face_x1=previous['face_x1'] + dx/width,
                       face_y0=previous['face_y0'] + dy/height,
                       face_y1=previous['face_y1'] + dy/height)
        support = {}
        for name in REGIONS:
            original = seed_grid(regional_bounds(previous, name, image.shape),
                                 image.shape, DEFAULT_CONFIG)
            old_reduced = seed_grid(regional_bounds(previous, name, (small_height, small_width)),
                                    (small_height, small_width), DEFAULT_CONFIG)
            self.assertGreaterEqual(len(original), 12, name)
            self.assertLess(len(old_reduced), 12, name)
            self.assertEqual(len(np.unique(original, axis=0)), len(original))
            left, top, right, bottom = regional_bounds(previous, name, image.shape)
            radius = DEFAULT_CONFIG.patch_radius
            self.assertTrue(((original[:, 0]-radius >= left) &
                             (original[:, 0]+radius <= right) &
                             (original[:, 1]-radius >= top) &
                             (original[:, 1]+radius <= bottom)).all())
            support[name] = len(original)
        tracker = RegionalMotionTracker()
        tracker.update(image, previous, 0)
        result = tracker.update(translate(image, dx, dy), current, 1)
        for name in REGIONS:
            self.assertEqual(result[f'{name}_seeds'], support[name])
            self.assertTrue(result[f'{name}_valid'], name)
            self.assertGreaterEqual(result[f'{name}_tracks'], 12, name)
            self.assertLessEqual(result[f'{name}_fb_error'], .5, name)
            np.testing.assert_allclose([result[f'{name}_dx'], result[f'{name}_dy']],
                                       [dx/height, dy/height], rtol=0, atol=.15/height)

    def test_six_local_regions_can_move_in_opposite_directions(self):
        image = texture(960, 720)
        current = image.copy()
        for index, name in enumerate(REGIONS):
            left, top, right, bottom = regional_bounds(record(), name, image.shape)
            x0, y0, x1, y1 = int(left), int(top), int(right), int(bottom)
            shifted = translate(image, 2 if index % 2 == 0 else -2, 0)
            current[y0:y1, x0:x1] = shifted[y0:y1, x0:x1]
        tracker = RegionalMotionTracker()
        tracker.update(image, record(), 0)
        result = tracker.update(current, record(1), 1)
        for index, name in enumerate(REGIONS):
            self.assertTrue(result[f'{name}_valid'], name)
            direction = 1 if index % 2 == 0 else -1
            self.assertGreater(result[f'{name}_dx'] * direction, 1.5/720)
            self.assertLess(abs(result[f'{name}_dy']), .2/720)

    def test_mask_loss_recovery_and_nonconsecutive_frame(self):
        image = texture()
        tracker = RegionalMotionTracker()
        tracker.update(image, record(), 0)
        lost = record(1)
        lost['forehead_valid'] = False
        missing = tracker.update(image, lost, 1)
        recovered_first = tracker.update(image, record(2), 2)
        recovered_second = tracker.update(image, record(3), 3)
        skipped = tracker.update(image, record(5), 5)
        for name in ('forehead_a', 'forehead_b'):
            for row in (missing, recovered_first):
                self.assertFalse(row[f'{name}_valid'])
                self.assertEqual(row[f'{name}_tracks'], 0)
                self.assertTrue(np.isnan(row[f'{name}_dx']))
                self.assertTrue(np.isnan(row[f'{name}_r']))
            self.assertTrue(recovered_second[f'{name}_valid'])
        self.assertTrue(missing['left_cheek_a_valid'])
        self.assertTrue(all(not skipped[f'{name}_valid'] for name in REGIONS))

    def test_current_rgb_and_displacement_use_identical_screened_keep(self):
        # Feed controlled correspondences to ensure rejected colour patches also
        # disappear from displacement, and that RGB is current measured colour.
        image = texture()
        tracker = RegionalMotionTracker()
        tracker.update(image, record(), 0)
        calls = []

        def flow(gray, points):
            moved = points.copy()
            moved[:, 0] += np.tile(np.arange(7), len(points)//7) * .02
            return moved, np.ones(len(points), bool), np.full(len(points), .1)

        def samples(rgb, points, radius):
            index = len(calls)
            calls.append(points.copy())
            if index % 2 == 0:
                return np.full((len(points), 3), 100.)
            values = np.full((len(points), 3), 110.)
            values[:len(values)//2] = 240.  # coherent minority outliers are rejected
            return values

        with patch.object(tracker.flow, '_flow', side_effect=flow), \
                patch('regional_motion.sample_patches', side_effect=samples):
            result = tracker.update(image, record(1), 1)
        for index, name in enumerate(REGIONS):
            old, new = calls[2*index:2*index+2]
            kept = slice(len(new)//2, None)
            expected = np.median(new[kept] - old[kept], axis=0) / image.shape[0]
            self.assertTrue(result[f'{name}_valid'])
            self.assertEqual(result[f'{name}_r'], 110.)
            self.assertEqual(result[f'{name}_tracks'], len(new)-len(new)//2)
            np.testing.assert_allclose([result[f'{name}_dx'], result[f'{name}_dy']], expected)


class CacheExtractionTests(unittest.TestCase):
    def make_cache(self, root, trace_length=4, video_frames=4, metadata_fps=30.):
        image = texture()
        video = root / 'synthetic.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (640, 360))
        self.assertTrue(writer.isOpened())
        for index in range(video_frames):
            writer.write(cv2.cvtColor(translate(image, index, 0), cv2.COLOR_RGB2BGR))
        writer.release()
        trace_path = root / 'frame_trace.csv'
        pd.DataFrame([record(i, metadata_fps) for i in range(trace_length)]).to_csv(trace_path, index=False)
        identity = dict(video=str(video.resolve()), size=video.stat().st_size,
                        mtime_ns=video.stat().st_mtime_ns, max_seconds=None)
        metadata = dict(identity=identity, fps=metadata_fps, n_frames=trace_length,
                        trace_sha256=sha256(trace_path), video_sha256=sha256(video))
        trace_path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        return trace_path, metadata

    def test_roundtrip_complete_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.make_cache(root)
            output = root / 'sidecar'
            trace, metadata = extract_regional_trace(path, output)
            self.assertEqual(len(trace), 4)
            self.assertEqual(metadata['n_frames'], 4)
            self.assertEqual(metadata['fps'], 30.)
            self.assertEqual(metadata['resolution'], [640, 360])
            self.assertTrue(metadata['original_video_eof_confirmed'])
            self.assertFalse(metadata['reference_used'])
            self.assertFalse(metadata['interpolation'])
            self.assertEqual(metadata['seed_grid_units'], 'original_image_pixels')
            self.assertEqual(metadata['patch_radius_units'], 'original_image_pixels')
            self.assertEqual(metadata['fb_error_units'], 'tracking_image_pixels')
            self.assertEqual(metadata['trace_sha256'], sha256(output / 'regional_trace.csv'))
            saved = pd.read_csv(output / 'regional_trace.csv')
            self.assertFalse(saved.loc[0, 'forehead_a_valid'])
            self.assertTrue(np.isnan(saved.loc[0, 'forehead_a_dx']))
            with self.assertRaises(FileExistsError):
                extract_regional_trace(path, output)

    def test_rejects_digest_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.make_cache(root)
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'trace SHA256 mismatch'):
                validate_geometry_cache(path)

    def test_rejects_extra_or_missing_video_frames_and_fps(self):
        for trace_length, video_frames, fps, expected in [
            (3, 4, 30., 'extra frames'), (5, 4, 30., 'ended before'),
            (4, 4, 29., 'FPS differs'),
        ]:
            with self.subTest(trace_length=trace_length, video_frames=video_frames, fps=fps), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, _ = self.make_cache(root, trace_length, video_frames, fps)
                with self.assertRaisesRegex(ValueError, expected):
                    extract_regional_trace(path, root / 'sidecar')
                self.assertFalse((root / 'sidecar').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)


