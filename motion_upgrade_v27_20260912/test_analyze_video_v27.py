"""Bounded CLI integration tests on a generated video and synthetic photometry.

The frontend is a declared test double; the actual anchoring, BVP preparation,
patch inference, saved-wave HR readout and plotting execute normally. These
fixtures do not represent extraction accuracy on real faces.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import analyze_video_v27 as cli
import run_v27
from stable_patch_frontend_v27 import StableAnchors
from test_prepare_patch_signals_v27 import fixture
from test_stable_patch_frontend_v27 import landmarks


def write_video(path, fps=30, seconds=18):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (64, 64))
    if not writer.isOpened():
        raise RuntimeError('Synthetic integration requires local MJPEG video encoding')
    try:
        for i in range(round(fps*seconds)):
            value = round(120+2*np.sin(2*np.pi*1.4*i/fps))
            writer.write(np.full((64, 64, 3), value, np.uint8))
    finally:
        writer.release()


def write_test_frontend(video, directory, fps=30, seconds=18):
    """Source-bound synthetic test double; not a MediaPipe prediction."""
    directory = Path(directory)
    directory.mkdir()
    patches, frames = fixture(fps, seconds)
    n = len(frames)
    patches['tracked_valid'] = patches.valid
    patches['pixel_reset'] = patches.frame == 0
    patches.loc[patches.frame == 0, [f'pixel_log_delta_{c}' for c in 'rgb']] = np.nan
    frames['source'] = 'mesh'
    frames['face_detected'] = True
    frames['flow_available'] = True
    frames['bridge_age_frames'] = 0
    patches.to_csv(directory/'patch_trace.csv', index=False)
    frames.to_csv(directory/'frame_trace.csv', index=False)
    points = landmarks()
    anchors = StableAnchors()
    anchors.initialize(points, 0)
    (directory/'patch_anchors.json').write_text(json.dumps(anchors.metadata()))
    np.savez_compressed(directory/'landmark_trace.npz', frame=np.arange(n), time_s=np.arange(n)/fps,
                        points_px=np.repeat(points[None], n, axis=0))
    metadata = dict(fps=fps, frames=n, declared_frames=n, max_seconds=None,
                    video_sha256=cli.sha(video), video_bytes=Path(video).stat().st_size,
                    source_hashes=cli.frontend_sources(), config=cli.asdict(cli.PatchConfig()),
                    patch_ids=list(cli.PATCH_IDS), reference_used=False,
                    temporal_RGB_filling=False, RGB_aggregation_across_patches=False,
                    synthetic_photometry_test_double=True,
                    output_hashes={name: cli.sha(directory/name) for name in cli.FRONTEND_FILES})
    (directory/'metadata.json').write_text(json.dumps(metadata))
    return metadata


class VideoCLIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root/'synthetic.avi'
        cls.cache = cls.root/'cache'
        write_video(cls.video)
        write_test_frontend(cls.video, cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def altered_cache(self, name):
        target = self.root/name
        shutil.copytree(self.cache, target)
        return target

    def rewrite_metadata(self, cache, change):
        path = cache/'metadata.json'
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_cached_all_eight_variants_without_batch_environment(self):
        out = self.root/'all_outputs'
        with patch.object(cli, 'extract_patches', side_effect=AssertionError('Cache must not redetect')), \
             patch.object(run_v27, 'frozen', side_effect=AssertionError('Never consult batch protocol')):
            result = cli.analyze_video(self.video, out, variant='all', frontend_cache=self.cache)
        self.assertFalse(result['reference_used'])
        self.assertFalse(result['physiological_accuracy_evaluated'])
        self.assertEqual(set(result['computed_variants']), set(cli.VARIANTS))
        self.assertEqual(result['frontend_binding']['probe']['independently_decoded_frames'], 540)
        for name in cli.VARIANTS:
            directory = out/'variants'/name
            manifest = json.loads((directory/'manifest.json').read_text())
            self.assertEqual(manifest['primary_is_displayed_fused_waveform_HR'], name in cli.READOUT_VARIANTS)
            self.assertTrue(manifest['experimental'])
            primary = pd.read_csv(directory/'primary_hr.csv')
            secondary = pd.read_csv(directory/'secondary_hr.csv')
            self.assertEqual(len(primary), 9)
            self.assertGreater(primary.accepted.sum(), 0)
            self.assertTrue(primary.loc[~primary.accepted, 'ridge_bpm'].isna().all())
            self.assertTrue(secondary.loc[~secondary.accepted, 'ridge_bpm'].isna().all())
            png = cv2.imread(str(directory/'ppg_hr.png'))
            self.assertIsNotNone(png)
            self.assertEqual(png.shape[:2], (1280, 1920))
            if name in cli.READOUT_VARIANTS:
                self.assertEqual(cli.sha(directory/'waveform.csv'), cli.sha(out/'variants/late_psd_cluster/waveform.csv'))
        for name, expected in result['outputs'].items():
            self.assertEqual(cli.sha(out/name), expected)

    def test_fresh_default_and_readout_only_support_variant(self):
        fresh = self.root/'fresh_output'
        with patch.object(cli, 'extract_patches', side_effect=write_test_frontend) as extraction:
            result = cli.analyze_video(self.video, fresh)
        self.assertEqual(extraction.call_count, 1)
        self.assertEqual(result['requested_variants'], ['late_psd_cluster'])
        binding = json.loads((fresh/'frontend_binding.json').read_text())
        self.assertEqual(binding['extraction_mode'], 'fresh_full_video_extraction')
        self.assertIsNone(binding['probe']['independently_decoded_frames'])
        secondary_only = self.root/'readout_only'
        result = cli.analyze_video(self.video, secondary_only, variant='fusion_nomotion_local', frontend_cache=self.cache)
        self.assertEqual(result['requested_variants'], ['fusion_nomotion_local'])
        self.assertEqual(result['support_variants'], ['late_psd_cluster'])

    def test_cache_rejects_different_video_bytes(self):
        other = self.root/'other.avi'
        other.write_bytes(self.video.read_bytes()+b'different content')
        with self.assertRaisesRegex(ValueError, 'different video bytes'):
            cli.validate_frontend(self.cache, other)

    def test_cache_rejects_changed_sources_and_unbound_output(self):
        cache = self.altered_cache('changed_source')
        self.rewrite_metadata(cache, lambda m: m['source_hashes'].update({'pixel_tracking.py': '0'*64}))
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            cli.validate_frontend(cache, self.video)
        cache = self.altered_cache('changed_output')
        with (cache/'patch_trace.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'output hash mismatch'):
            cli.validate_frontend(cache, self.video)

    def test_self_rehashed_bad_clock_still_rejected(self):
        cache = self.altered_cache('bad_clock')
        frames = pd.read_csv(cache/'frame_trace.csv')
        frames.loc[3, 'time_s'] += .01
        frames.to_csv(cache/'frame_trace.csv', index=False)
        self.rewrite_metadata(cache, lambda m: m['output_hashes'].update({'frame_trace.csv': cli.sha(cache/'frame_trace.csv')}))
        with self.assertRaises((ValueError, AssertionError)):
            cli.validate_frontend(cache, self.video)

    def test_actual_decode_count_and_fps_are_checked(self):
        cache = self.altered_cache('bad_frame_count')
        self.rewrite_metadata(cache, lambda m: m.update(frames=541))
        with self.assertRaisesRegex(ValueError, 'decoded video frame count'):
            cli.validate_frontend(cache, self.video)
        cache = self.altered_cache('bad_fps')
        self.rewrite_metadata(cache, lambda m: m.update(fps=31))
        with self.assertRaisesRegex(ValueError, 'FPS differs'):
            cli.validate_frontend(cache, self.video)

    def test_partial_cache_and_output_overwrite_are_rejected(self):
        cache = self.altered_cache('partial_cache')
        self.rewrite_metadata(cache, lambda m: m.update(max_seconds=18.))
        with self.assertRaisesRegex(ValueError, 'complete reference-free'):
            cli.validate_frontend(cache, self.video)
        existing = self.root/'protected_output'
        existing.mkdir()
        marker = existing/'keep.txt'
        marker.write_text('unchanged')
        with self.assertRaises(FileExistsError):
            cli.analyze_video(self.video, existing)
        self.assertEqual(marker.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
