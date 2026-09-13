"""Synthetic contracts for the V28 new-video CLI; no human reference labels."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import analyze_video_v28 as cli


def colour_pulse(fps=30., seconds=30.):
    t = np.arange(round(seconds*fps))/fps
    trace = pd.DataFrame(dict(frame=np.arange(len(t)), time_s=t, rgb_valid=True,
        motion_x=0., motion_y=0., face_x0=.2, face_x1=.6, face_y0=.2, face_y1=.6,
        pixel_rgb_kind='anchored_paired_relative_colour', anchor_hz=.15, anchor_order=4))
    values = []
    for k, roi in enumerate(cli.ROIS):
        pulse = np.sin(2*np.pi*1.4*t+.02*k)
        rgb = np.column_stack([130*(1+.002*pulse), 100*(1+.008*pulse), 80*(1+.001*pulse)])
        values.append(rgb)
        for j, channel in enumerate('rgb'):
            trace[f'{roi}_{channel}'] = trace[f'baseline_{roi}_{channel}'] = rgb[:, j]
            trace[f'{roi}_pixel_log_delta_{channel}'] = np.r_[0., np.diff(np.log(rgb[:, j]))]
        trace[f'{roi}_valid'] = True
        trace[f'{roi}_quality'] = 1.
        trace[f'{roi}_pixel_source'] = 'tracked_ratio'
        trace[f'{roi}_pixel_tracks'] = 50
    merged = np.mean(values, axis=0)
    for j, channel in enumerate('rgb'):
        trace[channel] = trace[f'baseline_{channel}'] = merged[:, j]
    return trace


class EntryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='v28_cli_contract_')
        cls.root = Path(cls.temp.name)
        cls.video = cls.root/'no_face.avi'
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (96, 96))
        if not writer.isOpened():
            raise RuntimeError('Cannot create bounded synthetic video')
        for _ in range(60):
            writer.write(np.zeros((96, 96, 3), np.uint8))
        writer.release()
        cls.out = cls.root/'fresh'
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--video', str(cls.video), '--out', str(cls.out)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_short_no_face_has_nan_waveform_and_empty_readable_tables(self):
        wave = pd.read_csv(self.out/'waveform.csv')
        self.assertEqual(len(wave), 60)
        np.testing.assert_allclose(wave.time_s, np.arange(60)/30, atol=1e-12)
        self.assertTrue(wave.base.isna().all())
        self.assertFalse(wave[['covered', 'observed', 'interpolated']].any().any())
        for rel in ('heart_rate.csv', 'routing_decisions.csv', 'v25_fallback/heart_rate.csv',
                    'v26_harmonic_candidate/heart_rate.csv', 'v26_harmonic_candidate/component_proposals.csv'):
            self.assertTrue(pd.read_csv(self.out/rel).empty)
        manifest = json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertFalse(manifest['reference_used'])
        self.assertEqual(manifest['video_probe']['decoded_frames'], 60)
        self.assertEqual(manifest['video']['sha256'], cli.sha(self.video))
        self.assertTrue((self.out/'ppg_and_hr.png').is_file())

    def test_cache_roundtrip_and_overwrite_rejection(self):
        reused = self.root/'cached'
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--video', str(self.video), '--out', str(reused), '--trace-cache', str(self.out)])
        pd.testing.assert_frame_equal(pd.read_csv(self.out/'waveform.csv'),
                                      pd.read_csv(reused/'waveform.csv'), check_exact=True)
        before = cli.sha(reused/'manifest.json')
        with self.assertRaises(FileExistsError):
            cli.main(['--video', str(self.video), '--out', str(reused)])
        self.assertEqual(before, cli.sha(reused/'manifest.json'))

    def test_cache_rejects_unbound_or_foreign_content(self):
        identity = cli.frontend_identity(self.video)
        with self.assertRaisesRegex(ValueError, 'content SHA256'):
            cli.load_trace_cache(self.out, self.video, identity, '0'*64)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            cli.load_trace_cache(self.out, self.video, dict(identity, pixel_mode='disabled'), cli.sha(self.video))
        isolated = self.root/'unbound'
        isolated.mkdir()
        (isolated/'frame_trace.csv').write_bytes((self.out/'frame_trace.csv').read_bytes())
        meta = json.loads((self.out/'frame_trace.json').read_text())
        del meta['video_sha256']
        cli.write_json(isolated/'frame_trace.json', meta)
        with self.assertRaisesRegex(ValueError, 'no video SHA256'):
            cli.load_trace_cache(isolated, self.video, identity, cli.sha(self.video))

    def test_cache_rejects_tampered_content_and_clock(self):
        copy = self.root/'bad_clock'
        copy.mkdir()
        table = pd.read_csv(self.out/'frame_trace.csv')
        meta = json.loads((self.out/'frame_trace.json').read_text())
        table.loc[1, 'time_s'] = table.loc[0, 'time_s']
        table.to_csv(copy/'frame_trace.csv', index=False)
        cli.write_json(copy/'frame_trace.json', meta)
        with self.assertRaisesRegex(ValueError, 'cache SHA256'):
            cli.load_trace_cache(copy, self.video, cli.frontend_identity(self.video), cli.sha(self.video))
        meta['trace_sha256'] = cli.sha(copy/'frame_trace.csv')
        cli.write_json(copy/'frame_trace.json', meta)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            cli.load_trace_cache(copy, self.video, cli.frontend_identity(self.video), cli.sha(self.video))

    def test_sequential_decode_count_overrides_incorrect_header(self):
        real_capture = cv2.VideoCapture
        class BadCount:
            def __init__(self, path): self.inner = real_capture(path)
            def isOpened(self): return self.inner.isOpened()
            def get(self, prop):
                return 9999 if prop == cv2.CAP_PROP_FRAME_COUNT else self.inner.get(prop)
            def read(self): return self.inner.read()
            def release(self): self.inner.release()
        with patch.object(cv2, 'VideoCapture', BadCount):
            probe = cli.probe_video(self.video)
        self.assertEqual(probe['container_reported_frames'], 9999)
        self.assertEqual(probe['decoded_frames'], 60)
        trace = pd.read_csv(self.out/'frame_trace.csv')
        cli.check_decoded_clock(trace, 30., probe)
        with self.assertRaisesRegex(ValueError, 'frame count'):
            cli.check_decoded_clock(trace.iloc[:-1].copy(), 30., probe)
        with self.assertRaisesRegex(ValueError, 'FPS differs'):
            cli.check_decoded_clock(trace, 29., probe)

    def test_only_one_variant_and_no_reference_or_target_inputs(self):
        for forbidden in ('--reference', '--reference-ubfc', '--target-hr', '--variant'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(['--video', 'x', '--out', 'y', forbidden, '84'])
        self.assertEqual(cli.VARIANT, 'direct_guard')

    def test_synthetic_pulse_pipeline_preserves_fallback_and_reestimates_saved_hr(self):
        trace = colour_pulse()
        out = self.root/'pulse'
        out.mkdir()
        wave, hr, decisions = cli.run_pipeline(trace, 30., out)
        old = pd.read_csv(out/'v25_fallback/waveform.csv')
        pd.testing.assert_series_equal(wave.base, old.base, check_exact=True)
        np.testing.assert_array_equal(wave.covered, old.covered)
        self.assertFalse(decisions.route_eligible.any())
        self.assertEqual(len(hr), 21)
        self.assertGreaterEqual(int(hr.accepted.sum()), 10)
        self.assertLess(abs(hr.loc[hr.accepted, 'ridge_bpm'].median()-84), 2)
        replay = cli.read_saved_hr(out/'waveform.csv', trace, 30., 'saved_v28_direct_guard_waveform_offline')
        np.testing.assert_allclose(hr.ridge_bpm, replay.ridge_bpm, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(hr.loc[~hr.accepted, 'ridge_bpm'].isna().all())
        for branch in ('v25_fallback', 'v26_harmonic_candidate'):
            self.assertTrue((out/branch/'waveform.csv').is_file())
            self.assertTrue((out/branch/'heart_rate.csv').is_file())


if __name__ == '__main__':
    unittest.main()
