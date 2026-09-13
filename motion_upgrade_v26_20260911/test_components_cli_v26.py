"""Small new-video CLI contracts; no human pulse labels or six-video fitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import analyze_components_v26 as cli


class EntryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='v26_cli_contract_')
        cls.root = Path(cls.temp.name)
        cls.video = cls.root/'no_face.avi'
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*'MJPG'),
                                 30, (96, 96))
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

    def test_short_no_face_has_complete_time_and_missing_outputs(self):
        wave = pd.read_csv(self.out/'waveform.csv')
        self.assertEqual(len(wave), 60)
        np.testing.assert_allclose(wave.time_s, np.arange(60)/30, atol=1e-12)
        self.assertTrue(wave.base.isna().all())
        self.assertFalse(wave[['covered', 'observed', 'interpolated']].any().any())
        self.assertTrue(pd.read_csv(self.out/'heart_rate.csv').empty)
        self.assertTrue(pd.read_csv(self.out/'component_proposals.csv').empty)
        manifest = json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertFalse(manifest['reference_used'])
        self.assertEqual(manifest['video']['sha256'], cli.sha(self.video))

    def test_reuse_new_cache_and_no_output_overwrite(self):
        reused = self.root/'cached'
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['--video', str(self.video), '--out', str(reused),
                      '--trace-cache', str(self.out)])
        pd.testing.assert_frame_equal(pd.read_csv(self.out/'waveform.csv'),
                                      pd.read_csv(reused/'waveform.csv'), check_exact=True)
        before = cli.sha(reused/'manifest.json')
        with self.assertRaises(FileExistsError):
            cli.main(['--video', str(self.video), '--out', str(reused)])
        self.assertEqual(before, cli.sha(reused/'manifest.json'))

    def test_cache_rejects_foreign_content_or_changed_parameters(self):
        identity = cli.frontend_identity(self.video)
        with self.assertRaisesRegex(ValueError, 'content SHA256'):
            cli.load_trace_cache(self.out, self.video, identity, '0'*64)
        wrong = dict(identity, pixel_mode='disabled')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            cli.load_trace_cache(self.out, self.video, wrong, cli.sha(self.video))

    def test_cache_rejects_tampered_trace_and_bad_timeline(self):
        copy = self.root/'bad'
        copy.mkdir()
        table = pd.read_csv(self.out/'frame_trace.csv')
        table.to_csv(copy/'frame_trace.csv', index=False)
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

    def test_unbound_legacy_cache_cannot_invent_historical_video_hash(self):
        isolated = self.root/'unbound'
        isolated.mkdir()
        (isolated/'frame_trace.csv').write_bytes((self.out/'frame_trace.csv').read_bytes())
        meta = json.loads((self.out/'frame_trace.json').read_text())
        del meta['video_sha256']
        cli.write_json(isolated/'frame_trace.json', meta)
        with self.assertRaisesRegex(ValueError, 'no video SHA256'):
            cli.load_trace_cache(isolated, self.video, cli.frontend_identity(self.video), cli.sha(self.video))

    def test_reference_and_target_hr_are_not_cli_inputs(self):
        for forbidden in ('--reference', '--reference-ubfc', '--target-hr'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(['--video', 'x', '--out', 'y', forbidden, '84'])

    def test_both_variants_resolve_independently(self):
        self.assertEqual(cli.load_variant('component').__name__, 'component_fusion_v26')
        if (cli.HERE/'component_harmonics_v26.py').exists():
            self.assertEqual(cli.load_variant('harmonic').__name__, 'component_harmonics_v26')

    def test_known_colour_pulse_goes_through_saved_waveform_hr(self):
        fps = 30.
        t = np.arange(900)/fps
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
        cli.validate_component_trace(trace, fps)
        for variant in ('component', 'harmonic'):
            wave, proposals, _, _ = cli.infer_components(trace, fps, variant)
            path = self.root/f'pulse_{variant}.csv'
            wave.to_csv(path, index=False)
            hr = cli.read_saved_hr(path, trace, fps, proposals)
            self.assertEqual(len(hr), 21)
            self.assertGreaterEqual(int(hr.accepted.sum()), 10)
            self.assertLess(abs(hr.loc[hr.accepted, 'ridge_bpm'].median()-84), 2)
            self.assertTrue(hr.loc[~hr.accepted, 'ridge_bpm'].isna().all())


if __name__ == '__main__':
    unittest.main()
