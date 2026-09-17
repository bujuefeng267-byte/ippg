"""V32 CLI boundary/integration tests using generated RGB, no private recordings."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyze_v28_result_v32 as cli


def synthetic_trace(model, seconds=16., fps=30.):
    t = np.arange(round(seconds*fps))/fps
    pulse = np.sin(2*np.pi*1.5*t)+.18*np.sin(2*np.pi*3*t+.2)
    rows = dict(frame=np.arange(len(t)), time_s=t, rgb_valid=True,
                motion_x=0., motion_y=0., face_x0=.2, face_y0=.2,
                face_x1=.6, face_y1=.6,
                pixel_rgb_kind='anchored_paired_relative_colour',
                anchor_hz=.15, anchor_order=4)
    for color, mean, strength in zip('rgb', (120., 100., 80.), (.09, .30, .05)):
        rgb = mean+strength*pulse
        rows[color] = rgb
        rows['baseline_'+color] = rgb
        for roi in model.ROIS:
            rows[roi+'_'+color] = rgb
            rows['baseline_'+roi+'_'+color] = rgb
            rows[roi+'_pixel_log_delta_'+color] = 0.
    for roi in model.ROIS:
        rows[roi+'_valid'] = True
        rows[roi+'_quality'] = .9
        rows[roi+'_pixel_source'] = 'tracked_ratio'
        rows[roi+'_pixel_tracks'] = 50
    return pd.DataFrame(rows)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.old = cls.folder/'original_v28'
        cls.old.mkdir()
        cls.legacy, cls.model, _, _ = cli.load_runtime()
        cls.fps = 30.
        trace = synthetic_trace(cls.model, fps=cls.fps)
        cls.trace_path = cls.old/'frame_trace.csv'
        cls.meta_path = cls.old/'frame_trace.json'
        trace.to_csv(cls.trace_path, index=False)
        saved = pd.read_csv(cls.trace_path)
        cls.model.validate_component_trace(saved, cls.fps)
        cls.model.run_pipeline(saved, cls.fps, cls.old)
        source_hashes = cls.model.frozen_sources()
        cls.legacy.save(cls.meta_path, dict(
            identity=dict(video='synthetic_no_video_file.avi', size=0, mtime_ns=0,
                rgb_reconstruction='anchored_log', pixel_mode='tracking_screened',
                frontend_hashes={name: source_hashes[name] for name in
                                 ('motion_frontend.py', 'anchored_reconstruction.py')}),
            fps=cls.fps, n_frames=len(trace), trace_sha256=cli.sha(cls.trace_path)))
        cls.legacy.save(cls.old/'summary.json', dict(status='complete', variant='direct_guard',
            reference_used=False, fps=cls.fps, frames=len(trace), source_hashes=source_hashes,
            input_hashes={str(path):cli.sha(path) for path in (cls.trace_path, cls.meta_path)},
            output_hashes={name:cli.sha(cls.old/name) for name in ('waveform.csv','heart_rate.csv')}))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cli_runs_real_pipeline_from_synthetic_trace_and_binds_every_window(self):
        out = self.folder/'cli_integration'
        original_hashes = {p:cli.sha(p) for p in self.old.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            summary = cli.main(['--v28-output', str(self.old), '--out', str(out)])
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(summary['motion_guard'])
        self.assertTrue(summary['hr_from_saved_window_waveforms'])
        self.assertFalse(summary['hr_from_single_continuous_waveform'])
        self.assertFalse(summary['hr_from_global_waveform'])
        self.assertFalse(summary['continuous_V32_waveform_available'])
        self.assertFalse(summary['reference_used'])
        self.assertFalse(summary['video_decoded'])
        self.assertEqual(summary['recommendation'], 'V28')
        self.assertFalse(summary['promoted'])
        self.assertGreater(summary['saved_window_count'], 0)
        self.assertEqual(cli.sha(out/'waveform.csv'), cli.sha(self.old/'waveform.csv'))
        self.legacy.verify(original_hashes)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['summary_sha256'], cli.sha(out/'summary.json'))
        for name, expected in summary['output_hashes'].items():
            self.assertEqual(cli.sha(out/name), expected)
        for path in ('candidate/processed_trace.csv', 'candidate/waveform.csv',
                     'candidate/heart_rate.csv', 'windows_manifest.csv'):
            self.assertTrue((out/path).is_file())
        windows = pd.read_csv(out/'windows_manifest.csv')
        first = out/windows.iloc[0]['waveform_file']
        original = first.read_bytes()
        try:
            first.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'file/path/hash'):
                cli.validate_window_bank(out, pd.read_csv(out/'heart_rate.csv'), self.old/'waveform.csv')
        finally:
            first.write_bytes(original)

    def test_changed_trace_rejected_even_when_new_metadata_describes_it(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder)/'frame_trace.csv'
            meta = trace.with_suffix('.json')
            trace.write_bytes(self.trace_path.read_bytes()+b'\n')
            metadata = json.loads(self.meta_path.read_text())
            metadata['trace_sha256'] = cli.sha(trace)
            self.legacy.save(meta, metadata)
            with self.assertRaisesRegex(ValueError, 'not bound'):
                self.legacy.verified_input(self.old, trace, meta, self.model)

    def test_modified_original_waveform_rejected(self):
        wave = self.old/'waveform.csv'
        original = wave.read_bytes()
        try:
            wave.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'original hash binding'):
                self.legacy.verified_input(self.old, self.trace_path, self.meta_path, self.model)
        finally:
            wave.write_bytes(original)

    def test_reference_candidate_and_guard_options_are_not_exposed(self):
        for forbidden in ('--reference', '--candidate', '--mode', '--no-motion-guard'):
            with self.subTest(argument=forbidden), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parser().parse_args(['--v28-output','old','--out','new',forbidden,'x'])

    def test_existing_output_rejected_before_runtime_or_input_work(self):
        with patch.object(cli, 'load_runtime', side_effect=AssertionError('runtime must not start')):
            with self.assertRaises(FileExistsError):
                cli.main(['--v28-output',str(self.old),'--out',str(self.old)])

    def test_nested_output_rejected_without_creating_it(self):
        nested = self.old/'nested'
        with self.assertRaisesRegex(ValueError, 'outside original'):
            cli.main(['--v28-output',str(self.old),'--out',str(nested)])
        self.assertFalse(nested.exists())


if __name__ == '__main__':
    unittest.main()
