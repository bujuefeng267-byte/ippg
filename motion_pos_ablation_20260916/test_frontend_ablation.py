"""Synthetic-only checks; no physiological reference or real cache is opened."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import frontend_ablation as module


def synthetic_trace(seconds=36, fps=30.):
    time = np.arange(round(seconds*fps))/fps
    trace = pd.DataFrame(dict(frame=np.arange(len(time)), time_s=time,
        rgb_valid=True, pixel_rgb_kind='anchored_paired_relative_colour',
        anchor_hz=.15, anchor_order=4))
    wave = 2*np.pi*1.2*time
    rgb = np.column_stack([150*(1+.004*np.sin(wave)),
                           130*(1+.008*np.sin(wave+.6)),
                           100*(1+.003*np.sin(wave-.6))])
    for j, region in enumerate(module.REGIONS):
        trace[region+'_valid'] = True
        for source in module.SOURCES:
            prefix = '' if source == 'anchored' else source+'_'
            for channel, c in enumerate('rgb'):
                trace[f'{prefix}{region}_{c}'] = rgb[:, channel]*(1+.05*j)
    return trace


class FrontendAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(os.environ.get('RPPG_PROJECT', '/home/fengbujue/项目/rppg识别'))
        cls.runtime = module.load_runtime(cls.project)
        cls.trace = synthetic_trace()
        cls.products, cls.sources, cls.records, cls.masks, cls.native = \
            module.extract_products(cls.trace, 30., cls.runtime)

    def test_identical_representations_give_identical_arms(self):
        for reducer in module.REDUCERS:
            first = self.products[f'front_baseline_{reducer}']
            for source in module.SOURCES[1:]:
                pd.testing.assert_frame_equal(first, self.products[f'front_{source}_{reducer}'])

    def test_real_estimator_recovers_synthetic_frequency_without_motion(self):
        for arm, wave in self.products.items():
            hr = module.estimate_waveform(wave, 30., self.runtime)
            self.assertGreater(int(hr.accepted.sum()), 20, arm)
            self.assertTrue(np.all(np.abs(hr.loc[hr.accepted, 'ridge_bpm']-72) <= 2), arm)
            self.assertFalse(hr.evidence_motion_available.any())

    def test_aggregation_uses_three_real_regions_or_one_actual_mixture(self):
        for arm, sources in self.sources.items():
            expected = module.REGIONS if arm.endswith('region_median') else ('mixed_rgb',)
            self.assertEqual([c for c in sources if c.endswith('_base')], [r+'_base' for r in expected])
            self.assertTrue(self.records[arm].status.eq('generated').all())

    def test_mask_intersection_preserves_native_rates_and_long_gap(self):
        trace = self.trace.copy()
        # One source/one ROI controls the common availability; none are ignored.
        trace.loc[450:479, 'unanchored_left_cheek_r'] = np.nan
        products, _, _, masks, native = module.extract_products(trace, 30., self.runtime)
        self.assertEqual(native['baseline']['all_three_pct'], 100.)
        self.assertLess(native['unanchored']['all_three_pct'], 100.)
        self.assertEqual(int((~masks.common_observed).sum()), 30)
        self.assertFalse(masks.common_interpolated.any())
        for wave in products.values():
            self.assertTrue(wave.base.iloc[450:480].isna().all())
            self.assertFalse(wave.observed.iloc[450:480].any())
            hr = module.estimate_waveform(wave, 30., self.runtime)
            touching = (hr.window_start_s < 16) & (hr.window_end_s > 15)
            self.assertFalse(hr.loc[touching, 'accepted'].any())

    def test_only_bounded_internal_short_gaps_are_filled(self):
        trace = self.trace.copy()
        trace.loc[[0, 300, 301, 302, len(trace)-1], 'baseline_forehead_g'] = np.nan
        products, _, _, masks, _ = module.extract_products(trace, 30., self.runtime)
        self.assertEqual(np.flatnonzero(masks.common_interpolated).tolist(), [300, 301, 302])
        for wave in products.values():
            self.assertTrue(wave.base.iloc[[0, len(trace)-1]].isna().all())
            self.assertTrue(wave.interpolated.iloc[300:303].all())
            self.assertFalse(wave.observed.iloc[300:303].any())

    def test_flat_color_does_not_invent_waveform(self):
        trace = self.trace.copy()
        for source in module.SOURCES:
            prefix = '' if source == 'anchored' else source+'_'
            for region in module.REGIONS:
                for c in 'rgb':
                    trace[f'{prefix}{region}_{c}'] = 100.
        products, _, _, _, _ = module.extract_products(trace, 30., self.runtime)
        for wave in products.values():
            self.assertTrue(wave.base.isna().all())
            self.assertFalse(wave.covered.any())

    def test_rounding_retains_actual_high_fps_grid(self):
        trace = synthetic_trace(seconds=7, fps=120.00048)
        products, _, records, _, _ = module.extract_products(trace, 120.00048, self.runtime)
        for arm, wave in products.items():
            np.testing.assert_array_equal(wave.time_s, trace.time_s)
            self.assertTrue(records[arm].start_frame.diff().dropna().eq(10).all())
            self.assertTrue((records[arm].stop_frame-records[arm].start_frame).eq(600).all())

    def test_end_to_end_exact_saved_inputs_and_freeze_enforcement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root/'frame_trace.csv'
            self.trace.to_csv(trace_path, index=False)
            metadata = dict(fps=30., n_frames=len(self.trace), trace_sha256=module.sha(trace_path),
                identity={'video': 'synthetic_no_video_exists', 'max_seconds': None})
            module.save_json(root/'frame_trace.json', metadata)
            freeze = dict(frontend_protocol=module.protocol(), reference_used=False,
                source_hashes=self.runtime.source_hashes,
                input_hashes={str(p): module.sha(p) for p in (trace_path, root/'frame_trace.json')})
            global_path = root/'protocol.json'
            module.save_json(global_path, dict(reference_used_by_inference=False,
                arms={arm: {} for arm in module.ARMS}))
            freeze['global_protocol_path'] = str(global_path)
            freeze['global_protocol_sha256'] = module.sha(global_path)
            freeze_path = root/'freeze.json'
            generated = module.make_stage_freeze(self.project, [trace_path], global_path, freeze_path)
            for key in freeze:
                self.assertEqual(generated[key], freeze[key])
            out = root/'output'
            result = module.run_case(trace_path, out, case='synthetic', project=self.project, freeze_path=freeze_path)
            self.assertEqual(set(result), set(module.ARMS))
            for arm in module.ARMS:
                folder = out/arm
                wave = pd.read_csv(folder/'waveform.csv')
                actual = pd.read_csv(folder/'heart_rate.csv')
                expected = module.estimate_waveform(wave, 30., self.runtime)
                # CSV may coerce window IDs to integers; numerical HR must be exact.
                np.testing.assert_array_equal(actual.ridge_bpm, expected.ridge_bpm)
                manifest = pd.read_csv(folder/'hr_windows_manifest.csv')
                for row in manifest.itertuples(index=False):
                    saved = pd.read_csv(folder/row.waveform_file)
                    pd.testing.assert_frame_equal(saved, wave.iloc[row.start_frame:row.stop_frame].reset_index(drop=True))
                    self.assertEqual(row.sha256, module.sha(folder/row.waveform_file))
                arm_manifest = json.loads((folder/'manifest.json').read_text())
                self.assertFalse(arm_manifest['reference_used'])
                self.assertFalse(arm_manifest['raw_video_read'])
            with self.assertRaises(FileExistsError):
                module.run_case(trace_path, out, case='synthetic', project=self.project, freeze_path=freeze_path)
            freeze['frontend_protocol']['min_bpm'] = 50.
            wrong = root/'wrong_freeze.json'
            module.save_json(wrong, freeze)
            with self.assertRaises(ValueError):
                module.run_case(trace_path, root/'wrong_output', case='synthetic', project=self.project, freeze_path=wrong)
            self.assertFalse((root/'wrong_output').exists())


if __name__ == '__main__':
    unittest.main()
