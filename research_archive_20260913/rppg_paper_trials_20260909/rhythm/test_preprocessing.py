"""Torch-free independent audit of the RhythmMamba project adapter.

Persistent outputs are this test and preprocessing_verification.json only.
Mock video frames exercise real OpenCV crop/resize/channel conversion; temporary
run_case outputs are isolated and automatically removed.
"""
from pathlib import Path
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import run_rhythm_trial as trial

HERE = Path(__file__).resolve().parent
EVIDENCE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_of(n, fps=30., box=(.2, .2, .6, .6)):
    trace = pd.DataFrame(np.tile(box, (n, 1)),
                         columns=['face_x0', 'face_y0', 'face_x1', 'face_y1'])
    trace['time_s'] = np.arange(n)/fps
    trace['rgb_valid'] = True
    return trace


class Capture:
    def __init__(self, frames, fps=30.):
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return self.fps

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        value = self.frames[self.index].copy()
        self.index += 1
        return True, value

    def release(self):
        self.released = True


class PreprocessingChecks(unittest.TestCase):
    def test_normalized_box_expands_width_height_by_one_point_five(self):
        self.assertEqual(trial.crop_bounds(np.array([.25, .25, .75, .75]), 100, 80),
                         (12, 10, 88, 70))
        self.assertEqual(trial.crop_bounds(np.array([.2, .2, .6, .6]), 10, 8),
                         (1, 0, 7, 6))

    def test_bounds_clamp_and_reject_no_frame_intersection(self):
        self.assertEqual(trial.crop_bounds(np.array([0., 0., .2, .2]), 100, 80),
                         (0, 0, 25, 20))
        self.assertEqual(trial.crop_bounds(np.array([.8, .8, 1., 1.]), 100, 80),
                         (75, 60, 100, 80))
        with self.assertRaises(ValueError):
            trial.crop_bounds(np.array([2., .2, 2.1, .4]), 100, 80)
        with self.assertRaises(ValueError):
            trial.crop_bounds(np.array([.4, .4, .4, .4]), 100, 80)

    def test_rgb_valid_and_bad_geometry_mask_before_interpolation(self):
        trace = trace_of(10)
        trace.loc[:3, 'rgb_valid'] = False  # leading gap must never extrapolate
        trace.loc[4, 'face_x1'] = .1  # reversed geometry
        trace.loc[5, 'face_x0'] = np.nan
        boxes, observed, filled = trial.prepare_boxes(trace, 30.)
        self.assertTrue(np.isnan(boxes[:6]).all())
        self.assertFalse(observed[:6].any())
        self.assertFalse(filled[:6].any())
        self.assertTrue(observed[6:].all())

    def test_floor_gap_limit_interpolates_geometry_only(self):
        trace = trace_of(12, fps=29.999)
        trace.loc[2:3, 'rgb_valid'] = False  # 2 frames allowed
        trace.loc[7:9, 'rgb_valid'] = False  # 3 frames too long
        boxes, observed, filled = trial.prepare_boxes(trace, 29.999)
        self.assertTrue(np.isfinite(boxes[2:4]).all())
        self.assertTrue(filled[2:4].all())
        self.assertFalse(observed[2:4].any())
        self.assertTrue(np.isnan(boxes[7:10]).all())
        self.assertFalse(filled[7:10].any())
        exact = trace_of(8)
        exact.loc[2:4, 'rgb_valid'] = False
        self.assertTrue(trial.prepare_boxes(exact, 30.)[2][2:5].all())

    def test_bounded_bbox_interpolation_positions_and_no_tail_extrapolation(self):
        trace = trace_of(7)
        trace.loc[0, ['face_x0','face_y0','face_x1','face_y1']] = [.1,.1,.5,.5]
        trace.loc[3, ['face_x0','face_y0','face_x1','face_y1']] = [.4,.4,.8,.8]
        trace.loc[1:2, 'rgb_valid'] = False
        trace.loc[5:6, 'rgb_valid'] = False
        boxes, _, filled = trial.prepare_boxes(trace, 30.)
        np.testing.assert_allclose(boxes[1], [.2,.2,.6,.6], atol=1e-15)
        np.testing.assert_allclose(boxes[2], [.3,.3,.7,.7], atol=1e-15)
        self.assertTrue(np.isnan(boxes[5:]).all())
        self.assertFalse(filled[5:].any())

    def test_real_frame_crop_rgb_order_and_short_gap_pixels(self):
        frames = []
        for index in range(3):
            frame = np.empty((8, 10, 3), np.uint8)
            frame[:,:,0] = 10+np.arange(10)[None,:]
            frame[:,:,1] = 30+np.arange(8)[:,None]
            frame[:,:,2] = 100+index*20
            frames.append(frame)
        trace = trace_of(3)
        trace.loc[1, 'rgb_valid'] = False
        capture = Capture(frames)
        with patch.object(trial.cv2, 'VideoCapture', return_value=capture):
            crops, valid, observed, filled = trial.read_crops('mock.avi', trace, 30.)
        self.assertTrue(capture.released)
        self.assertEqual(capture.index, 3)
        self.assertTrue(valid.all())
        self.assertEqual(observed.tolist(), [True,False,True])
        self.assertEqual(filled.tolist(), [False,True,False])
        for index in range(3):
            # Independently specified expected rectangle and BGR->RGB order.
            rgb = frames[index][0:6, 1:7, ::-1]
            expected = cv2.resize(rgb, (128,128), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(crops[index], expected)
        self.assertTrue((crops[1,:,:,0] == 120).all())
        EVIDENCE['rgb_crop'] = {'real_mock_frame_pixels_used': True,
            'interpolated_bbox_middle_frame_red_value': 120,
            'geometry_interpolation_does_not_synthesize_pixels': True}

    def test_long_gap_crops_remain_invalid_and_do_not_affect_statistics(self):
        frames = [np.full((8,10,3), i+1, np.uint8) for i in range(8)]
        trace = trace_of(8)
        trace.loc[2:5, 'rgb_valid'] = False
        with patch.object(trial.cv2, 'VideoCapture', return_value=Capture(frames)):
            crops, valid, observed, filled = trial.read_crops('mock.avi', trace, 30.)
        self.assertFalse(valid[2:6].any())
        self.assertTrue((crops[2:6] == 0).all())
        self.assertFalse((observed | filled)[2:6].any())
        self.assertEqual(trial.runs(valid), [(0,2),(6,8)])

    def test_statistics_matches_numpy_ddof_zero_and_includes_tail(self):
        rng = np.random.default_rng(20260909)
        crops = rng.integers(0, 100, size=(323,5,7,3), dtype=np.uint8)
        crops[320:] = 250
        mean, std = trial.segment_statistics(crops)
        ref = crops.astype(np.float64)
        self.assertAlmostEqual(mean, float(ref.mean()), places=11)
        self.assertAlmostEqual(std, float(ref.std(ddof=0)), places=11)
        self.assertGreater(abs(mean-float(ref[:320].mean())), 1.)
        self.assertGreater(abs(std-float(ref.std(ddof=1))), 1e-5)
        EVIDENCE['scalar_statistics'] = {'frames': len(crops), 'discarded_tail_for_160_clips': 3,
            'mean': mean, 'numpy_mean': float(ref.mean()),
            'std': std, 'numpy_std_ddof0': float(ref.std(ddof=0)),
            'whole_segment_including_tail': True}
        with self.assertRaises(ValueError):
            trial.segment_statistics(np.full((161,3,3,3), 127, np.uint8))

    def test_postprocess_gaps_and_future_segment_independence(self):
        fps = 30.
        t = np.arange(1200)/fps
        raw = np.sin(2*np.pi*1.2*t)
        raw[570:600] = np.nan
        changed = raw.copy()
        changed[600:] = 3*np.cos(2*np.pi*2.3*t[600:])
        first = trial.postprocess(raw, fps)
        second = trial.postprocess(changed, fps)
        np.testing.assert_allclose(first[:570], second[:570], rtol=0, atol=0, equal_nan=True)
        edge = round(1.6*fps)
        self.assertTrue(np.isnan(first[570-edge:600+edge]).all())
        self.assertTrue(np.isnan(first[:edge]).all())
        self.assertTrue(np.isnan(first[-edge:]).all())
        self.assertTrue(np.isfinite(first[edge:570-edge]).all())
        expected = trial.bandpass(raw[:570], fps, 42, 210)
        np.testing.assert_allclose(first[edge:570-edge], expected[edge:-edge], rtol=0, atol=0)
        self.assertTrue(np.isnan(trial.postprocess(np.full(600,np.nan), fps)).all())

    def test_run_case_saved_masks_and_original_time_axis_without_torch(self):
        fps = 29.999
        trace = trace_of(1050, fps=fps)
        trace.loc[500:529, 'rgb_valid'] = False
        trace.loc[100:101, 'rgb_valid'] = False
        boxes, observed, interpolated = trial.prepare_boxes(trace, fps)
        valid = np.isfinite(boxes).all(axis=1)
        raw = np.sin(2*np.pi*1.2*trace.time_s.to_numpy())
        raw[~valid] = np.nan
        with tempfile.TemporaryDirectory(prefix='rhythm_preproc_audit_') as temporary:
            root = Path(temporary)
            here = root/'rhythm'
            source = root/'rppg_motion_v22/validation/trimmed_gap10/mock'
            source.mkdir(parents=True)
            trace.to_csv(source/'frame_trace.csv', index=False)
            (source/'frame_trace.json').write_text(json.dumps({'fps':fps,
                'trace_sha256':sha(source/'frame_trace.csv')}))
            video = root/'mock.avi'
            video.write_bytes(b'video mock: decoder supplied by unittest')
            identity = {'video':str(video), 'sha256':sha(video)}
            fake_crops = np.zeros((len(trace),1,1,3), np.uint8)
            with patch.object(trial, 'ROOT', root), patch.object(trial, 'HERE', here), \
                 patch.object(trial, 'read_crops', return_value=(fake_crops,valid,observed,interpolated)), \
                 patch.object(trial, 'infer_crops', return_value=(raw,[])):
                trial.run_case('mock', None, {}, {}, identity, 'unused_no_torch')
            saved = pd.read_csv(here/'results/mock/waveform.csv')
            hr = pd.read_csv(here/'results/mock/heart_rate.csv')
            np.testing.assert_allclose(saved.time_s, np.arange(len(trace))/fps, rtol=0, atol=1e-12)
            covered = np.isfinite(saved.base)
            np.testing.assert_array_equal(saved.covered, covered)
            np.testing.assert_array_equal(saved.observed, observed & covered)
            np.testing.assert_array_equal(saved.interpolated, interpolated & covered)
            np.testing.assert_array_equal(saved.observed | saved.interpolated, covered)
            self.assertFalse((saved.observed & saved.interpolated).any())
            self.assertTrue(hr.loc[~hr.accepted,['spectral_peak_bpm','ridge_bpm']].isna().all().all())
            recomputed = trial.final_hr(saved, fps)
            pd.testing.assert_frame_equal(hr, recomputed, check_dtype=False, rtol=0, atol=1e-9)
            EVIDENCE['output_masks_and_time'] = {'original_fps':fps, 'frames':len(saved),
                'observed_or_interpolated_equals_covered':True,
                'missing_hr_columns_are_nan':True, 'saved_wave_hr_recomputed_equal':True,
                'torch_or_neural_model_used':False}


if __name__ == '__main__':
    before = sha(HERE/'run_rhythm_trial.py')
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PreprocessingChecks)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = sha(HERE/'run_rhythm_trial.py')
    record = {'passed':result.wasSuccessful() and before==after,
              'tests_run':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
              'runner_sha256_before':before, 'runner_sha256_after':after,
              'test_sha256':sha(__file__), 'torch_used':False,
              'reference_or_model_weights_read':False,
              'test_output':output.getvalue(), 'evidence':EVIDENCE,
              'scope':'Crop geometry, real RGB pixels, bbox gap handling, scalar statistics including tail, per-run filtering, saved masks/time/HR; no model accuracy or torch backend test.',
              'nonblocking_observations':[
                  'read_crops releases VideoCapture on normal completion; exceptions before cap.release lack a finally cleanup.',
                  'Segment statistics intentionally use future frames and discarded tail; therefore preprocessing remains offline.',
                  'Postprocessing preserves gap separation but may mix adjacent finite network clips, as explicitly designed.']}
    (HERE/'preprocessing_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(output.getvalue(), flush=True)
    print(json.dumps({'passed':record['passed'], 'tests_run':result.testsRun,
                      'runner_sha256':after}), flush=True)
    raise SystemExit(0 if record['passed'] else 1)
