import copy
import json
from pathlib import Path
import unittest

import numpy as np

from backend import ROIS, StreamingBackend, resample_values, motion_to_grid_units

P = Path(str(Path(__file__).resolve().parent.parent))


def row(t, bpm=90., valid=True):
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    result = dict(time_s=t, motion_x=0., motion_y=0., face_x0=.2,
                  face_y0=.2, face_x1=.8, face_y1=.8)
    for roi in ROIS:
        result.update({f'{roi}_valid':valid, f'{roi}_quality':.9,
                       f'{roi}_pixel_source':'baseline_ratio_fallback'})
        for c, baseline, gain in zip('rgb', [130, 100, 80], [.002, .015, .004]):
            result[f'{roi}_{c}'] = baseline * (1 + gain * pulse) if valid else np.nan
            result[f'baseline_{roi}_{c}'] = result[f'{roi}_{c}']
    return result


def numeric_event(event):
    return {k:v for k,v in event.items() if k != 'processing_ms'}


class BackendTests(unittest.TestCase):
    def backend(self):
        return StreamingBackend(P)

    def test_warmup_exact_measured_hr_and_json(self):
        b = self.backend()
        for t in np.arange(300) / 30:
            self.assertIsNone(b.append(row(t)))
        e = b.append(row(10.))
        self.assertIsNotNone(e)
        self.assertTrue(e['accepted'])
        self.assertLessEqual(abs(e['hr_bpm'] - 90), 2)
        self.assertEqual((e['window_start_s'],e['window_end_s']), (0.,10.))
        self.assertEqual(len(e['waveform']['values']), 300)
        self.assertTrue(all(x is not None for x in e['waveform']['values']))
        json.dumps(e, allow_nan=False)

    def test_prefix_causality_and_immutable_output(self):
        a, b = self.backend(), self.backend()
        first = []
        for t in np.arange(331) / 30:
            ea, eb = a.append(row(t)), b.append(row(t))
            self.assertEqual(ea is None, eb is None)
            if ea:
                self.assertEqual(numeric_event(ea), numeric_event(eb))
                first.append(ea)
        saved = copy.deepcopy(first)
        for t in np.arange(331,391) / 30:
            a.append(row(t, 150.))
            b.append(row(t, 60.))
        self.assertEqual(first, saved)

    def test_missing_measurements_remain_null(self):
        b = self.backend()
        event = None
        for t in np.arange(301)/30:
            event = b.append(row(t, valid=not (3 <= t < 5))) or event
        self.assertFalse(event['accepted'])
        self.assertIsNone(event['hr_bpm'])
        self.assertTrue(all(x is None for x in event['waveform']['values']))
        json.dumps(event, allow_nan=False)

    def test_long_capture_gap_never_interpolated(self):
        t = np.array([0., .03, .06, .5, .53])
        grid = np.array([.03, .2, .5])
        values, observed, filled = resample_values(t, t, grid, .03)
        self.assertTrue(np.isnan(values[1,0]))
        self.assertFalse(observed[1])
        self.assertFalse(filled[1])

    def test_short_missing_sample_is_labeled(self):
        t = np.arange(5)/30
        v = t.copy(); v[2] = np.nan
        values, observed, filled = resample_values(t, v, t, 1/30)
        self.assertAlmostEqual(values[2,0],t[2])
        self.assertTrue(filled[2]); self.assertFalse(observed[2])

    def test_nonmonotonic_time_rejected(self):
        b = self.backend(); b.append(row(0))
        for t in [0, -1, np.nan]:
            with self.assertRaises(ValueError): b.append(row(t))

    def test_motion_velocity_units_independent_of_capture_cadence(self):
        dt = np.asarray([1/20, 1/30, 1/180])
        velocity = .25
        scaled = motion_to_grid_units(velocity * dt, dt, 30.)
        np.testing.assert_allclose(scaled * 30, velocity, rtol=0, atol=1e-12)
        unknown = motion_to_grid_units([1., 1., 1.], [0., np.nan, .5], 30.)
        self.assertTrue(np.isnan(unknown).all())

    def test_rate_too_low_rejected_not_padded_to_30(self):
        b = self.backend()
        for t in np.arange(51)/5:
            event = b.append(row(t))
        self.assertFalse(event['accepted'])
        self.assertEqual(event['status'], 'insufficient_sampling')

    def test_buffer_bounded_and_gap_skips_updates(self):
        b = self.backend()
        for t in np.arange(300)/30: b.append(row(t))
        event = b.append(row(100.))
        self.assertEqual(event['skipped_update_count'],90)
        self.assertEqual(event['window_end_s'],100.)
        self.assertLess(len(b.rows),3)
        self.assertFalse(event['accepted'])

    def test_extra_reference_fields_do_not_change_inference(self):
        a,b=self.backend(),self.backend()
        for t in np.arange(301)/30:
            x=row(t); y=dict(x,reference_bpm=180.,video='data1',target_bpm=55.)
            ea,eb=a.append(x),b.append(y)
        self.assertEqual(numeric_event(ea),numeric_event(eb))


if __name__ == '__main__':
    unittest.main()
