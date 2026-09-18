import copy
import json
import os
from pathlib import Path
import unittest

import numpy as np

from retained_backend import ROIS, StreamingBackend, StreamingConfig


PROJECT = Path(os.environ.get("RPPG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))


def rows(seconds=16, fps=30):
    output = []
    for i in range(round(seconds*fps)+1):
        t = i/fps
        pulse = np.sin(2*np.pi*1.5*t)
        item = dict(time_s=t, motion_x=.002*np.cos(2*np.pi*.85*t), motion_y=.001*np.sin(2*np.pi*.85*t),
                    face_x0=.3,face_y0=.2,face_x1=.7,face_y1=.8)
        for roi in ROIS:
            item[f"{roi}_valid"]=True
            item[f"{roi}_quality"]=1.
            item[f"{roi}_pixel_source"]="tracked_ratio"
            for c,base,amplitude in zip("rgb",(120.,90.,65.),(.15,.6,.1)):
                item[f"{roi}_{c}"]=base+amplitude*pulse
                item[f"baseline_{roi}_{c}"]=base+amplitude*pulse
        output.append(item)
    return output


def without_runtime(event):
    output = copy.deepcopy(event)
    output.pop("processing_ms",None)
    return output


class RetainedStreamingTests(unittest.TestCase):
    def test_fixed_configuration(self):
        with self.assertRaises(ValueError):
            StreamingConfig(fixed_delay_s=0.)
        with self.assertRaises(ValueError):
            StreamingConfig(history_s=30.)

    def test_strict_clock_and_payload_exclusion(self):
        backend=StreamingBackend(PROJECT)
        row=rows(.1)[0]
        row["reference_hr"]=999
        backend.append(row)
        self.assertNotIn("reference_hr",backend.rows[0])
        with self.assertRaises(ValueError):
            backend.append(row)
        with self.assertRaises(ValueError):
            backend.append(dict(time_s=float("nan")))

    def test_future_prefix_invariance_and_immutable_snapshots(self):
        source=rows(16)
        short=StreamingBackend(PROJECT)
        expected=[e for row in source[:421] if (e:=short.append(row)) is not None]
        full=StreamingBackend(PROJECT)
        actual=[]
        frozen=None
        for row in source:
            event=full.append(row)
            if event is not None:
                actual.append(event)
                if len(actual)==3:
                    frozen=copy.deepcopy(event)
        self.assertEqual([without_runtime(e) for e in expected],
                         [without_runtime(e) for e in actual[:len(expected)]])
        self.assertEqual(actual[2],frozen)
        self.assertEqual(actual[0]["emitted_at_source_s"],12.)
        self.assertEqual(actual[0]["window_end_s"],10.)
        self.assertFalse(actual[0]["accepted"])
        self.assertTrue(actual[2]["accepted"])
        for event in actual:
            self.assertLessEqual(event["buffer_end_s"],event["emitted_at_source_s"]+1e-7)
            self.assertGreaterEqual(event["input_delay_s"],2.-1e-7)
            self.assertEqual(len(event["waveform"]["values"]),300)
            self.assertEqual(event["accepted"],event["hr_bpm"] is not None)
            json.dumps(event,allow_nan=False)

    def test_gap_never_becomes_constant_valid_ppg(self):
        backend=StreamingBackend(PROJECT)
        events=[]
        for row in rows(14):
            if 6. <= row["time_s"] <= 8.:
                for roi in ROIS:
                    row[f"{roi}_valid"]=False
            event=backend.append(row)
            if event is not None:
                events.append(event)
        self.assertTrue(events)
        self.assertTrue(all(not event["accepted"] for event in events))
        self.assertTrue(all(event["hr_bpm"] is None for event in events))

    def test_sampling_gate_and_bounded_storage(self):
        backend=StreamingBackend(PROJECT)
        # No inference kernel runs below the declared 8 Hz source floor.
        events=[e for row in rows(100,fps=4) if (e:=backend.append(row)) is not None]
        self.assertTrue(all(e["status"]=="insufficient_sampling" for e in events))
        self.assertLessEqual(len(backend.rows),102)
        self.assertGreaterEqual(backend.rows[0]["time_s"],75.)


if __name__ == "__main__":
    unittest.main()
