import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import frontend


def row(rgb=(100., 120., 90.), delta=(0., 0., 0.), source="tracked_ratio", valid=True):
    out = dict(zip("rgb", rgb))
    for roi in frontend.baseline.REGION_NAMES:
        out[f"{roi}_valid"] = valid
        out[f"{roi}_pixel_source"] = source
        out[f"{roi}_pixel_tracks"] = 50
        for j, c in enumerate("rgb"):
            out[f"{roi}_{c}"] = rgb[j]
            out[f"baseline_{roi}_{c}"] = rgb[j]
            out[f"{roi}_pixel_log_delta_{c}"] = delta[j]
    return out


FACE = SimpleNamespace(landmark=[SimpleNamespace(x=float(x), y=float(y))
                                for y in np.linspace(.15, .9, 22) for x in np.linspace(.15, .85, 22)])


class FakeMesh:
    def __init__(self, *args, **kwargs):
        self.found = True
        self.closed = False

    def process(self, image):
        return SimpleNamespace(multi_face_landmarks=[FACE] if self.found else [])

    def close(self):
        self.closed = True


class TestAnchor(unittest.TestCase):
    def test_common_signal_preserved_at_irregular_intervals(self):
        anchor = frontend.TimeAwareAnchor()
        t, previous = 0.0, None
        for i in range(180):
            dt = (.031, .029, .041)[i % 3]
            t += dt
            rgb = np.array([100., 120., 90.]) * (1 + .005 * np.sin(2 * np.pi * 1.6 * t))
            delta = np.zeros(3) if previous is None else np.log(rgb / previous)
            output = anchor.update(row(rgb, delta), None if i == 0 else dt)
            np.testing.assert_allclose([output[c] for c in "rgb"], rgb, rtol=1e-12)
            previous = rgb

    def test_missing_remains_nan_and_resets(self):
        anchor = frontend.TimeAwareAnchor()
        anchor.update(row(), None)
        result = anchor.update(row((np.nan,) * 3, valid=False), .03)
        self.assertTrue(np.isnan(result["g"]))
        reset = anchor.update(row((110., 130., 90.)), .03)
        self.assertEqual(reset["forehead_pixel_source"], "baseline_reset")
        self.assertEqual(reset["g"], 130.)

    def test_slow_drift_is_bounded(self):
        anchor = frontend.TimeAwareAnchor()
        anchor.update(row(), None)
        for i in range(900):
            result = anchor.update(row(delta=(.0005,) * 3), 1 / 30)
        self.assertLess(result["g"], 126.)
        self.assertGreater(result["g"], 120.)

    def test_bad_clock_rejected(self):
        anchor = frontend.TimeAwareAnchor()
        for dt in (0, -.1, .2, float("nan")):
            with self.assertRaises(ValueError):
                anchor.update(row(), dt)


class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.mesh_patch = patch.object(frontend.legacy.mp.solutions.face_mesh, "FaceMesh", FakeMesh)
        self.mesh_patch.start()
        self.image = np.random.default_rng(3).integers(70, 170, (480, 640, 3), dtype=np.uint8)
        self.front = frontend.RealtimeFrontend(max_width=320)

    def tearDown(self):
        self.front.close()
        self.mesh_patch.stop()

    def test_emitted_row_has_both_actual_raw_and_anchored_signal(self):
        result = self.front.process(self.image, 0.)
        self.assertEqual(result["sampling_width"], 320)
        self.assertEqual(result["source_width"], 640)
        self.assertTrue(result["rgb_valid"])
        self.assertTrue(result["face_detected"])
        for roi in frontend.baseline.REGION_NAMES:
            for c in "rgb":
                self.assertTrue(np.isfinite(result[f"baseline_{roi}_{c}"]))
                self.assertAlmostEqual(result[f"baseline_{roi}_{c}"], result[f"{roi}_{c}"])

    def test_bridge_uses_elapsed_time_and_stops_after_200ms(self):
        self.front.process(self.image, 0.)
        self.front.mesh.found = self.front.fallback.found = False
        with patch.object(frontend.legacy, "flow_affine", side_effect=lambda p, c, pts: pts):
            for t in (.08, .16, .2):
                result = self.front.process(self.image, t)
                self.assertEqual(result["source"], "flow_tracked")
            result = self.front.process(self.image, .21)
            self.assertEqual(result["source"], "missing")
            self.assertFalse(result["rgb_valid"])
            self.assertTrue(np.isnan(result["g"]))

    def test_discontinuity_resets_flow_and_anchor(self):
        self.front.process(self.image, 0.)
        result = self.front.process(self.image, .3)
        self.assertTrue(result["frame_discontinuity"])
        self.assertFalse(result["flow_available"])
        self.assertEqual(result["forehead_pixel_source"], "baseline_reset")

    def test_repeated_and_backwards_time_does_not_mutate_state(self):
        self.front.process(self.image, 1.)
        for t in (1., .9, float("nan")):
            with self.assertRaises(ValueError):
                self.front.process(self.image, t)
        self.assertEqual(self.front.index, 1)
        self.assertEqual(self.front.last_time, 1.)

    def test_emitted_rows_are_immutable_on_future_input(self):
        result = self.front.process(self.image, 0.)
        snapshot = copy.deepcopy(result)
        self.front.process(self.image, .03)
        for key in snapshot:
            if isinstance(snapshot[key], float) and np.isnan(snapshot[key]):
                self.assertTrue(np.isnan(result[key]))
            else:
                self.assertEqual(snapshot[key], result[key])

    def test_native_sampling_and_close(self):
        with frontend.RealtimeFrontend(max_width=None) as native:
            result = native.process(self.image, 0.)
            self.assertEqual(result["sampling_width"], 640)
        self.assertTrue(native.mesh.closed)
        with self.assertRaises(RuntimeError):
            native.process(self.image, .03)


if __name__ == "__main__":
    unittest.main()
