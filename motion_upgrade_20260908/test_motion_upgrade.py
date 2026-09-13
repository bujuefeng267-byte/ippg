import unittest
import numpy as np
import pandas as pd
import cv2
import tempfile
from pathlib import Path
from analyze_rppg_motion import fill_short_gaps, nlms_residual, ridge_path, estimate, flow_affine, reference_metrics


class MotionTests(unittest.TestCase):
    def test_only_short_internal_gaps_filled(self):
        x = np.repeat(np.arange(20.)[:, None], 3, axis=1)
        x[[0, 3, 4, 8, 9, 10, 19]] = np.nan
        result, filled = fill_short_gaps(x, 2)
        np.testing.assert_array_equal(np.flatnonzero(filled), [3, 4])
        self.assertTrue(np.isnan(result[[0, 8, 9, 10, 19]]).all())
        np.testing.assert_allclose(result[3:5, 0], [3, 4])

    def test_nlms_reduces_independent_motion_not_pulse(self):
        t = np.arange(3600) / 30
        pulse = np.sin(2 * np.pi * 1.2 * t)
        motion = np.column_stack([np.sin(2 * np.pi * 1.8 * t), np.zeros(len(t))])
        observed = pulse + 2.0 * motion[:, 0]
        cleaned = nlms_residual(observed, motion)
        raw_error = np.mean((observed[300:] - pulse[300:]) ** 2)
        error = np.mean((cleaned[300:] - pulse[300:]) ** 2)
        self.assertLess(error, raw_error * 0.3)
        self.assertGreater(np.corrcoef(cleaned[300:], pulse[300:])[0, 1], 0.85)

    def test_zero_or_missing_motion_passes_signal(self):
        y = np.sin(np.arange(300))
        for m in [np.zeros((300, 2)), np.full((300, 2), np.nan)]:
            np.testing.assert_allclose(nlms_residual(y, m), y)

    def test_ridge_ignores_single_transient_peak(self):
        grid = np.arange(50., 151.)
        p = np.full((20, len(grid)), 0.001)
        p[:, 30] = 1  # 80 bpm
        p[10, 80] = 3  # isolated false 130 bpm peak
        np.testing.assert_array_equal(grid[ridge_path(p, grid, 1)], np.full(20, 80.))

    def test_constant_signal_not_accepted(self):
        table = estimate(np.zeros(600), pd.DataFrame(dict(rgb_valid=np.ones(600, bool))),
                         np.zeros(600, bool), 30)
        self.assertFalse(table.accepted.any())
        self.assertTrue(table.ridge_bpm.isna().all())

    def test_missing_window_never_gets_tracked_hr(self):
        y = np.sin(2 * np.pi * 1.2 * np.arange(1200) / 30)
        y[500:560] = np.nan
        table = estimate(y, pd.DataFrame(dict(rgb_valid=np.ones(1200, bool))),
                         np.zeros(1200, bool), 30)
        self.assertTrue(table.loc[~table.accepted, "ridge_bpm"].isna().all())
        self.assertLess(abs(table.loc[table.accepted, "ridge_bpm"].median() - 72), 2)

    def test_flow_tracks_translation_and_rejects_blank(self):
        rng = np.random.default_rng(123)
        img = cv2.GaussianBlur(rng.integers(0, 255, (360, 360), dtype=np.uint8), (5, 5), 1)
        xy = np.array([(x, y) for y in range(70, 290, 6) for x in range(70, 290, 6)], float)
        moved = cv2.warpAffine(img, np.array([[1., 0., 3.], [0., 1., 2.]]), (360, 360))
        result = flow_affine(img, moved, xy)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(np.median(result - xy, axis=0), [3, 2], atol=0.3)
        self.assertIsNone(flow_affine(np.zeros_like(img), np.zeros_like(img), xy))

    def test_reference_duplicate_timestamp_and_no_extrapolation(self):
        t = np.r_[np.arange(0., 21.), 20.]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gt.txt'
            np.savetxt(path, np.vstack([t * 0, t * 0 + 80, t]))
            table = pd.DataFrame(dict(time_s=[10., 40.], spectral_peak_bpm=[80., 80.],
                                      ridge_bpm=[80., 80.], accepted=[True, True]))
            result = reference_metrics(table, path, 10)
            self.assertEqual(result['ridge_bpm']['n'], 1)
            self.assertEqual(result['ridge_bpm']['mae_bpm'], 0)
            self.assertTrue(np.isnan(table.reference_bpm.iloc[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
