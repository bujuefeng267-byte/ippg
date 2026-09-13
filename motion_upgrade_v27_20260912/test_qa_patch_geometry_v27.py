"""Synthetic controls for the independent geometry receipt."""
import copy
from dataclasses import asdict
import unittest

import numpy as np
import pandas as pd

from stable_patch_frontend_v27 import StablePatchSampler, DEFAULT_CONFIG
from test_stable_patch_frontend_v27 import image, landmarks
from qa_patch_geometry_v27 import audit_arrays


def synthetic_arrays(fps=30):
    sampler = StablePatchSampler(fps)
    base = landmarks()
    line = np.linspace(80, 560, len(base))
    geometries = [base, np.column_stack([line, line]), None, base+[1., .5], base+[2., 1.]]
    sources = ['mesh', 'mesh', 'missing', 'redetected', 'flow_tracked']
    frames, patches, saved_points = [], [], []
    pixels = image()
    for index, (points, source) in enumerate(zip(geometries, sources)):
        age = 1 if source in ('flow_tracked', 'missing') else 0
        patches.extend(sampler.update(pixels, points, index, source, age))
        if points is None:
            bounds = np.full(4, np.nan)
            saved_points.append(np.full_like(base, np.nan))
        else:
            low, high = np.percentile(points, [2, 98], axis=0)
            bounds = np.r_[low, high]/640
            saved_points.append(points)
        frames.append(dict(frame=index, time_s=index/fps, source=source,
            face_detected=source in ('mesh', 'redetected'), is_bridge=source == 'flow_tracked',
            bridge_age_frames=age, face_x0=bounds[0], face_y0=bounds[1], face_x1=bounds[2], face_y1=bounds[3]))
    return pd.DataFrame(frames), pd.DataFrame(patches), np.stack(saved_points), sampler.anchors.metadata(), asdict(DEFAULT_CONFIG), fps, (640, 640)


class GeometryAuditTests(unittest.TestCase):
    def test_QR_replay_identity_missing_degenerate_and_bridge_at_both_FPS(self):
        for fps in (30, 180):
            with self.subTest(fps=fps):
                result = audit_arrays(*synthetic_arrays(fps))
                self.assertEqual(result['stable_patch_count'], 12)
                self.assertEqual(result['parent_region_count'], 3)
                self.assertLess(max(row['maximum_corner_difference_px'] for row in result['patches']), 1e-8)

    def test_changed_corner_and_quality_are_detected(self):
        values = synthetic_arrays()
        for field, delta in [('corner0_x', 1.), ('quality', -.2)]:
            with self.subTest(field=field):
                changed = values[1].copy(deep=True)
                changed.loc[0, field] += delta
                with self.assertRaises(AssertionError):
                    audit_arrays(values[0], changed, *values[2:])

    def test_reselected_anchor_identity_is_detected(self):
        values = synthetic_arrays()
        anchors = copy.deepcopy(values[3])
        ids = anchors['patches']['forehead_0']['landmark_indices']
        ids[0], ids[1] = ids[1], ids[0]
        with self.assertRaises(AssertionError):
            audit_arrays(*values[:3], anchors, *values[4:])


if __name__ == '__main__':
    unittest.main()
