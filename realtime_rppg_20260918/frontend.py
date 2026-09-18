"""Single-pass, past-only face/skin extraction for live camera frames.

Uses the retained V28 detector, skin sampling and paired-pixel screening.
The default samples a width<=960 image (an explicit speed/accuracy tradeoff);
max_width=None samples native resolution. Geometry/LK still use width<=960.
Unlike offline V28, the 0.15 Hz fourth-order anchor uses actual elapsed time
and a continuous Butterworth filter discretized by zero-order hold. This is
causal but not numerically identical to the old fixed-FPS bilinear filter.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.signal import butter, cont2discrete, tf2ss

PROJECT = Path(os.environ.get("RPPG_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
V28 = PROJECT / "motion_upgrade_v28_20260912"
if not V28.is_dir():
    raise RuntimeError("Cannot locate retained V28 modules; set RPPG_PROJECT_ROOT")
sys.path.insert(0, str(V28))
import baseline_frontend as baseline
import legacy_motion as legacy
from pixel_tracking import PixelConfig, PixelTracker


class TimeAwareAnchor:
    """Finite-memory state; never changes previously emitted rows."""

    def __init__(self, cutoff_hz=0.15):
        if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
            raise ValueError("Anchor cutoff must be positive")
        self.cutoff_hz = float(cutoff_hz)
        b, a = butter(4, 2 * np.pi * self.cutoff_hz, analog=True)
        self.analog = tf2ss(b, a)
        self.reset()

    def reset(self):
        self.level = {r: None for r in baseline.REGION_NAMES}
        self.previous = {r: None for r in baseline.REGION_NAMES}
        self.state = {r: np.zeros((4, 3)) for r in baseline.REGION_NAMES}

    @lru_cache(maxsize=256)
    def _discrete(self, rounded_dt):
        return cont2discrete(self.analog, rounded_dt, method="zoh")[:4]

    def update(self, row, dt):
        out = dict(row)
        values = []
        if dt is not None:
            if not np.isfinite(dt) or dt <= 0 or dt > 0.100001:
                raise ValueError("Anchor dt must be within (0, 0.1] seconds")
            ad, bd, cd, dd = self._discrete(max(0.0001, round(float(dt), 4)))
        for roi in baseline.REGION_NAMES:
            raw = np.asarray([row[f"baseline_{roi}_{c}"] for c in "rgb"], float)
            valid = bool(row[f"{roi}_valid"]) and np.isfinite(raw).all() and (raw > 0).all()
            correction = np.nan
            if not valid:
                self.level[roi] = self.previous[roi] = None
                self.state[roi].fill(0)
                value, source = np.full(3, np.nan), "missing"
            else:
                lograw = np.log(raw)
                if dt is None or self.level[roi] is None or self.previous[roi] is None:
                    self.level[roi] = lograw.copy()
                    self.state[roi].fill(0)
                    value, source, correction = raw.copy(), "baseline_reset", 0.0
                else:
                    delta = np.asarray([row[f"{roi}_pixel_log_delta_{c}"] for c in "rgb"], float)
                    old_source = str(row[f"{roi}_pixel_source"])
                    tracked = old_source == "tracked_ratio" or (
                        old_source == "numerical_reset" and int(row[f"{roi}_pixel_tracks"]) >= 12)
                    if tracked and np.isfinite(delta).all():
                        source = "tracked_ratio"
                    else:
                        delta = lograw - np.log(self.previous[roi])
                        source = "baseline_ratio_fallback"
                    self.level[roi] += delta
                    discrepancy = lograw - self.level[roi]
                    self.state[roi] = ad @ self.state[roi] + bd @ discrepancy[None, :]
                    drift = (cd @ self.state[roi] + dd @ discrepancy[None, :])[0]
                    corrected = self.level[roi] + drift
                    correction = float(np.linalg.norm(drift))
                    if not np.isfinite(corrected).all() or np.max(np.abs(corrected)) > 50:
                        self.level[roi] = lograw.copy()
                        self.state[roi].fill(0)
                        corrected, source = lograw.copy(), "numerical_reset"
                    value = np.exp(corrected)
                self.previous[roi] = raw.copy()
                values.append(value)
            for j, c in enumerate("rgb"):
                out[f"unanchored_{roi}_{c}"] = row[f"{roi}_{c}"]
                out[f"{roi}_{c}"] = float(value[j])
            out[f"{roi}_unanchored_pixel_source"] = row[f"{roi}_pixel_source"]
            out[f"{roi}_pixel_source"] = source
            out[f"{roi}_anchor_correction_log"] = correction
        merged = np.mean(values, axis=0) if values else np.full(3, np.nan)
        for j, c in enumerate("rgb"):
            out[f"unanchored_{c}"] = row[c]
            out[c] = float(merged[j])
        out.update(pixel_rgb_kind="anchored_paired_relative_colour",
                   anchor_hz=self.cutoff_hz, anchor_order=4,
                   anchor_clock="actual_dt_zoh_rounded_0.1ms")
        return out


class RealtimeFrontend:
    def __init__(self, max_width=960):
        if max_width is not None and (not isinstance(max_width, int) or max_width < 160):
            raise ValueError("max_width must be None or an integer >=160")
        self.max_width = max_width
        self.mesh = legacy.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.fallback = legacy.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.anchor = TimeAwareAnchor()
        self.index = 0
        self.last_time = None
        self.closed = False
        self._reset_tracking()

    def _reset_tracking(self):
        self.previous = None
        self.previous_points = None
        self.last_detection_time = None
        self.bridge_age_frames = 0
        self.pixel = PixelTracker(PixelConfig(), mode="tracking_screened")
        self.anchor.reset()

    def process(self, bgr, time_s):
        if self.closed:
            raise RuntimeError("Frontend already closed")
        t = float(time_s)
        if not np.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        image = np.asarray(bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise ValueError("Expected nonempty uint8 BGR frame")
        dt = None if self.last_time is None else t - self.last_time
        discontinuity = dt is not None and dt > 0.100001
        if discontinuity:
            self._reset_tracking()
            dt = None
        original_h, original_w = image.shape[:2]
        if self.max_width and original_w > self.max_width:
            image = cv2.resize(image, (self.max_width, round(original_h * self.max_width / original_w)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ratio = min(1.0, 960 / w)
        small = cv2.resize(rgb, (round(w * ratio), round(h * ratio))) if ratio < 1 else rgb
        sh, sw = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if self.previous is not None and self.previous.shape != gray.shape:
            self._reset_tracking()
            discontinuity, dt = True, None
        try:
            prediction = legacy.flow_affine(self.previous, gray, self.previous_points)
        except cv2.error:
            prediction = None
        result = self.mesh.process(small)
        source, points = "missing", None
        if result.multi_face_landmarks:
            source = "mesh"
        else:
            result = self.fallback.process(small)
            if result.multi_face_landmarks:
                source = "redetected"
        detected = bool(result.multi_face_landmarks)
        if detected:
            points = np.asarray([(p.x * sw, p.y * sh) for p in result.multi_face_landmarks[0].landmark])
            self.last_detection_time = t
            self.bridge_age_frames = 0
        elif (prediction is not None and self.last_detection_time is not None
              and t - self.last_detection_time <= 0.2 + 1e-9):
            points, source = prediction, "flow_tracked"
            self.bridge_age_frames += 1
        else:
            self.bridge_age_frames += 1
        sampling = baseline._empty_samples("bridge_limit_reached" if prediction is not None
                                          else "no_face_and_no_valid_flow", "trimmed_mean")
        if points is not None:
            full = points * np.array([w / sw, h / sh])
            sampling = baseline.sample_rois(rgb, legacy.mesh_object(full, w, h))
        motion = np.full(2, np.nan)
        if self.previous_points is not None and prediction is not None:
            motion = np.median(prediction - self.previous_points, axis=0) / sh
        bbox = [np.nan] * 4
        if points is not None:
            lo = np.percentile(points, 2, axis=0) / [sw, sh]
            hi = np.percentile(points, 98, axis=0) / [sw, sh]
            bbox = [lo[0], lo[1], hi[0], hi[1]]
        row = dict(frame=self.index, time_s=t, source=source,
                   motion_x=float(motion[0]), motion_y=float(motion[1]),
                   face_x0=bbox[0], face_y0=bbox[1], face_x1=bbox[2], face_y1=bbox[3])
        row.update(sampling)
        row.update(face_detected=detected, flow_available=prediction is not None,
                   bridge_age_frames=self.bridge_age_frames,
                   bridge_age_s=0.0 if detected else (t - self.last_detection_time if self.last_detection_time is not None else np.nan),
                   frame_discontinuity=discontinuity,
                   source_width=original_w, source_height=original_h,
                   sampling_width=w, sampling_height=h)
        row = self.pixel.update(rgb, row, self.index)
        row = self.anchor.update(row, dt)
        self.previous, self.previous_points = gray, points
        self.last_time = t
        self.index += 1
        return row

    def close(self):
        if not self.closed:
            self.mesh.close()
            self.fallback.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
