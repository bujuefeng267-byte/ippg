"""Robust detection/tracking with independently aggregated RGB skin patches.

The detector, fallback, LK/RANSAC bridge and 0.2 s bridge limit are inherited
from ``legacy_motion.extract(..., frontend='robust')``. There is no temporal
RGB filling here. ``r/g/b`` are the unweighted mean of valid patch estimates.
The default estimate is a per-channel float64 mean after dropping floor(0.1*N)
values from each sorted tail of the N valid pixels. ``rgb_aggregation='median'``
retains the previous sampler for a controlled ablation. The trim fraction is
fixed and is never selected using a physiological reference.
An empty geometric image intersection is rejected rather than squeezed into
a one-pixel border strip (the sole defensive difference at invalid geometry).

Region names follow the legacy image-coordinate boxes, not anatomical labels.
``*_quality`` is a sampling proxy, NOT a pulse-quality score or probability:
    valid_pixel_fraction * visible_fraction, if >=25 valid pixels, else 0.
Valid pixels have every RGB channel strictly between 20 and 245. Visible
fraction is clipped patch area / intended integer patch area. Brightness is
mean finite-pixel luma /255; Laplacian variance is reported independently and
does not prefer textured/light skin or change sample acceptance.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import legacy_motion as legacy


REGION_BOXES = {
    "forehead": (0.32, 0.12, 0.68, 0.30),
    "left_cheek": (0.16, 0.46, 0.40, 0.70),
    "right_cheek": (0.60, 0.46, 0.84, 0.70),
}
REGION_NAMES = tuple(REGION_BOXES)
RGB_AGGREGATIONS = ("trimmed_mean", "median")
TRIM_FRACTION = 0.10
QUALITY_DEFINITION = "valid_pixel_fraction * visible_fraction if valid_pixels >=25; else 0; sampling proxy only"
LEGACY_COLUMNS = [
    "frame", "time_s", "source", "rgb_valid", "r", "g", "b",
    "motion_x", "motion_y", "face_x0", "face_y0", "face_x1", "face_y1",
]


def _validate_aggregation(rgb_aggregation):
    if rgb_aggregation not in RGB_AGGREGATIONS:
        raise ValueError(f"rgb_aggregation must be one of {RGB_AGGREGATIONS}")


def _empty_region(status, rgb_aggregation):
    return dict(r=np.nan, g=np.nan, b=np.nan, valid=False, quality=0.0,
                valid_pixel_fraction=0.0, visible_fraction=0.0,
                pixel_count=0, valid_pixels=0, brightness=np.nan,
                laplacian_var=np.nan, status=status, rgb_aggregation=rgb_aggregation)


def _empty_samples(status, rgb_aggregation):
    row = dict(rgb_valid=False, r=np.nan, g=np.nan, b=np.nan,
               roi_status=status, roi_valid_regions=0)
    for name in REGION_NAMES:
        row.update({f"{name}_{key}": value for key, value in _empty_region(status, rgb_aggregation).items()})
    return row


def sample_rois(frame_rgb, landmarks, rgb_aggregation="trimmed_mean"):
    """Return merged RGB plus independent patch estimates and diagnostics.

    This pure-array entry point does not run detection or fill missing data.
    Invalid, too-small, outside-image or dark/saturated patches retain NaN RGB.
    Partly visible patches use the same integer crop and >=25 pixel threshold
    as legacy sampling; their geometry coverage only changes the proxy score.
    """
    _validate_aggregation(rgb_aggregation)
    image = np.asarray(frame_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if landmarks is None:
        return _empty_samples("no_face_or_track", rgb_aggregation)
    height, width = image.shape[:2]
    points = np.asarray([(p.x * width, p.y * height) for p in landmarks.landmark], dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or not np.isfinite(points).all():
        return _empty_samples("invalid_face_geometry", rgb_aggregation)
    x0, x1 = np.percentile(points[:, 0], [2, 98])
    y0, y1 = np.percentile(points[:, 1], [2, 98])
    fw, fh = x1 - x0, y1 - y0
    if fw < 40 or fh < 40:
        return _empty_samples("face_too_small", rgb_aggregation)

    result = _empty_samples("no_valid_skin_patch", rgb_aggregation)
    estimates = []
    for name, (rx0, ry0, rx1, ry1) in REGION_BOXES.items():
        planned = (int(x0 + rx0 * fw), int(y0 + ry0 * fh),
                   int(x0 + rx1 * fw), int(y0 + ry1 * fh))
        xa, ya = max(0, planned[0]), max(0, planned[1])
        xb, yb = min(width, planned[2]), min(height, planned[3])
        info = _empty_region("outside_image", rgb_aggregation)
        if xa < xb and ya < yb:
            patch = image[ya:yb, xa:xb]
            valid = np.all(np.isfinite(patch) & (patch > 20) & (patch < 245), axis=2)
            pixels = patch[valid]
            area = int(patch.shape[0] * patch.shape[1])
            intended_area = (planned[2] - planned[0]) * (planned[3] - planned[1])
            visible_fraction = float(area / intended_area)
            valid_fraction = float(len(pixels) / area)
            gray = (patch.astype(float) * [0.299, 0.587, 0.114]).sum(axis=2)
            finite_luma = gray[np.isfinite(gray)]
            brightness = float(np.clip(finite_luma.mean() / 255, 0, 1)) if finite_luma.size else np.nan
            sharpness = (float(cv2.Laplacian(gray, cv2.CV_64F).var())
                         if np.isfinite(gray).all() and min(gray.shape) >= 3 else np.nan)
            info.update(pixel_count=area, valid_pixels=int(len(pixels)),
                        valid_pixel_fraction=valid_fraction, visible_fraction=visible_fraction,
                        brightness=brightness, laplacian_var=sharpness,
                        status="insufficient_valid_pixels")
            if len(pixels) >= 25:
                values = pixels.astype(np.float64, copy=False)
                if rgb_aggregation == "median":
                    estimate = np.median(values, axis=0)
                else:
                    ordered = np.sort(values, axis=0)
                    trim = int(np.floor(TRIM_FRACTION * len(ordered)))
                    retained = ordered[trim:len(ordered)-trim]
                    estimate = np.mean(retained, axis=0, dtype=np.float64)
                estimates.append(estimate)
                info.update(r=float(estimate[0]), g=float(estimate[1]), b=float(estimate[2]),
                            valid=True, quality=float(valid_fraction * visible_fraction), status="sampled")
        result.update({f"{name}_{key}": value for key, value in info.items()})
    if estimates:
        value = np.mean(estimates, axis=0)
        result.update(rgb_valid=True, r=float(value[0]), g=float(value[1]), b=float(value[2]),
                      roi_status="sampled" if len(estimates) == len(REGION_NAMES) else "sampled_partial_regions",
                      roi_valid_regions=len(estimates))
    return result


def extract(video, max_seconds=None, qa_dir=None, rgb_aggregation="trimmed_mean"):
    """Return trace and FPS, retaining all legacy robust frontend columns.

    Detector settings and sequencing intentionally match the frozen frontend.
    New region diagnostics do not loosen detection, sample or bridge gates.
    QA writes a few overlays only when explicitly supplied a directory.
    """
    _validate_aggregation(rgb_aggregation)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 5:
        cap.release()
        raise ValueError("Invalid fps")
    mesh = legacy.mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    fallback = legacy.mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    previous, previous_points = None, None
    bridge_age, rows, index, qa_tracked = 0, [], 0, 0
    limit = float("inf") if max_seconds is None else round(max_seconds * fps)
    qa_dir = Path(qa_dir) if qa_dir is not None else None
    try:
        while index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            ratio = min(1.0, 960 / w)
            small = cv2.resize(rgb, (round(w * ratio), round(h * ratio))) if ratio < 1 else rgb
            sh, sw = small.shape[:2]
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            prediction = legacy.flow_affine(previous, gray, previous_points)
            result = mesh.process(small)
            source, points = "missing", None
            if result.multi_face_landmarks:
                source = "mesh"
            else:
                result = fallback.process(small)
                if result.multi_face_landmarks:
                    source = "redetected"
            detected = bool(result.multi_face_landmarks)
            if detected:
                points = np.array([(p.x * sw, p.y * sh) for p in result.multi_face_landmarks[0].landmark])
                bridge_age = 0
            elif prediction is not None and bridge_age < round(0.2 * fps):
                points, source = prediction, "flow_tracked"
                bridge_age += 1
            else:
                bridge_age += 1
            sampling = _empty_samples("bridge_limit_reached" if prediction is not None
                                      else "no_face_and_no_valid_flow", rgb_aggregation)
            if points is not None:
                full = points * np.array([w / sw, h / sh])
                sampling = sample_rois(rgb, legacy.mesh_object(full, w, h), rgb_aggregation=rgb_aggregation)
            motion = np.array([np.nan, np.nan])
            if previous_points is not None and prediction is not None:
                motion = np.median(prediction - previous_points, axis=0) / sh
            bbox = [np.nan] * 4
            if points is not None:
                lo = np.percentile(points, 2, axis=0) / [sw, sh]
                hi = np.percentile(points, 98, axis=0) / [sw, sh]
                bbox = [lo[0], lo[1], hi[0], hi[1]]
                if qa_dir and (index % 300 == 0 or (source == "flow_tracked" and qa_tracked < 6)):
                    qa_dir.mkdir(exist_ok=True)
                    overlay = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                    for x, y in points[::5]:
                        cv2.circle(overlay, (round(x), round(y)), 1, (0, 255, 0), -1)
                    cv2.putText(overlay, f"frame {index} {source}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imwrite(str(qa_dir / f"{index:05d}_{source}.jpg"), overlay)
                    if source == "flow_tracked":
                        qa_tracked += 1
            row = dict(frame=index, time_s=index / fps, source=source,
                       rgb_valid=sampling["rgb_valid"], r=sampling["r"], g=sampling["g"], b=sampling["b"],
                       motion_x=motion[0], motion_y=motion[1],
                       face_x0=bbox[0], face_y0=bbox[1], face_x1=bbox[2], face_y1=bbox[3])
            row.update({key: value for key, value in sampling.items() if key not in row})
            row.update(face_detected=detected, flow_available=prediction is not None,
                       bridge_age_frames=bridge_age)
            rows.append(row)
            previous, previous_points = gray, points
            index += 1
            if index % 300 == 0:
                print(f"multi_roi: extracted {index} frames", flush=True)
    finally:
        cap.release()
        mesh.close()
        fallback.close()
    if not rows:
        raise ValueError("No frames decoded")
    trace = pd.DataFrame(rows)
    trace.attrs["region_quality_definition"] = QUALITY_DEFINITION
    trace.attrs["rgb_aggregation"] = rgb_aggregation
    trace.attrs["rgb_trim_fraction_each_tail"] = TRIM_FRACTION if rgb_aggregation == "trimmed_mean" else None
    trace.attrs["detector_policy"] = "legacy robust: width<=960, default detector thresholds, forward/backward affine, max bridge 0.2s"
    return trace, fps
