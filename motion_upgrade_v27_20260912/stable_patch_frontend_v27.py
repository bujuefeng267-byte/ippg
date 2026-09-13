"""Stable landmark-anchored, independent patch photometry for V27.

Twelve spatial patches (four within each of the three inherited regions) retain
separate RGB histories. Tracking points are aggregated only WITHIN their patch.
No pulse waveform, reference, heart-rate frequency or video-specific setting is
accepted. Raw RGB is measured camera colour; tracked RGB is an unanchored
relative-colour reconstruction. Neither is a physiological pulse measurement.

The inherited detector/re-detector and forward/backward affine bridge policy is
unchanged. A patch's local landmark indices are selected once at the first
usable genuine detection, then remain fixed for the entire video, across gaps.
RGB state is reset after a gap; landmark identity is never reassigned.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import json
import platform

import cv2
import numpy as np

import legacy_motion as legacy
from baseline_frontend import REGION_BOXES
from pixel_tracking import PixelConfig, PixelTracker, paired_log_change, sample_patches


REGIONS = tuple(REGION_BOXES)
PATCH_IDS = tuple(f'{region}_{i}' for region in REGIONS for i in range(4))


@dataclass(frozen=True)
class PatchConfig:
    rows: int = 2
    columns: int = 2
    local_landmarks: int = 12
    cell_inset_fraction: float = .05
    max_anchor_condition: float = 100.
    max_local_axis_ratio: float = 6.
    max_anchor_residual_face_fraction: float = .05
    min_face_pixels: float = 40.
    min_valid_pixels: int = 25
    points_per_side: int = 5
    patch_radius: int = 3
    min_tracks: int = 12
    fb_max_px: float = .5
    screen_z: float = 3.
    screen_floor: float = .003
    tracking_width: int = 960
    max_bridge_s: float = .2
    trim_fraction: float = .10

    def __post_init__(self):
        if (self.rows, self.columns) != (2, 2):
            raise ValueError('The single frozen V27 layout is 2 x 2 per region')
        if self.local_landmarks < 3 or self.points_per_side < 1:
            raise ValueError('Invalid local anchor or sampling count')
        if not 0 <= self.cell_inset_fraction < .5 or not 0 <= self.trim_fraction < .5:
            raise ValueError('Invalid inset or trim fraction')
        if self.max_anchor_condition <= 1 or self.max_local_axis_ratio <= 1:
            raise ValueError('Invalid geometry condition limit')
        if any(not np.isfinite(x) or x <= 0 for x in (
                self.max_anchor_residual_face_fraction, self.min_face_pixels,
                self.min_valid_pixels, self.patch_radius, self.min_tracks,
                self.fb_max_px, self.screen_z, self.screen_floor, self.tracking_width)):
            raise ValueError('Sampling and geometry limits must be positive')
        if self.max_bridge_s != .2:
            raise ValueError('Preserve the inherited 0.2 second bridge limit')

    def pixel_config(self):
        return PixelConfig(max_points_per_roi=self.points_per_side**2,
                           patch_radius=self.patch_radius, min_tracks=self.min_tracks,
                           fb_max_px=self.fb_max_px, screen_z=self.screen_z,
                           screen_floor=self.screen_floor, tracking_width=self.tracking_width)


DEFAULT_CONFIG = PatchConfig()


def _face_bounds(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        return None
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return np.r_[lo, hi]


class StableAnchors:
    """Fixed landmark IDs and local offsets; no detector-specific point numbers."""
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.anchors = None
        self.initial_frame = None

    def initialize(self, points, frame):
        if self.anchors is not None:
            return True
        points = np.asarray(points, float)
        box = _face_bounds(points)
        if box is None or len(points) < self.config.local_landmarks:
            return False
        x0, y0, x1, y1 = box
        width, height = x1-x0, y1-y0
        if min(width, height) < self.config.min_face_pixels:
            return False
        anchors = {}
        for region, (rx0, ry0, rx1, ry1) in REGION_BOXES.items():
            for row in range(2):
                for col in range(2):
                    ident = f'{region}_{2*row+col}'
                    cell_w, cell_h = (rx1-rx0)*width/2, (ry1-ry0)*height/2
                    left, top = x0+rx0*width+col*cell_w, y0+ry0*height+row*cell_h
                    inset = self.config.cell_inset_fraction
                    quad = np.array([[left+inset*cell_w, top+inset*cell_h],
                                     [left+(1-inset)*cell_w, top+inset*cell_h],
                                     [left+(1-inset)*cell_w, top+(1-inset)*cell_h],
                                     [left+inset*cell_w, top+(1-inset)*cell_h]])
                    center = quad.mean(0)
                    # Stable sorting resolves equal distances without changing IDs.
                    ids = np.argsort(np.sum(((points-center)/height)**2, axis=1),
                                     kind='stable')[:self.config.local_landmarks]
                    local = (points[ids]-center)/height
                    design = np.column_stack([local, np.ones(len(ids))])
                    # Test geometric degeneracy without confusing a small local
                    # patch (small coordinate units) with an ill-shaped anchor.
                    centered = local-local.mean(0)
                    scale = float(np.sqrt(np.mean(centered**2)))
                    normalized = np.column_stack([centered/max(scale, 1e-15), np.ones(len(ids))])
                    condition = float(np.linalg.cond(normalized))
                    if np.linalg.matrix_rank(design) < 3 or condition > self.config.max_anchor_condition:
                        return False
                    anchors[ident] = dict(region=region, row=row, column=col,
                        landmark_indices=ids.tolist(), initial_center_px=center.tolist(),
                        initial_face_height_px=float(height), local_landmark_coordinates=local.tolist(),
                        local_quad_coordinates=((quad-center)/height).tolist(),
                        initial_quad_px=quad.tolist(), normalized_design_condition=condition)
        self.anchors, self.initial_frame = anchors, int(frame)
        return True

    def project(self, points):
        result = {ident: (None, np.nan, 'anchors_unavailable') for ident in PATCH_IDS}
        if self.anchors is None:
            return result
        if points is None:
            return {ident: (None, np.nan, 'landmarks_missing') for ident in PATCH_IDS}
        points = np.asarray(points, float)
        box = _face_bounds(points)
        if box is None or min(box[2:]-box[:2]) < self.config.min_face_pixels:
            return {ident: (None, np.nan, 'invalid_face_geometry') for ident in PATCH_IDS}
        face_height = box[3]-box[1]
        for ident, anchor in self.anchors.items():
            ids = np.asarray(anchor['landmark_indices'], int)
            if ids.max() >= len(points):
                result[ident] = None, np.nan, 'landmark_count_changed'
                continue
            design = np.column_stack([anchor['local_landmark_coordinates'], np.ones(len(ids))])
            mapping = np.linalg.lstsq(design, points[ids], rcond=None)[0]
            axes = np.linalg.svd(mapping[:2], compute_uv=False)
            residual = float(np.sqrt(np.mean(np.sum((design@mapping-points[ids])**2, axis=1)))/face_height)
            if (not np.isfinite(mapping).all() or axes[-1] <= 1e-8 or
                    axes[0]/axes[-1] > self.config.max_local_axis_ratio or
                    np.linalg.det(mapping[:2]) <= 0 or
                    residual > self.config.max_anchor_residual_face_fraction):
                result[ident] = None, residual, 'local_anchor_geometry_rejected'
                continue
            local_quad = np.asarray(anchor['local_quad_coordinates'])
            quad = np.column_stack([local_quad, np.ones(4)])@mapping
            result[ident] = quad, residual, 'anchored'
        return result

    def metadata(self):
        return dict(initial_frame=self.initial_frame, patches=self.anchors,
                    identity_policy='fixed first-detection local landmark indices and local offsets; never re-seeded',
                    coordinates='image pixels; local offsets normalized by initial face height')


def sample_polygon(image, quad, config=DEFAULT_CONFIG):
    """Mean of measured pixels inside the patch, with per-channel tail trimming."""
    info = dict(valid=False, quality=0., raw_r=np.nan, raw_g=np.nan, raw_b=np.nan,
                pixel_count=0, valid_pixels=0, visible_fraction=0., valid_pixel_fraction=0.,
                raw_status='invalid_geometry')
    if quad is None:
        return info
    quad = np.asarray(quad, float)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if not np.isfinite(quad).all() or area <= 0:
        return info
    h, w = image.shape[:2]
    lo = np.floor(quad.min(0)).astype(int)
    hi = np.ceil(quad.max(0)).astype(int)+1
    xa, ya = max(0, lo[0]), max(0, lo[1])
    xb, yb = min(w, hi[0]), min(h, hi[1])
    if xa >= xb or ya >= yb:
        info['raw_status'] = 'outside_image'
        return info
    mask = np.zeros((yb-ya, xb-xa), np.uint8)
    local = np.rint((quad-[xa, ya])*256).astype(np.int32)
    cv2.fillConvexPoly(mask, local, 1, shift=8)
    pixels = np.asarray(image[ya:yb, xa:xb], float)[mask.astype(bool)]
    valid = np.isfinite(pixels).all(1) & (pixels > 20).all(1) & (pixels < 245).all(1)
    # Continuous polygon clipping measures visibility independently of the
    # integer pixel-center rasterizer (which includes boundary pixels).
    viewport = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    visible_area, _ = cv2.intersectConvexConvex(quad.astype(np.float32), viewport)
    visible = float(np.clip(visible_area/area, 0., 1.))
    fraction = float(valid.mean()) if len(valid) else 0.
    info.update(pixel_count=len(pixels), valid_pixels=int(valid.sum()), visible_fraction=visible,
                valid_pixel_fraction=fraction, raw_status='insufficient_valid_pixels')
    if valid.sum() >= config.min_valid_pixels:
        values = np.sort(pixels[valid], axis=0)
        cut = int(config.trim_fraction*len(values))
        rgb = values[cut:len(values)-cut].mean(0, dtype=np.float64)
        info.update(valid=True, quality=fraction*visible, raw_status='sampled',
                    **{f'raw_{c}': float(value) for c, value in zip('rgb', rgb)})
    return info


def seed_patch(quad, config=DEFAULT_CONFIG):
    if quad is None:
        return np.empty((0, 2), np.float32)
    quad = np.asarray(quad, float)
    coordinates = np.linspace(.1, .9, config.points_per_side)
    uv = np.array([(u, v) for v in coordinates for u in coordinates])
    u, v = uv[:, 0:1], uv[:, 1:2]
    points = ((1-u)*(1-v)*quad[0] + u*(1-v)*quad[1] + u*v*quad[2] + (1-u)*v*quad[3])
    good = points_inside(points, quad, config.patch_radius+1)
    return points[good].astype(np.float32)


def points_inside(points, quad, margin=0.):
    if quad is None:
        return np.zeros(len(points), bool)
    contour = np.asarray(quad, np.float32)
    return np.array([np.isfinite(point).all() and
                     cv2.pointPolygonTest(contour, tuple(map(float, point)), True) >= margin
                     for point in points], bool)


class StablePatchSampler:
    """Array-level sampling entry point, independently testable without a face model."""
    def __init__(self, fps, config=DEFAULT_CONFIG):
        if not np.isfinite(fps) or fps < 5:
            raise ValueError('Invalid FPS')
        self.fps, self.config = float(fps), config
        self.anchors = StableAnchors(config)
        self.flow = PixelTracker(config.pixel_config(), mode='tracking_screened')
        self.previous_index = self.previous_rgb = self.previous_gray = None
        self.previous_rows = self.previous_quads = None
        self.level = {ident: None for ident in PATCH_IDS}

    def update(self, image, points, frame, source, bridge_age_frames=0):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or not isinstance(frame, (int, np.integer)) or frame < 0:
            raise ValueError('Expected RGB image and nonnegative frame index')
        if source not in ('mesh', 'redetected', 'flow_tracked', 'missing'):
            raise ValueError('Unknown geometry source')
        if (points is None) != (source == 'missing'):
            raise ValueError('Geometry source must agree with landmark availability')
        if points is not None and source in ('mesh', 'redetected'):
            self.anchors.initialize(points, frame)
        projection = self.anchors.project(points)
        h, w = image.shape[:2]
        ratio = min(1., self.config.tracking_width/w)
        small = cv2.resize(image, (round(w*ratio), round(h*ratio))) if ratio < 1 else image
        if small.dtype not in (np.uint8, np.uint16, np.float32):
            small = small.astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        sh, sw = gray.shape
        scale = np.array([w/sw, h/sh])
        consecutive = (self.previous_index is not None and frame == self.previous_index+1 and
                       self.previous_gray.shape == gray.shape and self.previous_rgb.shape == image.shape)
        if not consecutive:
            self.level = {ident: None for ident in PATCH_IDS}
        rows, seeds, parts = {}, [], {}
        cursor = 0
        for ident in PATCH_IDS:
            quad, residual, geometry_status = projection[ident]
            info = sample_polygon(image, quad, self.config)
            row = dict(frame=int(frame), time_s=frame/self.fps, patch_id=ident,
                       region=ident.rsplit('_', 1)[0], source=source,
                       face_detected=source in ('mesh', 'redetected'),
                       is_bridge=source == 'flow_tracked', bridge_age_frames=int(bridge_age_frames),
                       geometry_valid=quad is not None, geometry_status=geometry_status,
                       anchor_residual_face_fraction=residual, **info)
            for i in range(4):
                row[f'corner{i}_x'] = float(quad[i, 0]) if quad is not None else np.nan
                row[f'corner{i}_y'] = float(quad[i, 1]) if quad is not None else np.nan
            old_ok = consecutive and bool(self.previous_rows[ident]['valid'])
            points_old = (seed_patch(self.previous_quads[ident], self.config)
                          if old_ok and info['valid'] else np.empty((0, 2), np.float32))
            parts[ident] = slice(cursor, cursor+len(points_old))
            cursor += len(points_old)
            seeds.append(points_old/scale)
            rows[ident] = row
        all_points = np.vstack(seeds).astype(np.float32)
        self.flow.previous_gray = self.previous_gray
        moved, flow_ok, fb_errors = (self.flow._flow(gray, all_points) if consecutive else
            (np.full_like(all_points, np.nan), np.zeros(len(all_points), bool), np.full(len(all_points), np.nan)))
        for ident in PATCH_IDS:
            row, sl = rows[ident], parts[ident]
            quad = projection[ident][0]
            good = flow_ok[sl] & points_inside(moved[sl]*scale, quad, self.config.patch_radius+1)
            delta = paired_delta = raw_delta = None
            stats = dict(n_input=0, n_finite=0, n_kept=0, screen_rejected=0)
            if good.any():
                previous = sample_patches(self.previous_rgb, all_points[sl][good]*scale, self.config.patch_radius)
                current = sample_patches(np.asarray(image, np.float32), moved[sl][good]*scale, self.config.patch_radius)
                paired_delta, _, stats = paired_log_change(previous, current, True, self.config.pixel_config())
            raw = np.array([row[f'raw_{c}'] for c in 'rgb'])
            old = (np.array([self.previous_rows[ident][f'raw_{c}'] for c in 'rgb'])
                   if consecutive else np.full(3, np.nan))
            if row['valid'] and np.isfinite(old).all() and (old > 0).all():
                raw_delta = np.log(raw/old)
            src, reset = 'missing', False
            if row['valid']:
                if self.level[ident] is None or not consecutive or raw_delta is None:
                    self.level[ident] = raw.copy()
                    src, reset = 'baseline_reset', True
                else:
                    delta = paired_delta if paired_delta is not None else raw_delta
                    src = 'tracked_ratio' if paired_delta is not None else 'baseline_ratio_fallback'
                    updated = self.level[ident]*np.exp(delta)
                    if not np.isfinite(updated).all() or (updated <= 1e-6).any() or (updated > 1e6).any():
                        updated = raw.copy()
                        src, reset = 'numerical_reset', True
                        delta = None
                    self.level[ident] = updated
                tracked_rgb = self.level[ident]
            else:
                self.level[ident] = None
                tracked_rgb = np.full(3, np.nan)
            finite_errors = fb_errors[sl][np.isfinite(fb_errors[sl])]
            row.update(pixel_source=src, pixel_reset=reset, pixel_tracks=stats['n_kept'],
                       pixel_seeds=sl.stop-sl.start, pixel_screen_rejected=stats['screen_rejected'],
                       pixel_color_rejected=stats['n_input']-stats['n_finite'],
                       pixel_fb_rejected=int((~flow_ok[sl]).sum()),
                       pixel_geometry_rejected=int(flow_ok[sl].sum()-good.sum()),
                       pixel_fb_error_median=float(np.median(finite_errors)) if len(finite_errors) else np.nan,
                       tracked_valid=bool(np.isfinite(tracked_rgb).all()))
            for j, c in enumerate('rgb'):
                row[f'tracked_{c}'] = float(tracked_rgb[j])
                row[f'pixel_log_delta_{c}'] = float(delta[j]) if delta is not None else np.nan
                row[f'paired_log_delta_{c}'] = float(paired_delta[j]) if paired_delta is not None else np.nan
                row[f'raw_log_delta_{c}'] = float(raw_delta[j]) if raw_delta is not None else np.nan
        self.previous_index, self.previous_rgb, self.previous_gray = int(frame), np.asarray(image, np.float32).copy(), gray.copy()
        self.previous_rows = rows
        self.previous_quads = {ident: projection[ident][0] for ident in PATCH_IDS}
        return [rows[ident] for ident in PATCH_IDS]


def choose_geometry(detected_points, prediction, bridge_age, fps, config=DEFAULT_CONFIG):
    """The inherited detector-first, bounded-affine-bridge state transition."""
    if detected_points is not None:
        return detected_points, 0, False
    if prediction is not None and bridge_age < round(config.max_bridge_s*fps):
        return prediction, bridge_age+1, True
    return None, bridge_age+1, False


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def source_hashes():
    here = Path(__file__).resolve().parent
    return {name: digest(here/name) for name in (
        'stable_patch_frontend_v27.py', 'legacy_motion.py', 'analyze_rppg.py',
        'baseline_frontend.py', 'pixel_tracking.py')}


def extract_patches(video, out, config=DEFAULT_CONFIG, *, max_seconds=None):
    """Decode original video and save independent long-form patch/frame traces.

    Output must not exist. No old bbox-only geometry cache is accepted. Complete
    landmarks are freshly detected and saved in landmark_trace.npz, bound to the
    original video/source hashes. Missing samples remain NaN; no temporal fill.
    """
    video, out = Path(video).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing output; choose a new directory')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max_seconds must be positive')
    video_sha, frozen = digest(video), source_hashes()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 5:
        cap.release()
        raise ValueError('Invalid video FPS')
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mesh = legacy.mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
        min_detection_confidence=.5, min_tracking_confidence=.5)
    fallback = legacy.mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
        min_detection_confidence=.5)
    sampler = StablePatchSampler(fps, config)
    previous = previous_points = None
    bridge_age = frame = 0
    landmarks, sources = [], Counter()
    valid_counts = Counter()
    patch_sources = {ident: Counter() for ident in PATCH_IDS}
    limit = float('inf') if max_seconds is None else round(max_seconds*fps)
    out.mkdir(parents=True)
    try:
        with (out/'patch_trace.csv').open('w', newline='', encoding='utf-8') as patch_file, \
                (out/'frame_trace.csv').open('w', newline='', encoding='utf-8') as frame_file:
            patch_writer = frame_writer = None
            while frame < limit:
                ok, bgr = cap.read()
                if not ok:
                    break
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                ratio = min(1., config.tracking_width/w)
                small = cv2.resize(image, (round(w*ratio), round(h*ratio))) if ratio < 1 else image
                sh, sw = small.shape[:2]
                gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                prediction = legacy.flow_affine(previous, gray, previous_points)
                result = mesh.process(small)
                source = 'mesh'
                if not result.multi_face_landmarks:
                    result = fallback.process(small)
                    source = 'redetected'
                detected = bool(result.multi_face_landmarks)
                detected_points = (np.array([(p.x*sw, p.y*sh) for p in result.multi_face_landmarks[0].landmark])
                                   if detected else None)
                points, bridge_age, is_bridge = choose_geometry(detected_points, prediction, bridge_age, fps, config)
                if is_bridge:
                    source = 'flow_tracked'
                elif points is None:
                    source = 'missing'
                full = points*np.array([w/sw, h/sh]) if points is not None else None
                rows = sampler.update(image, full, frame, source, bridge_age)
                motion = np.array([np.nan, np.nan])
                if previous_points is not None and prediction is not None:
                    motion = np.median(prediction-previous_points, axis=0)/sh
                box = _face_bounds(points) if points is not None else None
                box = box/np.array([sw, sh, sw, sh]) if box is not None else np.full(4, np.nan)
                frame_row = dict(frame=frame, time_s=frame/fps, source=source,
                    motion_x=float(motion[0]), motion_y=float(motion[1]),
                    face_x0=box[0], face_y0=box[1], face_x1=box[2], face_y1=box[3],
                    face_detected=detected, flow_available=prediction is not None,
                    bridge_age_frames=bridge_age, is_bridge=is_bridge,
                    valid_patch_count=sum(row['valid'] for row in rows), anchor_initialized=sampler.anchors.anchors is not None)
                if patch_writer is None:
                    patch_writer = csv.DictWriter(patch_file, fieldnames=list(rows[0]))
                    frame_writer = csv.DictWriter(frame_file, fieldnames=list(frame_row))
                    patch_writer.writeheader()
                    frame_writer.writeheader()
                patch_writer.writerows(rows)
                frame_writer.writerow(frame_row)
                landmarks.append(full.copy() if full is not None else None)
                sources[source] += 1
                for row in rows:
                    valid_counts[row['patch_id']] += int(row['valid'])
                    patch_sources[row['patch_id']][row['pixel_source']] += 1
                previous, previous_points = gray, points
                frame += 1
                if frame % 300 == 0:
                    print(f'stable_patch_frontend: {frame} frames, 12 independent patches', flush=True)
    finally:
        cap.release()
        mesh.close()
        fallback.close()
    if frame == 0:
        raise ValueError('No frames decoded')
    counts = {len(points) for points in landmarks if points is not None}
    if len(counts) > 1:
        raise ValueError('Landmark count changed; preserve incomplete output for diagnosis')
    count = next(iter(counts), 0)
    landmark_array = np.full((frame, count, 2), np.nan, np.float64)
    for i, points in enumerate(landmarks):
        if points is not None:
            landmark_array[i] = points
    np.savez_compressed(out/'landmark_trace.npz', points_px=landmark_array,
                        frame=np.arange(frame), time_s=np.arange(frame)/fps)
    (out/'patch_anchors.json').write_text(json.dumps(sampler.anchors.metadata(), ensure_ascii=False,
                                                   indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if digest(video) != video_sha or source_hashes() != frozen:
        raise ValueError('Video or frontend sources changed during extraction')
    metadata = dict(video_path=str(video), video_sha256=video_sha, video_bytes=video.stat().st_size,
        fps=fps, frames=frame, declared_frames=declared_frames, duration_s=frame/fps,
        max_seconds=max_seconds, source_hashes=frozen, config=asdict(config),
        patch_ids=list(PATCH_IDS), physical_regions=list(REGIONS), patch_count=12,
        frame_source_counts=dict(sources), valid_patch_frames=dict(valid_counts),
        patch_pixel_source_counts={key: dict(value) for key, value in patch_sources.items()},
        versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                      mediapipe=legacy.mp.__version__), reference_used=False, temporal_RGB_filling=False,
        RGB_aggregation_across_patches=False, anchors_reinitialized_after_gap=False,
        quality_definition='valid_pixel_fraction * visible_polygon_area_fraction; >=25 valid measured pixels',
        tracked_RGB_kind='unanchored paired relative colour; raw anchor ratio fallback; gap resets',
        landmark_trace_precision='float64 image-pixel coordinates; identical to online anchor-geometry inputs',
        detector_policy='inherited robust: width<=960; mesh then static redetection; FB-affine bridge <=round(.2*fps) frames',
        output_hashes={name: digest(out/name) for name in (
            'patch_trace.csv', 'frame_trace.csv', 'patch_anchors.json', 'landmark_trace.npz')})
    (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                               allow_nan=False)+'\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float)
    arguments = parser.parse_args()
    print(json.dumps(extract_patches(arguments.video, arguments.output,
                                     max_seconds=arguments.max_seconds), ensure_ascii=False), flush=True)
