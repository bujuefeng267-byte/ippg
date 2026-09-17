"""Six measured RGB/local-motion channels on a frozen geometry trace.

This module never reads reference physiology. A row is valid only when its RGB
and signed displacement were measured from the same surviving paired patches.
No interpolation, frame-rate conversion, global-motion substitution, temporal
RGB reconstruction, or detector rerun is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

# Reuse the frozen V28 implementation rather than silently changing LK,
# forward/backward rejection, patch sampling, or colour screening.
HERE = Path(__file__).resolve().parent
FRONTEND_DIR = next((p for p in (
    HERE.parent / 'motion_upgrade_v28_20260912',
    HERE.parent / 'rppg_motion_v28',
    Path('/home/fengbujue/项目/rppg识别/motion_upgrade_v28_20260912'),
) if (p / 'pixel_tracking.py').is_file()), None)
if FRONTEND_DIR is None:
    raise ImportError('The frozen V28 pixel_tracking.py directory is required')
sys.path.insert(0, str(FRONTEND_DIR))
from pixel_tracking import (PixelConfig, PixelTracker, paired_log_change,
                            roi_bounds, sample_patches, seed_grid)
import pixel_tracking as _pixel_module
import baseline_frontend as _baseline_module

if Path(_pixel_module.__file__).resolve().parent != FRONTEND_DIR.resolve():
    raise ImportError('A different pixel_tracking module was already imported')
if Path(_baseline_module.__file__).resolve().parent != FRONTEND_DIR.resolve():
    raise ImportError('A different baseline_frontend module was already imported')

PARENT_REGIONS = ('forehead', 'left_cheek', 'right_cheek')
REGIONS = tuple(f'{parent}_{half}' for parent in PARENT_REGIONS for half in ('a', 'b'))
DEFAULT_CONFIG = PixelConfig(max_points_per_roi=49)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def flag(value):
    """Accept explicit booleans / 0 and 1, never bool(NaN) or bool('False')."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value) and value == 1)
    return isinstance(value, str) and value.strip().lower() in ('true', '1')


def regional_bounds(row, name, shape):
    parent, half = name.rsplit('_', 1)
    bound = roi_bounds(row, parent, shape)
    if bound is None:
        return None
    left, top, right, bottom = bound
    middle = (left + right) / 2
    return np.array([left if half == 'a' else middle, top,
                     middle if half == 'a' else right, bottom])


class RegionalMotionTracker:
    """Pair adjacent original frames; dx/dy are signed frame-height fractions.

    Region a is the image-left half of its parent's box and b the image-right
    half, independently of anatomical naming. Grid points are renewed in every
    preceding frame in original-image pixels; only their LK coordinates are
    mapped to the reduced tracking image. First frames, gaps, and missing parent
    masks remain NaN.
    """

    def __init__(self, config=DEFAULT_CONFIG):
        if not 12 <= config.max_points_per_roi <= 49:
            raise ValueError('This frozen six-region experiment permits 12-49 seeds per region')
        self.config = config
        self.flow = PixelTracker(config, mode='tracking_screened')
        self.previous_rgb = None
        self.previous_row = None
        self.previous_index = None

    def update(self, frame_rgb, baseline_row, frame_index):
        image = np.asarray(frame_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('Expected a nonempty H x W x 3 RGB image')
        height, width = image.shape[:2]
        ratio = min(1., self.config.tracking_width / width)
        small = (cv2.resize(image, (round(width * ratio), round(height * ratio)))
                 if ratio < 1 else image)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        small_height, small_width = gray.shape
        scale = np.array([width / small_width, height / small_height])
        current_rgb = np.asarray(image, np.float32)
        consecutive = (self.previous_index is not None
                       and frame_index == self.previous_index + 1
                       and self.flow.previous_gray.shape == gray.shape
                       and self.previous_rgb.shape == image.shape)
        row = {'frame': int(frame_index), 'time_s': float(baseline_row['time_s'])}
        parts, groups, total = {}, [], 0
        for name in REGIONS:
            parent = name.rsplit('_', 1)[0]
            active = (consecutive and flag(self.previous_row[f'{parent}_valid'])
                      and flag(baseline_row[f'{parent}_valid'])
                      and regional_bounds(baseline_row, name, image.shape) is not None)
            # patch_radius and seed_grid spacing are original-image pixels.
            # Seeding after reducing a 4K frame would enlarge those margins and
            # spacing fourfold, suppressing otherwise measurable small ROIs.
            points = (seed_grid(regional_bounds(self.previous_row, name, image.shape),
                                image.shape, self.config) if active
                      else np.empty((0, 2), np.float32))
            parts[name] = slice(total, total + len(points))
            groups.append(points)
            total += len(points)
        original_points = np.vstack(groups)
        points = (original_points / scale).astype(np.float32)
        moved, flow_ok, errors = self.flow._flow(gray, points) if consecutive else (
            np.full_like(points, np.nan), np.zeros(total, bool), np.full(total, np.nan))
        for name in REGIONS:
            section = parts[name]
            old_points = original_points[section]
            new_points = moved[section] * scale
            tracked = flow_ok[section].copy()
            bound = regional_bounds(baseline_row, name, image.shape)
            if len(tracked):
                if bound is None:
                    tracked[:] = False
                else:
                    # The complete original 7x7 sampled patch must fit within
                    # the current subregion, not merely its centre pixel.
                    left, top, right, bottom = bound
                    margin = self.config.patch_radius
                    tracked &= ((new_points[:, 0] >= left + margin)
                                & (new_points[:, 0] <= right - margin)
                                & (new_points[:, 1] >= top + margin)
                                & (new_points[:, 1] <= bottom - margin))
            rgb, displacement = np.full(3, np.nan), np.full(2, np.nan)
            valid, error = False, np.nan
            info = dict(n_input=0, n_finite=0, n_kept=0, screen_rejected=0)
            if tracked.any():
                old = old_points[tracked]
                new = new_points[tracked]
                previous_samples = sample_patches(self.previous_rgb, old, self.config.patch_radius)
                current_samples = sample_patches(current_rgb, new, self.config.patch_radius)
                estimate, keep, info = paired_log_change(
                    previous_samples, current_samples, screened=True, config=self.config)
                if estimate is not None:
                    rgb = np.mean(current_samples[keep], axis=0, dtype=np.float64)
                    displacement = np.median(new[keep] - old[keep], axis=0) / height
                    error = float(np.median(errors[section][tracked][keep]))
                    valid = True
            for channel, value in zip('rgb', rgb):
                row[f'{name}_{channel}'] = float(value)
            row.update({f'{name}_dx': float(displacement[0]),
                        f'{name}_dy': float(displacement[1]),
                        f'{name}_valid': valid, f'{name}_tracks': info['n_kept'],
                        f'{name}_fb_error': error, f'{name}_seeds': len(old_points),
                        f'{name}_fb_rejected': int(len(old_points) - flow_ok[section].sum()),
                        f'{name}_geometry_rejected': int(flow_ok[section].sum() - tracked.sum()),
                        f'{name}_color_rejected': info['n_input'] - info['n_finite'],
                        f'{name}_screen_rejected': info['screen_rejected']})
        self.previous_rgb = current_rgb.copy()
        self.flow.previous_gray = gray.copy()
        self.previous_row = dict(baseline_row)
        self.previous_index = frame_index
        return row


def _legacy_video_identity(trace_path, video):
    for parent in list(trace_path.parents)[:4]:
        registry = parent / 'inputs_manifest.json'
        if not registry.is_file():
            continue
        records = json.loads(registry.read_text(encoding='utf-8'))
        if not isinstance(records, list):
            raise ValueError('Legacy input manifest must be a list')
        matches = [record['video'] for record in records
                   if isinstance(record, dict) and isinstance(record.get('video'), dict)
                   and Path(record['video'].get('path', '')).resolve() == video]
        if len(matches) != 1:
            raise ValueError('Legacy input manifest does not uniquely identify the video')
        match = matches[0]
        if match.get('bytes') != video.stat().st_size:
            raise ValueError('Legacy input manifest video size mismatch')
        return match.get('sha256'), dict(path=str(registry), sha256=sha256(registry),
            use='video path/bytes/SHA256 only; no reference file read')
    raise ValueError('Trace requires its original nearby inputs_manifest.json for video SHA256')


def validate_geometry_cache(trace_path):
    """Check cache digest, original video stat/digest, frame/time grid, and masks.

    The original frontend identity is retained as provenance. It is not replaced
    with the new sidecar's algorithm identity, since detection is not rerun.
    """
    path = Path(trace_path).resolve()
    if path.is_dir():
        path = path / 'frame_trace.csv'
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    trace_hash = sha256(path)
    if metadata.get('trace_sha256') != trace_hash:
        raise ValueError('Geometry trace SHA256 mismatch')
    identity = metadata['identity']
    video = Path(identity['video']).resolve()
    stat = video.stat()
    if identity.get('size') != stat.st_size or identity.get('mtime_ns') != stat.st_mtime_ns:
        raise ValueError('Geometry trace video size/mtime identity mismatch')
    if identity.get('max_seconds') is not None:
        raise ValueError('This study requires a complete original video trace')
    expected, registry = metadata.get('video_sha256'), None
    if expected is None:
        expected, registry = _legacy_video_identity(path, video)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Original video SHA256 is absent or malformed')
    raw_hash = sha256(video)
    if raw_hash != expected:
        raise ValueError('Geometry trace original video content SHA256 mismatch')
    trace = pd.read_csv(path)
    fps = float(metadata['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid original FPS')
    if len(trace) == 0 or len(trace) != metadata['n_frames']:
        raise ValueError('Geometry trace frame count mismatch or empty trace')
    required = ['frame', 'time_s', 'face_x0', 'face_y0', 'face_x1', 'face_y1']
    required += [f'{parent}_valid' for parent in PARENT_REGIONS]
    if set(required) - set(trace.columns):
        raise ValueError(f'Geometry trace columns missing: {sorted(set(required)-set(trace.columns))}')
    if not np.array_equal(trace.frame.to_numpy(), np.arange(len(trace))):
        raise ValueError('Geometry trace has missing, duplicated, or reordered original frames')
    if not np.allclose(trace.time_s, np.arange(len(trace)) / fps, rtol=0, atol=1e-7):
        raise ValueError('Geometry trace timestamps differ from original CFR grid')
    for parent in PARENT_REGIONS:
        values = trace[f'{parent}_valid']
        allowed = values.map(lambda value: isinstance(value, (bool, np.bool_)) or
                             (isinstance(value, (int, float, np.number)) and
                              np.isfinite(value) and value in (0, 1)) or
                             (isinstance(value, str) and value.strip().lower() in
                              ('true', 'false', '0', '1')))
        if not allowed.all():
            raise ValueError(f'Geometry trace contains invalid {parent} masks')
    cache_info = dict(path=str(path), metadata_path=str(metadata_path),
                      trace_sha256=trace_hash, metadata_sha256=sha256(metadata_path),
                      original_identity=identity, video_identity_registry=registry)
    return trace, fps, video, raw_hash, cache_info


def extract_regional_trace(trace_path, out_dir, config=DEFAULT_CONFIG, threads=1):
    """Write a new directory containing regional_trace.csv and its .json metadata."""
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output path: {out}')
    if threads not in (1, 2):
        raise ValueError('Use one or two OpenCV threads for this extraction')
    cv2.setNumThreads(threads)
    trace, fps, video, raw_hash, cache_info = validate_geometry_cache(trace_path)
    source_paths = [Path(__file__).resolve(), FRONTEND_DIR / 'pixel_tracking.py',
                    FRONTEND_DIR / 'baseline_frontend.py']
    sources = {str(path): sha256(path) for path in source_paths}
    tracker = RegionalMotionTracker(config)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    rows, resolution = [], None
    try:
        if not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, rtol=0, atol=1e-6):
            raise ValueError('Decoded video FPS differs from geometry trace')
        for index, baseline_row in enumerate(trace.to_dict('records')):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Video ended before the last cached geometry frame')
            current_resolution = [int(bgr.shape[1]), int(bgr.shape[0])]
            if resolution is None:
                resolution = current_resolution
            elif resolution != current_resolution:
                raise ValueError('Video changes dimensions within the geometry trace')
            rows.append(tracker.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), baseline_row, index))
            if (index + 1) % 300 == 0:
                print(f'regional_motion: {index + 1}/{len(trace)} original frames', flush=True)
        ok, _ = cap.read()
        if ok:
            raise ValueError('Video has extra frames beyond the cached geometry trace')
    finally:
        cap.release()
    # Detect input/source mutation during extraction before committing results.
    if sha256(video) != raw_hash or sha256(cache_info['path']) != cache_info['trace_sha256']:
        raise ValueError('Video or geometry trace changed during extraction')
    if sha256(cache_info['metadata_path']) != cache_info['metadata_sha256']:
        raise ValueError('Geometry metadata changed during extraction')
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError('Extraction source changed during execution')
    result = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=False)
    csv_path = out / 'regional_trace.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        result.to_csv(stream, index=False)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        fps=fps, n_frames=len(result), resolution=resolution,
        trace_sha256=sha256(csv_path), raw_video_sha256=raw_hash,
        video_sha256=raw_hash, video=str(video), old_cache_path=cache_info['path'],
        geometry_cache=cache_info, config=asdict(config), opencv_threads=threads,
        regions=list(REGIONS), region_boxes=_baseline_module.REGION_BOXES,
        subregion_split='image-horizontal midpoint; a=image-left half; b=image-right half',
        rgb_kind='mean_current_RGB_of_identical_screened_corresponding_patches',
        motion_kind='median_current_minus_previous_xy_of_same_patches / original_frame_height',
        motion_units='original_frame_height_fraction_per_original_frame',
        fb_error_units='tracking_image_pixels', first_frame_valid=False,
        seed_grid_units='original_image_pixels',
        patch_radius_units='original_image_pixels',
        sampling_revision='full_resolution_seed_grid_v2',
        original_frames_decoded=len(result), original_video_eof_confirmed=True,
        reference_used=False, interpolation=False, frame_rate_conversion=False,
        source_sha256=sources, opencv_version=cv2.__version__, status='complete')
    with (out / 'regional_trace.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--threads', type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    result, metadata = extract_regional_trace(args.trace, args.out, threads=args.threads)
    print(json.dumps(dict(out=str(args.out.resolve()), fps=metadata['fps'],
                          n_frames=len(result), trace_sha256=metadata['trace_sha256']),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()


