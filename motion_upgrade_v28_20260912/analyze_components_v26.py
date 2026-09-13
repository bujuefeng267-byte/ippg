#!/usr/bin/env python3
"""Reference-free experimental V26 component entry point for one local video.

Fixed V25 geometry / screened pixel sampling -> preserved baseline and anchored
tracked RGB -> POS/CHROM -> physical-ROI component consensus -> saved-wave HR.
This is offline, adaptively band-limited measured signal, not morphology proof.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from analyze_motion_v2 import per_roi_signals, source_hashes, validate_trace
from anchored_reconstruction import AnchorConfig, anchored_reconstruction, baseline_view
from evidence_hr import DEFAULT_CONFIG, estimate_evidence
from motion_frontend import extract
from pixel_tracking import PixelConfig

HERE = Path(__file__).resolve().parent
ROIS = ('forehead', 'left_cheek', 'right_cheek')
PARAMETERS = dict(rgb_aggregation='trimmed_mean', pixel_mode='tracking_screened',
                  max_gap_s=.1, window_s=10., step_s=1., min_bpm=42., max_bpm=210.,
                  max_seconds=None, rgb_reconstruction='anchored_log')
PROPOSAL_COLUMNS = ['window_index', 'time_s', 'proposal_bpm', 'proposal_supported',
                    'reacquired', 'generated', 'contributing_rois', 'channels']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def frontend_identity(video):
    video = Path(video).resolve()
    stat = video.stat()
    return dict(video=str(video), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                max_seconds=None, rgb_aggregation=PARAMETERS['rgb_aggregation'],
                pixel_mode=PARAMETERS['pixel_mode'], pixel_config=asdict(PixelConfig()),
                rgb_reconstruction='anchored_log', anchor_config=asdict(AnchorConfig()),
                frontend_hashes={name: sha(HERE/name) for name in
                    ['motion_frontend.py', 'baseline_frontend.py', 'pixel_tracking.py',
                     'legacy_motion.py', 'analyze_rppg.py', 'anchored_reconstruction.py']})


def load_variant(name):
    module_name = {'component': 'component_fusion_v26',
                   'harmonic': 'component_harmonics_v26'}[name]
    if not (HERE/f'{module_name}.py').is_file():
        raise ValueError(f'Variant module is unavailable: {module_name}.py')
    module = importlib.import_module(module_name)
    return module


def frozen_sources(variant):
    names = ['analyze_components_v26.py', 'component_fusion_v26.py']
    if variant == 'harmonic':
        names += ['component_harmonics_v26.py', 'motion_harmonic_evidence_v26.py']
    return {**source_hashes(), **{name: sha(HERE/name) for name in names}}


def legacy_video_digest(cache, video):
    """Read video identity only from a nearby original input manifest.

    Historical frame_trace metadata has only stat identity. Do not pretend that
    newly hashing the current video proves its identity when that trace was made.
    Such a cache needs its original input manifest; no reference file is opened.
    """
    for parent in list(Path(cache).resolve().parents)[:4]:
        registry = parent/'inputs_manifest.json'
        if not registry.is_file():
            continue
        rows = json.loads(registry.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Legacy video identity manifest must contain a list')
        matches = [row['video'] for row in rows if isinstance(row, dict)
                   and isinstance(row.get('video'), dict)
                   and Path(row['video'].get('path', '')).resolve() == video]
        if len(matches) != 1:
            raise ValueError('Legacy cache manifest does not uniquely identify this video')
        record = matches[0]
        if record.get('bytes') != video.stat().st_size:
            raise ValueError('Legacy input manifest video size mismatch')
        expected = record.get('sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Legacy input manifest video SHA256 is missing')
        return expected, dict(path=str(registry), sha256=sha(registry),
                              use='video path/bytes/SHA256 only; no reference file read')
    raise ValueError('Legacy trace has no video SHA256. Keep its original nearby '
                     'inputs_manifest.json, or omit --trace-cache for fresh extraction.')


def load_trace_cache(cache, video, identity, video_sha256):
    cache = Path(cache).resolve()
    if cache.is_dir():
        cache = cache/'frame_trace.csv'
    metadata_path = cache.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('identity') != identity:
        raise ValueError('Trace cache video/stat/parameters/frontend identity mismatch')
    if metadata.get('trace_sha256') != sha(cache):
        raise ValueError('Trace cache SHA256 mismatch')
    expected_video = metadata.get('video_sha256')
    registry = None
    if expected_video is None:
        expected_video, registry = legacy_video_digest(cache, video)
    if expected_video != video_sha256:
        raise ValueError('Trace cache video content SHA256 mismatch')
    trace = pd.read_csv(cache)
    fps = float(metadata['fps'])
    if len(trace) != metadata['n_frames']:
        raise ValueError('Trace cache frame count mismatch')
    validate_component_trace(trace, fps)
    return trace, fps, dict(path=str(cache), metadata_path=str(metadata_path),
                            trace_sha256=sha(cache), metadata_sha256=sha(metadata_path),
                            video_identity_registry=registry)


def validate_component_trace(trace, fps):
    validate_trace(trace, fps)
    if fps <= PARAMETERS['max_bpm']/30:
        raise ValueError('Fixed 42-210 bpm experiment requires FPS > 7 (Nyquist)')
    # These preserved raw measurements and source labels must belong to this
    # same trace. Missing diagnostics must fail, never become perfect tracking.
    baseline = baseline_view(trace)
    validate_trace(baseline, fps)
    required = ['pixel_rgb_kind', 'anchor_hz', 'anchor_order', 'motion_x', 'motion_y',
                'face_x0', 'face_y0', 'face_x1', 'face_y1']
    required += [f'{roi}_{suffix}' for roi in ROIS for suffix in
                 ['pixel_source', 'pixel_tracks', 'pixel_log_delta_r',
                  'pixel_log_delta_g', 'pixel_log_delta_b']]
    missing = sorted(set(required)-set(trace.columns))
    if missing:
        raise ValueError(f'Trace lacks screened/anchored sampling diagnostics: {missing}')
    if not trace.pixel_rgb_kind.eq('anchored_paired_relative_colour').all():
        raise ValueError('Expected anchored paired relative colour trace')
    if not trace.anchor_hz.eq(AnchorConfig().anchor_hz).all() or not trace.anchor_order.eq(4).all():
        raise ValueError('Trace anchor parameters differ from fixed experiment')
    for roi in ROIS:
        if not trace[f'{roi}_pixel_source'].isin(
                ['missing', 'baseline_reset', 'tracked_ratio',
                 'baseline_ratio_fallback', 'numerical_reset']).all():
            raise ValueError(f'Unknown pixel source for {roi}')


def prepare_roi_channels(trace, fps):
    channels, tables = {}, {}
    for branch, view in [('baseline', baseline_view(trace)), ('tracked', trace)]:
        _, roi_wave = per_roi_signals(view, fps, PARAMETERS['max_gap_s'],
                                     PARAMETERS['min_bpm'], PARAMETERS['max_bpm'])
        tables[branch] = roi_wave
        for roi in ROIS:
            for method in ('pos', 'chrom'):
                channels[f'{branch}/{roi}/{method}'] = roi_wave[f'{roi}_{method}'].to_numpy(float)
    for roi in ROIS:
        for tag in ('observed', 'interpolated'):
            np.testing.assert_array_equal(tables['baseline'][f'{roi}_{tag}'],
                                          tables['tracked'][f'{roi}_{tag}'])
    return channels, tables


def infer_components(trace, fps, variant):
    channels, roi_tables = prepare_roi_channels(trace, fps)
    module = load_variant(variant)
    kwargs = {key: PARAMETERS[key] for key in ('window_s', 'step_s', 'min_bpm', 'max_bpm')}
    if len(trace) < round(PARAMETERS['window_s']*fps):
        # The frozen component module's empty proposal table has no schema.
        # A short new video has no complete windows, not a fictitious estimate.
        wave = pd.DataFrame(dict(time_s=trace.time_s, base=np.nan, covered=False,
                                 observed=False, interpolated=False))
        proposals = pd.DataFrame(columns=PROPOSAL_COLUMNS)
        candidates = []
    else:
        wave, proposals, candidates = module.fuse_components(
            channels, trace, roi_tables['baseline'], fps, **kwargs)
    return wave, proposals, candidates, roi_tables


def read_saved_hr(waveform_path, trace, fps, proposals):
    saved = pd.read_csv(waveform_path)
    if len(saved) != len(trace):
        raise ValueError('Saved waveform lost original frames')
    np.testing.assert_allclose(saved.time_s, trace.time_s, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(np.isfinite(saved.base), saved.covered.to_numpy(bool))
    if (saved.observed & ~saved.covered).any() or (saved.interpolated & ~saved.covered).any():
        raise ValueError('Saved sampling provenance outside waveform support')
    hr = estimate_evidence(saved.base.to_numpy(float), pd.DataFrame({'rgb_valid': saved.observed}),
                           saved.interpolated.to_numpy(bool), fps, motion_trace=trace,
                           **{key: PARAMETERS[key] for key in ('window_s', 'step_s', 'min_bpm', 'max_bpm')})
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    starts = np.arange(0, len(trace)-round(10*fps)+1, round(fps))
    if len(hr) != len(starts) or len(proposals) != len(starts):
        raise ValueError('HR/proposal windows do not cover the fixed full plan')
    hr['window_start_s'] = starts/fps
    hr['window_end_s'] = (starts+round(10*fps))/fps
    hr['component_proposal_bpm'] = proposals.proposal_bpm.to_numpy(float)
    hr['hr_source'] = 'saved_measured_component_offline'
    return hr


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True, type=Path, help='One complete local CFR video')
    ap.add_argument('--out', required=True, type=Path, help='New result directory; existing paths are rejected')
    ap.add_argument('--trace-cache', type=Path, help='Exact frame_trace.csv with matching .json, or its directory')
    ap.add_argument('--variant', choices=['component', 'harmonic'], default='component')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    video, out = args.video.resolve(), args.out.resolve()
    if not video.is_file():
        raise ValueError(f'Video is not a file: {video}')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an existing result path: {out}')
    module = load_variant(args.variant)
    frozen = frozen_sources(args.variant)
    identity = frontend_identity(video)
    video_hash = sha(video)
    started = time.perf_counter()
    trace, fps, cache_info = None, None, None
    if args.trace_cache:
        trace, fps, cache_info = load_trace_cache(args.trace_cache, video, identity, video_hash)
    import cv2
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        probe = dict(fps=actual_fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     container_reported_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    if not np.isfinite(actual_fps) or actual_fps <= 7:
        raise ValueError('Fixed 42-210 bpm experiment requires video FPS > 7')
    if fps is not None and not np.isclose(actual_fps, fps, rtol=0, atol=1e-8):
        raise ValueError('Cached FPS differs from current video')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running',
        variant=args.variant, parameters=PARAMETERS, component_config=asdict(module.ComponentConfig()),
        pixel_config=asdict(PixelConfig()), anchor_config=asdict(AnchorConfig()),
        hr_config=asdict(DEFAULT_CONFIG), source_hashes=frozen,
        video=dict(path=str(video), bytes=identity['size'], sha256=video_hash),
        container_probe=probe, trace_cache=cache_info, reference_used=False,
        offline=True, hr_from_saved_waveform=True,
        status_of_algorithm='experimental; does not replace the V25 recommended entry point',
        limitations=['Fixed frame/FPS time axis assumes constant frame rate; VFR PTS are not modeled.',
          'Adaptively band-limited measured components do not validate full PPG morphology.',
          'Sampling and HR coverage are availability, not reference accuracy.',
          'Quality and motion scores are engineering evidence, not accuracy probabilities.',
          'Observation masks describe contributing samples, not the full filter receptive field.',
          'No reference file, target HR, per-video tuning, or HR interpolation is used.'])
    write_json(out/'manifest.json', manifest)
    if trace is None:
        trace, fps = extract(video, qa_dir=out/'qa', rgb_aggregation='trimmed_mean',
                             pixel_mode='tracking_screened')
        trace = anchored_reconstruction(trace, fps)
        validate_component_trace(trace, fps)
        if not np.isclose(fps, actual_fps, rtol=0, atol=1e-8):
            raise ValueError('Extractor FPS differs from video probe')
        trace.to_csv(out/'frame_trace.csv', index=False)
    else:
        shutil.copyfile(cache_info['path'], out/'frame_trace.csv')
    # Canonical persisted trace is the input in both fresh/cache workflows.
    # Small float serialization differences from historical unsaved ROI arrays
    # are measured in the regression receipt rather than silently hidden.
    trace = pd.read_csv(out/'frame_trace.csv')
    validate_component_trace(trace, fps)
    write_json(out/'frame_trace.json', dict(identity=identity, fps=fps, n_frames=len(trace),
               trace_sha256=sha(out/'frame_trace.csv'), video_sha256=video_hash))
    wave, proposals, candidates, roi_tables = infer_components(trace, fps, args.variant)
    for branch, name in [('baseline', 'baseline_roi_waveforms.csv'), ('tracked', 'roi_waveforms.csv')]:
        roi_tables[branch].to_csv(out/name, index=False)
    wave.to_csv(out/'waveform.csv', index=False)
    hr = read_saved_hr(out/'waveform.csv', trace, fps, proposals)
    hr.to_csv(out/'heart_rate.csv', index=False)
    proposals.to_csv(out/'component_proposals.csv', index=False)
    write_json(out/'all_roi_candidates.json', candidates)
    if frozen_sources(args.variant) != frozen:
        raise RuntimeError('Source changed during inference; result is not a frozen run')
    if frontend_identity(video) != identity or sha(video) != video_hash:
        raise RuntimeError('Input video changed during inference')
    summary = dict(variant=args.variant, fps=fps, frames=len(trace),
        waveform_coverage_pct=100*float(wave.covered.mean()),
        planned_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
        hr_coverage_pct=100*float(hr.accepted.mean()) if len(hr) else None,
        reference_used=False, offline=True, hr_from_saved_waveform=True,
        elapsed_s=time.perf_counter()-started,
        extraction_path='verified_trace_cache' if cache_info else 'fresh_v25_frontend',
        source_hashes=frozen)
    write_json(out/'summary.json', summary)
    manifest.update(status='complete', summary=summary,
        outputs={path.name: sha(path) for path in sorted(out.iterdir())
                 if path.is_file() and path.name != 'manifest.json'})
    write_json(out/'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
