#!/usr/bin/env python3
"""Analyze one local CFR video with V28 direct-motion guarded V26 components.

The fixed V25 waveform is the fallback. A candidate can replace it only when
the old frequency has direct motion evidence as well as the original V26 gates.
All HR tables are independently read from saved measured waveforms. Missing
samples remain NaN. No pulse reference or target HR is an inference input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

import analyze_components_v26 as components
from analyze_motion_v2 import source_hashes
from anchored_reconstruction import AnchorConfig, anchored_reconstruction
from evidence_hr import DEFAULT_CONFIG, estimate_evidence
from guarded_fusion import GuardConfig, guarded_fuse
from motion_frontend import extract
from motion_fusion import FusionConfig
from pixel_tracking import PixelConfig
from waveform_hr import estimate_fused_waveform

HERE = Path(__file__).resolve().parent
PARAMETERS = dict(components.PARAMETERS)
VARIANT = 'direct_guard'
sha = components.sha
write_json = components.write_json
frontend_identity = components.frontend_identity
load_trace_cache = components.load_trace_cache
validate_component_trace = components.validate_component_trace
ROIS = components.ROIS


def frozen_sources():
    names = ['analyze_video_v28.py', 'analyze_components_v26.py',
             'component_fusion_v26.py', 'component_harmonics_v26.py',
             'motion_harmonic_evidence_v26.py', 'conservative_component_router_v26.py',
             'direct_guard_router_v28.py']
    return {**source_hashes(), **{name: sha(HERE/name) for name in names}}


def probe_video(video):
    """Decode every frame: container frame-count metadata is not the clock."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= PARAMETERS['max_bpm']/30:
            raise ValueError('Fixed 42-210 bpm experiment requires video FPS > 7')
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count, dimensions = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError('Decoded video frame is not a three-channel image')
            current = (int(frame.shape[1]), int(frame.shape[0]))
            if dimensions is None:
                dimensions = current
            elif dimensions != current:
                raise ValueError('Changing decoded dimensions are unsupported')
            count += 1
        if count == 0:
            raise ValueError('Video has no decodable frames')
        return dict(fps=fps, width=dimensions[0], height=dimensions[1],
                    container_reported_width=reported_width,
                    container_reported_height=reported_height,
                    container_reported_frames=reported, decoded_frames=count,
                    decoded_duration_s=count/fps, count_method='complete_sequential_decode',
                    clock='uniform_frame_index_over_reported_fps_CFR_assumption')
    finally:
        cap.release()


def check_decoded_clock(trace, fps, probe):
    if not np.isclose(fps, probe['fps'], rtol=0, atol=1e-8):
        raise ValueError('Trace FPS differs from complete video decode')
    if len(trace) != probe['decoded_frames']:
        raise ValueError('Trace frame count differs from complete video decode')
    validate_component_trace(trace, fps)


def read_saved_hr(path, trace, fps, source):
    """Final readout accepts saved samples, never proposal/old/reference HR."""
    saved = pd.read_csv(path)
    if len(saved) != len(trace):
        raise ValueError('Saved waveform lost original frames')
    np.testing.assert_allclose(saved.time_s, trace.time_s, atol=1e-8, rtol=0)
    for field in ('covered', 'observed', 'interpolated'):
        vals = pd.to_numeric(saved[field], errors='coerce').to_numpy(float)
        if not np.isfinite(vals).all() or not np.isin(vals, [0., 1.]).all():
            raise ValueError(f'{field} must have explicit Boolean flags')
        saved[field] = vals.astype(bool)
    np.testing.assert_array_equal(np.isfinite(saved.base), saved.covered)
    if (saved.observed & ~saved.covered).any() or (saved.interpolated & ~saved.covered).any():
        raise ValueError('Saved sample provenance extends outside waveform support')
    kwargs = {key: PARAMETERS[key] for key in ('window_s', 'step_s', 'min_bpm', 'max_bpm')}
    hr = estimate_evidence(saved.base.to_numpy(float),
        pd.DataFrame({'rgb_valid': saved.observed}), saved.interpolated.to_numpy(bool),
        fps, motion_trace=trace, **kwargs)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    window, step = round(PARAMETERS['window_s']*fps), round(PARAMETERS['step_s']*fps)
    starts = np.arange(0, max(0, len(trace)-window+1), step, dtype=int)
    if len(hr) != len(starts):
        raise ValueError('HR windows differ from the fixed complete plan')
    hr['window_start_s'] = starts/fps
    hr['window_end_s'] = (starts+window)/fps
    hr['hr_source'] = source
    return hr


def generate_v25(trace, fps, out):
    """Reproduce the V25 legacy/legacy fusion and evidence readout on this trace."""
    channels, tables = components.prepare_roi_channels(trace, fps)
    for branch, table in tables.items():
        table.to_csv(out/f'{branch}_roi_waveforms.csv', index=False)
    config, guard = FusionConfig(motion_evidence_mode='legacy'), GuardConfig(routing_mode='legacy')
    window = round(PARAMETERS['window_s']*fps)
    if len(trace) < window:
        wave = pd.DataFrame(dict(time_s=trace.time_s, base=np.nan, covered=False,
                                 observed=False, interpolated=False))
        pd.DataFrame(columns=['window_index', 'time_s', 'accepted', 'ridge_bpm']).to_csv(
            out/'fusion_proposals.csv', index=False)
        pd.DataFrame(columns=['window_index', 'roi', 'waveform_weight']).to_csv(
            out/'fusion_diagnostics.csv', index=False)
        pd.DataFrame(columns=['window_index', 'time_s', 'selected_branch']).to_csv(
            out/'branch_routing.csv', index=False)
    else:
        signals = {branch: {f'{roi}_{method}': channels[f'{branch}/{roi}/{method}']
                    for roi in ROIS for method in ('pos', 'chrom')}
                   for branch in ('baseline', 'tracked')}
        kwargs = {key: PARAMETERS[key] for key in ('window_s', 'step_s', 'min_bpm', 'max_bpm')}
        proposals, values, diagnostics, routing, _ = guarded_fuse(
            signals['baseline'], signals['tracked'], trace, fps,
            config=config, guard=guard, **kwargs)
        proposals['waveform_generated'] = proposals.accepted & (proposals.waveform_roi_count >= config.min_rois)
        # Use the unchanged contributor-provenance implementation. The returned
        # in-memory HR is deliberately discarded; saved samples are read below.
        _, wave = estimate_fused_waveform(values, trace, tables['tracked'], proposals,
                                         diagnostics, fps, hr_mode='evidence', **kwargs)
        np.testing.assert_array_equal(np.isfinite(values), wave.covered)
        wave.insert(1, 'base', values)
        proposals.to_csv(out/'fusion_proposals.csv', index=False)
        diagnostics.to_csv(out/'fusion_diagnostics.csv', index=False)
        routing.to_csv(out/'branch_routing.csv', index=False)
    wave.to_csv(out/'waveform.csv', index=False)
    hr = read_saved_hr(out/'waveform.csv', trace, fps, 'saved_v25_fallback_waveform_offline')
    hr.to_csv(out/'heart_rate.csv', index=False)
    return pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv')


def run_pipeline(trace, fps, out):
    """Write one fixed variant and both measurable intermediate waveforms."""
    from direct_guard_router_v28 import route_components
    validate_component_trace(trace, fps)
    old_dir, candidate_dir = out/'v25_fallback', out/'v26_harmonic_candidate'
    old_dir.mkdir(exist_ok=False)
    candidate_dir.mkdir(exist_ok=False)
    old_wave, old_hr = generate_v25(trace, fps, old_dir)
    candidate, proposals, records, roi_tables = components.infer_components(trace, fps, 'harmonic')
    candidate.to_csv(candidate_dir/'waveform.csv', index=False)
    proposals.to_csv(candidate_dir/'component_proposals.csv', index=False)
    write_json(candidate_dir/'all_roi_candidates.json', records)
    for branch, table in roi_tables.items():
        table.to_csv(candidate_dir/f'{branch}_roi_waveforms.csv', index=False)
    candidate_hr = read_saved_hr(candidate_dir/'waveform.csv', trace, fps,
                                 'saved_v26_harmonic_candidate_waveform_offline')
    candidate_hr.to_csv(candidate_dir/'heart_rate.csv', index=False)
    wave, decisions = route_components(old_wave, old_hr,
        pd.read_csv(candidate_dir/'waveform.csv'),
        pd.read_csv(candidate_dir/'component_proposals.csv'), trace, fps)
    if decisions.empty and not len(decisions.columns):
        # No complete 10 s windows is an empty result, with a readable schema.
        decisions = pd.DataFrame(columns=['window_index', 'time_s', 'window_start_s',
            'window_end_s', 'old_bpm', 'candidate_bpm', 'old_direct_motion_risk',
            'route_eligible', 'reason'])
    np.testing.assert_array_equal(np.isfinite(wave.base), np.isfinite(old_wave.base))
    np.testing.assert_array_equal(wave.covered, old_wave.covered)
    wave.to_csv(out/'waveform.csv', index=False)
    decisions.to_csv(out/'routing_decisions.csv', index=False)
    final_hr = read_saved_hr(out/'waveform.csv', trace, fps, 'saved_v28_direct_guard_waveform_offline')
    final_hr.to_csv(out/'heart_rate.csv', index=False)
    return pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv'), decisions


def plot_outputs(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wave, hr = pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv')
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout='constrained')
    axes[0].plot(wave.time_s, wave.base, lw=.75, color='#24736d')
    axes[0].set(ylabel='Measured signal (a.u.)',
                title='V28 direct-motion guard: measured waveform and saved-wave HR')
    axes[0].text(.01, .97, 'May include adaptive narrowband components; PPG morphology is not validated',
                 transform=axes[0].transAxes, va='top', fontsize=9,
                 bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    axes[1].plot(hr.time_s, hr.ridge_bpm.where(hr.accepted), lw=1.3, color='#285aa7')
    axes[1].set(xlabel='Video elapsed time (s); HR at 10 s window centers', ylabel='HR (bpm)',
                ylim=(PARAMETERS['min_bpm']-3, PARAMETERS['max_bpm']+3))
    axes[1].text(.01, .03, 'Missing output stays blank; no reference is supplied',
                 transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    if len(wave):
        axes[1].set_xlim(0, max(1., float(wave.time_s.iloc[-1])))
    fig.savefig(out/'ppg_and_hr.png', dpi=160)
    plt.close(fig)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', required=True, type=Path, help='One complete local CFR video')
    ap.add_argument('--out', required=True, type=Path, help='New output directory; existing paths are rejected')
    ap.add_argument('--trace-cache', type=Path, help='Exact anchored frame_trace.csv and matching .json, or directory')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    video, out = args.video.resolve(), args.out.resolve()
    if not video.is_file():
        raise ValueError(f'Video is not a file: {video}')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output path: {out}')
    import cv2
    cv2.setNumThreads(2)
    fixed = frozen_sources()
    identity, video_hash = frontend_identity(video), sha(video)
    started = time.perf_counter()
    trace, fps, cache_info = None, None, None
    if args.trace_cache:
        trace, fps, cache_info = load_trace_cache(args.trace_cache, video, identity, video_hash)
    probe = probe_video(video)
    if trace is not None:
        check_decoded_clock(trace, fps, probe)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running',
        variant=VARIANT, parameters=PARAMETERS, source_hashes=fixed,
        video=dict(path=str(video), bytes=identity['size'], sha256=video_hash),
        video_probe=probe, trace_cache=cache_info, reference_used=False, offline=True,
        hr_from_saved_waveform=True, v25_finite_mask_preserved=True,
        fusion_config=asdict(FusionConfig(motion_evidence_mode='legacy')),
        guard_config=asdict(GuardConfig(routing_mode='legacy')),
        component_config=asdict(components.load_variant('harmonic').ComponentConfig()),
        pixel_config=asdict(PixelConfig()), anchor_config=asdict(AnchorConfig()),
        hr_config=asdict(DEFAULT_CONFIG),
        limitations=['Uniform frame/FPS axis assumes CFR; variable frame timestamps are not modeled.',
          'Mixed measured waveform can contain adaptively narrowband components; PPG morphology is unvalidated.',
          'Motion overlap is engineering evidence, not proof of interference or an accuracy probability.',
          'Waveform and HR coverage measure availability; reference accuracy is not measured by this CLI.',
          'No reference file, target HR, per-video thresholds, or HR interpolation is used.',
          'Hann overlap can affect adjacent windows whose own route_eligible flag is false.',
          'The finite sample mask is preserved; observed/interpolated describe all/any actual contributors.'])
    write_json(out/'manifest.json', manifest)
    if trace is None:
        trace, fps = extract(video, qa_dir=out/'qa', rgb_aggregation='trimmed_mean',
                             pixel_mode='tracking_screened')
        trace = anchored_reconstruction(trace, fps)
        check_decoded_clock(trace, fps, probe)
        trace.to_csv(out/'frame_trace.csv', index=False)
    else:
        shutil.copyfile(cache_info['path'], out/'frame_trace.csv')
    trace = pd.read_csv(out/'frame_trace.csv')
    check_decoded_clock(trace, fps, probe)
    write_json(out/'frame_trace.json', dict(identity=identity, fps=fps, n_frames=len(trace),
        trace_sha256=sha(out/'frame_trace.csv'), video_sha256=video_hash,
        video_probe=probe, source_hashes=fixed))
    wave, hr, decisions = run_pipeline(trace, fps, out)
    plot_outputs(out)
    if frozen_sources() != fixed:
        raise RuntimeError('Inference source changed during run')
    if frontend_identity(video) != identity or sha(video) != video_hash:
        raise RuntimeError('Input video changed during run')
    summary = dict(variant=VARIANT, fps=fps, frames=len(trace),
        waveform_coverage_pct=100*float(wave.covered.mean()),
        planned_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
        hr_coverage_pct=100*float(hr.accepted.mean()) if len(hr) else None,
        route_eligible_windows=int(decisions.route_eligible.sum()) if len(decisions) else 0,
        reference_used=False, offline=True, hr_from_saved_waveform=True,
        v25_finite_mask_preserved=True, elapsed_s=time.perf_counter()-started,
        extraction_path='verified_trace_cache_full_decode' if cache_info else 'fresh_v25_frontend',
        source_hashes=fixed)
    write_json(out/'summary.json', summary)
    manifest.update(status='complete', summary=summary,
        outputs={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*'))
                 if path.is_file() and path != out/'manifest.json'})
    write_json(out/'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
