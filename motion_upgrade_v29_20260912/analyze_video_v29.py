#!/usr/bin/env python3
"""Two fixed offline 6 s profiles of the complete V28 measurement pipeline.

Both preserve the original frontend, real component reconstruction and direct
motion guard. Only short6_relaxed changes fusion ambiguity_ratio (.80 -> .90).
HR sampling/concentration gates, two physical ROI support, and gap rules remain
unchanged. A short window's end is support timing, not a real-time latency claim.
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
import analyze_video_v28 as v28
from anchored_reconstruction import AnchorConfig, anchored_reconstruction
from component_harmonics_v26 import ComponentConfig, fuse_components
from direct_guard_router_v29 import route_components
from evidence_hr import DEFAULT_CONFIG
from guarded_fusion import GuardConfig, guarded_fuse
from motion_frontend import extract
from motion_fusion import FusionConfig
from pixel_tracking import PixelConfig
from quality_readout_v29 import SamplingConfig, estimate_evidence
from waveform_provenance_v29 import fused_provenance

HERE = Path(__file__).resolve().parent
ROIS = components.ROIS
sha, write_json = components.sha, components.write_json
frontend_identity, load_trace_cache = components.frontend_identity, components.load_trace_cache
validate_component_trace = components.validate_component_trace


def _profile(ambiguity):
    return dict(window_s=6.0, step_s=1.0, min_bpm=42.0, max_bpm=210.0,
        fusion_config=asdict(FusionConfig(motion_evidence_mode='legacy', ambiguity_ratio=ambiguity)),
        guard_config=asdict(GuardConfig(routing_mode='legacy')),
        component_config=asdict(ComponentConfig()), hr_sampling=asdict(SamplingConfig()),
        evidence_config=asdict(DEFAULT_CONFIG), max_gap_s=.1,
        pixel_config=asdict(PixelConfig()), anchor_config=asdict(AnchorConfig()),
        original_strict_readout_window_s=10.0, min_physical_rois=2,
        direct_router=dict(min_old_motion_risk=.50, min_old_direct_motion_risk=.50,
                           min_motion_risk_decrease=.30), offline=True)


# Normalize nested tuples for exact JSON protocol round trips and hash binding.
PROFILE_CONFIGS = json.loads(json.dumps({'short6': _profile(.80),
                                        'short6_relaxed': _profile(.90)}))


def config_for(profile):
    if profile not in PROFILE_CONFIGS:
        raise ValueError('Choose the fixed short6 or short6_relaxed profile')
    return PROFILE_CONFIGS[profile]


def frozen_sources():
    names = ('analyze_video_v29.py', 'quality_readout_v29.py',
             'waveform_provenance_v29.py', 'direct_guard_router_v29.py')
    return {**v28.frozen_sources(), **{name: sha(HERE/name) for name in names}}


def _window_kwargs(config):
    return {key: config[key] for key in ('window_s', 'step_s', 'min_bpm', 'max_bpm')}


def read_saved_hr(path, trace, fps, profile, source, *, window_s=None):
    config = config_for(profile)
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
    kwargs = _window_kwargs(config)
    if window_s is not None:
        kwargs['window_s'] = float(window_s)
    hr = estimate_evidence(saved.base.to_numpy(float),
        pd.DataFrame({'rgb_valid': saved.observed}), saved.interpolated.to_numpy(bool), fps,
        motion_trace=trace, sampling=SamplingConfig(**config['hr_sampling']), **kwargs)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    window, step = round(kwargs['window_s']*fps), round(kwargs['step_s']*fps)
    starts = np.arange(0, max(0, len(trace)-window+1), step, dtype=int)
    if len(hr) != len(starts):
        raise ValueError('HR rows do not match the complete window plan')
    hr['window_index'] = np.arange(len(starts))
    hr['window_start_s'], hr['window_end_s'] = starts/fps, (starts+window)/fps
    hr['profile'], hr['analysis_window_s'], hr['hr_source'] = profile, kwargs['window_s'], source
    # Both profiles use exactly the original HR gates. This is not a statement
    # about the relaxed fusion's quality, physiological correctness or accuracy.
    assert config['hr_sampling'] == asdict(SamplingConfig())
    hr['hr_passes_original_gates'] = hr.accepted.to_numpy(bool)
    hr['strict_accepted'] = hr.accepted.to_numpy(bool)
    tier = ('original_hr_gates_with_relaxed_fusion' if profile == 'short6_relaxed'
            else 'original_hr_gates_with_short6_fusion')
    hr['quality_tier'] = np.where(hr.accepted, tier, 'rejected')
    hr.attrs.update(offline=True, hr_gate_label_is_accuracy_guarantee=False,
                    window_end_is_support_time_not_runtime_latency=True)
    return hr


def infer_components(trace, fps, profile, prepared=None):
    config = config_for(profile)
    channels, roi_tables = prepared if prepared is not None else components.prepare_roi_channels(trace, fps)
    if len(trace) < round(config['window_s']*fps):
        wave = pd.DataFrame(dict(time_s=trace.time_s, base=np.nan, covered=False,
                                 observed=False, interpolated=False))
        proposals, records = pd.DataFrame(columns=components.PROPOSAL_COLUMNS), []
    else:
        wave, proposals, records = fuse_components(channels, trace, roi_tables['baseline'], fps,
            config=ComponentConfig(**config['component_config']), **_window_kwargs(config))
    window, step = round(config['window_s']*fps), round(config['step_s']*fps)
    starts = np.arange(0, max(0, len(trace)-window+1), step, dtype=int)
    proposals['window_start_s'], proposals['window_end_s'] = starts/fps, (starts+window)/fps
    return wave, proposals, records, roi_tables


def generate_fallback(trace, fps, out, profile, prepared):
    """Original V25 fusion logic with the declared 6 s window/ambiguity setting."""
    profile_config = config_for(profile)
    channels, tables = prepared
    for branch, table in tables.items():
        table.to_csv(out/f'{branch}_roi_waveforms.csv', index=False)
    config = FusionConfig(**profile_config['fusion_config'])
    guard = GuardConfig(**profile_config['guard_config'])
    if len(trace) < round(profile_config['window_s']*fps):
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
        kwargs = _window_kwargs(profile_config)
        proposals, values, diagnostics, routing, _ = guarded_fuse(
            signals['baseline'], signals['tracked'], trace, fps,
            config=config, guard=guard, **kwargs)
        proposals['waveform_generated'] = proposals.accepted & (proposals.waveform_roi_count >= config.min_rois)
        wave = fused_provenance(values, trace, tables['tracked'], proposals, diagnostics, fps, **kwargs)
        np.testing.assert_array_equal(np.isfinite(values), wave.covered)
        wave.insert(1, 'base', values)
        proposals.to_csv(out/'fusion_proposals.csv', index=False)
        diagnostics.to_csv(out/'fusion_diagnostics.csv', index=False)
        routing.to_csv(out/'branch_routing.csv', index=False)
    wave.to_csv(out/'waveform.csv', index=False)
    read_saved_hr(out/'waveform.csv', trace, fps, profile,
                  'saved_short_window_v25_logic_fallback_offline').to_csv(out/'heart_rate.csv', index=False)
    return pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv')


def run_pipeline(trace, fps, out, profile):
    """Write one profile's 6 s and 10 s readouts from the same saved waveform."""
    config = config_for(profile)
    validate_component_trace(trace, fps)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old_dir, candidate_dir = out/'short_window_fallback', out/'v26_harmonic_candidate'
    old_dir.mkdir(exist_ok=False)
    candidate_dir.mkdir(exist_ok=False)
    prepared = components.prepare_roi_channels(trace, fps)
    old_wave, old_hr = generate_fallback(trace, fps, old_dir, profile, prepared)
    candidate, proposals, records, roi_tables = infer_components(trace, fps, profile, prepared)
    candidate.to_csv(candidate_dir/'waveform.csv', index=False)
    proposals.to_csv(candidate_dir/'component_proposals.csv', index=False)
    write_json(candidate_dir/'all_roi_candidates.json', records)
    for branch, table in roi_tables.items():
        table.to_csv(candidate_dir/f'{branch}_roi_waveforms.csv', index=False)
    read_saved_hr(candidate_dir/'waveform.csv', trace, fps, profile,
                  'saved_short_window_v26_harmonic_candidate_offline').to_csv(candidate_dir/'heart_rate.csv', index=False)
    wave, decisions = route_components(old_wave, old_hr,
        pd.read_csv(candidate_dir/'waveform.csv'), pd.read_csv(candidate_dir/'component_proposals.csv'),
        trace, fps, window_s=config['window_s'], step_s=config['step_s'])
    if decisions.empty and not len(decisions.columns):
        decisions = pd.DataFrame(columns=['window_index', 'time_s', 'window_start_s',
            'window_end_s', 'old_bpm', 'candidate_bpm', 'old_motion_risk_direct', 'route_eligible', 'reason'])
    np.testing.assert_array_equal(np.isfinite(wave.base), np.isfinite(old_wave.base))
    np.testing.assert_array_equal(wave.covered, old_wave.covered)
    wave.to_csv(out/'waveform.csv', index=False)
    decisions.to_csv(out/'routing_decisions.csv', index=False)
    final_hr = read_saved_hr(out/'waveform.csv', trace, fps, profile,
                            'saved_v29_short_window_direct_guard_waveform_offline')
    final_hr.to_csv(out/'heart_rate.csv', index=False)
    # Independent original 10 s strict reader, not interpolation of 6 s HR.
    strict10 = v28.read_saved_hr(out/'waveform.csv', trace, fps,
                                'saved_v29_waveform_original_10s_strict_readout_offline')
    strict10['profile'] = profile
    strict10['analysis_window_s'] = 10.0
    strict10['strict_accepted'] = strict10.accepted
    strict10['hr_passes_original_gates'] = strict10.accepted
    strict10['quality_tier'] = np.where(strict10.accepted,
        'original_10s_hr_gates_on_'+profile+'_fusion', 'rejected')
    strict10.to_csv(out/'heart_rate_10s.csv', index=False)
    write_json(out/'pipeline_config.json', config)
    return pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv'), decisions


def plot_outputs(out, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wave, hr = pd.read_csv(out/'waveform.csv'), pd.read_csv(out/'heart_rate.csv')
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout='constrained')
    axes[0].plot(wave.time_s, wave.base, lw=.75, color='#24736d')
    axes[0].set(ylabel='Measured signal (a.u.)', title=f'V29 {profile}: saved waveform and offline HR')
    axes[0].text(.01, .97, 'May include measured narrowband components; PPG morphology is unvalidated',
                 transform=axes[0].transAxes, va='top', fontsize=9,
                 bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    axes[1].plot(hr.time_s, hr.ridge_bpm.where(hr.accepted), lw=1.3, color='#285aa7')
    axes[1].set(xlabel='Video time (s); HR at 6 s window centers', ylabel='HR (bpm)', ylim=(39,213))
    axes[1].text(.01,.03,'Missing outputs stay blank. Gate acceptance is not an accuracy guarantee.',
                 transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    if len(wave):
        axes[1].set_xlim(0, max(1., float(wave.time_s.iloc[-1])))
    fig.savefig(out/'ppg_and_hr.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trace-cache', type=Path)
    ap.add_argument('--profile', choices=tuple(PROFILE_CONFIGS), required=True)
    args = ap.parse_args(argv)
    video, out = args.video.resolve(), args.out.resolve()
    if not video.is_file():
        raise ValueError('Video must be a complete local file')
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output')
    import cv2
    cv2.setNumThreads(2)
    started, fixed = time.perf_counter(), frozen_sources()
    identity, video_hash = frontend_identity(video), sha(video)
    trace, fps, cache = None, None, None
    if args.trace_cache:
        trace, fps, cache = load_trace_cache(args.trace_cache, video, identity, video_hash)
    probe = v28.probe_video(video)
    if trace is not None:
        v28.check_decoded_clock(trace, fps, probe)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running',
        profile=args.profile, config=config_for(args.profile), source_hashes=fixed,
        video=dict(path=str(video), sha256=video_hash, bytes=identity['size']),
        reference_used=False, offline=True, video_probe=probe, trace_cache=cache,
        hr_from_saved_waveform=True,
        limitations=['Gate acceptance and finite coverage are not accuracy probabilities.',
            'Full-record zero-phase filters and offline DP remain; 6 s does not state runtime latency.',
            'Measured narrowband components may contribute; waveform morphology is not validated.',
            'CFR frame-index/fps clock, exact cache identity and video hash are required.',
            'Original gap limit and contributor provenance are preserved; no HR filling.'])
    write_json(out/'manifest.json', manifest)
    if trace is None:
        trace, fps = extract(video, qa_dir=out/'qa', rgb_aggregation='trimmed_mean', pixel_mode='tracking_screened')
        trace = anchored_reconstruction(trace, fps)
        v28.check_decoded_clock(trace, fps, probe)
        trace.to_csv(out/'frame_trace.csv', index=False)
    else:
        shutil.copyfile(cache['path'], out/'frame_trace.csv')
    trace = pd.read_csv(out/'frame_trace.csv')
    v28.check_decoded_clock(trace, fps, probe)
    write_json(out/'frame_trace.json', dict(identity=identity, fps=fps, n_frames=len(trace),
        trace_sha256=sha(out/'frame_trace.csv'), video_sha256=video_hash, video_probe=probe, source_hashes=fixed))
    wave, hr, decisions = run_pipeline(trace, fps, out, args.profile)
    plot_outputs(out, args.profile)
    if frozen_sources() != fixed or frontend_identity(video) != identity or sha(video) != video_hash:
        raise RuntimeError('Inference source or input video changed during run')
    summary = dict(status='complete', profile=args.profile, fps=fps, frames=len(trace),
        waveform_coverage_pct=100*float(wave.covered.mean()), planned_windows=len(hr),
        accepted_windows=int(hr.accepted.sum()),
        hr_coverage_pct=100*float(hr.accepted.mean()) if len(hr) else None,
        route_eligible_windows=int(decisions.route_eligible.sum()),
        reference_used=False, offline=True, hr_from_saved_waveform=True,
        elapsed_s=time.perf_counter()-started, source_hashes=fixed)
    write_json(out/'summary.json', summary)
    manifest.update(status='complete', summary=summary,
        outputs={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*'))
                 if path.is_file() and path != out/'manifest.json'})
    write_json(out/'manifest.json', manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
