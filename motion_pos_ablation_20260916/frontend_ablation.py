"""Controlled three-physical-ROI POS ablation on a frozen V28 RGB cache.

This is not the six-subregion POS reproduction. No video or physiology is read.
The six arms differ only in saved RGB representation and the declared reducer;
all use the same intersection mask, fixed POS implementation and HR estimator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

REGIONS = ('forehead', 'left_cheek', 'right_cheek')
SOURCES = ('baseline', 'unanchored', 'anchored')
REDUCERS = ('region_median', 'rgb_mean')
ARMS = tuple(f'front_{source}_{reducer}' for source in SOURCES for reducer in REDUCERS)


def protocol():
    return dict(version=1, arms=list(ARMS), regions=list(REGIONS),
        rgb_fields={'baseline': 'baseline_{region}_{channel}',
                    'unanchored': 'unanchored_{region}_{channel}',
                    'anchored': '{region}_{channel}'},
        input='Existing V28 trace; preserve cached tracking fallbacks and resets.',
        availability='One per-frame intersection across all three sources and all three physical ROIs; record native masks separately.',
        valid_rgb='Cached physical ROI valid AND all RGB finite and positive.',
        extraction_window_s=5., extraction_hop_s=5./60.,
        extraction_rounding='round(seconds*actual_fps), hop at least one frame',
        max_internal_gap_s=.1, gap_rounding='floor(seconds*fps+1e-9)',
        min_observed_fraction=.9, require_all_three_regions=True,
        projection='Frozen motion_dis_hybrid_v2_20260915.dis_core.preprocess and project(method=pos); no motion input or region ranking.',
        region_median='Preprocess three regions jointly per RGB channel as in frozen POS, project each region, samplewise median of all three standardized region waveforms, then standardize.',
        rgb_mean='Equal-weight mean of the three observed physical ROI RGB vectors, same preprocessing and POS projection as one mixed signal, then standardize.',
        overlap_add='Uniform sums divided by actual generated-window counts.',
        min_bpm=42., max_bpm=210., hr_window_s=10., hr_step_s=1.,
        hr_estimator='Frozen V28 estimate_evidence, motion_trace=None, unchanged default config.',
        hr_gates=dict(all_samples_finite=True, min_observed=.9,
            max_interpolated=.1, min_std=1e-8, min_peak_concentration=.12),
        selection='No new HR threshold, no reference-dependent routing or tuning.',
        comparisons=[['front_unanchored_region_median', 'front_baseline_region_median'],
                     ['front_anchored_region_median', 'front_baseline_region_median'],
                     ['front_anchored_region_median', 'front_unanchored_region_median']] +
                    [[f'front_{source}_rgb_mean', f'front_{source}_region_median'] for source in SOURCES],
        interpretation='Controlled engineering ablation; native deployment coverage and six-subregion reproduction are separate questions.',
        attribution_limits={
            'baseline_vs_unanchored': 'Whole RGB input pathway: pixel sampling, tracking/screening and log integration change together; cannot isolate pure optical-flow causation.',
            'unanchored_vs_anchored': 'Isolates the existing fixed slow anchor on the tracked reconstruction.',
            'region_median_vs_rgb_mean': 'Declared aggregation order; preprocessing normalization operates on three regions versus one mixture.',
            'six_region_vs_three_region': 'Whole pipeline comparison; six-region POS also tracks/screens patches but does not integrate log changes.'},
        reference_used=False, raw_video_read=False, video_resampling=False,
        default_replacement=False, offline=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def save_csv(path, frame):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)


def load_runtime(project):
    project = Path(project).resolve()
    folders = dict(dis_core=project/'motion_dis_hybrid_v2_20260915',
                   evidence_hr=project/'motion_upgrade_v28_20260912')
    modules = {}
    for name, folder in folders.items():
        sys.path.insert(0, str(folder))
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != folder.resolve():
            raise ImportError(f'Wrong frozen module already imported: {name}')
        modules[name] = module
    dependencies = [Path(__file__).resolve(), folders['dis_core']/'dis_core.py']
    dependencies += [folders['evidence_hr']/name for name in
                     ('evidence_hr.py', 'legacy_motion.py', 'analyze_rppg.py')]
    return SimpleNamespace(core=modules['dis_core'],
        estimate=modules['evidence_hr'].estimate_evidence,
        source_hashes={str(path): sha(path) for path in dependencies})


def explicit_mask(values, label):
    values = pd.Series(values)
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f'Explicit Boolean mask required: {label}')
    return values.to_numpy(bool)


def read_sources(trace, fps):
    n = len(trace)
    if n < 1 or not np.isfinite(fps) or fps <= 7:
        raise ValueError('Nonempty cache and FPS above the fixed band Nyquist required')
    if not np.array_equal(trace.frame.to_numpy(), np.arange(n)):
        raise ValueError('Cache frame indices must be complete and in original order')
    np.testing.assert_allclose(trace.time_s, np.arange(n)/fps, rtol=0, atol=1e-7)
    masks = pd.DataFrame({'frame': trace.frame, 'time_s': trace.time_s})
    values, observed = [], []
    for source in SOURCES:
        prefix = '' if source == 'anchored' else source+'_'
        rgb, valid = [], []
        for region in REGIONS:
            channels = trace[[f'{prefix}{region}_{c}' for c in 'rgb']].to_numpy(float)
            if np.isinf(channels).any():
                raise ValueError(f'Infinite cached RGB: {source}/{region}')
            ob = explicit_mask(trace[region+'_valid'], region+'_valid') & \
                np.isfinite(channels).all(axis=1) & (channels > 0).all(axis=1)
            masks[f'{source}_{region}_observed'] = ob
            rgb.append(channels)
            valid.append(ob)
        values.append(rgb)
        observed.append(valid)
    values, observed = np.asarray(values), np.asarray(observed)
    common = observed.all(axis=(0, 1))
    masks['common_observed'] = common
    native = {source: dict(all_three_pct=100*float(observed[k].all(axis=0).mean()),
                          any_region_pct=100*float(observed[k].any(axis=0).mean()),
                          regions_pct={region: 100*float(observed[k, j].mean())
                                       for j, region in enumerate(REGIONS)})
              for k, source in enumerate(SOURCES)}
    native['common_all_sources_all_regions_pct'] = 100*float(common.mean())
    if 'rgb_valid' in trace:
        native['original_trace_rgb_valid_pct'] = 100*float(explicit_mask(trace.rgb_valid, 'rgb_valid').mean())
    values[:, :, ~common, :] = np.nan
    return values, common, masks, native


def _standardize(signal):
    scale = np.std(signal, ddof=1)
    if not np.isfinite(signal).all() or not np.isfinite(scale) or scale < 1e-12:
        return None
    return (signal-np.mean(signal))/scale


def extract_products(trace, fps, runtime):
    """Pure cache-to-waveform stage; also used by synthetic tests before freeze."""
    values, observed, masks, native = read_sources(trace, fps)
    n = len(trace)
    core = runtime.core
    width, hop = round(5*fps), max(1, round(5./60.*fps))
    gap = int(np.floor(.1*fps+1e-9))
    filled = np.zeros(n, bool)
    for source in range(len(SOURCES)):
        for region in range(len(REGIONS)):
            values[source, region], current = core.fill_bounded(values[source, region], gap)
            if source == 0 and region == 0:
                filled = current
            elif not np.array_equal(current, filled):
                raise AssertionError('Common intersection masks must fill identically')
    masks['common_interpolated'] = filled
    records, products, source_products = {}, {}, {}
    for source_index, source in enumerate(SOURCES):
        state = {reducer: dict(total=np.zeros(n), count=np.zeros(n, int),
            source_total=np.zeros((3 if reducer == 'region_median' else 1, n)),
            source_count=np.zeros((3 if reducer == 'region_median' else 1, n), int),
            records=[]) for reducer in REDUCERS}
        for start in range(0, n-width+1, hop):
            stop = start+width
            segment = values[source_index, :, start:stop]
            observed_fraction = float(observed[start:stop].mean())
            shared_status = ('missing_or_long_gap' if not np.isfinite(segment).all() else
                             'insufficient_observed_rgb' if observed_fraction < .9 else None)
            for reducer in REDUCERS:
                bucket = state[reducer]
                status, projected, fused = shared_status, None, None
                if status is None:
                    try:
                        input_rgb = segment if reducer == 'region_median' else segment.mean(axis=0, keepdims=True)
                        color = np.stack([core.preprocess(input_rgb[:, :, j], fps) for j in range(3)], axis=2)
                        projected = core.project(color, np.zeros(color.shape[:2]+(2,)), 'pos')
                        if not np.isfinite(projected).all():
                            status = 'degenerate_projection'
                        else:
                            fused = _standardize(np.median(projected, axis=0) if reducer == 'region_median' else projected[0])
                            status = 'generated' if fused is not None else 'degenerate_fusion'
                    except ValueError:
                        status = 'normalization_failure'
                if status == 'generated':
                    bucket['total'][start:stop] += fused
                    bucket['count'][start:stop] += 1
                    bucket['source_total'][:, start:stop] += projected
                    bucket['source_count'][:, start:stop] += 1
                bucket['records'].append(dict(start_frame=start, stop_frame=stop,
                    start_s=start/fps, stop_s=stop/fps, status=status,
                    observed_fraction=observed_fraction,
                    interpolated_fraction=float(filled[start:stop].mean()),
                    source=source, reducer=reducer, regions=';'.join(REGIONS),
                    input_signal_count=3 if reducer == 'region_median' else 1))
        for reducer, bucket in state.items():
            arm = f'front_{source}_{reducer}'
            cover = bucket['count'] > 0
            wave = np.full(n, np.nan)
            np.divide(bucket['total'], bucket['count'], out=wave, where=cover)
            products[arm] = pd.DataFrame(dict(frame=trace.frame, time_s=trace.time_s,
                base=wave, covered=cover, observed=cover & observed,
                interpolated=cover & filled, overlap_count=bucket['count']))
            sources = pd.DataFrame(dict(frame=trace.frame, time_s=trace.time_s))
            names = REGIONS if reducer == 'region_median' else ('mixed_rgb',)
            for j, name in enumerate(names):
                c = bucket['source_count'][j] > 0
                y = np.full(n, np.nan)
                np.divide(bucket['source_total'][j], bucket['source_count'][j], out=y, where=c)
                sources[name+'_base'] = y
                sources[name+'_covered'] = c
                sources[name+'_observed'] = c & observed
                sources[name+'_interpolated'] = c & filled
                sources[name+'_overlap_count'] = bucket['source_count'][j]
            source_products[arm] = sources
            records[arm] = pd.DataFrame(bucket['records'], columns=['start_frame', 'stop_frame',
                'start_s', 'stop_s', 'status', 'observed_fraction', 'interpolated_fraction',
                'source', 'reducer', 'regions', 'input_signal_count'])
    return products, source_products, records, masks, native


def estimate_waveform(wave, fps, runtime):
    hr = runtime.estimate(wave.base.to_numpy(float), pd.DataFrame({'rgb_valid': wave.observed}),
        wave.interpolated.to_numpy(bool), fps, 10, 1, 42, 210, motion_trace=None)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['ridge_bpm', 'spectral_peak_bpm']] = np.nan
    width, hop = round(10*fps), round(fps)
    starts = np.arange(0, len(wave)-width+1, hop)
    if len(hr) != len(starts):
        raise AssertionError('HR output and planned windows disagree')
    hr['window_start_s'], hr['window_end_s'] = starts/fps, (starts+width)/fps
    hr['hr_source'] = 'saved_continuous_POS_waveform_evidence_no_motion'
    hr['window_id'] = [f'{i:04d}' for i in range(len(hr))]
    return hr


def load_cache(trace_path):
    path = Path(trace_path).resolve()
    if path.is_dir():
        path /= 'frame_trace.csv'
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    inputs = {str(p): sha(p) for p in (path, metadata_path)}
    if metadata.get('trace_sha256') != inputs[str(path)]:
        raise ValueError('Cached trace hash mismatch')
    trace = pd.read_csv(path)
    if len(trace) != metadata['n_frames']:
        raise ValueError('Cached frame count mismatch')
    if metadata.get('identity', {}).get('max_seconds') is not None:
        raise ValueError('Complete source video cache required')
    if set(trace.pixel_rgb_kind.dropna()) != {'anchored_paired_relative_colour'}:
        raise ValueError('Expected frozen anchored V28 cache including prior RGB representations')
    if not np.allclose(trace.anchor_hz, .15) or not np.all(trace.anchor_order == 4):
        raise ValueError('Unexpected cached anchor configuration')
    return trace, float(metadata['fps']), metadata, inputs


def _verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Frozen file changed: '+path)


def make_stage_freeze(project, trace_paths, global_protocol_path, out_path):
    """Bind completed implementation to the already frozen experiment design."""
    runtime = load_runtime(project)
    global_path = Path(global_protocol_path).resolve()
    global_protocol = json.loads(global_path.read_text(encoding='utf-8'))
    if global_protocol.get('reference_used_by_inference') is not False or \
            not set(ARMS).issubset(global_protocol.get('arms', {})):
        raise ValueError('Global experiment must predeclare these reference-free six arms')
    inputs = {}
    for path in trace_paths:
        _, _, _, current = load_cache(path)
        inputs.update(current)
    value = dict(status='source_and_input_frozen_before_frontend_inference',
        created_utc=datetime.now(timezone.utc).isoformat(), reference_used=False,
        frontend_protocol=protocol(), source_hashes=runtime.source_hashes,
        input_hashes=inputs, global_protocol_path=str(global_path),
        global_protocol_sha256=sha(global_path))
    save_json(out_path, value)
    return value


def run_case(trace_path, out_dir, *, case, project, freeze_path):
    """Run six declared arms after a global freeze; refuse existing output dirs.

    Freeze JSON must contain reference_used:false, frontend_protocol:protocol(),
    and source_hashes/input_hashes mappings including this runtime and cache,
    plus global_protocol_path/global_protocol_sha256 from the design freeze.
    Extra global-protocol fields or other experiment branches are permitted.
    """
    out, freeze_path = Path(out_dir).resolve(), Path(freeze_path).resolve()
    if out.exists():
        raise FileExistsError(out)
    runtime = load_runtime(project)
    trace, fps, metadata, inputs = load_cache(trace_path)
    freeze_digest = sha(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if freeze.get('reference_used') is not False or freeze.get('frontend_protocol') != protocol():
        raise ValueError('Global freeze must bind this exact reference-free frontend protocol')
    global_path = Path(freeze['global_protocol_path']).resolve()
    global_digest = freeze['global_protocol_sha256']
    if sha(global_path) != global_digest:
        raise ValueError('Global experiment protocol changed since stage freeze')
    global_protocol = json.loads(global_path.read_text(encoding='utf-8'))
    if global_protocol.get('reference_used_by_inference') is not False or \
            not set(ARMS).issubset(global_protocol.get('arms', {})):
        raise ValueError('Global design does not predeclare the reference-free six arms')
    for field, bindings in [('source_hashes', runtime.source_hashes), ('input_hashes', inputs)]:
        if any(freeze.get(field, {}).get(path) != digest for path, digest in bindings.items()):
            raise ValueError(f'Global freeze must bind exact {field} of this case/runtime')
    products, sources, records, masks, native = extract_products(trace, fps, runtime)
    _verify(inputs); _verify(runtime.source_hashes)
    if sha(freeze_path) != freeze_digest:
        raise ValueError('Global freeze changed during extraction')
    if sha(global_path) != global_digest:
        raise ValueError('Global experiment protocol changed during extraction')
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out/'input_masks.csv', masks)
    common = dict(case=case, fps=fps, n_frames=len(trace), protocol=protocol(),
        native_input_validity=native, input_hashes=inputs, source_hashes=runtime.source_hashes,
        stage_freeze_path=str(freeze_path), stage_freeze_sha256=freeze_digest,
        global_protocol_path=str(global_path), global_protocol_sha256=global_digest,
        cached_video_identity=metadata.get('identity'),
        video_sha256_from_cache=metadata.get('video_sha256'),
        reference_used=False, raw_video_read=False, default_replacement=False,
        interpretation='Experimental three-ROI controlled ablation; not six-subregion reproduction.',
        created_utc=datetime.now(timezone.utc).isoformat())
    results = {}
    for arm in ARMS:
        folder = out/arm
        folder.mkdir()
        save_csv(folder/'waveform.csv', products[arm])
        save_csv(folder/'region_sources.csv', sources[arm])
        save_csv(folder/'extraction_windows.csv', records[arm])
        # Estimate on a reread of the actual saved waveform, including CSV masks.
        wave = pd.read_csv(folder/'waveform.csv')
        waveform_digest = sha(folder/'waveform.csv')
        hr = estimate_waveform(wave, fps, runtime)
        save_csv(folder/'heart_rate.csv', hr)
        bank = folder/'window_waveforms'
        bank.mkdir()
        width, hop = round(10*fps), round(fps)
        manifest = []
        for index, start in enumerate(range(0, len(wave)-width+1, hop)):
            filename = f'window_{index:04d}.csv'
            save_csv(bank/filename, wave.iloc[start:start+width])
            manifest.append(dict(window_id=f'{index:04d}', start_frame=start,
                stop_frame=start+width, time_s=float(hr.iloc[index].time_s),
                window_start_s=start/fps, window_end_s=(start+width)/fps,
                waveform_file='window_waveforms/'+filename, sha256=sha(bank/filename),
                source_waveform_sha256=waveform_digest,
                source_arm=arm, accepted=bool(hr.iloc[index].accepted),
                status=str(hr.iloc[index].status)))
        save_csv(folder/'hr_windows_manifest.csv', pd.DataFrame(manifest, columns=[
            'window_id', 'start_frame', 'stop_frame', 'time_s', 'window_start_s', 'window_end_s',
            'waveform_file', 'sha256', 'source_waveform_sha256', 'source_arm', 'accepted', 'status']))
        outputs = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        results[arm] = dict(accepted_windows=int(hr.accepted.sum()), planned_windows=len(hr),
            waveform_coverage_pct=100*float(wave.covered.mean()))
        save_json(folder/'manifest.json', dict(common, arm=arm, status='complete',
            **results[arm], output_hashes=outputs,
            hr_window_source='Exact slice of saved waveform; per-region OLA diagnostic sources are not a reconstructed HR input.'))
    _verify(inputs); _verify(runtime.source_hashes)
    if sha(freeze_path) != freeze_digest:
        raise ValueError('Global freeze changed during inference')
    if sha(global_path) != global_digest:
        raise ValueError('Global experiment protocol changed during inference')
    save_json(out/'manifest.json', dict(common, status='complete', arms=results,
        output_hashes={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_case(args.trace, args.out, case=args.case, project=args.project, freeze_path=args.freeze)
    print(json.dumps(dict(case=args.case, status='complete', arms=result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
