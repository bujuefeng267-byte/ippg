#!/usr/bin/env python3
"""Run the experimental, not-promoted V31 protection layer on verified V28 results.

No video decoding or reference measurements are used. An original anchored RGB
trace with matching SHA metadata and a completed V28 output manifest is required.
The original inputs remain untouched; every run needs a new output directory.
"""
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import argparse
import hashlib
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
RUNTIME = HERE.parent/'motion_upgrade_v28_20260912'
if not RUNTIME.is_dir():
    RUNTIME = PROJECT/'motion_upgrade_v28_20260912'
CDF_SHA256 = 'c07d1660ea27d0f7ebb197f3317f6910ef478bdbae069ee2023fd49713ee5ce7'
OWN_SOURCES = ('analyze_v28_result_v31.py', 'cdf_preprocess.py',
    'proposal_evidence_v31.py', 'protected_inference_v31.py', 'protected_waveform_router.py')
PUBLISHED = ('waveform.csv', 'heart_rate.csv', 'routing_decisions.csv',
    'proposal_evidence.csv', 'guard_history.json', 'preservation_audit.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError('Bound input/source changed: '+str(path))


def save(path, value):
    # Inference helpers return only standard finite JSON values after cleaning.
    from protected_inference_v31 import clean
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')


def load_model():
    if sha(HERE/'cdf_preprocess.py') != CDF_SHA256:
        raise ValueError('CDF must be the unchanged, frozen V30 implementation')
    sys.path.insert(0, str(RUNTIME))
    import analyze_video_v28 as model
    if Path(model.__file__).resolve().parent != RUNTIME.resolve():
        raise ValueError('Unexpected V28 runtime module')
    return model


def sources(model):
    return {**{str((RUNTIME/name).resolve()): digest for name, digest in model.frozen_sources().items()},
        **{str(HERE/name): sha(HERE/name) for name in OWN_SOURCES}}


def trace_paths(cache, old):
    selected = Path(cache).resolve() if cache else old/'frame_trace.csv'
    path = selected/'frame_trace.csv' if selected.is_dir() else selected
    if path.name != 'frame_trace.csv':
        raise ValueError('Use the original frame_trace.csv, or its directory')
    metadata = path.with_suffix('.json')
    if not path.is_file() or not metadata.is_file():
        raise ValueError('Both original frame_trace.csv and matching frame_trace.json are required')
    return path, metadata


def verified_input(old, trace_path, metadata_path, model):
    """Accept original batch and installed single-video V28 manifest schemas."""
    import numpy as np
    import pandas as pd
    summary_path, manifest_path = old/'summary.json', old/'manifest.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    if summary.get('status', manifest.get('status')) != 'complete':
        raise ValueError('V28 result must have completed successfully')
    if summary.get('variant') != 'direct_guard' or summary.get('reference_used') is not False:
        raise ValueError('An original reference-free V28 direct_guard result is required')
    output_hashes = summary.get('output_hashes', manifest.get('outputs', {}))
    input_hashes = summary.get('input_hashes', {})
    consumed = [summary_path, trace_path, metadata_path]
    if manifest:
        consumed.append(manifest_path)
    for name in ('waveform.csv', 'heart_rate.csv'):
        if output_hashes.get(name) != sha(old/name):
            raise ValueError('V28 output is missing its original hash binding: '+name)
        consumed.append(old/name)
    for path in (trace_path, metadata_path):
        expected = {digest for name, digest in input_hashes.items() if Path(name).name == path.name}
        if path.name in output_hashes:
            expected.add(output_hashes[path.name])
        if sha(path) not in expected:
            raise ValueError('Trace is not bound to the supplied V28 result: '+path.name)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('trace_sha256') != sha(trace_path):
        raise ValueError('Trace SHA differs from its cache metadata')
    fps = float(metadata['fps'])
    trace = pd.read_csv(trace_path)
    if len(trace) != int(metadata['n_frames']) or len(trace) != int(summary['frames']):
        raise ValueError('Trace frame count differs from metadata or V28 result')
    if not np.isclose(fps, float(summary['fps']), rtol=0, atol=1e-8):
        raise ValueError('Trace FPS differs from V28 result')
    identity = metadata.get('identity', {})
    if identity.get('rgb_reconstruction') != 'anchored_log' or identity.get('pixel_mode') != 'tracking_screened':
        raise ValueError('Original anchored tracking-screened RGB cache is required')
    current_sources = model.frozen_sources()
    for name, digest in identity.get('frontend_hashes', {}).items():
        if current_sources.get(name) != digest:
            raise ValueError('Trace frontend does not match unchanged V28: '+name)
    if not identity.get('frontend_hashes'):
        raise ValueError('Trace metadata needs original frontend source bindings')
    # Historical batch wrappers and installed CLI have different wrapper names;
    # compare all inherited runtime files that their manifests bind in common.
    declared = {Path(name).name: digest for name, digest in summary.get('source_hashes', {}).items()}
    for name in set(declared) & set(current_sources):
        if declared[name] != current_sources[name]:
            raise ValueError('V28 core source differs from original output: '+name)
    if 'direct_guard_router_v28.py' not in declared:
        raise ValueError('Missing original V28 direct-guard source binding')
    model.validate_component_trace(trace, fps)
    if len(trace) < round(10*fps):
        raise ValueError('This experimental result adapter requires at least one complete 10s window')
    return trace, fps, {str(path): sha(path) for path in consumed}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v28-output', required=True, type=Path, help='Completed original V28 output directory')
    ap.add_argument('--trace-cache', type=Path, help='Original frame_trace.csv or directory; default: V28 output/frame_trace.csv')
    ap.add_argument('--out', required=True, type=Path, help='New experiment output directory; existing paths rejected')
    ap.add_argument('--mode', choices=('direct_motion', 'raw_consensus'), default='direct_motion',
        help='One fixed experimental mechanism; default: direct_motion')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    old, out = args.v28_output.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(out))
    if not old.is_dir():
        raise ValueError('V28 output is not a directory')
    trace_path, metadata_path = trace_paths(args.trace_cache, old)
    if out.is_relative_to(old) or out.is_relative_to(trace_path.parent):
        raise ValueError('Keep experimental output outside original V28 and trace directories')
    model = load_model()
    import pandas as pd
    from cdf_preprocess import DEFAULT_CONFIG, preprocess_trace
    from proposal_evidence_v31 import protocol_description
    from protected_inference_v31 import infer_protected
    fixed_sources = sources(model)
    trace, fps, fixed_inputs = verified_input(old, trace_path, metadata_path, model)
    verify(fixed_sources); verify(fixed_inputs)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running',
        variant='protected_cdf_motion' if args.mode == 'direct_motion' else 'protected_cdf_consensus',
        mode=args.mode, experimental=True, promoted=False, recommendation='V28',
        reference_used=False, reference_input_supported=False, video_decoded=False,
        offline=True, fps=fps, frames=len(trace), source_hashes=fixed_sources,
        input_hashes=fixed_inputs, original_V28_output=str(old), inherited_runtime=str(RUNTIME),
        cdf_source_sha256=CDF_SHA256, cdf_config=asdict(DEFAULT_CONFIG),
        original_V28_parameters=model.PARAMETERS, proposal_protocol=protocol_description(),
        scope='Reference-free reuse of verified original RGB cache. No accuracy or waveform-morphology claim.',
        recommendation_policy='Experimental adapter only; does not change the installed recommended V28.')
    save(out/'protocol_before_run.json', record)
    protocol_hash = sha(out/'protocol_before_run.json')
    save(out/'manifest.json', record)
    try:
        candidate = out/'candidate'; candidate.mkdir()
        processed, processing, windows = preprocess_trace(trace, fps)
        model.validate_component_trace(processed, fps)
        processed.to_csv(candidate/'processed_trace.csv', index=False)
        windows.to_csv(candidate/'cdf_windows.csv', index=False)
        save(candidate/'cdf_preprocessing.json', processing)
        save(candidate/'processed_trace_metadata.json', dict(
            kind='derived_CDF_RGB_view_not_frontend_cache', original_trace_path=str(trace_path),
            original_trace_sha256=sha(trace_path), derived_trace_sha256=sha(candidate/'processed_trace.csv'),
            reference_used=False, offline=True, cdf_config=asdict(DEFAULT_CONFIG)))
        saved_processed = pd.read_csv(candidate/'processed_trace.csv')
        model.validate_component_trace(saved_processed, fps)
        model.run_pipeline(saved_processed, fps, candidate)
        save(candidate/'summary.json', dict(status='complete', variant='cdf_v28',
            experimental=True, promoted=False, reference_used=False, fps=fps, frames=len(trace),
            output_hashes={str(p.relative_to(candidate)): sha(p) for p in sorted(candidate.rglob('*')) if p.is_file()}))
        wave, hr, decisions, audit = infer_protected(old/'waveform.csv', old/'heart_rate.csv',
            candidate/'waveform.csv', candidate/'heart_rate.csv', trace, fps, out/'protected_run', args.mode)
        for name in PUBLISHED:
            shutil.copyfile(out/'protected_run'/name, out/name)
        verify(fixed_sources); verify(fixed_inputs)
        if sources(model) != fixed_sources or sha(out/'protocol_before_run.json') != protocol_hash:
            raise ValueError('Runtime sources or protocol changed during inference')
        summary = dict(record, status='complete', protocol_path=str(out/'protocol_before_run.json'),
            protocol_sha256=protocol_hash, planned_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
            waveform_coverage_pct=100*float(wave.covered.mean()),
            hr_coverage_pct=100*float(hr.accepted.mean()), hr_from_saved_waveform=True,
            preservation=audit, original_inputs_unchanged=True, sources_unchanged=True,
            elapsed_s=time.perf_counter()-started,
            output_hashes={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p.name != 'manifest.json'})
        save(out/'summary.json', summary)
        save(out/'manifest.json', dict(record, status='complete', summary_sha256=sha(out/'summary.json'),
            protocol_sha256=protocol_hash, output_hashes=summary['output_hashes']))
        print(json.dumps({key: summary[key] for key in ('status', 'variant', 'experimental', 'promoted',
            'planned_windows', 'accepted_windows', 'waveform_coverage_pct', 'hr_coverage_pct', 'elapsed_s')},
            ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        save(out/'manifest.json', dict(record, status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
