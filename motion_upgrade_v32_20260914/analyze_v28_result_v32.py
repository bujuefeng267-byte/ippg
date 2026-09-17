#!/usr/bin/env python3
"""Experimental V32 window-based HR from a completed, verified V28 result.

The original anchored RGB trace is reused without decoding the video or opening
reference measurements. CDF candidates are regenerated locally. Each HR window
is bound to its own saved signal; waveform.csv is unchanged V28 context, not a
new V32 continuous waveform. This adapter does not change the recommended V28.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT_FALLBACK = Path('/home/fengbujue/项目/rppg识别')
V31_ADAPTER_SHA256 = 'b1380c6fe5e7f4ff1bab52a16de9312043fc4e1d50539b9fd4edf3ebfecbbcbb'
CDF_SHA256 = 'c07d1660ea27d0f7ebb197f3317f6910ef478bdbae069ee2023fd49713ee5ce7'
OWN_SOURCES = ('analyze_v28_result_v32.py', 'window_inference_v32.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_root():
    for root in (HERE.parent, PROJECT_FALLBACK):
        if all((root/name).is_dir() for name in
               ('motion_upgrade_v28_20260912', 'motion_upgrade_v31_20260912')):
            return root.resolve()
    raise FileNotFoundError('Install beside the unchanged V28 and V31 runtime directories')


def load_runtime():
    """Reuse the frozen V31 input validation instead of maintaining another copy."""
    root = runtime_root()
    v31 = root/'motion_upgrade_v31_20260912'
    path = v31/'analyze_v28_result_v31.py'
    if sha(path) != V31_ADAPTER_SHA256:
        raise ValueError('V31 input adapter differs from the frozen implementation')
    if sha(v31/'cdf_preprocess.py') != CDF_SHA256:
        raise ValueError('CDF must be the unchanged, frozen V30 implementation')
    spec = importlib.util.spec_from_file_location('_frozen_v31_adapter_for_v32', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    sys.path.insert(0, str(v31))
    model = legacy.load_model()
    import cdf_preprocess as cdf
    if Path(cdf.__file__).resolve() != (v31/'cdf_preprocess.py').resolve():
        raise ValueError('Unexpected CDF module; start V32 in a fresh Python process')
    sys.path.insert(0, str(HERE))
    import window_inference_v32 as core
    if Path(core.__file__).resolve() != (HERE/'window_inference_v32.py').resolve():
        raise ValueError('Unexpected V32 window runtime module')
    return legacy, model, cdf, core


def bound_sources(legacy, model):
    return {**legacy.sources(model),
            **{str(HERE/name): sha(HERE/name) for name in OWN_SOURCES}}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v28-output', required=True, type=Path,
                    help='Completed original V28 direct_guard output directory')
    ap.add_argument('--trace-cache', type=Path,
                    help='Original frame_trace.csv or directory; default: V28 output/frame_trace.csv')
    ap.add_argument('--out', required=True, type=Path,
                    help='New experiment directory; existing paths are rejected')
    return ap


def validate_window_bank(out, hr, old_wave_path):
    """Verify published paths/hashes and forbid a misleading global-wave claim."""
    import pandas as pd
    manifest = pd.read_csv(out/'windows_manifest.csv')
    if len(manifest) != len(hr) or sorted(manifest.window_index.tolist()) != list(range(len(hr))):
        raise ValueError('Every planned HR window must have a unique saved-window record')
    if set(manifest.waveform_source) - {'original_v28', 'independent_cdf'}:
        raise ValueError('Unknown saved-window source')
    bank = (out/'window_waveforms').resolve()
    for row in manifest.itertuples(index=False):
        path = (out/row.waveform_file).resolve()
        expected = bank/f'window_{int(row.window_index):04d}.csv'
        if path != expected or not path.is_file() or sha(path) != row.sha256:
            raise ValueError('Saved-window file/path/hash differs from its manifest')
    if sha(out/'waveform.csv') != sha(old_wave_path):
        raise ValueError('Global waveform.csv must remain byte-identical original V28 context')
    if 'waveform_file' not in hr or 'waveform_source' not in hr:
        raise ValueError('HR rows must identify their saved-window file and source')
    ordered = manifest.sort_values('window_index').reset_index(drop=True)
    if (list(hr.waveform_file) != list(ordered.waveform_file) or
            list(hr.waveform_source) != list(ordered.waveform_source)):
        raise ValueError('HR rows and the saved-window manifest disagree')
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    old, out = args.v28_output.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(out))
    if not old.is_dir():
        raise ValueError('V28 output is not a directory')
    legacy, model, cdf, core = load_runtime()
    trace_path, metadata_path = legacy.trace_paths(args.trace_cache, old)
    if out.is_relative_to(old) or out.is_relative_to(trace_path.parent):
        raise ValueError('Keep experimental output outside original V28 and trace directories')
    trace, fps, fixed_inputs = legacy.verified_input(old, trace_path, metadata_path, model)
    fixed_sources = bound_sources(legacy, model)
    legacy.verify(fixed_sources)
    legacy.verify(fixed_inputs)
    started = time.perf_counter()
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running',
        variant='independent_window_cdf_motion_guard', motion_guard=True,
        experimental=True, promoted=False, recommendation='V28',
        reference_used=False, reference_input_supported=False, video_decoded=False,
        offline=True, fps=fps, frames=len(trace), source_hashes=fixed_sources,
        input_hashes=fixed_inputs, original_V28_output=str(old),
        inherited_runtime=str(legacy.RUNTIME), inherited_input_adapter=str(legacy.HERE),
        cdf_source_sha256=CDF_SHA256, cdf_config=asdict(cdf.DEFAULT_CONFIG),
        original_V28_parameters=model.PARAMETERS,
        window_protocol=core.protocol_description(),
        hr_from_saved_window_waveforms=True, hr_from_single_continuous_waveform=False,
        hr_from_global_waveform=False,
        global_waveform_role='unchanged_original_V28_context',
        continuous_V32_waveform_available=False,
        scope='Reference-free experimental window HR. No new continuous PPG or morphology claim.',
        recommendation_policy='Experimental adapter only; the recommended version remains V28.')
    out.mkdir(parents=True, exist_ok=False)
    legacy.save(out/'protocol_before_run.json', record)
    protocol_hash = sha(out/'protocol_before_run.json')
    legacy.save(out/'manifest.json', record)
    try:
        import pandas as pd
        candidate = out/'candidate'
        candidate.mkdir()
        processed, processing, windows = cdf.preprocess_trace(trace, fps)
        model.validate_component_trace(processed, fps)
        processed.to_csv(candidate/'processed_trace.csv', index=False)
        windows.to_csv(candidate/'cdf_windows.csv', index=False)
        legacy.save(candidate/'cdf_preprocessing.json', processing)
        legacy.save(candidate/'processed_trace_metadata.json', dict(
            kind='derived_CDF_RGB_view_not_frontend_cache',
            original_trace_path=str(trace_path), original_trace_sha256=sha(trace_path),
            derived_trace_sha256=sha(candidate/'processed_trace.csv'),
            reference_used=False, offline=True, cdf_config=asdict(cdf.DEFAULT_CONFIG)))
        saved_processed = pd.read_csv(candidate/'processed_trace.csv')
        model.validate_component_trace(saved_processed, fps)
        model.run_pipeline(saved_processed, fps, candidate)
        legacy.save(candidate/'summary.json', dict(status='complete', variant='cdf_v28',
            experimental=True, promoted=False, reference_used=False, fps=fps, frames=len(trace),
            output_hashes={str(p.relative_to(candidate)): sha(p)
                for p in sorted(candidate.rglob('*')) if p.is_file()}))
        hr, decisions, audit = core.infer_windows(old/'waveform.csv', old/'heart_rate.csv',
            candidate/'waveform.csv', candidate/'heart_rate.csv', trace, fps, out,
            motion_guard=True)
        bank = validate_window_bank(out, hr, old/'waveform.csv')
        legacy.verify(fixed_sources)
        legacy.verify(fixed_inputs)
        if bound_sources(legacy, model) != fixed_sources or sha(out/'protocol_before_run.json') != protocol_hash:
            raise ValueError('Runtime sources or protocol changed during inference')
        context_wave = pd.read_csv(out/'waveform.csv')
        summary = dict(record, status='complete', protocol_path=str(out/'protocol_before_run.json'),
            protocol_sha256=protocol_hash, planned_windows=len(hr),
            accepted_windows=int(hr.accepted.sum()),
            selected_cdf_windows=int((bank.waveform_source == 'independent_cdf').sum()),
            saved_window_count=len(bank), hr_coverage_pct=100*float(hr.accepted.mean()),
            original_V28_context_waveform_coverage_pct=100*float(context_wave.covered.mean()),
            preservation=audit, original_inputs_unchanged=True, sources_unchanged=True,
            elapsed_s=time.perf_counter()-started,
            output_hashes={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p not in (out/'manifest.json', out/'summary.json')})
        legacy.save(out/'summary.json', summary)
        legacy.save(out/'manifest.json', dict(record, status='complete',
            summary_sha256=sha(out/'summary.json'), protocol_sha256=protocol_hash,
            output_hashes=summary['output_hashes']))
        print(json.dumps({key: summary[key] for key in ('status', 'variant', 'experimental',
            'promoted', 'planned_windows', 'accepted_windows', 'selected_cdf_windows',
            'hr_coverage_pct', 'continuous_V32_waveform_available', 'elapsed_s')},
            ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        legacy.save(out/'manifest.json', dict(record, status='failed',
                    error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
