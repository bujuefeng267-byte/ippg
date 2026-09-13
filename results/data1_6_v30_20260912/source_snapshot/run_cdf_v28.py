"""One frozen CDF preprocessing experiment on the unchanged installed V28 pipeline."""
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from cdf_preprocess import DEFAULT_CONFIG, preprocess_trace

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
R28 = P/'results/data1_6_v28_20260912'
ROOT = P/'results/data1_6_v30_20260912'
RUNTIME = P/'motion_upgrade_v28_20260912'
PROTOCOL = ROOT/'freeze_cdf.json'
VARIANT = 'cdf_v28'
CASES = tuple(f'data{i}' for i in range(1, 7))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, (bool,np.bool_)): return bool(x)
    if isinstance(x, (int,np.integer)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return str(x) if isinstance(x, Path) else x


def save(path, x):
    Path(path).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def model_module():
    # Import the installed, unchanged model explicitly, not another experiment's
    # similarly named files in this workspace. CDF itself was imported above.
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    import analyze_video_v28 as model
    if Path(model.__file__).resolve().parent != RUNTIME:
        raise RuntimeError('Unexpected V28 module location')
    return model


def sources():
    model = model_module()
    inherited = {str(RUNTIME/name): value for name,value in model.frozen_sources().items()}
    own = {str(HERE/name):sha(HERE/name) for name in
           ('cdf_preprocess.py','run_cdf_v28.py','test_cdf_preprocess.py')}
    return {**inherited, **own}


def inputs(case):
    return [B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json']


def verify(hashes):
    for path, value in hashes.items():
        assert sha(path) == value, 'Frozen file changed: '+str(path)


def freeze():
    if PROTOCOL.exists() or (HERE/'freeze_cdf.json').exists():
        raise FileExistsError('Preserve an existing CDF freeze record')
    model = model_module()
    previous_path = R28/'protocol_before_run.json'
    previous = json.loads(previous_path.read_text())
    bound_inputs = {}
    for case in CASES:
        bound_inputs[case] = {str(path):sha(path) for path in inputs(case)}
        for path,value in bound_inputs[case].items():
            assert previous['input_hashes'][case][path] == value, 'Input differs from previously bound V28 cache'
    fixed_sources = sources()
    verify(fixed_sources)
    protected_paths = [P/'run.sh', P/'run_motion_v25.sh', P/'run_motion_v28.sh',
                       RUNTIME/'installation_manifest.json', previous_path]
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), variant=VARIANT,
        maximum_candidates=1, cdf_config=asdict(DEFAULT_CONFIG),
        source_hashes=fixed_sources, input_hashes=bound_inputs,
        protected_hashes={str(path):sha(path) for path in protected_paths if path.exists()},
        original_V28_parameters=model.PARAMETERS, inherited_runtime=str(RUNTIME),
        source='https://sstuijk.estue.nl/publications/fg17.pdf',
        only_change='CDF independently preprocesses baseline and existing anchored tracked RGB in three physical ROIs; then unchanged complete V28 pipeline.',
        fixed_filter='u=[-1,2,-1]/sqrt(6); W=|FFT(AC/DC_RGB)@u|^2/sum(|FFT(AC/DC_RGB)|^2), clipped to [0,1]; common real W for RGB.',
        frequency_policy='rFFT/irFFT keeps conjugate negative frequencies symmetric, suppressing outside actual 42-210bpm bins; mean RGB restored.',
        window_policy='round(10*fps) complete windows, round(fps) hop from each observed run start; positive Hann overlap normalization.',
        gap_policy='No new interpolation. Input NaN entries unchanged. Short valid runs, uncovered segment tails and nonpositive/nonfinite output windows retain actual original RGB.',
        quality_policy='All original tracking, fallback, observation, quality, geometry, ambiguity, motion, component and HR gates unchanged.',
        output_policy='Save processed_trace.csv as an explicitly derived view, not frame_trace.csv/cache; save/reload final waveform before unchanged V28 HR.',
        mask_limit='Final router preserves its CDF-derived V25 fallback finite mask; original unprocessed V28 waveform coverage is not guaranteed unchanged.',
        reference_used=False, reference_HR_read=False, offline=True,
        no_post_reference_tuning=True, no_motion_scoring_experiment_combined=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    save(PROTOCOL, record)
    save(HERE/'freeze_cdf.json', record)
    print(json.dumps(dict(protocol=str(PROTOCOL),sources=len(fixed_sources),cases=list(CASES)),ensure_ascii=False),flush=True)


def frozen(case):
    record = json.loads(PROTOCOL.read_text())
    assert record['variant'] == VARIANT and record['cdf_config'] == asdict(DEFAULT_CONFIG)
    assert record['source_hashes'] == sources()
    assert set(record['input_hashes'][case]) == {str(path) for path in inputs(case)}
    verify(record['source_hashes']); verify(record['input_hashes'][case]); verify(record['protected_hashes'])
    return record


def run(case):
    fixed, started = frozen(case), time.perf_counter()
    model = model_module()
    trace_path, metadata_path = inputs(case)
    metadata = json.loads(metadata_path.read_text())
    assert metadata['trace_sha256'] == sha(trace_path)
    trace = pd.read_csv(trace_path)
    fps = float(metadata['fps'])
    assert len(trace) == metadata['n_frames']
    model.validate_component_trace(trace, fps)
    out = ROOT/VARIANT/case
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running',case=case,variant=VARIANT,reference_used=False,
        source_hashes=fixed['source_hashes'], input_hashes=fixed['input_hashes'][case],
        protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL), fps=fps,frames=len(trace))
    save(out/'manifest.json',manifest)
    processed, process_summary, process_windows = preprocess_trace(trace,fps)
    model.validate_component_trace(processed, fps)
    processed.to_csv(out/'processed_trace.csv', index=False)
    process_windows.to_csv(out/'cdf_windows.csv', index=False)
    save(out/'cdf_preprocessing.json',process_summary)
    save(out/'processed_trace_metadata.json',dict(kind='derived_CDF_RGB_view_not_frontend_cache',
        original_trace_path=str(trace_path), original_trace_sha256=sha(trace_path),
        original_metadata_path=str(metadata_path), original_metadata_sha256=sha(metadata_path),
        derived_trace_sha256=sha(out/'processed_trace.csv'), cdf_config=asdict(DEFAULT_CONFIG),
        protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL), reference_used=False, offline=True))
    saved_processed = pd.read_csv(out/'processed_trace.csv')
    model.validate_component_trace(saved_processed,fps)
    wave,hr,decisions = model.run_pipeline(saved_processed,fps,out)
    frozen(case)
    summary = dict(status='complete',case=case,variant=VARIANT,reference_used=False,
        fps=fps,frames=len(trace),source_hashes=fixed['source_hashes'],input_hashes=fixed['input_hashes'][case],
        protocol_path=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),
        output_hashes={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'manifest.json'},
        planned_windows=len(hr),accepted_windows=int(hr.accepted.sum()),
        hr_coverage_pct=100*float(hr.accepted.mean()) if len(hr) else None,
        waveform_coverage_pct=100*float(wave.covered.mean()),
        route_eligible_windows=int(decisions.route_eligible.sum()) if len(decisions) else 0,
        cdf_stream_statuses=process_summary['streams'], cdf_config=asdict(DEFAULT_CONFIG),
        hr_from_saved_waveform=True,offline=True,elapsed_s=time.perf_counter()-started,
        original_frontend_cache_unchanged=True,observation_masks_preserved=True,
        relative_mask_scope='Final waveform finite mask is preserved relative to the CDF-derived fallback only, not the original V28 output.',
        inference_only='No reference HR, reference alignment or accuracy report opened during CDF inference.')
    save(out/'summary.json',summary)
    manifest.update(status='complete',summary_sha256=sha(out/'summary.json'),output_hashes=summary['output_hashes'])
    save(out/'manifest.json',manifest)
    print(json.dumps(clean({key:summary[key] for key in ('case','variant','planned_windows','accepted_windows',
        'hr_coverage_pct','waveform_coverage_pct','route_eligible_windows','elapsed_s')}),ensure_ascii=False),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','infer','all'])
    parser.add_argument('--case',choices=CASES)
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'all':
        for case in CASES: run(case)
    else:
        if not args.case: parser.error('--case required for infer')
        run(args.case)
