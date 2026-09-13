"""Two predetermined short-window experiments on the original cached videos."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
R28 = P/'results/data1_6_v28_20260912'
ROOT = P/'results/data1_6_v29_20260912'
HERE = Path(__file__).resolve().parent
CASES = [f'data{i}' for i in range(1, 7)]
PROFILES = ['short6', 'short6_relaxed']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def read(path):
    return json.loads(Path(path).read_text())

def inputs(case):
    return [B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json',
            R28/'direct_guard'/case/'waveform.csv', R28/'direct_guard'/case/'heart_rate.csv']

def freeze():
    import analyze_video_v29 as model
    import evaluate_v29 as evaluator
    old = read(R28/'protocol_before_run.json')
    for case in CASES:
        for path in inputs(case)[:2]:
            assert sha(path) == old['input_hashes'][case][str(path)]
    assert sha(B/'inputs_manifest.json') == old['fixed_inputs_manifest_sha256']
    evaluation_inputs = evaluator.freeze_evaluation_inputs()
    protected = dict(old['protected_entries'])
    for path in [P/'run_motion_v28.sh', P/'motion_upgrade_v28_20260912/installation_manifest.json']:
        protected[str(path)] = sha(path)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment='V29 shorter windows and one relaxed ambiguity gate',
        max_candidates=2, profiles=model.PROFILE_CONFIGS,
        source_hashes=model.frozen_sources(),
        runner_sha256=sha(HERE/'run_batch_v29.py'),
        evaluation_hashes={str(HERE/'evaluate_v29.py'):sha(HERE/'evaluate_v29.py')},
        evaluation_inputs=evaluation_inputs,
        input_hashes={case:{str(path):sha(path) for path in inputs(case)} for case in CASES},
        fixed_inputs_manifest_sha256=sha(B/'inputs_manifest.json'),
        previous_protocol_sha256=sha(R28/'protocol_before_run.json'),
        protected_entries=protected,
        changes=['Both profiles use 6 second windows and 1 second steps.',
                 'Only short6_relaxed changes fusion ambiguity_ratio from .80 to .90.',
                 'No other quality gates, frontend edge trims or gap interpolation limits change.'],
        reference_policy='Inference cannot access references; scoring uses unchanged original estimated alignment.',
        comparison_policy='Native 6s plans assessed separately; saved-wave strict 10s readout uses original 309 windows.',
        interpretation='Offline processing, not verified real-time latency. More output is not higher accuracy.',
        selection_policy='Report both predetermined profiles across all six seen development videos; no threshold search or per-video selection.')
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT/'protocol_before_run.json'
    if target.exists():
        raise FileExistsError(target)
    save(target, protocol)
    print(json.dumps(dict(protocol=str(target), profiles=protocol['profiles']), ensure_ascii=False))

def frozen(case):
    import analyze_video_v29 as model
    protocol = read(ROOT/'protocol_before_run.json')
    assert model.frozen_sources() == protocol['source_hashes']
    assert model.PROFILE_CONFIGS == protocol['profiles']
    assert sha(HERE/'run_batch_v29.py') == protocol['runner_sha256']
    assert sha(B/'inputs_manifest.json') == protocol['fixed_inputs_manifest_sha256']
    for path, h in protocol['input_hashes'][case].items():
        assert sha(path) == h, path
    for path, h in protocol['protected_entries'].items():
        assert sha(path) == h, path
    return protocol

def infer(case, profile):
    import analyze_video_v29 as model
    protocol = frozen(case)
    started = time.perf_counter()
    out = ROOT/profile/case
    out.mkdir(parents=True, exist_ok=False)
    meta = read(B/case/'inference/frame_trace.json')
    assert sha(B/case/'inference/frame_trace.csv') == meta['trace_sha256']
    trace = pd.read_csv(B/case/'inference/frame_trace.csv')
    fps = meta['fps']
    assert len(trace) == meta['n_frames']
    np.testing.assert_allclose(trace.time_s, np.arange(len(trace))/fps, atol=1e-8, rtol=0)
    model.run_pipeline(trace, fps, out, profile)
    wave = pd.read_csv(out/'waveform.csv')
    hr = pd.read_csv(out/'heart_rate.csv')
    finite = np.isfinite(wave.base)
    np.testing.assert_array_equal(finite, wave.covered)
    summary = dict(status='complete', case=case, profile=profile,
        parameters=model.PROFILE_CONFIGS[profile], frames=len(trace), fps=fps,
        planned_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
        hr_coverage_pct=100*float(hr.accepted.mean()) if len(hr) else None,
        waveform_coverage_pct=100*float(finite.mean()),
        first_wave_sample_s=float(wave.time_s[finite].iloc[0]) if finite.any() else None,
        reference_used=False, offline=True, hr_from_saved_waveform=True,
        source_hashes=protocol['source_hashes'], input_hashes=protocol['input_hashes'][case],
        protocol_sha256=sha(ROOT/'protocol_before_run.json'),
        elapsed_s=time.perf_counter()-started,
        output_hashes={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    frozen(case)
    save(out/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ('case','profile','planned_windows','accepted_windows','hr_coverage_pct','waveform_coverage_pct','first_wave_sample_s','elapsed_s')}, ensure_ascii=False), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze', 'infer', 'all'])
    ap.add_argument('--case', choices=CASES)
    ap.add_argument('--profile', choices=PROFILES)
    args = ap.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'all':
        def launch(case, profile):
            result = subprocess.run([sys.executable, '-B', str(HERE/'run_batch_v29.py'),
                'infer', '--case', case, '--profile', profile], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'{case}/{profile}: {result.stdout}\n{result.stderr}')
            return result.stdout
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, case, profile) for case in CASES for profile in PROFILES]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    else:
        if not args.case or not args.profile:
            ap.error('--case and --profile are required for inference')
        infer(args.case, args.profile)

if __name__ == '__main__':
    main()
