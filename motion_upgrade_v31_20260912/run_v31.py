"""Two predeclared reference-free protection mechanisms on fixed saved inputs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
RUNTIME = P/'motion_upgrade_v28_20260912'
B = P/'results/data1_6_20260911'
V28 = P/'results/data1_6_v28_20260912/direct_guard'
CDF = P/'results/data1_6_v30_20260912/cdf_v28'
ROOT = P/'results/data1_6_v31_20260912'
PROTOCOL = ROOT/'freeze_inference_v31.json'
VARIANTS = {'protected_cdf_motion': 'direct_motion', 'protected_cdf_consensus': 'raw_consensus'}
CASES = tuple(f'data{i}' for i in range(1, 7))
sys.path.insert(0, str(RUNTIME))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def hashes(paths): return {str(p): sha(p) for p in paths}


def verify(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, 'Frozen file changed: '+path


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def input_paths(case):
    return [B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json',
        V28/case/'waveform.csv', V28/case/'heart_rate.csv', V28/case/'summary.json',
        CDF/case/'waveform.csv', CDF/case/'heart_rate.csv', CDF/case/'summary.json']


def freeze():
    if PROTOCOL.exists(): raise FileExistsError(PROTOCOL)
    own = [HERE/name for name in ('run_v31.py', 'protected_inference_v31.py',
        'proposal_evidence_v31.py', 'protected_waveform_router.py',
        'test_protected_inference_v31.py', 'test_proposal_evidence_v31.py',
        'test_protected_waveform_router.py', 'test_v31_integration.py')]
    protected = [P/'run_recommended.sh', P/'run_motion_v28.sh', P/'recommended_version.json']
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        variants=VARIANTS, source_hashes=hashes(own+sorted(RUNTIME.glob('*.py'))),
        input_hashes={case: hashes(input_paths(case)) for case in CASES},
        protected_entries=hashes(protected),
        reference_used=False, previously_seen_development_data=True,
        candidate_config='Previously frozen V30 CDF 10s/1s 42-210; cached complete CDF V28 wave and HR, unchanged.',
        fixed_window=dict(window_s=10, step_s=1, min_bpm=42, max_bpm=210),
        modes=dict(direct_motion='Original direct motion risk>=0.50 and decrease>=0.30; at least two physical raw ROIs support candidate; both HR accepted/full windows, >6bpm disagreement.',
            raw_consensus='Same direct-motion route OR at least two original baseline ROIs have BOTH POS and CHROM maximum-power candidates within +/-3bpm of CDF HR and >6bpm from old HR. No half/double relation rule or reference input.'),
        protection='Union of every noneligible complete HR window protects all contained original V28 samples. Modifications only in remaining eligible cores, with positive Hann edge taper. Noneligible accepted HR states anchored to original V28 after verifying support in actual saved final wave; old rejected HR remain rejected.',
        runtime_rollback='Monotonic withdrawal on old accepted HR loss or >12bpm jump count greater than old or maximum jump greater than max(12,old maximum). Withdraw overlapping admissions and recompute full saved waveform and constrained DP; fallback to all-original if needed.',
        no_post_reference_threshold_search=True, maximum_modes=2,
        no_video_specific_selection=True, no_reference_used_for_admission=True,
        no_HR_interpolation=True, original_recommendation_retained_during_experiment=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    save(PROTOCOL, record)
    print(json.dumps({'frozen':str(PROTOCOL),'variants':list(VARIANTS)},ensure_ascii=False),flush=True)


def run(variant, case):
    from analyze_motion_v2 import validate_trace
    from protected_inference_v31 import infer_protected
    record = json.loads(PROTOCOL.read_text())
    verify(record['source_hashes']); verify(record['input_hashes'][case]); verify(record['protected_entries'])
    assert record['variants'] == VARIANTS
    for folder in (V28, CDF):
        info = json.loads((folder/case/'summary.json').read_text())
        assert info['status'] == 'complete' and info['reference_used'] is False
        for name in ('waveform.csv','heart_rate.csv'):
            assert sha(folder/case/name) == info['output_hashes'][name]
    metadata = json.loads((B/case/'inference/frame_trace.json').read_text())
    assert sha(B/case/'inference/frame_trace.csv') == metadata['trace_sha256']
    fps = float(metadata['fps'])
    trace = pd.read_csv(B/case/'inference/frame_trace.csv')
    assert len(trace) == metadata['n_frames']
    validate_trace(trace, fps)
    started = time.perf_counter()
    out = ROOT/variant/case
    wave, hr, decisions, audit = infer_protected(V28/case/'waveform.csv', V28/case/'heart_rate.csv',
        CDF/case/'waveform.csv', CDF/case/'heart_rate.csv', trace, fps, out, VARIANTS[variant])
    verify(record['source_hashes']); verify(record['input_hashes'][case]); verify(record['protected_entries'])
    summary = dict(status='complete', case=case, variant=variant, fps=fps, frames=len(trace),
        reference_used=False, offline=True, hr_from_saved_waveform=True,
        source_hashes=record['source_hashes'], input_hashes=record['input_hashes'][case],
        protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
        output_hashes={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()},
        planned_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
        preservation=audit, elapsed_s=time.perf_counter()-started)
    save(out/'summary.json', summary)
    print(json.dumps({'variant':variant,'case':case,'accepted':int(hr.accepted.sum()),
        'initial_eligible':audit['initial_eligible_windows'],'final_eligible':audit['final_eligible_windows'],
        'modified_seconds':audit['modified_seconds'],'guard_attempts':audit['guard_attempts'],
        'elapsed_s':summary['elapsed_s']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','infer'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    else:
        for variant in ([args.variant] if args.variant else VARIANTS):
            for case in ([args.case] if args.case else CASES): run(variant,case)
