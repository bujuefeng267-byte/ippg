"""One fixed motion-normalization ablation on unchanged, saved V28 waveforms.

No physiological reference is loaded by this program. Freeze before all six
predictions; outputs are deliberately experimental and never change V28.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
V28 = P/'motion_upgrade_v28_20260912'
B = P/'results/data1_6_20260911'
OLD = P/'results/data1_6_v28_20260912/direct_guard'
ROOT = P/'results/data1_6_v30_20260912'
PROTOCOL = ROOT/'freeze_motion.json'
CASES = tuple(f'data{i}' for i in range(1, 7))
sys.path.insert(0, str(V28))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def hashes(paths):
    return {str(p): sha(p) for p in paths}


def verify(mapping):
    for name, expected in mapping.items():
        assert sha(name) == expected, 'Frozen file changed: '+name


def inputs(case):
    return [OLD/case/'waveform.csv', OLD/case/'heart_rate.csv',
            B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json']


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    files = sorted(V28.glob('*.py')) + [HERE/'run_motion_readout.py',
        HERE/'motion_evidence_full_spectrum.py']
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        variant='motion_full_spectrum', reference_used=False,
        source_hashes=hashes(files),
        input_hashes={case: hashes(inputs(case)) for case in CASES},
        protected_entries=hashes([P/'run_recommended.sh', P/'run_motion_v28.sh',
                                 P/'recommended_version.json']),
        fixed_settings=dict(window_s=10, step_s=1, min_bpm=42, max_bpm=210),
        modification='Only HR readout motion PSD normalization uses the complete one-sided spectrum including DC, after the unchanged constant detrending. The saved V28 waveform is copied byte for byte; V28 extraction, fusion, routing, validity gates, periodic/harmonic scores and DP are unchanged.',
        scope='Readout-only ablation: upstream fusion/routing still used original V28 motion evidence.',
        no_reference_tuning=True, maximum_configurations=1,
        development_data_previously_seen=True, offline=True)
    save(PROTOCOL, record)
    print(json.dumps({'frozen': str(PROTOCOL), 'cases': list(CASES)}), flush=True)


def run(case):
    from evidence_hr import estimate_evidence
    from analyze_motion_v2 import validate_trace
    from motion_evidence_full_spectrum import motion_evidence
    fixed = json.loads(PROTOCOL.read_text())
    verify(fixed['source_hashes'])
    verify(fixed['input_hashes'][case])
    verify(fixed['protected_entries'])
    started = time.perf_counter()
    output = ROOT/'motion_full_spectrum'/case
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((B/case/'inference/frame_trace.json').read_text())
    fps = float(meta['fps'])
    assert sha(B/case/'inference/frame_trace.csv') == meta['trace_sha256']
    trace = pd.read_csv(B/case/'inference/frame_trace.csv')
    validate_trace(trace, fps)
    assert len(trace) == meta['n_frames']
    shutil.copyfile(OLD/case/'waveform.csv', output/'waveform.csv')
    wave = pd.read_csv(output/'waveform.csv')
    np.testing.assert_allclose(wave.time_s, trace.time_s, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(wave.covered, np.isfinite(wave.base))
    hr = estimate_evidence(wave.base.to_numpy(float),
        pd.DataFrame({'rgb_valid': wave.observed}), wave.interpolated.to_numpy(bool),
        fps, motion_trace=trace, motion_provider=motion_evidence)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    starts = np.arange(0, len(wave)-round(10*fps)+1, round(fps))
    hr['window_start_s'] = starts/fps
    hr['window_end_s'] = (starts+round(10*fps))/fps
    hr['hr_source'] = 'saved_V28_waveform_full_spectrum_motion_normalization_offline'
    old_hr = pd.read_csv(OLD/case/'heart_rate.csv')
    np.testing.assert_array_equal(hr.accepted, old_hr.accepted)
    np.testing.assert_allclose(hr.time_s, old_hr.time_s, rtol=0, atol=1e-8)
    hr.to_csv(output/'heart_rate.csv', index=False)
    verify(fixed['source_hashes'])
    verify(fixed['input_hashes'][case])
    verify(fixed['protected_entries'])
    summary = dict(status='complete', case=case, variant='motion_full_spectrum',
        reference_used=False, fps=fps, frames=len(trace), offline=True,
        source_hashes=fixed['source_hashes'], input_hashes=fixed['input_hashes'][case],
        output_hashes={name: sha(output/name) for name in ('waveform.csv', 'heart_rate.csv')},
        protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
        saved_waveform_sha256_preserved=sha(output/'waveform.csv') == sha(OLD/case/'waveform.csv'),
        HR_accepted_mask_preserved=True, planned_windows=len(hr),
        accepted_windows=int(hr.accepted.sum()), elapsed_s=time.perf_counter()-started,
        hr_from_saved_waveform=True)
    save(output/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('case','accepted_windows','elapsed_s')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'infer'])
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        for case in ([args.case] if args.case else CASES):
            run(case)
