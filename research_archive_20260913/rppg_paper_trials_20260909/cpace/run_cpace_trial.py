"""Isolated unmodified cPACE core trial on frozen V2.2 ROI traces.

No reference is accepted by this interface. Author default seed comes from
video GREEN, independently per finite ROI-set segment. The scalar waveform is
the fixed Forehead output; no invented ROI fusion or oracle selection.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AUTHOR = HERE/'author_source'
ESTIMATOR = HERE/'project_estimator'
sys.path.insert(0, str(ESTIMATOR))
from legacy_motion import estimate, fill_short_gaps, gap_frame_limit, runs
sys.path.insert(0, str(AUTHOR))
from pipeline import run_method
from projections import _cpace_candidates_one_roi

COMMIT = '4cd7a438bf9dd30af8ec3afc365dad53861b6c97'
ROIS = {'forehead': 'Forehead', 'left_cheek': 'L_Cheek', 'right_cheek': 'R_Cheek'}
CASES = ('user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1')
PARAMETERS = {
    'method': 'cPACE', 'cardiac_band_hz': [0.7, 3.0], 'eigen_bw_hz': 0.30,
    'seed': 'unmodified author default: Forehead GREEN PSD; no external seed',
    'max_gap_s': 0.10, 'edge_mask_s': 1.6, 'minimum_segment_s': 4.0,
    'roi_segment_rule': 'constant available ROI set; Forehead required plus at least one cheek',
    'primary_waveform': 'fixed Forehead cardiac_signals; no ROI averaging or selection',
    'final_hr': 'V2.2 legacy estimate on CSV-roundtripped saved waveform; 42–210 bpm, 10 s windows, 1 s step, offline DP',
    'provenance': 'ALL used ROIs observed and ANY used ROI interpolated; masked waveform samples have both flags false',
    'reference_used': False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def code_hashes():
    files = [HERE/'run_cpace_trial.py', HERE/'test_cpace_trial.py']
    files += sorted(AUTHOR.glob('*.py')) + [AUTHOR/'LICENSE', AUTHOR/'README.md']
    files += sorted(ESTIMATOR.glob('*.py'))
    return {str(p.relative_to(HERE)): sha(p) for p in files}


def check_trace(trace, fps):
    expected = np.arange(len(trace))/fps
    if not np.allclose(trace.time_s.to_numpy(float), expected, rtol=0, atol=1e-8):
        raise ValueError('Trace must retain the original uniform frame clock')
    for roi in ROIS:
        rgb = trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy(float)
        valid = trace[f'{roi}_valid'].to_numpy()
        if valid.dtype != bool:
            raise ValueError('ROI validity must be Boolean')
        if not np.array_equal(valid, np.isfinite(rgb).all(axis=1)):
            raise ValueError('ROI validity and RGB finite mask differ')


def infer_trace(trace, fps):
    """Return two modes, all real ROI waveforms, and segment diagnostics."""
    check_trace(trace, fps)
    n = len(trace)
    filled_rgb, interpolation = {}, {}
    codes = np.zeros(n, np.uint8)
    for bit, roi in enumerate(ROIS):
        rgb = trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy(float)
        filled_rgb[roi], interpolation[roi] = fill_short_gaps(rgb, gap_frame_limit(.10, fps))
        codes |= np.isfinite(filled_rgb[roi]).all(axis=1).astype(np.uint8) << bit
    changes = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1])+1, n]
    waves = {mode: {roi: np.full(n, np.nan) for roi in ROIS}
             for mode in ('default', 'no_homodyne')}
    observed = np.zeros(n, bool)
    interpolated = np.zeros(n, bool)
    records = []
    edge = round(1.6*fps)
    for sid, (a, b) in enumerate(zip(changes[:-1], changes[1:])):
        active = [roi for bit, roi in enumerate(ROIS) if int(codes[a]) & (1 << bit)]
        record = {'segment_index': sid, 'start_frame': int(a), 'end_frame_exclusive': int(b),
                  'start_s': a/fps, 'end_s': b/fps, 'roi_names': active}
        if 'forehead' not in active or len(active) < 2:
            record['status'] = 'insufficient_rois_with_fixed_forehead'
            records.append(record)
            continue
        if b-a < max(round(4*fps), 2*edge+1):
            record['status'] = 'segment_too_short'
            records.append(record)
            continue
        roi_rgb = {ROIS[roi]: tuple(filled_rgb[roi][a:b].T) for roi in active}
        outputs = {}
        for mode, homodyne in [('default', True), ('no_homodyne', False)]:
            # Deliberately omit hr_seed_hz: inference cannot receive reference HR.
            outputs[mode] = run_method(roi_rgb, fps, method='cPACE',
                                       apply_homodyne=homodyne,
                                       cardiac_band=(0.7, 3.0), eigen_bw=.30,
                                       seed_roi='Forehead')
        if outputs['default']['hr_seed_hz'] != outputs['no_homodyne']['hr_seed_hz']:
            raise RuntimeError('Same video input produced different author seeds')
        record['status'] = 'processed'
        record['seed_bpm'] = outputs['default']['hr_seed_bpm']
        record['seed_source'] = 'video Forehead GREEN only, recomputed per segment'
        record['retained_start_frame'] = int(a+edge)
        record['retained_end_frame_exclusive'] = int(b-edge)
        record['native_author_results'] = {}
        for mode, result in outputs.items():
            record['native_author_results'][mode] = {k: v for k, v in result.items()
                                                     if k != 'cardiac_signals'}
            record['native_author_results'][mode]['scope'] = 'whole finite segment before edge mask; not final window HR'
            record['native_author_results'][mode]['roi_waveform_std'] = {}
            for roi in active:
                actual = np.asarray(result['cardiac_signals'][ROIS[roi]], float)
                if actual.shape != (b-a,) or not np.isfinite(actual).all():
                    raise ValueError('Author output is nonfinite or has wrong length')
                waves[mode][roi][a+edge:b-edge] = actual[edge:-edge]
                record['native_author_results'][mode]['roi_waveform_std'][roi] = float(np.std(actual))
        # Diagnostic equality check; this does not choose or alter the output.
        r, g, blue = roi_rgb['Forehead']
        candidates = _cpace_candidates_one_roi(r, g, blue, fps,
                                               outputs['default']['hr_seed_hz'], .30)
        actual = outputs['no_homodyne']['cardiac_signals']['Forehead']
        record['forehead_selected_eigenvector_matches'] = [key for key, val in candidates.items()
                                                          if np.allclose(val, actual, rtol=1e-10, atol=1e-14)]
        observed[a+edge:b-edge] = np.logical_and.reduce([
            trace[f'{roi}_valid'].to_numpy(bool)[a+edge:b-edge] for roi in active])
        interpolated[a+edge:b-edge] = np.logical_or.reduce([
            interpolation[roi][a+edge:b-edge] for roi in active])
        records.append(record)
    return waves, observed, interpolated, records


def final_hr(waveform, fps):
    table = estimate(waveform.base.to_numpy(float),
                     pd.DataFrame({'rgb_valid': waveform.observed.to_numpy(bool)}),
                     waveform.interpolated.to_numpy(bool), fps, 10, 1, 42, 210)
    table['raw_spectral_peak_bpm'] = table.spectral_peak_bpm
    table.loc[~table.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    table['hr_source'] = 'saved_fixed_forehead_cpace_waveform_offline'
    return table


def run_case(case, frozen):
    started = time.perf_counter()
    source = ROOT/'rppg_motion_v22/validation/trimmed_gap10'/case
    trace = pd.read_csv(source/'frame_trace.csv')
    meta = json.loads((source/'frame_trace.json').read_text(encoding='utf-8'))
    fps = float(meta['fps'])
    waves, observed, interpolated, records = infer_trace(trace, fps)
    case_summaries = {}
    for mode, directory in [('default', 'results'), ('no_homodyne', 'results_no_homodyne')]:
        output = HERE/directory/case
        output.mkdir(parents=True, exist_ok=False)
        pd.DataFrame({'time_s': trace.time_s, 'base': waves[mode]['forehead'],
                      'observed': observed, 'interpolated': interpolated,
                      'covered': np.isfinite(waves[mode]['forehead'])}).to_csv(output/'waveform.csv', index=False)
        # Final HR is computed from the actual serialized samples that are shared.
        saved = pd.read_csv(output/'waveform.csv')
        table = final_hr(saved, fps)
        table.to_csv(output/'heart_rate.csv', index=False)
        pd.DataFrame({'time_s': trace.time_s, **waves[mode]}).to_csv(output/'roi_waveforms.csv', index=False)
        dump(output/'segment_diagnostics.json', records)
        summary = {
            'case': case, 'method': 'author_cPACE_fixed_forehead', 'variant': mode,
            'author_commit': COMMIT, 'author_math_unmodified': True, 'license': 'MIT',
            'fps': fps, 'frames': len(trace), 'parameters': PARAMETERS,
            'source_trace': str(source/'frame_trace.csv'),
            'trace_sha256': sha(source/'frame_trace.csv'),
            'video_identity': meta.get('identity'),
            'source_video_sha256': json.loads((ROOT/'rppg_motion_v22/validation/protocol_before_validation.json').read_text(encoding='utf-8'))['cases'][case]['sha256'],
            'native_hr_scope': 'per finite segment, before edge masking; diagnostics only',
            'waveform_scope': 'real per-ROI author core output; fixed Forehead, no sine synthesis',
            'homodyne_changes_morphology': mode == 'default',
            'offline': True, 'reference_used': False,
            'accepted_windows': int(table.accepted.sum()), 'total_windows': len(table),
            'status_counts': table.status.value_counts().to_dict(),
            'waveform_coverage': float(saved.base.notna().mean()),
            'observed_fraction': float(saved.observed.mean()),
            'interpolated_fraction': float(saved.interpolated.mean()),
            'processed_segments': int(sum(r['status']=='processed' for r in records)),
            'seed_bpm_per_segment': [r['seed_bpm'] for r in records if r['status']=='processed'],
            'wall_seconds_shared_two_modes': time.perf_counter()-started,
            'code_hashes': frozen,
            'output_hashes': {name: sha(output/name) for name in
                              ['waveform.csv', 'heart_rate.csv', 'roi_waveforms.csv', 'segment_diagnostics.json']},
            'limitations': ['Not a full paper protocol reproduction',
                            'Original .7–3.0 Hz core differs from 42–210 bpm final HR search',
                            'Whole-segment seed, projection and filtering use future samples',
                            'Common brightness can survive original normalized-coordinate projection',
                            'Input includes only V2.2 trimmed-gap10 cached video RGB; no reference data'],
        }
        dump(output/'summary.json', summary)
        case_summaries[mode] = {k: summary[k] for k in ['accepted_windows','total_windows','waveform_coverage','processed_segments','seed_bpm_per_segment']}
    print(json.dumps({'case': case, **case_summaries}), flush=True)
    return case_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-all', action='store_true')
    args = parser.parse_args()
    if not args.run_all:
        parser.error('Use --run-all after synthetic self-checks')
    if (HERE/'results').exists() or (HERE/'results_no_homodyne').exists():
        raise FileExistsError('Trial outputs already exist; do not overwrite')
    test_record = json.loads((HERE/'synthetic_selfchecks.json').read_text(encoding='utf-8'))
    if not test_record['successful']:
        raise RuntimeError('Synthetic self-checks did not pass')
    frozen = code_hashes()
    if test_record['code_hashes'] != frozen:
        raise RuntimeError('Sources changed after self-checks')
    source = ROOT/'rppg_motion_v22/validation/trimmed_gap10'
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'author_commit': COMMIT, 'parameters': PARAMETERS, 'code_hashes': frozen,
                'command': [sys.executable, '-B', str(Path(__file__).resolve()), '--run-all'],
                'source_trace_hashes': {case: sha(source/case/'frame_trace.csv') for case in CASES},
                'source_metadata_hashes': {case: sha(source/case/'frame_trace.json') for case in CASES},
                'selfchecks_sha256': sha(HERE/'synthetic_selfchecks.json'),
                'frozen_before_real_inference': True}
    dump(HERE/'protocol_before_inference.json', protocol)
    summaries = {case: run_case(case, frozen) for case in CASES}
    if code_hashes() != frozen:
        raise RuntimeError('Source changed during inference')
    dump(HERE/'run_summary.json', {'cases': summaries, 'all_six_successful': True,
                                  'code_hashes': frozen, 'protocol_sha256': sha(HERE/'protocol_before_inference.json')})


if __name__ == '__main__':
    main()
