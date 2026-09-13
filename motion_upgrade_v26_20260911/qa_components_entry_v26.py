"""Saved data2 entry smoke vs frozen component experiment; no reference reads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_json(a, b, label='root', tolerance=1e-8):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f'JSON keys differ at {label}')
        return max((compare_json(a[k], b[k], f'{label}.{k}', tolerance) for k in a), default=0.)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f'JSON length differs at {label}')
        return max((compare_json(x, y, f'{label}[{i}]', tolerance) for i, (x, y) in enumerate(zip(a, b))), default=0.)
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        delta = abs(float(a)-float(b))
        if tolerance is not None and delta > tolerance:
            raise AssertionError(f'JSON numeric difference {delta} at {label}')
        return delta
    if a != b:
        raise AssertionError(f'JSON differs at {label}')
    return 0.


def compare_csv(a, b, strict_numeric=False):
    aa, bb = pd.read_csv(a), pd.read_csv(b)
    if list(aa.columns) != list(bb.columns) or len(aa) != len(bb):
        raise AssertionError(f'CSV shape/schema differs: {a.name}')
    max_abs, exact, json_delta = {}, [], {}
    for col in aa:
        x, y = aa[col], bb[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            np.testing.assert_array_equal(x.isna(), y.isna())
            valid = x.notna()
            max_abs[col] = float(np.max(np.abs(x[valid]-y[valid]))) if valid.any() else 0.
            if strict_numeric:
                np.testing.assert_allclose(x, y, atol=1e-8, rtol=0, equal_nan=True)
            if np.array_equal(x, y, equal_nan=True):
                exact.append(col)
        elif col.endswith('_json'):
            json_delta[col] = max((compare_json(json.loads(p), json.loads(q), col, None)
                                   for p, q in zip(x, y)), default=0.)
        else:
            np.testing.assert_array_equal(x.fillna('__MISSING__'), y.fillna('__MISSING__'))
            exact.append(col)
    return dict(rows=len(aa), max_abs_difference=max_abs, exact_columns=exact,
                structured_json_max_difference=json_delta,
                a_sha256=sha(a), b_sha256=sha(b))


def path_ties(records, proposals, other_proposals, step_s):
    """Independent joint emissions/transition arithmetic from saved candidates."""
    grid = np.arange(42., 211.)
    physical = ('forehead', 'left_cheek', 'right_cheek')
    scores = np.full((len(proposals), 3, len(grid)), -np.inf)
    for row in records:
        window = row['window_index']
        roi = physical.index(row['channel'].split('/')[1])
        for bpm in row['supported_bpm']:
            k = int(bpm-42)
            score = row['evidence_score']-.025*(bpm-row['bpm'])**2
            scores[window, roi, k] = max(scores[window, roi, k], score)
    count = np.isfinite(scores).sum(axis=1)
    joint = np.full((len(proposals), len(grid)), -np.inf)
    good = count >= 2
    joint[good] = np.where(np.isfinite(scores), scores, 0).sum(axis=1)[good]/count[good]
    usable = np.isfinite(joint).any(axis=1)
    edges = np.diff(np.r_[False, usable, False].astype(int))
    groups = []
    def objective(start, stop, path):
        values = path[start:stop]
        picked = np.round(values-42).astype(int)
        total = float(joint[np.arange(start, stop), picked].sum())
        delta = np.diff(values)
        normal = np.where(np.abs(delta) <= 12*step_s+1e-9, -.05*(delta/step_s)**2, -np.inf)
        return total + float(np.maximum(normal, -3.).sum())
    own = proposals.proposal_bpm.to_numpy(float)
    other = other_proposals.proposal_bpm.to_numpy(float)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        own_score, other_score = objective(start, stop, own), objective(start, stop, other)
        groups.append(dict(start_window=int(start), stop_window_exclusive=int(stop),
            own_score=own_score, alternative_score=other_score,
            difference=abs(own_score-other_score), equivalent_within_1e_10=abs(own_score-other_score)<=1e-10))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', required=True, type=Path)
    ap.add_argument('--frozen', required=True, type=Path)
    ap.add_argument('--trace-source', required=True, type=Path)
    args = ap.parse_args()
    report = dict(passed=False, reference_used=False, roi_reconstruction_tolerance=1e-8,
                  scope='one data2 cache smoke; fresh no-face synthetic CLI tested separately', files={})
    for name in ('waveform.csv', 'heart_rate.csv', 'component_proposals.csv'):
        report['files'][name] = compare_csv(args.result/name, args.frozen/name)
    for name in ('baseline_roi_waveforms.csv', 'roi_waveforms.csv'):
        report['files'][name] = compare_csv(args.result/name, args.trace_source/name, strict_numeric=True)
    candidate_a, candidate_b = [json.loads(p.read_text()) for p in
        (args.result/'all_roi_candidates.json', args.frozen/'all_roi_candidates.json')]
    report['all_candidates_max_abs_difference'] = compare_json(candidate_a, candidate_b)
    report['trace_bytes_exact'] = sha(args.result/'frame_trace.csv') == sha(args.trace_source/'frame_trace.csv')
    if not report['trace_bytes_exact']:
        raise AssertionError('Original trace CSV bytes changed')
    hr = report['files']['heart_rate.csv']
    for key in ('ridge_bpm', 'spectral_peak_bpm', 'accepted'):
        if key not in hr['exact_columns']:
            raise AssertionError(f'Main HR/acceptance not exact: {key}')
    w0, w1 = [pd.read_csv(p/'waveform.csv') for p in (args.result, args.frozen)]
    valid = w0.base.notna()
    error = w0.base[valid]-w1.base[valid]
    report['waveform_relative_rmse'] = float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(w1.base[valid]**2)))
    report['waveform_equal_within_1e_8'] = bool(error.abs().max() <= 1e-8)
    p0, p1 = [pd.read_csv(p/'component_proposals.csv') for p in (args.result, args.frozen)]
    differences = p0.proposal_bpm.notna() & p0.proposal_bpm.ne(p1.proposal_bpm)
    report['proposal_differences'] = [dict(window_index=int(i), new_bpm=float(p0.loc[i, 'proposal_bpm']),
                                         historical_bpm=float(p1.loc[i, 'proposal_bpm']))
                                    for i in p0.index[differences]]
    meta = json.loads((args.result/'frame_trace.json').read_text())
    step = round(meta['fps'])/meta['fps']
    report['path_comparison_using_new_input_candidates'] = path_ties(candidate_a, p0, p1, step)
    report['path_comparison_using_historical_input_candidates'] = path_ties(candidate_b, p1, p0, step)
    report['pass_criteria'] = 'Original trace exact; recomputed ROI inputs within 1e-8; all masks/channels exact; final local/DP HR exact. Waveform/proposals are separately quantified, not required exact.'
    report['passed'] = True
    (args.result/'entry_qa_receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(passed=True, waveform_max_abs_difference=max(report['files']['waveform.csv']['max_abs_difference'].values()),
          main_hr_and_acceptance_exact=True, all_candidates_max_abs_difference=report['all_candidates_max_abs_difference'])))


if __name__ == '__main__':
    main()
