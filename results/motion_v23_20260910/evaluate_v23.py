"""Frozen, reference-blind V22/V23 regression scoring of saved output files.

This evaluator imports only the hash-pinned previous evaluator's pure scoring
helpers, never inference code. It does not retime references, rerun inference,
tune thresholds, choose a model, or promote an executable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V22 = ROOT / 'rppg_motion_v22'
CORE_SHA = '691a6c594c926892fab94fe69e197a391f39d27b22b3b28eaf972bc7aa7f733e'
CRITERIA_SHA = '234b52a58868329105992236f7affdae24d0c745b3f21f4b62d7ba79c0ab8a68'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if sha(V22 / 'evaluate_v22.py') != CORE_SHA:
    raise RuntimeError('Frozen evaluation dependency changed')
spec = importlib.util.spec_from_file_location('frozen_v22_scoring', V22 / 'evaluate_v22.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

CASES = core.CASES
BASELINES = ('trimmed_gap10', 'trimmed_gap15')
CANDIDATES = ('tracking_only_gap10', 'tracking_only_gap15',
              'tracking_screened_gap10', 'tracking_screened_gap15')
VERSIONS = BASELINES + CANDIDATES
VARIANTS = ('fusion', 'pos', 'chrom')
ESTIMATORS = core.ESTIMATORS
GEOMETRY = ('face_detected', 'flow_available', 'source', 'time_s',
            'face_x0', 'face_y0', 'face_x1', 'face_y1', 'motion_x', 'motion_y',
            'rgb_valid', 'forehead_valid', 'left_cheek_valid', 'right_cheek_valid')
FROZEN_BACKEND_FILES = ('analyze_rppg.py', 'motion_fusion.py', 'stable_groups.py', 'waveform_hr.py')
METRIC_COLUMNS = ('Nplanned', 'Noutput', 'Nref', 'Nvalid', 'MAE_bpm', 'RMSE_bpm',
                  'Bias_bpm', 'P5_valid_pct', 'R5_all_reference_pct')


def load_version(case, version, directory, evidence):
    summary = evidence.json(directory / 'summary.json')
    metadata = evidence.json(directory / 'frame_trace.json')
    trace = evidence.csv(directory / 'frame_trace.csv')
    frames, fps = int(summary['frames']), float(summary['fps'])
    assert frames == len(trace) == metadata['n_frames']
    assert sha(directory / 'frame_trace.csv') == metadata['trace_sha256']
    np.testing.assert_allclose(trace.time_s, np.arange(frames) / fps, rtol=0, atol=1e-8)
    config = summary['config']
    assert config['window'] == 10 and config['step'] == 1
    assert (config['min_bpm'], config['max_bpm']) == (42, 210)
    assert config['max_gap'] == (.1 if version.endswith('gap10') else .15)
    assert summary['reference_identity'] is None and config['reference_ubfc'] is None
    window, step = round(10 * fps), round(fps)
    starts = np.arange(0, frames - window + 1, step)
    centers = (starts + window / 2) / fps
    data = dict(directory=directory, summary=summary, metadata=metadata, trace=trace,
                frames=frames, fps=fps, starts=starts, centers=centers, window=window, step=step,
                variants={})
    for variant in VARIANTS:
        hr = evidence.csv(directory / f'{variant}_heart_rate.csv')
        wave = evidence.csv(directory / f'{variant}_waveform.csv')
        assert len(wave) == frames and len(hr) == len(starts)
        np.testing.assert_allclose(wave.time_s, trace.time_s, rtol=0, atol=1e-8)
        np.testing.assert_allclose(hr.time_s, centers, rtol=0, atol=1e-8)
        accepted = core.bools(hr.accepted, f'{case}/{version}/{variant}/accepted')
        for column in ESTIMATORS.values():
            assert np.isfinite(hr.loc[accepted, column]).all()
            if variant == 'fusion' or column == 'ridge_bpm':
                assert hr.loc[~accepted, column].isna().all(), 'Rejected final HR must remain missing'
            else:
                # Frozen shared POS/CHROM readout retains its pre-gate local
                # peak for diffuse-spectrum diagnostics; it is NOT output HR.
                diagnostic = ~accepted & hr[column].notna().to_numpy()
                assert hr.loc[diagnostic, 'status'].eq('diffuse_spectrum').all()
        finite = np.isfinite(wave.base.to_numpy(float))
        finite_windows = np.array([finite[a:a + window].all() for a in starts])
        assert np.all(finite_windows[accepted]), 'Accepted HR requires a finite saved waveform segment'
        assert summary['variants'][variant]['accepted_windows'] == int(accepted.sum())
        np.testing.assert_allclose(summary['variants'][variant]['finite_waveform_fraction'], finite.mean(), atol=1e-12)
        if variant == 'fusion':
            np.testing.assert_array_equal(core.bools(wave.covered, 'covered'), finite)
            assert not np.any(core.bools(wave.observed, 'observed') & ~finite)
            assert not np.any(core.bools(wave.interpolated, 'interpolated') & ~finite)
            generated = core.bools(hr.waveform_generated, 'waveform_generated')
            np.testing.assert_array_equal(core.bools(hr.neighbor_covered, 'neighbor_covered'),
                                          finite_windows & ~generated)
        else:
            generated = None
        data['variants'][variant] = dict(hr=hr, wave=wave, fps=fps, frames=frames, starts=starts,
            centers=centers, window=window, step=step, accepted=accepted,
            generated=generated, finite_samples=finite, finite_windows=finite_windows)
    return data


def verify_geometry(case, baseline, candidate, before, after):
    rows = []
    for column in GEOMETRY:
        a, b = before['trace'][column], after['trace'][column]
        if pd.api.types.is_numeric_dtype(a):
            exact = (a.to_numpy() == b.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())
            equal = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
            both = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
            maximum = float(np.max(np.abs(a.to_numpy(float)[both] - b.to_numpy(float)[both]))) if both.any() else 0.
        else:
            equal = a.fillna('<NA>').astype(str).to_numpy() == b.fillna('<NA>').astype(str).to_numpy()
            exact, maximum = equal, None
        count = int((~equal).sum())
        rows.append(dict(case=case, before=baseline, after=candidate, field=column,
                         differing_frames_beyond_serialization_tolerance=count, max_abs_difference=maximum,
                         exact_differing_frames=int((~exact).sum()), numerically_verified=count == 0))
        assert count == 0, f'Frozen geometry changed: {case}/{candidate}/{column}: {count} frames'
    return rows


def load_case(case, versions, args, evidence, frozen22, frozen23):
    data = {}
    for version in versions:
        directory = (args.baseline_root if version in BASELINES else args.validation_root) / version / case
        data[version] = load_version(case, version, directory, evidence)
    baseline = data['trimmed_gap10']
    geometry_rows = []
    for version, item in data.items():
        assert item['frames'] == baseline['frames']
        np.testing.assert_allclose(item['fps'], baseline['fps'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(item['centers'], baseline['centers'], rtol=0, atol=1e-8)
        for key in ('video', 'size', 'mtime_ns', 'max_seconds'):
            assert item['summary']['identity'][key] == baseline['summary']['identity'][key], (case, version, key)
        assert item['summary']['fusion_config'] == baseline['summary']['fusion_config']
        expected = frozen22['source_hashes'] if version in BASELINES else frozen23['source_hashes']
        assert item['summary']['source_hashes'] == expected, f'Unfrozen inference source: {case}/{version}'
        for filename in FROZEN_BACKEND_FILES:
            assert item['summary']['source_hashes'][filename] == frozen22['source_hashes'][filename]
        geometry_rows.extend(verify_geometry(case, 'trimmed_gap10', version, baseline, item))
    for before, after in [('trimmed_gap10', 'trimmed_gap15'),
                          ('tracking_only_gap10', 'tracking_only_gap15'),
                          ('tracking_screened_gap10', 'tracking_screened_gap15')]:
        if before in data and after in data:
            a, b = data[before]['trace'], data[after]['trace']
            assert list(a) == list(b)
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]):
                    np.testing.assert_allclose(a[column], b[column], rtol=0, atol=1e-9, equal_nan=True,
                        err_msg=f'Gap modes must share frontend: {case}/{before}/{after}/{column}')
                else:
                    np.testing.assert_array_equal(a[column].fillna('<NA>'), b[column].fillna('<NA>'))
    return data, geometry_rows


def continuity_row(data):
    row = core.continuity(data['accepted'], data['starts'], data['window'], data['fps'])
    rejected = row['longest_rejected_run_windows']
    row['longest_rejected_grid_run_s'] = rejected * data['step'] / data['fps']
    row['longest_rejected_center_span_s'] = max(0, rejected - 1) * data['step'] / data['fps']
    return row


def waveform_row(data, values, mask):
    finite = data['finite_samples']
    observed = core.bools(data['wave'].observed, 'wave observed') & finite
    interpolated = core.bools(data['wave'].interpolated, 'wave interpolated') & finite
    spans = core.spans(finite)
    missing = core.spans(~finite)
    internal = [(a, b) for a, b in missing if a > 0 and b < len(finite)]
    selected = values[mask & np.isfinite(values)]
    row = dict(Nplanned=len(data['accepted']), Naccepted=int(data['accepted'].sum()),
        Nfinite_waveform_windows=int(data['finite_windows'].sum()), Nscored=len(selected),
        finite_sample_count=int(finite.sum()), finite_sample_pct=100 * finite.mean(),
        finite_sample_duration_s=float(finite.sum() / data['fps']),
        longest_finite_run_s=max((b-a for a, b in spans), default=0) / data['fps'],
        longest_missing_run_s=max((b-a for a, b in missing), default=0) / data['fps'],
        longest_internal_missing_run_s=max((b-a for a, b in internal), default=0) / data['fps'],
        finite_runs=len(spans), observed_finite_samples=int(observed.sum()),
        observed_all_samples_pct=100 * observed.mean(), interpolated_samples=int(interpolated.sum()),
        interpolated_all_samples_pct=100 * interpolated.mean(),
        interpolated_covered_samples_pct=100 * interpolated.sum() / finite.sum() if finite.any() else None,
        snr_mean_db=float(selected.mean()) if len(selected) else None,
        snr_median_db=float(np.median(selected)) if len(selected) else None,
        peak_concentration_median_accepted=float(data['hr'].loc[data['accepted'], 'peak_concentration'].median())
            if data['accepted'].any() else None,
        r_wave=None, morphology_reference_status='no_qualified_synchronized_waveform_truth')
    if data['generated'] is not None:
        row.update(waveform_generated_count=int(data['generated'].sum()),
            waveform_generated_but_HR_rejected_count=int((data['generated'] & ~data['accepted']).sum()),
            neighbor_covered_count=int(core.bools(data['hr'].neighbor_covered, 'neighbor_covered').sum()),
            accepted_neighbor_covered_count=int((core.bools(data['hr'].neighbor_covered, 'neighbor_covered') & data['accepted']).sum()))
    return row


def pair_rows(case, variant, estimator, before, after, data, reference, shift):
    aa, bb = data[before]['variants'][variant], data[after]['variants'][variant]
    both = aa['accepted'] & bb['accepted']
    added = ~aa['accepted'] & bb['accepted']
    lost = aa['accepted'] & ~bb['accepted']
    common = dict(case=case, variant=variant, estimator=estimator, before=before, after=after,
        reference_kind=core.REFERENCE_KIND.get(case, 'none'), shift_s=shift,
        Ncommon_output=int(both.sum()), Nadded_output=int(added.sum()), Nlost_output=int(lost.sum()))
    rows = []
    for scope, version, mask in [('common_before', before, both), ('common_after', after, both),
                                 ('added_after', after, added), ('lost_before', before, lost)]:
        pred = data[version]['variants'][variant]['hr'][ESTIMATORS[estimator]].to_numpy(float)
        rows.append(dict(**common, scope=scope, version=version, **core.metrics(pred, mask, reference)))
    return rows


def audit_baseline(case, version, estimator, row, old_metrics):
    prior = old_metrics[old_metrics.case.eq(case) & old_metrics.version.eq(version)
                        & old_metrics.estimator.eq(estimator)].iloc[0]
    for key in METRIC_COLUMNS:
        np.testing.assert_allclose(row[key] if row[key] is not None else np.nan, prior[key],
                                   rtol=0, atol=1e-9, equal_nan=True,
                                   err_msg=f'Frozen V22 mismatch: {case}/{version}/{estimator}/{key}')


def check_receipt(path, evidence, args):
    if not path.exists():
        return {'passed': False, 'reason': 'Saved waveform replay QA receipt is missing'}
    receipt = evidence.json(path)
    reasons = []
    for key, expected in [('passed', True), ('checked_cases', 6), ('checked_modes', 4)]:
        if receipt.get(key) != expected:
            reasons.append(f'{key} must equal {expected}')
    if set(receipt.get('checked_variants', [])) != set(VARIANTS):
        reasons.append('All three waveform variants must be replayed')
    if set(receipt.get('checked_estimators', [])) != set(ESTIMATORS):
        reasons.append('Both HR estimators must be replayed')
    hashes = receipt.get('hr_sha256', {})
    expected_keys = {f'{mode}/{case}/{variant}_heart_rate.csv'
                     for mode, case, variant in itertools.product(CANDIDATES, CASES, VARIANTS)}
    if set(hashes) != expected_keys:
        reasons.append('Receipt must hash all 72 candidate HR output files')
    else:
        for key, expected in hashes.items():
            if sha(evidence.register(args.validation_root / key)) != expected:
                reasons.append(f'Replayed HR changed: {key}')
    return dict(passed=not reasons, reasons=reasons, receipt=receipt)


def upgrade_gate(tables, criteria, baseline, candidate, engineering_pass):
    """Apply the frozen rule; never search for a better candidate or shift."""
    checks = []
    tolerance = criteria['noninferiority_numerical_tolerance']

    def check(name, passed, **detail):
        checks.append(dict(check=name, passed=bool(passed), **detail))

    def one(table, **filters):
        found = [row for row in tables[table] if all(row.get(k) == v for k, v in filters.items())]
        assert len(found) == 1, (table, filters, len(found))
        return found[0]

    def no_worse(name, before, after, lower_better=True):
        valid = before is not None and after is not None and np.isfinite(before) and np.isfinite(after)
        delta = after - before if valid else None
        passed = valid and (delta <= tolerance if lower_better else delta >= -tolerance)
        check(name, passed, before=before, after=after, delta=delta)

    check('engineering_integrity_and_replay_QA', engineering_pass)
    for case in criteria['reference_cases']:
        for estimator in criteria['required_estimators']:
            filters = dict(case=case, variant='fusion', estimator=estimator)
            own_a = one('metrics', version=baseline, **filters)
            own_b = one('metrics', version=candidate, **filters)
            pa = one('paired_metrics', before=baseline, after=candidate, scope='common_before', **filters)
            pb = one('paired_metrics', before=baseline, after=candidate, scope='common_after', **filters)
            check(f'{case}/{estimator}/nonempty_common', pa['Nvalid'] > 0 and pb['Nvalid'] > 0)
            for name in ('MAE_bpm', 'RMSE_bpm'):
                no_worse(f'{case}/{estimator}/own/{name}', own_a[name], own_b[name])
                no_worse(f'{case}/{estimator}/common/{name}', pa[name], pb[name])
            no_worse(f'{case}/{estimator}/R5', own_a['R5_all_reference_pct'], own_b['R5_all_reference_pct'], False)
        pair = one('waveform_deltas', case=case, variant='fusion', before=baseline, after=candidate)
        check(f'{case}/nonempty_paired_reference_H1_SNR', pair['Npaired_snr'] > 0)
        no_worse(f'{case}/paired_reference_H1_SNR', 0, pair['mean_paired_delta_snr_db'], False)
    meaningful = []
    for case in CASES:
        a = one('metrics', case=case, version=baseline, variant='fusion', estimator='offline_ridge')
        b = one('metrics', case=case, version=candidate, variant='fusion', estimator='offline_ridge')
        wa = one('waveform_metrics', case=case, version=baseline, variant='fusion', scope='own')
        wb = one('waveform_metrics', case=case, version=candidate, variant='fusion', scope='own')
        ca = one('continuity', case=case, version=baseline, variant='fusion')
        cb = one('continuity', case=case, version=candidate, variant='fusion')
        no_worse(f'{case}/HR_coverage', a['C_out_pct'], b['C_out_pct'], False)
        no_worse(f'{case}/finite_waveform_coverage', wa['finite_sample_pct'], wb['finite_sample_pct'], False)
        no_worse(f'{case}/longest_HR_rejection_grid_s', ca['longest_rejected_grid_run_s'], cb['longest_rejected_grid_run_s'])
        no_worse(f'{case}/longest_waveform_missing_s', wa['longest_missing_run_s'], wb['longest_missing_run_s'])
        if case != 'synthetic72':
            coverage_gain = max(b['C_out_pct'] - a['C_out_pct'], wb['finite_sample_pct'] - wa['finite_sample_pct'])
            if coverage_gain >= 5 - tolerance:
                meaningful.append(dict(case=case, improvement='coverage_percentage_points', value=coverage_gain))
        if case in criteria['reference_cases']:
            for estimator in criteria['required_estimators']:
                aa = one('metrics', case=case, version=baseline, variant='fusion', estimator=estimator)
                bb = one('metrics', case=case, version=candidate, variant='fusion', estimator=estimator)
                if aa['MAE_bpm'] is not None and bb['MAE_bpm'] is not None:
                    gain = aa['MAE_bpm'] - bb['MAE_bpm']
                    if gain >= 1 - tolerance:
                        meaningful.append(dict(case=case, estimator=estimator, improvement='MAE_bpm', value=gain))
    for estimator in criteria['required_estimators']:
        row = one('metrics', case='synthetic72', version=candidate, variant='fusion', estimator=estimator)
        check(f'synthetic72/{estimator}/MAE_at_most_1bpm',
              row['MAE_bpm'] is not None and row['MAE_bpm'] <= 1 + tolerance, value=row['MAE_bpm'])
    check('at_least_one_frozen_meaningful_improvement', bool(meaningful), improvements=meaningful)
    return dict(baseline=baseline, candidate=candidate, primary_shift_s=0,
        eligible_for_named_upgrade=all(row['passed'] for row in checks),
        failed_checks=[row for row in checks if not row['passed']], checks=checks,
        note='Development/regression gate only. data1 timing is estimated. Evaluator does not install or switch defaults.')


def self_check():
    result = core.self_check()
    data = dict(accepted=np.array([True, False, False, True]), starts=np.arange(4) * 30,
                window=300, fps=30., step=30)
    c = continuity_row(data)
    assert c['longest_rejected_grid_run_s'] == 2
    assert c['longest_rejected_center_span_s'] == 1
    # Retain original reference denominators on added/lost/common subsets.
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    added = core.metrics([101, np.nan, 120, np.nan], ~a & b, [100] * 4)
    lost = core.metrics([101, 102, np.nan, np.nan], a & ~b, [100] * 4)
    assert added['Nref'] == lost['Nref'] == 4 and added['MAE_bpm'] == 20
    assert lost['R5_all_reference_pct'] == 25
    result['v23_checks'] = ['HR rejection duration uses explicit decision-grid convention',
        'added/lost errors preserve full planned reference denominator']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=V22 / 'validation')
    parser.add_argument('--validation-root', type=Path, default=HERE / 'validation')
    parser.add_argument('--output', type=Path, default=HERE / 'evaluation')
    parser.add_argument('--baseline-only', action='store_true', help='Reproduce both V22 baselines before candidate results exist')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    qa = self_check()
    if args.self_check:
        print(json.dumps(qa)); return
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing evidence is never overwritten')
    evidence = core.Evidence()
    evidence.register(Path(__file__))
    evidence.register(V22 / 'evaluate_v22.py')
    criteria_path = HERE / 'evaluation_upgrade_criteria.json'
    assert sha(criteria_path) == CRITERIA_SHA
    criteria = evidence.json(criteria_path)
    alignment = evidence.json(ROOT / 'rppg_data1_20260909/alignment_protocol.json')
    assert alignment['video_start_utc_ns'] == 1788939171166718500
    assert alignment['sensitivity_shifts_s'] == [-5, -2, -1, 0, 1, 2, 5]
    frozen22 = evidence.json(args.baseline_root / 'protocol_before_validation.json')
    assert sha(ROOT / 'rppg_data1_20260909/alignment_protocol.json') == frozen22['data1_alignment_protocol_sha256']
    for filename, expected in frozen22['source_hashes'].items():
        assert sha(evidence.register(V22 / filename)) == expected
    frozen23, replay_qa, runs, tests = None, {'passed': False, 'reason': 'baseline-only precheck'}, [], {}
    versions = BASELINES if args.baseline_only else VERSIONS
    if not args.baseline_only:
        frozen23 = evidence.json(args.validation_root / 'protocol_before_validation.json')
        assert set(frozen23['cases']) == set(CASES)
        assert frozen23['data1_alignment_protocol_sha256'] == frozen22['data1_alignment_protocol_sha256']
        assert frozen23.get('upgrade_criteria_sha256', frozen23.get('criteria_sha256')) == CRITERIA_SHA
        for filename, expected in frozen23['source_hashes'].items():
            assert sha(evidence.register(HERE / filename)) == expected
        for case in CASES:
            for key in ('sha256', 'bytes'):
                assert frozen23['cases'][case][key] == frozen22['cases'][case][key]
            assert frozen23['geometry_caches'][case]['sha256'] == \
                sha(evidence.register(args.baseline_root / 'trimmed_gap10' / case / 'frame_trace.csv'))
        runs = evidence.json(args.validation_root / 'runs.json')
        assert len(runs) == 24 and all(row['exit_code'] == 0 for row in runs)
        assert {(row['case'], row['mode']) for row in runs} == set(itertools.product(CASES, CANDIDATES))
        tests = evidence.json(args.validation_root / 'tests.json')
        assert tests['failures'] == tests['errors'] == 0 and tests['run'] > 0
        assert tests['source_hashes'] == frozen23['source_hashes']
        replay_qa = check_receipt(args.validation_root / 'waveform_replay_qa.json', evidence, args)
    prior_metrics = evidence.csv(V22 / 'evaluation/metrics.csv')
    prior_wave = evidence.csv(V22 / 'evaluation/waveform_metrics.csv')
    tables = {name: [] for name in ('metrics', 'common_metrics', 'paired_metrics', 'deltas', 'continuity',
        'waveform_metrics', 'waveform_deltas', 'waveform_windows', 'aligned_windows', 'reference_windows',
        'sensitivity', 'sensitivity_common', 'sensitivity_pairs', 'waveform_sensitivity', 'waveform_pair_sensitivity',
        'geometry_audit', 'frontend_metrics', 'pixel_source_metrics', 'baseline_reproduction')}
    case_metadata = {}
    pairs = list(itertools.combinations(versions, 2))
    for case in CASES:
        data, geometry_rows = load_case(case, versions, args, evidence, frozen22, frozen23)
        tables['geometry_audit'].extend(geometry_rows)
        base = data['trimmed_gap10']
        for version, item in data.items():
            trace = item['trace']
            front = dict(case=case, version=version, frames=item['frames'])
            for col in ('face_detected', 'flow_available', 'rgb_valid', 'forehead_valid', 'left_cheek_valid', 'right_cheek_valid'):
                mask = core.bools(trace[col], col)
                front[col + '_count'] = int(mask.sum())
                front[col + '_pct'] = 100 * mask.mean()
            tables['frontend_metrics'].append(front)
            for column in trace:
                if column.endswith('_pixel_source'):
                    counts = trace[column].fillna('<missing_field>').astype(str).value_counts()
                    for source, count in counts.items():
                        tables['pixel_source_metrics'].append(dict(case=case, version=version,
                            roi=column.removesuffix('_pixel_source'), source=source, frames=item['frames'],
                            count=int(count), all_video_frames_pct=100 * int(count) / item['frames']))
            for variant in VARIANTS:
                tables['continuity'].append(dict(case=case, version=version, variant=variant,
                                                **continuity_row(item['variants'][variant])))
        shifts = alignment['sensitivity_shifts_s'] if case == 'data1' else [0]
        for shift in shifts:
            reference, ref_detail = core.reference_for(case, base['centers'], shift, evidence, alignment)
            for i, center in enumerate(base['centers']):
                tables['reference_windows'].append(dict(case=case, shift_s=shift, window_index=i, time_s=center,
                    reference_bpm=reference[i], reference_kind=core.REFERENCE_KIND.get(case, 'none'),
                    reference_eligible=bool(np.isfinite(reference[i])), **ref_detail[i],
                    video_window_center_utc_ns=alignment['video_start_utc_ns'] + round((float(center) + shift) * 1e9)
                        if case == 'data1' else None))
            for variant in VARIANTS:
                common = np.logical_and.reduce([data[v]['variants'][variant]['accepted'] for v in versions])
                own_rows = {}
                scores = {v: core.wave_scores(data[v]['variants'][variant], reference) for v in versions}
                common_snr = np.logical_and.reduce([np.isfinite(scores[v]) for v in versions])
                for version, item in data.items():
                    d = item['variants'][variant]
                    for estimator, column in ESTIMATORS.items():
                        keys = dict(case=case, version=version, variant=variant, estimator=estimator,
                            branch=f'{version}/{variant}/{estimator}', shift_s=shift,
                            reference_kind=core.REFERENCE_KIND.get(case, 'none'))
                        prediction = d['hr'][column].to_numpy(float)
                        own = dict(**keys, scope='own', **core.metrics(prediction, d['accepted'], reference))
                        shared = dict(**keys, scope='strict_common_all_versions_same_variant',
                                      **core.metrics(prediction, common, reference))
                        own_rows[(version, estimator)] = own
                        if case == 'data1':
                            tables['sensitivity'].append(own)
                            tables['sensitivity_common'].append(shared)
                        if shift == 0:
                            tables['metrics'].append(own)
                            tables['common_metrics'].append(shared)
                            if variant == 'fusion' and version in BASELINES:
                                audit_baseline(case, version, estimator, own, prior_metrics)
                                tables['baseline_reproduction'].append(dict(case=case, version=version,
                                    estimator=estimator, metrics_reproduced=True))
                            for i, center in enumerate(base['centers']):
                                valid = bool(d['accepted'][i] and np.isfinite(reference[i]))
                                tables['aligned_windows'].append(dict(**keys, window_index=i, time_s=center,
                                    actual_window_start_s=base['starts'][i] / base['fps'],
                                    actual_window_end_s=(base['starts'][i] + base['window']) / base['fps'],
                                    prediction_bpm=prediction[i] if d['accepted'][i] else np.nan,
                                    diagnostic_raw_peak_bpm=prediction[i] if not d['accepted'][i] else np.nan,
                                    reference_bpm=reference[i],
                                    valid_output=bool(d['accepted'][i]), paired_valid=valid,
                                    all_versions_common_output=bool(common[i]),
                                    error_bpm=prediction[i] - reference[i] if valid else np.nan,
                                    status=d['hr'].status.iloc[i]))
                    for scope, mask in [('own', np.isfinite(scores[version])),
                                        ('strict_common_all_versions_same_variant', common_snr)]:
                        row = dict(case=case, version=version, variant=variant, scope=scope, shift_s=shift,
                            reference_kind=core.REFERENCE_KIND.get(case, 'none'), Nref=int(np.isfinite(reference).sum()),
                            **waveform_row(d, scores[version], mask))
                        if case == 'data1':
                            tables['waveform_sensitivity'].append(row)
                        if shift == 0:
                            tables['waveform_metrics'].append(row)
                            if variant == 'fusion' and scope == 'own' and version in BASELINES:
                                old = prior_wave[prior_wave.case.eq(case) & prior_wave.version.eq(version)
                                                 & prior_wave.scope.eq('own')].iloc[0]
                                for key in ('snr_mean_db', 'snr_median_db', 'finite_sample_pct'):
                                    np.testing.assert_allclose(row[key] if row[key] is not None else np.nan,
                                        old[key], rtol=0, atol=1e-8, equal_nan=True)
                    if shift == 0:
                        for i, center in enumerate(base['centers']):
                            tables['waveform_windows'].append(dict(case=case, version=version, variant=variant,
                                window_index=i, time_s=center, accepted=bool(d['accepted'][i]),
                                finite_segment=bool(d['finite_windows'][i]), reference_bpm=reference[i],
                                snr_ref_h1_db=scores[version][i], all_versions_common_snr=bool(common_snr[i]),
                                waveform_generated=bool(d['generated'][i]) if d['generated'] is not None else None))
                for before, after in pairs:
                    for estimator in ESTIMATORS:
                        paired = pair_rows(case, variant, estimator, before, after, data, reference, shift)
                        if case == 'data1':
                            tables['sensitivity_pairs'].extend(paired)
                        if shift == 0:
                            tables['paired_metrics'].extend(paired)
                            own_a, own_b = own_rows[(before, estimator)], own_rows[(after, estimator)]
                            pa, pb, added, lost = paired
                            assert own_b['Nwithin5'] - own_a['Nwithin5'] == \
                                pb['Nwithin5'] - pa['Nwithin5'] + added['Nwithin5'] - lost['Nwithin5']
                            delta = dict(case=case, variant=variant, estimator=estimator, before=before, after=after,
                                Ncommon_output=pa['Ncommon_output'], Nadded_output=pa['Nadded_output'],
                                Nlost_output=pa['Nlost_output'])
                            for key in ('MAE_bpm', 'RMSE_bpm', 'Bias_bpm', 'P5_valid_pct', 'R5_all_reference_pct', 'C_out_pct'):
                                delta['own_delta_' + key] = core.difference(own_b[key], own_a[key])
                                delta['common_delta_' + key] = core.difference(pb[key], pa[key])
                            tables['deltas'].append(delta)
                    mask = np.isfinite(scores[before]) & np.isfinite(scores[after])
                    changes = scores[after][mask] - scores[before][mask]
                    row = dict(case=case, variant=variant, before=before, after=after, shift_s=shift,
                        Npaired_snr=len(changes), mean_paired_delta_snr_db=float(changes.mean()) if len(changes) else None,
                        median_paired_delta_snr_db=float(np.median(changes)) if len(changes) else None,
                        positive_delta_pct=100 * float(np.mean(changes > 1e-9)) if len(changes) else None)
                    if case == 'data1':
                        tables['waveform_pair_sensitivity'].append(row)
                    if shift == 0:
                        tables['waveform_deltas'].append(row)
        case_metadata[case] = dict(frames=base['frames'], fps=base['fps'], duration_s=base['frames'] / base['fps'],
            actual_window_s=base['window'] / base['fps'], actual_step_s=base['step'] / base['fps'],
            Nplanned=len(base['centers']), directories={v: str(d['directory']) for v, d in data.items()},
            reference_kind=core.REFERENCE_KIND.get(case, 'none'))
    assert len(tables['metrics']) == len(CASES) * len(versions) * len(VARIANTS) * len(ESTIMATORS)
    assert len(tables['baseline_reproduction']) == 24
    assert len(tables['sensitivity']) == 7 * len(versions) * len(VARIANTS) * len(ESTIMATORS)
    gate = {'evaluated': False, 'reason': 'baseline-only precheck'}
    if not args.baseline_only:
        gate = dict(evaluated=True,
            default=upgrade_gate(tables, criteria, 'trimmed_gap10', 'tracking_screened_gap10', replay_qa['passed']),
            separate_continuity_option=upgrade_gate(tables, criteria, 'trimmed_gap15', 'tracking_screened_gap15', replay_qa['passed']))
    evidence.verify()
    args.output.mkdir(parents=True, exist_ok=False)
    output_manifest = {}
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        integer_times = [col for col in frame if col.endswith('_utc_ns')]
        for col in integer_times:
            frame[col] = pd.array([row.get(col) for row in rows], dtype='Int64')
        path = args.output / f'{name}.csv'
        frame.to_csv(path, index=False)
        if len(frame.columns):
            reread = pd.read_csv(path, dtype={col: 'Int64' for col in integer_times})
            assert frame.shape == reread.shape and list(frame) == list(reread)
            np.testing.assert_array_equal(frame.isna(), reread.isna())
            for col in integer_times:
                pd.testing.assert_series_equal(frame[col], reread[col], check_names=False)
        output_manifest[path.name] = dict(rows=len(frame), sha256=sha(path), columns=list(frame))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluator_sha256=sha(Path(__file__)),
        frozen_v22_evaluator_sha256=CORE_SHA, upgrade_criteria_sha256=CRITERIA_SHA,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        cases=list(CASES), versions=list(versions), variants=list(VARIANTS), case_metadata=case_metadata,
        primary_variant='fusion', primary_shift_s=0, alignment_protocol=alignment,
        source_protocol_v22=frozen22, source_protocol_v23=frozen23, run_records=runs,
        upgrade_criteria=criteria, upgrade_gate=gate,
        QA=dict(self_check=qa, baseline_metrics_reproduced=True, baseline_SNR_reproduced=True,
            geometry_flags_and_sources_exact_and_numeric_values_verified=True,
            gap_pair_trace_values_verified_with_1e_9_serialization_tolerance=True,
            tests=tests, waveform_replay=replay_qa, input_hashes_unchanged=True,
            CSV_shape_missingness_and_integer_timestamp_roundtrip=True),
        definitions=dict(
            error='Predicted minus reference BPM. Own windows differ; paired common/new/lost retain full Nplanned and Nref denominators.',
            P5='100 * Nwithin5 / Nvalid; R5 = 100 * Nwithin5 / Nref. No reference produces NA.',
            primary_HR='Final HR read from saved waveform output; offline_ridge and local_peak both reported and guarded.',
            rejected_diagnostics='Fusion rejected HR columns stay missing. POS/CHROM local-peak column may retain a pre-gate diffuse-spectrum diagnostic; accepted=False always excludes it from coverage and scoring, and aligned output separates that diagnostic from prediction.',
            SNR='Frozen V22 reference-H1 periodic Hann full window, nfft>=8192, .7-3.5Hz, reference H1 +/- .1Hz versus remaining band; accepted HR, finite full waveform and eligible reference required.',
            common='Strict common across all versions is descriptive; upgrade uses pairwise common against its same-gap baseline.',
            waveform_coverage='Finite saved base samples / all original video frames. It includes marked interpolation and neighboring-window contributions and is not physiological accuracy.',
            interruption='HR longest rejected grid run is rejected decision count * actual step duration; first-to-last rejected-center span is also reported. Waveform longest missing run is missing sample count / FPS. Both include boundary missing runs.',
            support='Accepted HR support intervals are unioned; overlapping ten-second windows are never summed into validated duration.',
            proxies='Peak concentration is an engineering frequency concentration proxy, not a probability of correctness.',
            frontend='Geometry source/flags and missingness are identical; numeric fields allow only absolute 1e-9 CSV round-trip tolerance, with maxima disclosed. Pixel-source percentages use all video frames per ROI and distinguish tracked ratios, baseline fallback, reset, and missing samples.',
            LoA='Bias +/- 1.96 sample SD, descriptive only; highly overlapping windows are not independent.'),
        limitations=[
            'All six sources were previously inspected development/regression data; not held-out generalization evidence.',
            'Five human recordings plus one synthetic generator do not establish six independent participants.',
            'UBFC uses paired device HR, not ECG. data1 absolute synchronization remains estimated.',
            'All seven data1 shifts are disclosed without selecting a best shift or fitting timing.',
            'No-reference videos have no measurable HR accuracy. Synthetic 72 bpm is an engineering control.',
            'No qualified waveform morphology reference; r_wave is NA. Coverage or reference-H1 SNR does not establish morphology recovery.'],
        input_manifest=evidence.manifest, output_manifest=output_manifest)
    (args.output / 'summary.json').write_text(json.dumps(core.clean(report), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'upgrade_gate.json').write_text(json.dumps(core.clean(gate), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    primary = pd.DataFrame(tables['metrics'])
    print(primary[primary.variant.eq('fusion') & primary.estimator.eq('offline_ridge')]
          [['case', 'version', 'Noutput', 'Nplanned', 'MAE_bpm', 'RMSE_bpm', 'R5_all_reference_pct']].to_string(index=False))
    print(json.dumps(core.clean(gate), ensure_ascii=True))
    print(f'Completed: {args.output}')


if __name__ == '__main__':
    main()
