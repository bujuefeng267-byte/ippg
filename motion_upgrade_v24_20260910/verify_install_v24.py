"""Fresh UBFC + synthetic72 verification of the separately installed V24 entry.

No arguments: read-only plan. --self-check: in-memory comparator tests only.
--run: two full original-video executions, no RGB/geometry/tracking cache.
The comparison contract below is fixed before running. FB median diagnostics
remain fully visible and may fail strict_all_fields_match independently of
core_outputs_match. No other numeric tolerance is relaxed after seeing output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from deploy_v24 import (HERE, PROJECT, CODE_NAME, RESULT_NAME, LAUNCHER,
                       CANDIDATE, LAUNCHER_ARGS, checked, digest, load,
                       require, require_wsl, verify)

CASES = ('ubfc', 'synthetic72')
ATOL = 1e-9
ROIS = ('forehead', 'left_cheek', 'right_cheek')
DIAGNOSTIC_ONLY = {f'{roi}_pixel_fb_error_median' for roi in ROIS}
FB_DIAGNOSTICS = DIAGNOSTIC_ONLY | {f'{roi}_pixel_fb_rejected' for roi in ROIS}
CSVS = ('frame_trace.csv', 'roi_waveforms.csv', 'pos_waveform.csv',
        'chrom_waveform.csv', 'fusion_waveform.csv', 'pos_heart_rate.csv',
        'chrom_heart_rate.csv', 'fusion_heart_rate.csv', 'fusion_proposals.csv',
        'fusion_diagnostics.csv', 'baseline_roi_waveforms.csv', 'branch_routing.csv',
        'baseline_branch_proposals.csv', 'baseline_branch_waveform.csv',
        'tracked_branch_proposals.csv', 'tracked_branch_waveform.csv')
SUMMARY_RUNTIME_FIELDS = {'elapsed_processing_seconds', 'extraction_path', 'opencv_threads'}
CONFIG_PATH_FIELDS = {'output', 'cache', 'geometry_cache', 'tracking_cache'}
CONTRACT = dict(schema_version=1, candidate=CANDIDATE, cases=list(CASES),
    launcher_args=LAUNCHER_ARGS, files=list(CSVS), numeric_atol=ATOL, numeric_rtol=0,
    discrete_values='exact', missingness='exact in every field including diagnostics',
    schema='row count and ordered column names exact; complete CSV inventory exact',
    core_outputs='All CSV values, including actual waveforms, HR, proposals, routing, source/time, masks, baseline and new RGB, except finite differences in the three named FB medians.',
    diagnostic_only_numeric_columns=sorted(DIAGNOSTIC_ONLY),
    diagnostic_reporting='Every frame/value of all six FB median/rejected columns exported; count/flag changes still fail core output matching.',
    no_cache=True, full_video=True, no_reference_during_inference=True,
    summary_runtime_fields_separate=sorted(SUMMARY_RUNTIME_FIELDS),
    config_path_fields_separate=sorted(CONFIG_PATH_FIELDS),
    pass_fields_separate=['core_outputs_match', 'diagnostic_values_match',
                          'strict_all_fields_match', 'runtime_integrity_passed'],
    no_automatic_default_promotion=True)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def compare_table(left, right, filename):
    report = dict(expected_shape=list(left.shape), actual_shape=list(right.shape),
                  columns_equal=list(left) == list(right), columns={})
    if left.shape != right.shape or list(left) != list(right):
        return {**report, 'core_outputs_match': False, 'strict_all_fields_match': False,
                'diagnostic_values_match': False, 'reason': 'schema/row count mismatch'}, []
    diagnostic_rows = []
    for name in left:
        a, b = left[name], right[name]
        missing_a, missing_b = a.isna().to_numpy(), b.isna().to_numpy()
        missing_mismatch = missing_a != missing_b
        present = ~missing_a & ~missing_b
        numeric = is_numeric_dtype(a.dtype) and is_numeric_dtype(b.dtype)
        discrete = (is_bool_dtype(a.dtype) or is_bool_dtype(b.dtype) or
                    is_integer_dtype(a.dtype) or is_integer_dtype(b.dtype) or not numeric)
        errors = missing_mismatch.copy()
        maximum = None
        delta = np.full(len(a), np.nan)
        if discrete:
            errors[present] |= a[present].to_numpy() != b[present].to_numpy()
            if is_bool_dtype(a.dtype) != is_bool_dtype(b.dtype):
                errors[present] = True
        else:
            av, bv = a.to_numpy(float), b.to_numpy(float)
            finite = present & np.isfinite(av) & np.isfinite(bv)
            delta[finite] = np.abs(av[finite] - bv[finite])
            maximum = float(delta[finite].max()) if finite.any() else None
            errors[present] |= ~np.isclose(av[present], bv[present], atol=ATOL, rtol=0)
        numeric_diagnostic = filename == 'frame_trace.csv' and name in DIAGNOSTIC_ONLY
        # Missing diagnostic values remain part of the core missingness contract.
        if numeric_diagnostic:
            finite_pair = np.zeros(len(a), dtype=bool)
            if numeric and not is_bool_dtype(a.dtype) and not is_bool_dtype(b.dtype):
                finite_pair = present & np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
            core_pass = not (missing_mismatch | (errors & ~finite_pair)).any()
        else:
            core_pass = not errors.any()
        indexes = np.flatnonzero(errors)
        report['columns'][name] = dict(passed=not errors.any(), core_passed=core_pass,
            diagnostic_numeric_only=numeric_diagnostic,
            comparison='exact' if discrete else 'absolute_tolerance_1e-9',
            mismatch_count=len(indexes), missingness_mismatches=int(missing_mismatch.sum()),
            max_abs_difference=maximum,
            first_differences=[dict(row=int(i), expected=repr(a.iloc[i]), actual=repr(b.iloc[i]))
                               for i in indexes[:5]])
        if filename == 'frame_trace.csv' and name in FB_DIAGNOSTICS:
            for i in range(len(a)):
                diagnostic_rows.append(dict(row=i, frame=left.frame.iloc[i],
                    time_s=left.time_s.iloc[i], field=name, expected=a.iloc[i], actual=b.iloc[i],
                    missingness_equal=not missing_mismatch[i],
                    absolute_difference=delta[i], matches_contract=not errors[i],
                    finite_numeric_differences_diagnostic_only=numeric_diagnostic))
    columns = report['columns']
    report['core_outputs_match'] = all(v['core_passed'] for v in columns.values())
    report['strict_all_fields_match'] = all(v['passed'] for v in columns.values())
    report['diagnostic_values_match'] = all(v['passed'] for k, v in columns.items()
                                           if filename == 'frame_trace.csv' and k in FB_DIAGNOSTICS)
    report['failed_columns'] = [k for k, v in columns.items() if not v['passed']]
    return report, diagnostic_rows


def compare_json(a, b, path=''):
    errors = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [dict(path=path, reason='keys differ', expected_keys=sorted(a), actual_keys=sorted(b))]
        for key in a:
            errors.extend(compare_json(a[key], b[key], path + '/' + key))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [dict(path=path, reason='list length differs')]
        for i, (av, bv) in enumerate(zip(a, b)):
            errors.extend(compare_json(av, bv, path + '/' + str(i)))
    elif isinstance(a, bool) or isinstance(b, bool):
        if type(a) is not type(b) or a != b:
            errors.append(dict(path=path, reason='Boolean differs', expected=a, actual=b))
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        ok = a == b if isinstance(a, int) and isinstance(b, int) else np.isclose(a, b, atol=ATOL, rtol=0)
        if not ok:
            errors.append(dict(path=path, reason='numeric differs', expected=a, actual=b))
    elif a != b:
        errors.append(dict(path=path, reason='value differs', expected=a, actual=b))
    return errors


def compare_outputs(reference, actual, sources):
    reference_csv = {p.name for p in reference.glob('*.csv')}
    actual_csv = {p.name for p in actual.glob('*.csv')}
    inventory_match = reference_csv == actual_csv == set(CSVS)
    checks, diagnostic_rows = {}, []
    for filename in CSVS:
        if not (reference / filename).is_file() or not (actual / filename).is_file():
            checks[filename] = dict(core_outputs_match=False, strict_all_fields_match=False,
                                    diagnostic_values_match=False, reason='file missing')
            continue
        checks[filename], rows = compare_table(pd.read_csv(reference / filename),
                                               pd.read_csv(actual / filename), filename)
        diagnostic_rows.extend(rows)
    expected, measured = load(reference / 'summary.json'), load(actual / 'summary.json')
    ignored_summary = SUMMARY_RUNTIME_FIELDS | {'config'}
    summary_errors = compare_json({k: v for k, v in expected.items() if k not in ignored_summary},
                                  {k: v for k, v in measured.items() if k not in ignored_summary})
    summary_errors.extend(compare_json(
        {k: v for k, v in expected['config'].items() if k not in CONFIG_PATH_FIELDS},
        {k: v for k, v in measured.get('config', {}).items() if k not in CONFIG_PATH_FIELDS}, '/config'))
    for name in ('cache', 'geometry_cache', 'tracking_cache', 'reference_ubfc', 'max_seconds'):
        if measured.get('config', {}).get(name) is not None:
            summary_errors.append(dict(path='/config/' + name, reason='fresh/reference-free/full-video requirement failed'))
    if measured.get('source_hashes') != sources or measured.get('reference_identity') is not None:
        summary_errors.append(dict(path='/source_hashes', reason='source/reference provenance differs'))
    if measured.get('extraction_path') != 'fresh_detector_and_pixels':
        summary_errors.append(dict(path='/extraction_path', reason='fresh detector/pixel path not confirmed'))
    metadata = load(actual / 'frame_trace.json')
    metadata_errors = []
    if metadata.get('trace_sha256') != digest(actual / 'frame_trace.csv'):
        metadata_errors.append('Trace checksum differs from its own metadata')
    if metadata.get('identity') != measured['identity'] or metadata.get('n_frames') != measured['frames']:
        metadata_errors.append('Trace identity/frame count differs from summary')
    non_csv_ok = inventory_match and not summary_errors and not metadata_errors
    core = non_csv_ok and all(v['core_outputs_match'] for v in checks.values())
    strict = non_csv_ok and all(v['strict_all_fields_match'] for v in checks.values())
    diagnostic = all(v['diagnostic_values_match'] for v in checks.values())
    return dict(core_outputs_match=core, diagnostic_values_match=diagnostic,
        strict_all_fields_match=strict, csv_inventory_match=inventory_match,
        expected_csv_inventory=sorted(reference_csv), actual_csv_inventory=sorted(actual_csv),
        csv_checks=checks, summary_errors=summary_errors, metadata_errors=metadata_errors,
        runtime_metadata={k: dict(validation=expected.get(k), fresh=measured.get(k))
                          for k in sorted(SUMMARY_RUNTIME_FIELDS)},
        diagnostic_only_fields=sorted(DIAGNOSTIC_ONLY)), diagnostic_rows


def self_check():
    a = pd.DataFrame(dict(frame=[0, 1], time_s=[0., .1], source=['mesh', 'flow'],
        rgb_valid=[True, False], baseline_r=[1., np.nan], r=[2., np.nan],
        forehead_pixel_fb_error_median=[.1, .2], forehead_pixel_fb_rejected=[0, 1]))
    result, values = compare_table(a, a.copy(), 'frame_trace.csv')
    require(result['core_outputs_match'] and result['strict_all_fields_match'], 'Equal table failed')
    require(len(values) == 4, 'Full diagnostic rows were not retained')
    b = a.copy(); b.loc[0, 'forehead_pixel_fb_error_median'] += .001
    result, values = compare_table(a, b, 'frame_trace.csv')
    require(result['core_outputs_match'] and not result['strict_all_fields_match'] and
            not result['diagnostic_values_match'], 'Diagnostic difference was hidden or misclassified')
    for field, value in [('rgb_valid', False), ('source', 'none'), ('frame', 9),
                         ('time_s', .001), ('baseline_r', 1.01), ('r', 2.01),
                         ('forehead_pixel_fb_rejected', 1), ('forehead_pixel_fb_error_median', np.nan)]:
        b = a.copy(); b.loc[0, field] = value
        require(not compare_table(a, b, 'frame_trace.csv')[0]['core_outputs_match'],
                f'Core mismatch accepted: {field}')
    b = a.copy(); b.loc[0, 'r'] += .5e-9
    require(compare_table(a, b, 'frame_trace.csv')[0]['core_outputs_match'], 'Fixed tolerance failed')
    b = a.copy(); b.loc[0, 'r'] += 2e-9
    require(not compare_table(a, b, 'frame_trace.csv')[0]['core_outputs_match'], 'Large numeric mismatch accepted')
    require(not compare_table(a, a.iloc[:1], 'frame_trace.csv')[0]['core_outputs_match'], 'Shortened video accepted')
    require(compare_json({'a': True}, {'a': 1}), 'Boolean/integer confusion')
    require(compare_json({'a': None}, {'a': 0}), 'Missing/zero confusion')
    b = a.copy(); b['frame'] = b.frame.astype(float); b.loc[0, 'frame'] += .5e-9
    require(not compare_table(a, b, 'frame_trace.csv')[0]['core_outputs_match'], 'Discrete identity tolerated as float')
    b = a.copy(); b.loc[0, 'forehead_pixel_fb_error_median'] = np.inf
    require(not compare_table(a, b, 'frame_trace.csv')[0]['core_outputs_match'], 'Nonfinite diagnostic silently exempted')
    return dict(passed=True, checks=19, video_inference_run=False, filesystem_writes=False,
                diagnostic_exception_tested=True, core_missingness_and_discrete_fail_closed=True)


def run():
    require_wsl()
    record_path = HERE / 'deployment.json'
    record = load(record_path)
    before = verify(record)
    installed_results = checked(PROJECT, RESULT_NAME)
    installed_code = checked(PROJECT, CODE_NAME)
    require(digest(installed_code / 'verify_install_v24.py') == digest(__file__),
            'Verifier changed since installation')
    require(digest(installed_code / 'deploy_v24.py') == digest(HERE / 'deploy_v24.py'),
            'Installer utilities changed since installation')
    protocol = load(installed_results / 'validation/protocol_before_validation.json')
    require(record['source_hashes'] == protocol['source_hashes'] and
            protocol['default_candidate'] == CANDIDATE, 'Installed candidate/source freeze differs')
    receipt_path = HERE / 'deployment_verification.json'
    base = checked(PROJECT, RESULT_NAME + '/installed_fresh')
    require(not receipt_path.exists() and not base.exists(), 'Existing fresh output/receipt is immutable')
    source_inputs, reference_inputs = {}, {}
    for case in CASES:
        item = protocol['cases'][case]
        video = Path(item['video'])
        require(video.is_file() and digest(video) == item['sha256'], f'Original video changed: {case}')
        source_inputs[case] = dict(path=str(video), sha256=item['sha256'], bytes=video.stat().st_size)
        reference = installed_results / 'validation' / CANDIDATE / case
        for name in (*CSVS, 'summary.json', 'frame_trace.json'):
            reference_inputs[str(reference / name)] = digest(reference / name)
    # This immutable contract is persisted BEFORE either video is run.
    base.mkdir(exist_ok=False)
    write_json(base / 'comparison_contract.json', dict(**CONTRACT,
        frozen_utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=digest(__file__),
        installation_manifest_sha256=digest(record_path), source_hashes=record['source_hashes'],
        validation_input_hashes=reference_inputs, original_video_hashes=source_inputs))
    cases = []
    for case in CASES:
        output = checked(base, case)
        reference = installed_results / 'validation' / CANDIDATE / case
        command = [str(PROJECT / LAUNCHER), source_inputs[case]['path'], '--output', str(output)]
        row = dict(case=case, candidate=CANDIDATE, command=command, source_video=source_inputs[case],
            used_cache=False, used_geometry_cache=False, used_tracking_cache=False,
            core_outputs_match=False, diagnostic_values_match=False, strict_all_fields_match=False)
        print(json.dumps({'event': 'fresh_start', 'case': case}), flush=True)
        started = time.perf_counter()
        try:
            with (base / (case + '.log')).open('x', encoding='utf-8') as log:
                proc = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            row['exit_code'] = proc.returncode
            if proc.returncode == 0:
                result, diagnostic_rows = compare_outputs(reference, output, record['source_hashes'])
                row.update(result)
                diagnostic_path = base / (case + '_fb_diagnostics_all_values.csv')
                pd.DataFrame(diagnostic_rows).to_csv(diagnostic_path, index=False, mode='x')
                row['full_fb_diagnostics'] = dict(path=str(diagnostic_path),
                    rows=len(diagnostic_rows), sha256=digest(diagnostic_path))
                row['output_hashes'] = {n: digest(output / n) for n in (*CSVS, 'summary.json', 'frame_trace.json')}
            else:
                row['error'] = 'Installed entry failed; preserved log contains details'
        except Exception as error:
            row.setdefault('exit_code', None)
            row['error'] = f'{type(error).__name__}: {error}'
        row['elapsed_s'] = time.perf_counter() - started
        write_json(base / (case + '_comparison.json'), row)
        cases.append(row)
        print(json.dumps(clean({k: row[k] for k in ('case', 'exit_code', 'core_outputs_match',
              'diagnostic_values_match', 'strict_all_fields_match')})), flush=True)
    after_error = None
    try:
        after = verify(record)
    except Exception as error:
        after = {'verified': False}; after_error = f'{type(error).__name__}: {error}'
    references_unchanged = all(digest(p) == h for p, h in reference_inputs.items())
    videos_unchanged = all(digest(v['path']) == v['sha256'] for v in source_inputs.values())
    integrity = after['verified'] and references_unchanged and videos_unchanged and all(r['exit_code'] == 0 for r in cases)
    core = integrity and all(r['core_outputs_match'] for r in cases)
    strict = integrity and all(r['strict_all_fields_match'] for r in cases)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate=CANDIDATE,
        runtime_integrity_passed=integrity, core_outputs_match=core,
        diagnostic_values_match=all(r['diagnostic_values_match'] for r in cases),
        strict_all_fields_match=strict, numeric_atol=ATOL, numeric_rtol=0,
        comparison_contract_sha256=digest(base / 'comparison_contract.json'),
        installation_manifest_sha256=digest(record_path), verifier_sha256=digest(__file__),
        source_hashes=record['source_hashes'], installed_hashes_before=before,
        installed_hashes_after=after, installed_hash_error=after_error,
        original_validation_inputs_unchanged=references_unchanged,
        original_videos_unchanged=videos_unchanged, cases=cases,
        performance_gate=record['performance_gate'], default_changed=False,
        interpretation='Exit status follows runtime integrity and core matching. strict_all_fields_match remains independent and must never be described as true when diagnostics differ. Performance eligibility is separate; no default is promoted.',
        limitations=['Two fresh runs verify installation, not held-out accuracy or generalization.',
                     'No failed run is replaced by cached replay.',
                     'Every FB diagnostic frame/value is retained, including matched values.'])
    write_json(receipt_path, report)
    write_json(base / 'verification.json', report)
    print(json.dumps({k: report[k] for k in ('runtime_integrity_passed', 'core_outputs_match',
          'diagnostic_values_match', 'strict_all_fields_match')}), flush=True)
    if not core:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    elif args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(dict(action='read_only_plan', modified=False, contract=CONTRACT,
            output=str(PROJECT / RESULT_NAME / 'installed_fresh'),
            installation_receipt_exists=(HERE / 'deployment.json').is_file()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
