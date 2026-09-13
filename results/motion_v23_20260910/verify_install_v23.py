"""Verify installed V2.3 by fresh, full-video extraction; never substitute cache replay.

Default invocation only prints a plan. Use --run after installation succeeds.
Discrete values and missingness must match exactly; finite numeric differences
are recorded and checked with fixed atol=1e-9, rtol=0. A fresh-path mismatch
produces a failed receipt and preserved diagnostics, not a changed tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
CASES = ('ubfc', 'synthetic72')
MODE = 'tracking_screened_gap10'
ATOL = 1e-9
CSVS = ('frame_trace.csv', 'roi_waveforms.csv', 'pos_waveform.csv',
        'chrom_waveform.csv', 'fusion_waveform.csv', 'pos_heart_rate.csv',
        'chrom_heart_rate.csv', 'fusion_heart_rate.csv', 'fusion_proposals.csv',
        'fusion_diagnostics.csv')
GEOMETRY = ('frame', 'time_s', 'source', 'face_detected', 'flow_available',
            'bridge_age_frames', 'face_x0', 'face_y0', 'face_x1', 'face_y1',
            'motion_x', 'motion_y')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_new_json(path, value):
    payload = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def safe_target(path):
    path = Path(path).absolute()
    require(path.resolve().is_relative_to(PROJECT.resolve()) and path.resolve() != PROJECT.resolve(),
            f'Path outside the explicit WSL project: {path}')
    probe = path
    while probe != PROJECT:
        require(not probe.is_symlink(), f'Symbolic link in verification target: {probe}')
        require(probe != probe.parent, f'Unexpected path root: {path}')
        probe = probe.parent
    return path


def verify_installed_hashes(record):
    """Check the exact installed evidence/code and preserved pre-existing files."""
    failures = []
    for collection in ('files', 'preserved_entry_hashes'):
        for relative, expected in record[collection].items():
            path = safe_target(PROJECT / relative)
            if not path.is_file() or digest(path) != expected:
                failures.append(f'{collection}: {relative}')
    launcher = safe_target(record['launcher'])
    if not launcher.is_file() or digest(launcher) != record['launcher_sha256']:
        failures.append('experimental launcher')
    return {'passed': not failures, 'failures': failures}


def compare_frames(expected, actual):
    result = dict(expected_shape=list(expected.shape), actual_shape=list(actual.shape),
                  columns_equal=list(expected) == list(actual), numeric_atol=ATOL,
                  numeric_rtol=0, discrete_and_missingness_exact=True, columns={})
    if expected.shape != actual.shape or list(expected) != list(actual):
        return {**result, 'passed': False, 'reason': 'row count or ordered schema differs'}
    for name in expected:
        left, right = expected[name], actual[name]
        missing_a, missing_b = left.isna().to_numpy(), right.isna().to_numpy()
        missing_mismatch = missing_a != missing_b
        present = ~missing_a & ~missing_b
        errors = missing_mismatch.copy()
        numeric = is_numeric_dtype(left.dtype) and is_numeric_dtype(right.dtype)
        discrete = (is_bool_dtype(left.dtype) or is_bool_dtype(right.dtype) or
                    (is_integer_dtype(left.dtype) and is_integer_dtype(right.dtype)) or not numeric)
        maximum = None
        if discrete:
            errors[present] |= left[present].to_numpy() != right[present].to_numpy()
            if is_bool_dtype(left.dtype) != is_bool_dtype(right.dtype):
                errors[present] = True
        else:
            a, b = left.to_numpy(float), right.to_numpy(float)
            finite = present & np.isfinite(a) & np.isfinite(b)
            maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
            errors[present] |= ~np.isclose(a[present], b[present], rtol=0, atol=ATOL, equal_nan=False)
        indices = np.flatnonzero(errors)
        examples = []
        for index in indices[:5]:
            examples.append(dict(row=int(index), expected=repr(left.iloc[index]), actual=repr(right.iloc[index])))
        result['columns'][name] = dict(passed=not len(indices), comparison='exact' if discrete else 'absolute_tolerance',
            missingness_mismatches=int(missing_mismatch.sum()), mismatch_count=int(len(indices)),
            max_abs_difference=maximum, first_differences=examples)
    result['passed'] = all(item['passed'] for item in result['columns'].values())
    result['failed_columns'] = [name for name, item in result['columns'].items() if not item['passed']]
    return result


def compare_json_values(expected, actual, path=''):
    errors = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [dict(path=path, reason='keys differ')]
        for key in expected:
            errors.extend(compare_json_values(expected[key], actual[key], path + '/' + key))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [dict(path=path, reason='list length differs')]
        for index, (left, right) in enumerate(zip(expected, actual)):
            errors.extend(compare_json_values(left, right, path + '/' + str(index)))
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            errors.append(dict(path=path, expected=expected, actual=actual, reason='Boolean mismatch'))
    elif isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        passed = expected == actual if isinstance(expected, int) and isinstance(actual, int) else \
            bool(np.isclose(expected, actual, rtol=0, atol=ATOL, equal_nan=False))
        if not passed:
            errors.append(dict(path=path, expected=expected, actual=actual, reason='numeric mismatch'))
    elif expected != actual:
        errors.append(dict(path=path, expected=expected, actual=actual, reason='value mismatch'))
    return errors


def compare_outputs(reference, actual, source_hashes):
    csv_checks = {}
    for filename in CSVS:
        if not (actual / filename).is_file():
            csv_checks[filename] = {'passed': False, 'reason': 'output missing'}
            continue
        csv_checks[filename] = compare_frames(pd.read_csv(reference / filename), pd.read_csv(actual / filename))
    expected, measured = load(reference / 'summary.json'), load(actual / 'summary.json')
    summary_keys = ('version', 'identity', 'source_hashes', 'fps', 'frames', 'effective_window_s',
                    'effective_step_s', 'requested_window_s', 'nominal_reference_window_s',
                    'reference_identity', 'observed_rgb_fraction', 'roi_observed_fraction',
                    'source_counts', 'fusion_config', 'variants', 'pixel_config', 'pixel_diagnostics')
    summary_errors = compare_json_values({key: expected[key] for key in summary_keys},
                                         {key: measured.get(key) for key in summary_keys})
    ignored_config = {'output', 'cache', 'geometry_cache'}
    summary_errors.extend(compare_json_values(
        {key: value for key, value in expected['config'].items() if key not in ignored_config},
        {key: value for key, value in measured['config'].items() if key not in ignored_config}, '/config'))
    for field in ('cache', 'geometry_cache', 'max_seconds', 'reference_ubfc'):
        if measured['config'].get(field) is not None:
            summary_errors.append(dict(path='/config/' + field, reason='fresh full-video reference-free path was not used'))
    if measured.get('source_hashes') != source_hashes or measured.get('reference_identity') is not None:
        summary_errors.append(dict(path='/source_hashes', reason='frozen/reference-free provenance mismatch'))
    metadata = load(actual / 'frame_trace.json')
    metadata_errors = []
    if metadata.get('trace_sha256') != digest(actual / 'frame_trace.csv'):
        metadata_errors.append('trace metadata checksum mismatch')
    if metadata.get('identity') != measured['identity'] or metadata.get('n_frames') != measured['frames']:
        metadata_errors.append('trace metadata identity/frame count mismatch')
    geometry_checks = csv_checks.get('frame_trace.csv', {}).get('columns', {})
    geometry_match = all(geometry_checks.get(column, {}).get('passed') is True for column in GEOMETRY)
    csv_match = all(item['passed'] for item in csv_checks.values())
    return dict(matches_validation=csv_match and not summary_errors and not metadata_errors,
                fresh_geometry_matches_validation=geometry_match, csv_checks=csv_checks,
                summary_errors=summary_errors, trace_metadata_errors=metadata_errors,
                expected_extraction_path='V23 original pixels with frozen V22 geometry cache',
                actual_extraction_path='fresh detector + fresh pixel tracking + frozen waveform/HR pipeline',
                ignored_noncomputational_differences=['output directory', 'geometry cache source path',
                    'processing wall time', 'generated tracking QA images and plot encoding'],
                note='A mismatch is preserved. It is not automatically explained away as detector randomness or replaced by cache replay.')


def self_check():
    a = pd.DataFrame({'frame': [0, 1, 2], 'accepted': [True, False, True],
                      'value': [1., np.nan, 3.], 'source': ['mesh', 'none', 'flow']})
    checks = []
    require(compare_frames(a, a.copy())['passed'], 'Equal data failed')
    checks.append('equal table')
    b = a.copy(); b.loc[0, 'value'] += .5e-9
    require(compare_frames(a, b)['passed'], 'Within-tolerance finite number failed')
    checks.append('finite round-trip tolerance')
    for column, row, value in [('accepted', 1, True), ('frame', 2, 3),
                               ('source', 0, 'flow'), ('value', 1, 0.), ('value', 0, 1.+2e-9)]:
        b = a.copy(); b.loc[row, column] = value
        require(not compare_frames(a, b)['passed'], f'Mismatch was accepted: {column}')
    checks.append('flags, frame identity, source, missingness and above-tolerance values rejected')
    b = a.copy(); b['accepted'] = b.accepted.astype(int)
    require(not compare_frames(a, b)['passed'], 'Integer encoding silently substituted for Boolean flags')
    require(not compare_frames(a, a.iloc[:-1])['passed'], 'Shortened stream accepted')
    require(compare_json_values({'flag': True}, {'flag': 1}), 'Boolean/numeric equality confused')
    checks.append('shortened streams and Boolean/numeric confusion rejected')
    return {'passed': True, 'checks': checks, 'video_inference_run': False, 'WSL_writes': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2)); return
    if not args.run:
        print(json.dumps(dict(action='read_only_plan', cases=list(CASES), mode=MODE,
            inference='fresh full-video extraction, no geometry or RGB cache',
            deployment_record_exists=(HERE / 'deployment.json').is_file(),
            output='results/motion_v23_20260910/installed_smoke/{case}',
            numeric_atol=ATOL, numeric_rtol=0, strict_discrete_and_missing_masks=True), ensure_ascii=False, indent=2))
        return
    require(sys.platform == 'linux', 'Run this verifier in the existing WSL project environment')
    require(PROJECT.is_dir() and PROJECT.resolve() == PROJECT, 'Project missing or aliased')
    record_path = HERE / 'deployment.json'
    require(record_path.is_file(), 'V2.3 has not been installed')
    record = load(record_path)
    frozen = load(HERE / 'validation/protocol_before_validation.json')
    require(record['source_hashes'] == frozen['source_hashes'], 'Installed and validated sources differ')
    receipt_path = HERE / 'deployment_verification.json'
    require(not receipt_path.exists(), 'Existing verification receipt is immutable')
    base = safe_target(PROJECT / 'results/motion_v23_20260910/installed_smoke')
    require(not os.path.lexists(base), 'Existing installed smoke evidence must not be overwritten')
    require(safe_target(record['results']) == base.parent, 'Unexpected installed results location')
    before = verify_installed_hashes(record)
    require(before['passed'], 'Installed source/evidence or old entry hash mismatch before verification')
    source_identity = {}
    reference_inputs = {}
    for case in CASES:
        video = Path(frozen['cases'][case]['video'])
        require(video.is_file(), f'Original video missing: {case}')
        actual_video_sha256 = digest(video)
        require(actual_video_sha256 == frozen['cases'][case]['sha256'], f'Original video changed: {case}')
        source_identity[case] = dict(path=str(video), sha256=actual_video_sha256, bytes=video.stat().st_size)
        reference = HERE / 'validation' / MODE / case
        for filename in (*CSVS, 'summary.json', 'frame_trace.json'):
            reference_inputs[str(reference / filename)] = digest(reference / filename)
    base.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in CASES:
        reference = HERE / 'validation' / MODE / case
        output = safe_target(base / case)
        log_path = base / (case + '.log')
        command = [record['launcher'], source_identity[case]['path'], '--output', str(output)]
        row = dict(case=case, mode=MODE, command=command, output=str(output),
                   inference_path='fresh_original_video', used_cache=False, used_geometry_cache=False,
                   matches_validation=False, source_video=source_identity[case])
        started = time.perf_counter()
        print(json.dumps({'event': 'fresh_install_verification_start', 'case': case}), flush=True)
        try:
            with log_path.open('x', encoding='utf-8') as log:
                process = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            row['exit_code'] = process.returncode
            if process.returncode == 0:
                row.update(compare_outputs(reference, output, record['source_hashes']))
            else:
                row['error'] = 'Installed launcher failed; inspect preserved log'
        except Exception as error:
            row.setdefault('exit_code', None)
            row['error'] = f'{type(error).__name__}: {error}'
        row['elapsed_s'] = time.perf_counter() - started
        cases.append(row)
        write_new_json(base / (case + '_comparison.json'), row)
        print(json.dumps({'event': 'fresh_install_verification_complete', 'case': case,
            'exit_code': row['exit_code'], 'matches_validation': row['matches_validation']}, ensure_ascii=False), flush=True)
    after = verify_installed_hashes(record)
    unchanged_references = all(digest(path) == expected for path, expected in reference_inputs.items())
    unchanged_videos = all(digest(item['path']) == item['sha256'] for item in source_identity.values())
    installed_entry_verified = after['passed'] and unchanged_references and unchanged_videos and all(
        row.get('exit_code') == 0 and row['matches_validation'] for row in cases)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        installed_entry_verified=installed_entry_verified, inference_path='fresh_original_video',
        source_hashes=record['source_hashes'], deployment_manifest_sha256=digest(record_path),
        verifier_sha256=digest(Path(__file__)), numeric_atol=ATOL, numeric_rtol=0,
        discrete_values_and_missingness='exact', cases=cases,
        old_entries_unchanged=not any('preserved_entry_hashes:' in item for item in after['failures']),
        installed_hashes_before=before, installed_hashes_after=after,
        original_validation_inputs_unchanged=unchanged_references,
        original_videos_unchanged=unchanged_videos,
        reference_input_hashes=reference_inputs,
        limitations=['This verifies two complete installed executions, not physiological generalization.',
            'A fresh detector path differs from frozen-geometry validation; any mismatch remains visible.',
            'No cache fallback or relaxed post-result tolerance is performed by this verifier.',
            'Default promotion still requires the separate fixed performance gate.'])
    write_new_json(receipt_path, report)
    write_new_json(base / 'verification.json', report)
    print(json.dumps({'installed_entry_verified': installed_entry_verified, 'receipt': str(receipt_path)}, ensure_ascii=False), flush=True)
    if not installed_entry_verified:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
