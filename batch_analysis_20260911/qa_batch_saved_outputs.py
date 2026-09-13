"""Independent CSV-only replay for the six data1--data6 batch results.

Reads no video or physiological reference and imports no production estimator.
Reuses the hash-pinned independent Welch/DP auditor, not its old fixed-case CLI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIN = '34484cc6ce73f9b35b3f49c54bffaa212c165df1d63980ff2713624bdb81b96b'
CASES = tuple(f'data{i}' for i in range(1, 7))
ROIS = ('forehead', 'left_cheek', 'right_cheek')
CONFIG = dict(algorithm_mode='guarded_fusion', pixel_mode='tracking_screened',
              rgb_aggregation='trimmed_mean', max_gap=.1, window=10, step=1,
              min_bpm=42, max_bpm=210)
GUARD = dict(min_tracked_fraction=.6, max_reset_fraction=.05,
             switch_score_ratio=1.15, max_branch_disagreement_bpm=12.)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, hashes):
    hashes[str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(path, hashes):
    hashes[str(path)] = sha(path)
    return pd.read_csv(path)


def supplementary_checks(directory, core, hashes, expected_frames):
    summary = read_json(directory / 'summary.json', hashes)
    metadata = read_json(directory / 'frame_trace.json', hashes)
    for name, value in CONFIG.items():
        assert summary['config'][name] == value, f'Unexpected {name}'
    assert summary['guard_config'] == GUARD
    assert summary['config']['max_seconds'] is None, 'Full-video batch was truncated by CLI'
    assert summary['reference_identity'] is None and summary['config']['reference_ubfc'] is None
    trace = read_csv(directory / 'frame_trace.csv', hashes)
    assert metadata['trace_sha256'] == sha(directory / 'frame_trace.csv')
    assert metadata['n_frames'] == summary['frames'] == len(trace)
    assert metadata['identity'] == summary['identity']
    fps, n = float(summary['fps']), len(trace)
    if expected_frames is not None:
        assert n == expected_frames, f'Decoded frames {n} != expected {expected_frames}'
    window, step = round(10 * fps), round(fps)
    starts = np.arange(0, max(0, n - window + 1), step, dtype=int)
    core.equal_numeric(summary['effective_window_s'], window / fps, 'Effective window')
    core.equal_numeric(summary['effective_step_s'], step / fps, 'Effective step')
    roi = read_csv(directory / 'roi_waveforms.csv', hashes)
    baseline_roi = read_csv(directory / 'baseline_roi_waveforms.csv', hashes)
    for name in ROIS:
        for tag in ('observed', 'interpolated'):
            np.testing.assert_array_equal(core.flag(roi[f'{name}_{tag}']),
                                          core.flag(baseline_roi[f'{name}_{tag}']))
    diagnostics = read_csv(directory / 'fusion_diagnostics.csv', hashes)
    routing = read_csv(directory / 'branch_routing.csv', hashes)
    assert set(diagnostics.roi).issubset(ROIS)
    assert len(routing) == len(starts)
    np.testing.assert_array_equal(routing.window_index, np.arange(len(starts)))
    core.equal_numeric(routing.time_s, (starts + window / 2) / fps, 'Routing centers')
    assert set(routing.selected_branch).issubset({'none', 'baseline', 'tracked'})
    for index, start in enumerate(starts):
        selected = routing.selected_branch.iloc[index]
        subset = diagnostics.loc[diagnostics.window_index.eq(index)]
        used = subset.loc[subset.waveform_weight.gt(0)]
        assert set(used.branch) == ({selected} if selected != 'none' else set())
        assert subset.loc[~subset.branch.eq(selected), 'waveform_weight'].eq(0).all()
        if selected == 'tracked':
            eligible = 0
            for name in set(used.roi):
                sources = trace[f'{name}_pixel_source'].iloc[start:start + window]
                tracked = float(sources.eq('tracked_ratio').mean())
                reset = float(sources.isin(['baseline_reset', 'numerical_reset']).mean())
                eligible += tracked >= GUARD['min_tracked_fraction'] and reset <= GUARD['max_reset_fraction']
            assert eligible >= 2, 'Selected tracked branch lacks two actually contributing qualified ROIs'
    selection_counts = {str(k): int(v) for k, v in routing.selected_branch.value_counts().items()}
    assert selection_counts == summary['branch_selection_counts']
    return dict(fps=fps, frames=n, duration_frame_clock_s=n / fps,
                expected_frames=expected_frames, container_frame_count_independently_checked=expected_frames is not None,
                window_frames=window, step_frames=step, planned_windows=len(starts),
                source_hashes=summary['source_hashes'], branch_selection_counts=selection_counts,
                selected_tracking_eligibility_verified=True,
                full_unselected_branch_routing_recomputed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--qa-core', type=Path, default=HERE.parent / 'rppg_motion_v24/qa_recompute_waveform_hr.py')
    parser.add_argument('--source-root', type=Path, help='Installed inference source whose hashes must match each run')
    parser.add_argument('--expected-frames', type=Path, help='Optional JSON mapping each dataN to independently measured full frame count')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directories = {case: args.batch_root / case / 'inference' for case in CASES}
    missing = [str(d / 'summary.json') for d in directories.values() if not (d / 'summary.json').is_file()]
    if missing:
        raise RuntimeError('Wait until all six runs have completed: ' + ', '.join(missing))
    output = args.output or args.batch_root / 'qa/waveform_replay_qa.json'
    if output.exists():
        raise FileExistsError(f'Existing QA receipt will not be overwritten: {output}')
    if sha(args.qa_core) != PIN:
        raise RuntimeError('Independent replay helper changed from audited source')
    spec = importlib.util.spec_from_file_location('independent_batch_replay', args.qa_core)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if args.source_root:
        core.HERE = args.source_root.resolve()
    hashes = {str(args.qa_core): PIN, str(Path(__file__).resolve()): sha(__file__)}
    expected = read_json(args.expected_frames, hashes) if args.expected_frames else {}
    if expected:
        assert set(expected) == set(CASES)
        assert all(isinstance(v, int) and v > 0 for v in expected.values())
    results, errors, shared_sources = [], [], None
    for case, directory in directories.items():
        try:
            checked = core.audit_case(directory, hashes)
            extra = supplementary_checks(directory, core, hashes, expected.get(case))
            if shared_sources is None:
                shared_sources = extra['source_hashes']
            assert extra['source_hashes'] == shared_sources, 'Inference sources differ between recordings'
            results.append(dict(case=case, passed=True, checked=checked, **extra))
            print(f'PASS {case}: {extra["frames"]} frames at {extra["fps"]} fps, {extra["planned_windows"]} windows; 3 saved waveforms -> local/DP HR', flush=True)
        except Exception as exc:
            errors.append(dict(case=case, error=f'{type(exc).__name__}: {exc}'))
            print(f'FAIL {case}: {type(exc).__name__}: {exc}', flush=True)
    changed = [p for p, old in hashes.items() if sha(p) != old]
    if changed:
        errors.append(dict(error='Evidence changed during audit', paths=changed))
    for name, expected_hash in (shared_sources or {}).items():
        path = core.HERE / name
        if sha(path) != expected_hash:
            errors.append(dict(error='Inference source changed during audit', path=str(path)))
        hashes[str(path)] = expected_hash
    report = dict(passed=not errors and len(results) == 6,
                  checked_utc=datetime.now(timezone.utc).isoformat(), checked_cases=len(results),
                  planned_cases=list(CASES), checked_waveform_branches=sum(len(x['checked']) for x in results),
                  checked_estimators=['local_peak', 'offline_ridge'], reference_read=False, videos_read=False,
                  production_estimator_imported=False, input_sha256=hashes, source_hashes=shared_sources,
                  methodology='Hash-pinned independent saved-waveform Welch/gate/offline-DP replay; time axes, summaries and actual ROI provenance checked; selected tracked ROI eligibility independently verified.',
                  limitations=['Full unselected-branch routing is not reconstructed from zeroed final weights.',
                               'Accuracy versus reference and synchronization are separate from this arithmetic QA.',
                               'Container completeness uses supplied independent counts when present; no source video is read here.'],
                  results=results, errors=errors)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=report['passed'], output=str(output), errors=errors), ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
