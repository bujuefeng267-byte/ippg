"""Prospective V28 evaluation with frozen references and three fixed comparators.

No reference value is passed to inference. Call freeze_evaluation_inputs before
prediction and evaluate_variant only after every planned prediction is saved.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
V25 = P/'results/data1_6_v25_20260911/stage4_preserve_waveform'
R26 = P/'results/data1_6_v26_20260911'
ROOT = P/'results/data1_6_v28_20260912'
CORE_PATH = P/'batch_analysis_20260911/evaluate_batch.py'
CASES = tuple(f'data{i}' for i in range(1, 7))
COMPARATORS = {'V25': V25, 'V26_harmonic': R26/'component_harmonics',
               'V26_conservative': R26/'conservative_components'}
SHIFTS = (-5, -2, -1, 0, 1, 2, 5)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def core_module():
    spec = importlib.util.spec_from_file_location('v28_frozen_reference_core', CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bools(series):
    assert series.notna().all() and series.isin([True, False, 0, 1]).all(), 'Explicit masks required'
    return series.to_numpy(bool)


def comparator_paths(name, case):
    d = COMPARATORS[name]/case
    return (d/'fusion_heart_rate.csv', d/'fusion_waveform.csv') if name == 'V25' else (
        d/'heart_rate.csv', d/'waveform.csv')


def freeze_evaluation_inputs():
    """Hash fixed clocks, references and ALL three comparator outputs, read only."""
    paths = {CORE_PATH, B/'inputs_manifest.json', V25/'stage_evaluation.json',
             R26/'component_harmonics/evaluation_summary.json',
             R26/'conservative_components/evaluation_summary.json'}
    for case in CASES:
        alignment_path = B/case/'evaluation/alignment.json'
        alignment = json.loads(alignment_path.read_text())
        paths.update([alignment_path, B/case/'evaluation/paired_windows.csv',
                      B/case/'inference/summary.json',
                      B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json',
                      Path(alignment['reference_path'])])
        for name, folder in COMPARATORS.items():
            paths.update([*comparator_paths(name, case), folder/case/'summary.json'])
        paths.add(R26/'conservative_components'/case/'routing_decisions.csv')
    return {str(p): sha(p) for p in sorted(paths, key=str)}


def verify_hashes(hashes):
    assert isinstance(hashes, dict) and hashes, 'Nonempty hash binding required'
    for path, expected in hashes.items():
        assert sha(path) == expected, 'Frozen input/source changed: '+str(path)


def aggregate(rows):
    n = sum(r['Nvalid'] for r in rows)
    nr = sum(r['Nref'] for r in rows)
    nc = sum(r.get('common_windows', 0) for r in rows)
    frames = sum(r['waveform_frames'] for r in rows)
    result = dict(Nplanned=sum(r['Nplanned'] for r in rows),
        Noutput=sum(r['Noutput'] for r in rows), Nref=nr, Nvalid=n,
        Nwithin5=sum(r['Nwithin5'] for r in rows),
        MAE_bpm=sum(r['Nvalid']*r['MAE_bpm'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['Nvalid']*r['RMSE_bpm']**2 for r in rows if r['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*sum(r['Nwithin5'] for r in rows)/n if n else np.nan,
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in rows)/nr if nr else np.nan,
        HR_coverage_pct=100*sum(r['Noutput'] for r in rows)/sum(r['Nplanned'] for r in rows),
        waveform_frame_coverage_pct=100*sum(r['waveform_finite_frames'] for r in rows)/frames,
        waveform_time_coverage_pct=100*sum(r['waveform_finite_frames']/r['fps'] for r in rows)/
            sum(r['waveform_frames']/r['fps'] for r in rows),
        waveform_coverage_definition='Time coverage weights frames by original 1/fps; frame coverage is separately named')
    if any('common_windows' in r for r in rows):
        result.update(common_windows=nc,
            common_MAE_bpm=sum(r['common_windows']*r['common_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan,
            common_V25_MAE_bpm=sum(r['common_windows']*r['common_V25_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan)
    return result


def compare(y, ok, r, oy, ook):
    common = ok & ook & np.isfinite(r) & (r > 0)
    new = ok & ~ook & np.isfinite(r) & (r > 0)
    lost = ~ok & ook & np.isfinite(r) & (r > 0)
    return dict(common_windows=int(common.sum()), new_windows=int(new.sum()), lost_windows=int(lost.sum()),
        common_MAE_bpm=float(np.mean(abs(y[common]-r[common]))) if common.any() else np.nan,
        common_comparator_MAE_bpm=float(np.mean(abs(oy[common]-r[common]))) if common.any() else np.nan,
        new_MAE_bpm=float(np.mean(abs(y[new]-r[new]))) if new.any() else np.nan,
        lost_comparator_MAE_bpm=float(np.mean(abs(oy[lost]-r[lost]))) if lost.any() else np.nan), (common, new, lost)


def verify_prediction(case, directory, protocol, protocol_sha):
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['reference_used'] is False
    assert summary['case'] == case and summary['variant'] == 'direct_guard'
    assert summary['source_hashes'] == protocol['source_hashes']
    assert summary['protocol_sha256'] == protocol_sha
    assert summary['input_hashes'] == protocol['input_hashes'][case]
    verify_hashes(summary['input_hashes'])
    for name in ('waveform.csv', 'heart_rate.csv', 'routing_decisions.csv'):
        assert summary['output_hashes'][name] == sha(directory/name), 'Output changed: '+name
    wave = pd.read_csv(directory/'waveform.csv')
    hr = pd.read_csv(directory/'heart_rate.csv')
    decisions = pd.read_csv(directory/'routing_decisions.csv')
    meta = json.loads((B/case/'inference/summary.json').read_text())
    fps, frames = float(meta['fps']), int(meta['frames'])
    assert fps == summary['fps'] and frames == summary['frames'] and len(wave) == frames
    window, hop = round(10*fps), round(fps)
    starts = np.arange(0, frames-window+1, hop)
    np.testing.assert_allclose(wave.time_s, np.arange(frames)/fps, atol=1e-8, rtol=0)
    for table in (hr, decisions):
        assert len(table) == len(starts)
        np.testing.assert_allclose(table.time_s, (starts+window/2)/fps, atol=1e-8, rtol=0)
        np.testing.assert_allclose(table.window_start_s, starts/fps, atol=1e-8, rtol=0)
        np.testing.assert_allclose(table.window_end_s, (starts+window)/fps, atol=1e-8, rtol=0)
    old = pd.read_csv(V25/case/'fusion_waveform.csv')
    candidate = pd.read_csv(R26/'component_harmonics'/case/'waveform.csv')
    previous = pd.read_csv(R26/'conservative_components'/case/'routing_decisions.csv')
    finite = np.isfinite(wave.base.to_numpy(float))
    np.testing.assert_array_equal(finite, np.isfinite(old.base))
    np.testing.assert_array_equal(finite, bools(wave.covered))
    assert not np.isinf(wave.base).any()
    eligible = bools(decisions.route_eligible)
    assert not (eligible & ~bools(previous.route_eligible)).any(), 'Direct guard can only veto prior routes'
    # Independent reconstruction of the convex combination and its source masks.
    total, replacement = np.zeros(frames), np.zeros(frames)
    taper = np.hanning(window+2)[1:-1]
    for i, start in enumerate(starts):
        total[start:start+window] += taper
        if eligible[i]:
            replacement[start:start+window] += taper
    weight = np.clip(np.divide(replacement, total, out=np.zeros(frames), where=total > 0), 0, 1)
    weight[~finite] = 0
    np.testing.assert_allclose(wave.candidate_weight, weight, atol=1e-12, rtol=0)
    use_candidate, use_old = weight > 0, finite & (weight < 1)
    expected = old.base.to_numpy(float).copy()
    expected[use_candidate] = ((1-weight[use_candidate])*expected[use_candidate] +
                              weight[use_candidate]*candidate.base.to_numpy(float)[use_candidate])
    np.testing.assert_allclose(wave.base, expected, atol=1e-10, rtol=1e-12, equal_nan=True)
    expected_observed = ((~use_old | bools(old.observed)) &
                         (~use_candidate | bools(candidate.observed)) & finite)
    expected_interpolated = ((use_old & bools(old.interpolated)) |
                             (use_candidate & bools(candidate.interpolated))) & finite
    np.testing.assert_array_equal(bools(wave.observed), expected_observed)
    np.testing.assert_array_equal(bools(wave.interpolated), expected_interpolated)
    ok, y = bools(hr.accepted), hr.ridge_bpm.to_numpy(float)
    assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
    assert all(finite[a:a+window].all() for a in starts[ok]), 'Accepted HR crosses missing waveform'
    # Re-read the actual saved broadband mixture; do not copy old/candidate HR.
    from evidence_hr import estimate_evidence
    trace = pd.read_csv(B/case/'inference/frame_trace.csv')
    replay = estimate_evidence(wave.base.to_numpy(float), pd.DataFrame({'rgb_valid': wave.observed}),
        bools(wave.interpolated), fps, motion_trace=trace)
    np.testing.assert_array_equal(ok, bools(replay.accepted))
    np.testing.assert_allclose(y, replay.ridge_bpm, rtol=0, atol=1e-10, equal_nan=True)
    for i in np.flatnonzero(ok):
        candidates = json.loads(hr.iloc[i].evidence_candidates_json)
        assert any(any(abs(float(v)-y[i]) < 1e-9 for v in c['supported_bpm']) for c in candidates), 'Unsupported HR'
        assert any(any(abs(float(v)-y[i]) < 1e-9 and float(p) >= .05-1e-12
            for v, p in zip(c['supported_bpm'], c['supported_relative_power'])) for c in candidates)
    return wave, hr, decisions, fps, frames


def evaluate_variant(name):
    assert name == 'direct_guard', 'Only the single prospectively declared V28 variant is permitted'
    core = core_module()
    protocol_path = ROOT/'protocol_before_run.json'
    protocol_sha = sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    assert protocol['evaluation_inputs'] == freeze_evaluation_inputs(), 'Evaluation inputs changed since freeze'
    verify_hashes(protocol['evaluation_hashes'])
    assert str(Path(__file__).resolve()) in protocol['evaluation_hashes'], 'Evaluator must be prospectively frozen'
    verify_hashes({str(HERE/name): value for name, value in protocol['source_hashes'].items()})
    folder = ROOT/name
    assert not (folder/'evaluation_summary.json').exists(), 'Preserve completed evaluations'
    # Require all six inference summaries before the first reference comparison.
    for case in CASES:
        assert json.loads((folder/case/'summary.json').read_text())['status'] == 'complete'
    rows, comparison_rows, comparator_rows = [], [], {key: [] for key in COMPARATORS}
    hashes = {str(protocol_path): protocol_sha, **protocol['evaluation_inputs'], **protocol['evaluation_hashes']}
    for case in CASES:
        directory = folder/case
        wave, hr, decisions, fps, frames = verify_prediction(case, directory, protocol, protocol_sha)
        ref = pd.read_csv(B/case/'evaluation/paired_windows.csv')
        np.testing.assert_allclose(hr.time_s, ref.time_s, atol=1e-8, rtol=0)
        ok, y, r = bools(hr.accepted), hr.ridge_bpm.to_numpy(float), ref.reference_bpm.to_numpy(float)
        finite = np.isfinite(wave.base)
        m = core.score(y, ok, r)
        row = dict(variant=name, case=case, **m, fps=fps, waveform_frames=frames,
            waveform_finite_frames=int(finite.sum()), waveform_coverage_pct=100*finite.mean(),
            mean_HR_bpm=float(np.mean(y[ok])) if ok.any() else np.nan,
            target_P5_95_met=bool(m['P5_valid_pct'] >= 95), status_counts=hr.status.value_counts().to_dict(),
            routed_windows=int(bools(decisions.route_eligible).sum()),
            routing_reason_counts=decisions.reason.value_counts().to_dict())
        paired = ref[['window_index', 'time_s', 'window_start_s', 'window_end_s',
                      'reference_bpm', 'reference_valid']].copy()
        paired['estimated_bpm'], paired['accepted'] = y, ok
        paired['error_bpm'], paired['abs_error_bpm'] = np.where(ok, y-r, np.nan), np.where(ok, abs(y-r), np.nan)
        paired['status'] = hr.status
        for comparator in COMPARATORS:
            old_hr_path, old_wave_path = comparator_paths(comparator, case)
            old, old_wave = pd.read_csv(old_hr_path), pd.read_csv(old_wave_path)
            np.testing.assert_allclose(old.time_s, ref.time_s, atol=1e-8, rtol=0)
            np.testing.assert_allclose(old_wave.time_s, wave.time_s, atol=1e-8, rtol=0)
            oy, ook = old.ridge_bpm.to_numpy(float), bools(old.accepted)
            old_metric = core.score(oy, ook, r)
            old_finite = np.isfinite(old_wave.base)
            comparator_rows[comparator].append(dict(case=case, **old_metric, fps=fps,
                waveform_frames=frames, waveform_finite_frames=int(old_finite.sum()), waveform_coverage_pct=100*old_finite.mean()))
            common, masks = compare(y, ok, r, oy, ook)
            comparison = dict(variant=name, comparator=comparator, case=case,
                delta_MAE_bpm=m['MAE_bpm']-old_metric['MAE_bpm'],
                delta_RMSE_bpm=m['RMSE_bpm']-old_metric['RMSE_bpm'],
                delta_P5_pp=m['P5_valid_pct']-old_metric['P5_valid_pct'],
                delta_R5_pp=m['R5_all_reference_pct']-old_metric['R5_all_reference_pct'],
                delta_hr_coverage_pp=m['hr_output_coverage_pct']-old_metric['hr_output_coverage_pct'],
                delta_waveform_coverage_pp=100*(finite.mean()-old_finite.mean()), **common)
            comparison_rows.append(comparison)
            paired[comparator+'_bpm'], paired[comparator+'_accepted'] = oy, ook
            if comparator == 'V25':
                row.update({k: v for k, v in comparison.items() if k.startswith('delta_')})
                row.update(common_windows=common['common_windows'], new_windows=common['new_windows'],
                    lost_windows=common['lost_windows'], common_MAE_bpm=common['common_MAE_bpm'],
                    common_V25_MAE_bpm=common['common_comparator_MAE_bpm'], new_MAE_bpm=common['new_MAE_bpm'],
                    lost_V25_MAE_bpm=common['lost_comparator_MAE_bpm'], V25_MAE_bpm=old_metric['MAE_bpm'])
                paired['common'], paired['new'], paired['lost'] = masks
        out = directory/'evaluation'
        out.mkdir()
        paired.to_csv(out/'paired_windows.csv', index=False)
        write(out/'metrics.json', row)
        rows.append(row)
        alignment = json.loads((B/case/'evaluation/alignment.json').read_text())
        raw = pd.read_csv(alignment['reference_path'])
        sensitivity = []
        for shift in SHIFTS:
            adjusted = core.reference_windows(raw, alignment['video_start_utc_ns'],
                ref.window_start_s, ref.window_end_s, shift)
            if shift == 0:
                np.testing.assert_allclose(r, adjusted.reference_bpm, atol=1e-9, rtol=0, equal_nan=True)
            sensitivity.append(dict(variant=name, case=case, shift_s=shift,
                **core.score(y, ok, adjusted.reference_bpm.to_numpy(float))))
        pd.DataFrame(sensitivity).to_csv(out/'alignment_sensitivity.csv', index=False)
        for file in ('waveform.csv', 'heart_rate.csv', 'routing_decisions.csv', 'summary.json'):
            hashes[str(directory/file)] = sha(directory/file)
    pooled = aggregate(rows)
    assert pooled['Nplanned'] == pooled['Nref'] == 309
    baselines = {key: dict(pooled=aggregate(value), cases=value) for key, value in comparator_rows.items()}
    prior = baselines['V25']['pooled']
    saved_prior = json.loads((V25/'stage_evaluation.json').read_text())['pooled']
    for key in ('MAE_bpm', 'RMSE_bpm', 'R5_all_reference_pct'):
        np.testing.assert_allclose(prior[key], saved_prior[key], atol=1e-9, rtol=0)
    checks = dict(pooled_MAE=pooled['MAE_bpm'] < prior['MAE_bpm'],
        pooled_R5_gain=pooled['R5_all_reference_pct'] >= prior['R5_all_reference_pct']+5,
        each_MAE=all(r['delta_MAE_bpm'] <= 3 for r in rows),
        each_P5=all(r['delta_P5_pp'] >= -3 for r in rows),
        each_R5=all(r['delta_R5_pp'] >= -3 for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp'] >= -3 for r in rows),
        each_waveform_coverage=all(r['delta_waveform_coverage_pp'] >= -3 for r in rows),
        common_MAE=pooled['common_MAE_bpm'] < pooled['common_V25_MAE_bpm'])
    pooled_comparisons = {}
    for name_, baseline in baselines.items():
        selected = [r for r in comparison_rows if r['comparator'] == name_]
        count = sum(r['common_windows'] for r in selected)
        pooled_comparisons[name_] = dict(
            **{'delta_'+key: pooled[key]-baseline['pooled'][key] for key in (
                'MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct', 'HR_coverage_pct', 'waveform_time_coverage_pct')},
            common_windows=count,
            common_MAE_bpm=sum(r['common_windows']*r['common_MAE_bpm'] for r in selected if r['common_windows'])/count if count else np.nan,
            common_comparator_MAE_bpm=sum(r['common_windows']*r['common_comparator_MAE_bpm'] for r in selected if r['common_windows'])/count if count else np.nan)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), variant=name, pooled=pooled, cases=rows,
        baselines=baselines, comparator_differences=comparison_rows, pooled_comparator_differences=pooled_comparisons,
        promotion_checks=checks, promotion_pass=all(checks.values()),
        near_all_within_5bpm_target_met=all(r['target_P5_95_met'] for r in rows), input_hashes=hashes,
        scope='Previously inspected development clips; overlapping windows; fixed estimated original alignment, not hardware synchronized or held-out validation',
        waveform_interpretation='Finite coverage of measured convex mixture; no waveform ground truth, morphology/SNR improvement unvalidated')
    verify_hashes(hashes)
    write(folder/'evaluation_summary.json', report)
    pd.DataFrame(rows).drop(columns=['status_counts', 'routing_reason_counts']).to_csv(folder/'metrics.csv', index=False)
    pd.DataFrame(comparison_rows).to_csv(folder/'comparator_differences.csv', index=False)
    write(folder/'baseline_recomputed_metrics.json', baselines)
    print(json.dumps(clean(dict(variant=name, pooled=pooled, promotion_checks=checks,
        promotion_pass=all(checks.values()), cases=[{k: r[k] for k in ('case', 'MAE_bpm', 'P5_valid_pct',
            'R5_all_reference_pct', 'hr_output_coverage_pct', 'waveform_coverage_pct', 'routed_windows')} for r in rows])), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['direct_guard'])
    parser.add_argument('--freeze-inputs', action='store_true')
    args = parser.parse_args()
    if args.freeze_inputs:
        print(json.dumps(freeze_evaluation_inputs(), ensure_ascii=False, indent=2))
    elif args.variant:
        evaluate_variant(args.variant)
    else:
        parser.error('Choose --freeze-inputs or --variant direct_guard')
