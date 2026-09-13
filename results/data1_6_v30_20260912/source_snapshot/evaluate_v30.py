"""Frozen V30 evaluation versus V28 on the original 309 ten-second windows.

Both declared variants and all six cases must finish before scoring. References
are evaluation-only; alignment is fixed and sensitivity never selects an offset.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
V28 = P/'results/data1_6_v28_20260912/direct_guard'
ROOT = P/'results/data1_6_v30_20260912'
OLD_CODE = P/'motion_upgrade_v28_20260912'
CORE_PATH = P/'batch_analysis_20260911/evaluate_batch.py'
CASES = tuple(f'data{i}' for i in range(1, 7))
VARIANTS = ('motion_full_spectrum', 'cdf_v28')
PROTOCOLS = {'motion_full_spectrum': ROOT/'freeze_motion.json', 'cdf_v28': ROOT/'freeze_cdf.json'}
SHIFTS = (-5, -2, -1, 0, 1, 2, 5)
FREEZE = ROOT/'freeze_evaluation_v30.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def bools(series):
    assert series.notna().all() and series.isin([True, False, 0, 1]).all(), 'Explicit Boolean masks required'
    return series.to_numpy(bool)


def verify_hashes(bindings):
    assert isinstance(bindings, dict) and bindings, 'Nonempty hash bindings required'
    for name, expected in bindings.items():
        assert Path(name).is_absolute(), 'Hash keys must be absolute paths'
        assert sha(name) == expected, 'Bound input/source changed: '+str(name)


def protocol_bindings(value):
    """Find nested absolute-path -> SHA256 pairs without imposing a top schema."""
    result = {}
    def visit(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if isinstance(key, str) and Path(key).is_absolute() and isinstance(val, str) and len(val) == 64:
                    if key in result: assert result[key] == val, 'Conflicting frozen file binding'
                    result[key] = val
                visit(val)
        elif isinstance(item, list):
            for val in item: visit(val)
    visit(value)
    return result


def frozen_inputs():
    paths = {CORE_PATH, B/'inputs_manifest.json', V28/'evaluation_summary.json'}
    for name in ('evidence_hr.py', 'legacy_motion.py', 'analyze_rppg.py'):
        paths.add(OLD_CODE/name)
    for case in CASES:
        alignment_path = B/case/'evaluation/alignment.json'
        alignment = json.loads(alignment_path.read_text())
        paths.update([alignment_path, Path(alignment['reference_path']),
            B/case/'evaluation/paired_windows.csv', B/case/'inference/summary.json',
            B/case/'inference/frame_trace.json', B/case/'inference/frame_trace.csv',
            V28/case/'waveform.csv', V28/case/'heart_rate.csv', V28/case/'summary.json'])
    return {str(path): sha(path) for path in sorted(paths, key=str)}


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), variants=list(VARIANTS),
        allowed_inference_protocols=PROTOCOLS,
        evaluation_sources={str(HERE/name): sha(HERE/name) for name in ('evaluate_v30.py', 'test_evaluate_v30.py')},
        evaluation_inputs=frozen_inputs(), shifts_s=list(SHIFTS),
        window_plan='Original complete 10 s / frame-rounded 1 s; 309 reference-qualified windows',
        promotion_rule='Pooled MAE decreases; R5 improves at least 5 percentage points; each case MAE worsens <=3 bpm and P5/R5/HR coverage/wave coverage falls <=3pp; common-window MAE decreases.',
        no_additional_parameter_trials=True,
        scope='Previously inspected development videos, fixed estimated Polar alignment, overlapping windows; no held-out validation')
    write_new(FREEZE, record)
    print(json.dumps(dict(freeze=str(FREEZE), sha256=sha(FREEZE), variants=VARIANTS), ensure_ascii=False))


def core_module():
    spec = importlib.util.spec_from_file_location('v30_reference_core', CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_signal_tables(wave, hr, frames, fps):
    assert len(wave) == frames and frames > 0 and np.isfinite(fps) and fps > 7
    np.testing.assert_allclose(wave.time_s, np.arange(frames)/fps, rtol=0, atol=1e-8)
    values = wave.base.to_numpy(float)
    assert not np.isinf(values).any()
    finite = np.isfinite(values)
    np.testing.assert_array_equal(finite, bools(wave.covered))
    observed, interpolated = bools(wave.observed), bools(wave.interpolated)
    assert not (observed & ~finite).any() and not (interpolated & ~finite).any()
    width, hop = round(10*fps), round(fps)
    starts = np.arange(0, frames-width+1, hop, dtype=int)
    assert len(hr) == len(starts)
    for key, expected in [('time_s', (starts+width/2)/fps), ('window_start_s', starts/fps),
                          ('window_end_s', (starts+width)/fps)]:
        np.testing.assert_allclose(hr[key], expected, rtol=0, atol=1e-8)
    accepted, y = bools(hr.accepted), hr.ridge_bpm.to_numpy(float)
    assert np.isfinite(y[accepted]).all() and np.isnan(y[~accepted]).all()
    assert ((y[accepted] >= 42) & (y[accepted] <= 210)).all()
    assert all(finite[a:a+width].all() for a in starts[accepted]), 'Accepted HR crosses missing samples'
    return values, finite, observed, interpolated, starts, accepted, y


def verify_measured_support(wave, hr, fps, starts, accepted):
    """Recompute original peak support from actual saved samples, without truth.

Motion scoring changes ranks, not peak detection or supported frequency grids.
This checks emitted HR has measured energy; it does not force V28 HR selection.
"""
    sys.path.insert(0, str(OLD_CODE))
    from evidence_hr import score_candidates
    from legacy_motion import estimate
    original_gates = estimate(wave.base.to_numpy(float), pd.DataFrame({'rgb_valid': bools(wave.observed)}),
        bools(wave.interpolated), fps, 10, 1, 42, 210)
    assert not (accepted & ~bools(original_gates.accepted)).any(), 'Accepted HR fails unchanged V28 validity gates'
    grid, width = np.arange(42., 211.), round(10*fps)
    for i in np.flatnonzero(accepted):
        claimed = json.loads(hr.iloc[i].evidence_candidates_json)
        y = float(hr.iloc[i].ridge_bpm)
        assert any(any(abs(float(bpm)-y) <= 1e-9 and float(power) >= .05-1e-12
            for bpm, power in zip(c['supported_bpm'], c['supported_relative_power'])) for c in claimed), 'Claimed HR lacks candidate support'
        measured, _ = score_candidates(wave.base.iloc[starts[i]:starts[i]+width].to_numpy(float), fps, grid)
        assert any(any(abs(float(bpm)-y) <= 1e-9 and float(power) >= .05-1e-12
            for bpm, power in zip(c['supported_bpm'], c['supported_relative_power'])) for c in measured), 'HR lacks actual saved-wave spectral support'


def verify_prediction(variant, case, evaluation_freeze):
    directory = ROOT/variant/case
    summary_path = directory/'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete' and summary['reference_used'] is False
    assert summary['case'] == case and summary['variant'] == variant
    protocol_path = Path(summary['protocol_path']).resolve()
    assert protocol_path == Path(evaluation_freeze['allowed_inference_protocols'][variant]).resolve()
    assert sha(protocol_path) == summary['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    bound = protocol_bindings(protocol)
    hashes = {str(summary_path): sha(summary_path), str(protocol_path): sha(protocol_path)}
    for group in ('source_hashes', 'input_hashes'):
        verify_hashes(summary[group])
        for name, expected in summary[group].items():
            assert bound.get(name) == expected, 'Summary file not bound by declared inference protocol: '+name
        hashes.update(summary[group])
    for filename in ('waveform.csv', 'heart_rate.csv'):
        assert summary['output_hashes'][filename] == sha(directory/filename)
        hashes[str(directory/filename)] = sha(directory/filename)
    meta = json.loads((B/case/'inference/summary.json').read_text())
    fps, frames = float(meta['fps']), int(meta['frames'])
    assert int(summary['frames']) == frames and abs(float(summary['fps'])-fps) <= 1e-8
    wave, hr = pd.read_csv(directory/'waveform.csv'), pd.read_csv(directory/'heart_rate.csv')
    _, _, _, _, starts, accepted, _ = validate_signal_tables(wave, hr, frames, fps)
    verify_measured_support(wave, hr, fps, starts, accepted)
    return wave, hr, fps, frames, hashes


def group_error(y, reference, mask):
    valid = np.asarray(mask, bool) & np.isfinite(reference) & (reference > 0) & np.isfinite(y)
    errors = np.asarray(y)[valid]-np.asarray(reference)[valid]
    return dict(Nvalid=int(valid.sum()), Nwithin5=int((abs(errors) <= 5).sum()),
        MAE_bpm=float(abs(errors).mean()) if len(errors) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        P5_valid_pct=100*float((abs(errors) <= 5).mean()) if len(errors) else np.nan)


def compare(y, accepted, reference, old_y, old_accepted):
    common, new, lost = accepted & old_accepted, accepted & ~old_accepted, ~accepted & old_accepted
    groups = dict(common_new=group_error(y, reference, common), common_V28=group_error(old_y, reference, common),
                  newly_covered=group_error(y, reference, new), lost_V28=group_error(old_y, reference, lost))
    return dict(common_output_windows=int(common.sum()), new_output_windows=int(new.sum()),
                lost_output_windows=int(lost.sum()), **groups), (common, new, lost)


def pool_errors(rows):
    n = sum(row['Nvalid'] for row in rows)
    n5 = sum(row['Nwithin5'] for row in rows)
    return dict(Nvalid=n, Nwithin5=n5,
        MAE_bpm=sum(row['Nvalid']*row['MAE_bpm'] for row in rows if row['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(row['Nvalid']*row['RMSE_bpm']**2 for row in rows if row['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*n5/n if n else np.nan)


def aggregate(rows):
    result = pool_errors(rows)
    planned, nref = sum(r['Nplanned'] for r in rows), sum(r['Nref'] for r in rows)
    output = sum(r['Noutput'] for r in rows)
    result.update(Nplanned=planned, Noutput=output, Nref=nref,
        R5_all_reference_pct=100*result['Nwithin5']/nref if nref else np.nan,
        HR_coverage_pct=100*output/planned if planned else np.nan,
        waveform_time_coverage_pct=100*sum(r['waveform_finite_frames']/r['fps'] for r in rows)/
            sum(r['waveform_frames']/r['fps'] for r in rows),
        waveform_frame_coverage_pct=100*sum(r['waveform_finite_frames'] for r in rows)/sum(r['waveform_frames'] for r in rows))
    return result


def promotion(pooled, baseline, rows, groups):
    return dict(pooled_MAE=bool(pooled['MAE_bpm'] < baseline['MAE_bpm']),
        pooled_R5_gain=bool(pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']+5),
        each_MAE=all(r['delta_MAE_bpm'] <= 3 for r in rows),
        each_P5=all(r['delta_P5_pp'] >= -3 for r in rows),
        each_R5=all(r['delta_R5_pp'] >= -3 for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp'] >= -3 for r in rows),
        each_waveform_coverage=all(r['delta_waveform_coverage_pp'] >= -3 for r in rows),
        common_MAE=bool(groups['common_new']['MAE_bpm'] < groups['common_V28']['MAE_bpm']))


def evaluate():
    frozen = json.loads(FREEZE.read_text())
    assert frozen['variants'] == list(VARIANTS) and frozen['shifts_s'] == list(SHIFTS)
    verify_hashes(frozen['evaluation_sources'])
    assert frozen['evaluation_inputs'] == frozen_inputs()
    assert str(Path(__file__).resolve()) in frozen['evaluation_sources']
    # Require all twelve complete runs before calculating any new accuracy.
    for variant in VARIANTS:
        assert not (ROOT/variant/'evaluation_summary.json').exists(), 'Preserve previous evaluation'
        for case in CASES:
            assert json.loads((ROOT/variant/case/'summary.json').read_text())['status'] == 'complete'
            assert not (ROOT/variant/case/'evaluation').exists(), 'Preserve previous/partial evaluation'
    assert not (ROOT/'evaluation_summary.json').exists()
    core = core_module()
    hashes = {str(FREEZE): sha(FREEZE), **frozen['evaluation_sources'], **frozen['evaluation_inputs']}
    reports, baseline_rows = {}, []
    for variant in VARIANTS:
        rows, differences, paired_tables, sensitivities = [], [], {}, {}
        for case in CASES:
            wave, hr, fps, frames, bindings = verify_prediction(variant, case, frozen)
            hashes.update(bindings)
            old_wave, old_hr = pd.read_csv(V28/case/'waveform.csv'), pd.read_csv(V28/case/'heart_rate.csv')
            _, old_finite, _, _, _, old_ok, old_y = validate_signal_tables(old_wave, old_hr, frames, fps)
            finite, ok, y = np.isfinite(wave.base), bools(hr.accepted), hr.ridge_bpm.to_numpy(float)
            if variant == 'motion_full_spectrum':
                np.testing.assert_allclose(wave.base, old_wave.base, rtol=0, atol=0, equal_nan=True)
                for key in ('covered', 'observed', 'interpolated'):
                    np.testing.assert_array_equal(bools(wave[key]), bools(old_wave[key]))
            reference = pd.read_csv(B/case/'evaluation/paired_windows.csv')
            np.testing.assert_allclose(hr.time_s, reference.time_s, rtol=0, atol=1e-8)
            r = reference.reference_bpm.to_numpy(float)
            m, old_m = core.score(y, ok, r), core.score(old_y, old_ok, r)
            diff, masks = compare(y, ok, r, old_y, old_ok)
            differences.append(diff)
            row = dict(variant=variant, case=case, **m, fps=fps, waveform_frames=frames,
                waveform_finite_frames=int(finite.sum()), waveform_coverage_pct=100*float(finite.mean()),
                waveform_new_seconds=int((finite & ~old_finite).sum())/fps,
                waveform_lost_seconds=int((~finite & old_finite).sum())/fps,
                status_counts=hr.status.value_counts().to_dict(), target_P5_95_met=bool(m['P5_valid_pct'] >= 95),
                delta_MAE_bpm=m['MAE_bpm']-old_m['MAE_bpm'], delta_RMSE_bpm=m['RMSE_bpm']-old_m['RMSE_bpm'],
                delta_P5_pp=m['P5_valid_pct']-old_m['P5_valid_pct'], delta_R5_pp=m['R5_all_reference_pct']-old_m['R5_all_reference_pct'],
                delta_hr_coverage_pp=m['hr_output_coverage_pct']-old_m['hr_output_coverage_pct'],
                delta_waveform_coverage_pp=100*(float(finite.mean())-float(old_finite.mean())), **diff)
            rows.append(row)
            if variant == VARIANTS[0]:
                baseline_rows.append(dict(case=case, **old_m, fps=fps, waveform_frames=frames,
                    waveform_finite_frames=int(old_finite.sum()), waveform_coverage_pct=100*float(old_finite.mean())))
            paired = reference[['window_index', 'time_s', 'window_start_s', 'window_end_s', 'reference_bpm', 'reference_valid']].copy()
            paired['estimated_bpm'], paired['accepted'], paired['status'] = y, ok, hr.status
            paired['V28_bpm'], paired['V28_accepted'] = old_y, old_ok
            paired['error_bpm'] = np.where(ok & bools(paired.reference_valid), y-r, np.nan)
            paired['abs_error_bpm'] = abs(paired.error_bpm)
            paired['common_output'], paired['new_output'], paired['lost_output'] = masks
            paired_tables[case] = paired
            alignment = json.loads((B/case/'evaluation/alignment.json').read_text())
            raw = pd.read_csv(alignment['reference_path'])
            sensitivity = []
            for shift in SHIFTS:
                adjusted = core.reference_windows(raw, alignment['video_start_utc_ns'], reference.window_start_s, reference.window_end_s, shift)
                if shift == 0:
                    np.testing.assert_allclose(r, adjusted.reference_bpm, rtol=0, atol=1e-9, equal_nan=True)
                sensitivity.append(dict(variant=variant, case=case, shift_s=shift,
                    **core.score(y, ok, adjusted.reference_bpm.to_numpy(float))))
            sensitivities[case] = sensitivity
        pooled, baseline = aggregate(rows), aggregate(baseline_rows)
        assert pooled['Nplanned'] == pooled['Nref'] == baseline['Nplanned'] == 309
        old_saved = json.loads((V28/'evaluation_summary.json').read_text())['pooled']
        for key in ('MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct'):
            np.testing.assert_allclose(baseline[key], old_saved[key], rtol=0, atol=1e-9)
        groups = {key: pool_errors([d[key] for d in differences]) for key in ('common_new', 'common_V28', 'newly_covered', 'lost_V28')}
        checks = promotion(pooled, baseline, rows, groups)
        report = dict(variant=variant, pooled=pooled, cases=rows, baseline_V28=dict(pooled=baseline, cases=baseline_rows),
            pooled_coverage_groups=groups, promotion_checks=checks, promotion_pass=all(checks.values()),
            near_all_within_5bpm_target_met=all(r['target_P5_95_met'] for r in rows),
            scope='Previously inspected development clips; overlapping windows; original estimated Polar alignment, not hardware synchronization or held-out validation',
            waveform_interpretation='Measured signal coverage only. No synchronous PPG morphology ground truth; SNR and waveform fidelity improvements are not established.')
        reports[variant] = report
        # Write both variants only after all statistics are calculated below.
        report['_paired_tables'], report['_sensitivities'] = paired_tables, sensitivities
    verify_hashes(hashes)
    outputs = {}
    for variant, report in reports.items():
        pairs, sensitivity = report.pop('_paired_tables'), report.pop('_sensitivities')
        for row in report['cases']:
            case = row['case']; directory = ROOT/variant/case/'evaluation'; directory.mkdir()
            pairs[case].to_csv(directory/'paired_windows.csv', index=False)
            pairs[case].loc[pairs[case].new_output].to_csv(directory/'newly_covered_windows.csv', index=False)
            pd.DataFrame(sensitivity[case]).to_csv(directory/'alignment_sensitivity.csv', index=False)
            write_new(directory/'metrics.json', row)
        write_new(ROOT/variant/'evaluation_summary.json', report)
        simple = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in report['cases']]
        pd.DataFrame(simple).to_csv(ROOT/variant/'metrics.csv', index=False)
        for path in (ROOT/variant).rglob('*'):
            if path.is_file() and (path.parent.name == 'evaluation' or path.name in ('evaluation_summary.json', 'metrics.csv')):
                outputs[str(path)] = sha(path)
    verify_hashes(hashes)
    final = dict(created_utc=datetime.now(timezone.utc).isoformat(), variants=reports,
        input_hashes=hashes, evaluation_output_hashes=outputs, all_variants_complete_before_scoring=True,
        fixed_shifts_reported_without_selection=list(SHIFTS), evaluation_freeze_sha256=sha(FREEZE))
    write_new(ROOT/'evaluation_summary.json', final)
    print(json.dumps(clean({name: dict(pooled=r['pooled'], promotion_checks=r['promotion_checks'], promotion_pass=r['promotion_pass'])
                            for name, r in reports.items()}), ensure_ascii=False, indent=2))
    return final


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    freeze() if args.freeze else evaluate()
