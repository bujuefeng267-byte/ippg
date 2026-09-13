"""Frozen evaluation of two protected CDF mechanisms on the original 309 windows.

All twelve inference runs must complete before either variant is scored. Neither
references nor alignment sensitivity may choose inference settings or a winner
per video. Protection and actual saved-wave support are verified before scoring.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np
import pandas as pd

from evaluation_common_v31 import (sha, clean, write_new, bools, verify_hashes,
    protocol_bindings, core_module, validate_signal_tables, verify_measured_support,
    compare, pool_errors, aggregate)

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
V28 = P/'results/data1_6_v28_20260912/direct_guard'
CDF = P/'results/data1_6_v30_20260912/cdf_v28'
ROOT = P/'results/data1_6_v31_20260912'
OLD_CODE = P/'motion_upgrade_v28_20260912'
CORE_PATH = P/'batch_analysis_20260911/evaluate_batch.py'
CASES = tuple(f'data{i}' for i in range(1, 7))
VARIANTS = ('protected_cdf_motion', 'protected_cdf_consensus')
PROTOCOL_PATH = ROOT/'freeze_inference_v31.json'
SHIFTS = (-5, -2, -1, 0, 1, 2, 5)
FREEZE = ROOT/'freeze_evaluation_v31.json'
TOL = 1e-9
PROMOTION_RULE = dict(
    no_original_correct_window_becomes_wrong_or_missing=True,
    no_new_absolute_error_at_least_20_bpm=True,
    each_case_nonregression=['MAE', 'P5', 'R5', 'HR_coverage', 'waveform_coverage'],
    pooled_nonregression=['MAE', 'P5', 'R5'], common_MAE_nonregression=True,
    each_case_large_jump_count_nonincrease=True,
    substantive_gain='pooled MAE decreases by at least 1 bpm OR R5 increases by at least 1 percentage point',
    numerical_tolerance=TOL, large_error_bpm=20, correct_error_bpm=5,
    large_jump_bpm_strict=12,
    jump_pairs='Consecutive planned windows, both accepted; never bridge missing windows',
    first_20_seconds='Primary: actual window center in [0,20); also report windows ending <=20 s',
    maximum_jump='Runtime guarantee: new maximum <= max(12 bpm, original maximum); ordinary changes <=12 bpm are not large jumps. Fixed before scoring.')


def frozen_inputs():
    paths = {CORE_PATH, B/'inputs_manifest.json', V28/'evaluation_summary.json'}
    paths.update(OLD_CODE.glob('*.py'))
    for case in CASES:
        alignment_path = B/case/'evaluation/alignment.json'
        alignment = json.loads(alignment_path.read_text())
        paths.update([alignment_path, Path(alignment['reference_path']),
            B/case/'evaluation/paired_windows.csv', B/case/'inference/summary.json',
            B/case/'inference/frame_trace.json', B/case/'inference/frame_trace.csv'])
        for directory in (V28/case, CDF/case):
            paths.update(directory/name for name in ('waveform.csv', 'heart_rate.csv', 'summary.json'))
    return {str(path): sha(path) for path in sorted(paths, key=str)}


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    names = ('evaluate_v31.py', 'evaluation_common_v31.py', 'test_evaluate_v31.py', 'qa_evaluation_v31.py')
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), variants=list(VARIANTS),
        allowed_inference_protocol=str(PROTOCOL_PATH),
        evaluation_sources={str(HERE/name): sha(HERE/name) for name in names},
        evaluation_inputs=frozen_inputs(), shifts_s=list(SHIFTS),
        window_plan='Original complete 10 s / frame-rounded 1 s; 309 reference-qualified windows',
        promotion_rule=PROMOTION_RULE, no_additional_parameter_trials=True,
        target_change_before_scoring='Previous +5 pp target replaced for this protected-path experiment by strict nonregression plus >=1 bpm MAE or >=1 pp R5 gain; fixed before V31 scoring.',
        scope='Previously inspected development videos; estimated Polar alignment; overlapping windows; no held-out validation')
    write_new(FREEZE, record)
    print(json.dumps(dict(freeze=str(FREEZE), sha256=sha(FREEZE), variants=VARIANTS), ensure_ascii=False))


def protection_mask(frames, starts, width, eligible, old_finite):
    """Reconstruct sample protection independently from the final window flags."""
    protected, in_plan = np.zeros(frames, bool), np.zeros(frames, bool)
    for start, permitted in zip(starts, eligible):
        in_plan[start:start+width] = True
        if not permitted:
            protected[start:start+width] = True
    return protected | ~in_plan | ~old_finite


def verify_protection(wave, hr, old_wave, old_hr, decisions, fps, starts):
    eligible = bools(decisions.eligible)
    assert len(decisions) == len(hr)
    np.testing.assert_array_equal(decisions.window_index, np.arange(len(hr)))
    for field in ('time_s', 'window_start_s', 'window_end_s'):
        np.testing.assert_allclose(decisions[field], hr[field], rtol=0, atol=1e-8)
    protected_window = bools(decisions.protected_window)
    np.testing.assert_array_equal(protected_window, ~eligible)
    old_ok, ok = bools(old_hr.accepted), bools(hr.accepted)
    assert not (eligible & ~old_ok).any(), 'Originally rejected HR must be protected'
    old_finite, finite = np.isfinite(old_wave.base), np.isfinite(wave.base)
    np.testing.assert_array_equal(finite, old_finite)
    expected = protection_mask(len(wave), starts, round(10*fps), eligible, old_finite)
    np.testing.assert_array_equal(bools(wave.protected_sample), expected)
    for field in ('base', 'time_s', 'covered', 'observed', 'interpolated'):
        np.testing.assert_allclose(wave.loc[expected, field], old_wave.loc[expected, field],
                                   rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(ok[protected_window], old_ok[protected_window])
    np.testing.assert_allclose(hr.ridge_bpm[protected_window], old_hr.ridge_bpm[protected_window],
                               rtol=0, atol=0, equal_nan=True)
    # Missing outputs cannot silently recover or disappear under this protected design.
    np.testing.assert_array_equal(ok, old_ok)
    changed = (finite & (wave.base.to_numpy(float).view(np.uint64) !=
                         old_wave.base.to_numpy(float).view(np.uint64)))
    for field in ('observed', 'interpolated'):
        changed |= bools(wave[field]) != bools(old_wave[field])
    assert not changed[expected].any()
    counts = np.array([changed[a:a+round(10*fps)].sum() for a in starts], int)
    np.testing.assert_array_equal(decisions.modified_samples_in_window, counts)
    assert not counts[protected_window].any()
    assert (counts[eligible] > 0).all(), 'Unmodified windows must remain protected HR states'
    if 'initial_eligible' in decisions:
        initial = bools(decisions.initial_eligible)
        assert not (eligible & ~initial).any(), 'Restoration must only revoke eligibility'
    else:
        initial = eligible.copy()
    return dict(protected_windows=int(protected_window.sum()), eligible_windows=int(eligible.sum()),
        initially_eligible_windows=int(initial.sum()), revoked_windows=int((initial & ~eligible).sum()),
        protected_samples=int(expected.sum()), modified_samples=int(changed.sum()),
        effective_changed_windows=int((counts > 0).sum()), protected_wave_exact=True,
        protected_HR_exact=True, original_missing_masks_exact=True)


def window_changes(y, accepted, reference, old_y, old_accepted, centers, ends):
    valid_ref = np.isfinite(reference) & (reference > 0)
    old_valid, new_valid = old_accepted & valid_ref, accepted & valid_ref
    old_error, new_error = abs(old_y-reference), abs(y-reference)
    old_correct = old_valid & (old_error <= 5)
    old_large = old_valid & (old_error >= 20)
    new_large = new_valid & (new_error >= 20)
    flags = dict(old_correct=old_correct,
        old_correct_to_wrong_or_missing=old_correct & (~new_valid | (new_error > 5)),
        old_large_error=old_large, new_large_error=new_large,
        new_large_error_introduced=new_large & ~old_large,
        old_large_error_reduced=old_large & new_valid & (new_error < old_error-TOL),
        old_large_error_below_20=old_large & new_valid & (new_error < 20),
        old_large_error_within_5=old_large & new_valid & (new_error <= 5))
    report = {key+'_windows': int(mask.sum()) for key, mask in flags.items()}
    early, ready = (centers >= 0) & (centers < 20), ends <= 20
    for name, mask in (('first20_center', early), ('first20_window_end', ready)):
        report[name] = dict(planned_windows=int(mask.sum()),
            V28_output_windows=int((mask & old_accepted).sum()), output_windows=int((mask & accepted).sum()),
            V28_large_error_windows=int((mask & old_large).sum()), large_error_windows=int((mask & new_large).sum()),
            newly_large_error_windows=int((mask & flags['new_large_error_introduced']).sum()))
    return report, flags


def jump_metrics(y, accepted):
    pairs = accepted[1:] & accepted[:-1]
    jumps = abs(np.diff(y))[pairs]
    return dict(adjacent_valid_pairs=int(pairs.sum()), large_jump_count=int((jumps > 12).sum()),
        max_jump_bpm=float(jumps.max()) if len(jumps) else 0.)


def promotion(pooled, baseline, rows, groups):
    return dict(
        no_original_correct_damage=all(r['old_correct_to_wrong_or_missing_windows'] == 0 for r in rows),
        no_new_large_errors=all(r['new_large_error_introduced_windows'] == 0 for r in rows),
        each_MAE=all(r['delta_MAE_bpm'] <= TOL for r in rows),
        each_P5=all(r['delta_P5_pp'] >= -TOL for r in rows),
        each_R5=all(r['delta_R5_pp'] >= -TOL for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp'] >= -TOL for r in rows),
        each_waveform_coverage=all(r['delta_waveform_coverage_pp'] >= -TOL for r in rows),
        each_large_jump_count=all(r['jumps']['large_jump_count'] <= r['V28_jumps']['large_jump_count'] for r in rows),
        pooled_MAE=bool(pooled['MAE_bpm'] <= baseline['MAE_bpm']+TOL),
        pooled_P5=bool(pooled['P5_valid_pct'] >= baseline['P5_valid_pct']-TOL),
        pooled_R5=bool(pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']-TOL),
        common_MAE=bool(groups['common_new']['MAE_bpm'] <= groups['common_V28']['MAE_bpm']+TOL),
        substantive_gain=bool(pooled['MAE_bpm'] <= baseline['MAE_bpm']-1+TOL or
            pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']+1-TOL))


def verify_prediction(variant, case, frozen):
    directory = ROOT/variant/case
    summary_path = directory/'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete' and summary['reference_used'] is False
    assert summary['case'] == case and summary['variant'] == variant
    protocol_path = Path(summary['protocol_path']).resolve()
    assert protocol_path == Path(frozen['allowed_inference_protocol']).resolve()
    assert sha(protocol_path) == summary['protocol_sha256']
    bound = protocol_bindings(json.loads(protocol_path.read_text()))
    hashes = {str(summary_path): sha(summary_path), str(protocol_path): sha(protocol_path)}
    for group in ('source_hashes', 'input_hashes'):
        verify_hashes(summary[group])
        for name, expected in summary[group].items():
            assert bound.get(name) == expected, 'Summary file absent from inference freeze: '+name
        hashes.update(summary[group])
    for filename in ('waveform.csv', 'heart_rate.csv', 'routing_decisions.csv'):
        assert summary['output_hashes'][filename] == sha(directory/filename)
    for filename, expected in summary['output_hashes'].items():
        target = (directory/filename).resolve()
        assert target.is_relative_to(directory.resolve()), 'Output binding escapes case directory'
        assert sha(target) == expected
        hashes[str(target)] = expected
    meta = json.loads((B/case/'inference/summary.json').read_text())
    fps, frames = float(meta['fps']), int(meta['frames'])
    assert int(summary['frames']) == frames and abs(float(summary['fps'])-fps) <= 1e-8
    wave, hr = pd.read_csv(directory/'waveform.csv'), pd.read_csv(directory/'heart_rate.csv')
    decisions = pd.read_csv(directory/'routing_decisions.csv')
    _, _, _, _, starts, accepted, _ = validate_signal_tables(wave, hr, frames, fps)
    verify_measured_support(wave, hr, fps, starts, accepted)
    old_wave, old_hr = pd.read_csv(V28/case/'waveform.csv'), pd.read_csv(V28/case/'heart_rate.csv')
    validate_signal_tables(old_wave, old_hr, frames, fps)
    protection = verify_protection(wave, hr, old_wave, old_hr, decisions, fps, starts)
    new_jump = jump_metrics(hr.ridge_bpm.to_numpy(float), bools(hr.accepted))
    old_jump = jump_metrics(old_hr.ridge_bpm.to_numpy(float), bools(old_hr.accepted))
    assert new_jump['large_jump_count'] <= old_jump['large_jump_count'], 'Runtime jump-count guarantee failed'
    assert new_jump['max_jump_bpm'] <= max(12., old_jump['max_jump_bpm'])+TOL, 'Runtime max-jump guarantee failed'
    return wave, hr, old_wave, old_hr, decisions, fps, frames, hashes, protection


def evaluate():
    frozen = json.loads(FREEZE.read_text())
    assert frozen['variants'] == list(VARIANTS) and frozen['shifts_s'] == list(SHIFTS)
    assert frozen['promotion_rule'] == PROMOTION_RULE
    verify_hashes(frozen['evaluation_sources'])
    assert frozen['evaluation_inputs'] == frozen_inputs()
    assert str(Path(__file__).resolve()) in frozen['evaluation_sources']
    for variant in VARIANTS:
        assert not (ROOT/variant/'evaluation_summary.json').exists(), 'Preserve prior evaluation'
        for case in CASES:
            assert json.loads((ROOT/variant/case/'summary.json').read_text())['status'] == 'complete'
            assert not (ROOT/variant/case/'evaluation').exists()
    assert not (ROOT/'evaluation_summary.json').exists()
    core = core_module()
    hashes = {str(FREEZE): sha(FREEZE), **frozen['evaluation_sources'], **frozen['evaluation_inputs']}
    reports, baseline_rows = {}, []
    for variant in VARIANTS:
        rows, differences, paired_tables, sensitivities = [], [], {}, {}
        for case in CASES:
            wave, hr, old_wave, old_hr, decisions, fps, frames, bindings, protection = verify_prediction(variant, case, frozen)
            hashes.update(bindings)
            finite, old_finite = np.isfinite(wave.base), np.isfinite(old_wave.base)
            ok, old_ok = bools(hr.accepted), bools(old_hr.accepted)
            y, old_y = hr.ridge_bpm.to_numpy(float), old_hr.ridge_bpm.to_numpy(float)
            reference = pd.read_csv(B/case/'evaluation/paired_windows.csv')
            for field in ('time_s', 'window_start_s', 'window_end_s'):
                np.testing.assert_allclose(hr[field], reference[field], rtol=0, atol=1e-8)
            r = reference.reference_bpm.to_numpy(float)
            m, old_m = core.score(y, ok, r), core.score(old_y, old_ok, r)
            diff, masks = compare(y, ok, r, old_y, old_ok)
            differences.append(diff)
            changes, flags = window_changes(y, ok, r, old_y, old_ok,
                hr.time_s.to_numpy(float), hr.window_end_s.to_numpy(float))
            row = dict(variant=variant, case=case, **m, fps=fps, waveform_frames=frames,
                waveform_finite_frames=int(finite.sum()), waveform_coverage_pct=100*float(finite.mean()),
                waveform_new_seconds=int((finite & ~old_finite).sum())/fps,
                waveform_lost_seconds=int((~finite & old_finite).sum())/fps,
                status_counts=hr.status.value_counts().to_dict(), target_P5_95_met=bool(m['P5_valid_pct'] >= 95),
                delta_MAE_bpm=m['MAE_bpm']-old_m['MAE_bpm'], delta_RMSE_bpm=m['RMSE_bpm']-old_m['RMSE_bpm'],
                delta_P5_pp=m['P5_valid_pct']-old_m['P5_valid_pct'], delta_R5_pp=m['R5_all_reference_pct']-old_m['R5_all_reference_pct'],
                delta_hr_coverage_pp=m['hr_output_coverage_pct']-old_m['hr_output_coverage_pct'],
                delta_waveform_coverage_pp=100*(float(finite.mean())-float(old_finite.mean())),
                **diff, **changes, protection=protection,
                jumps=jump_metrics(y, ok), V28_jumps=jump_metrics(old_y, old_ok))
            rows.append(row)
            if variant == VARIANTS[0]:
                baseline_rows.append(dict(case=case, **old_m, fps=fps, waveform_frames=frames,
                    waveform_finite_frames=int(old_finite.sum()), waveform_coverage_pct=100*float(old_finite.mean()),
                    jumps=jump_metrics(old_y, old_ok)))
            paired = reference[['window_index', 'time_s', 'window_start_s', 'window_end_s', 'reference_bpm', 'reference_valid']].copy()
            paired['estimated_bpm'], paired['accepted'], paired['status'] = y, ok, hr.status
            paired['V28_bpm'], paired['V28_accepted'] = old_y, old_ok
            paired['error_bpm'] = np.where(ok & bools(paired.reference_valid), y-r, np.nan)
            paired['abs_error_bpm'], paired['V28_abs_error_bpm'] = abs(paired.error_bpm), np.where(old_ok, abs(old_y-r), np.nan)
            paired['common_output'], paired['new_output'], paired['lost_output'] = masks
            paired['eligible'], paired['protected_window'] = decisions.eligible, decisions.protected_window
            paired['modified_samples_in_window'] = decisions.modified_samples_in_window
            for key, mask in flags.items(): paired[key] = mask
            paired_tables[case] = paired
            alignment = json.loads((B/case/'evaluation/alignment.json').read_text())
            raw = pd.read_csv(alignment['reference_path'])
            sensitivity = []
            for shift in SHIFTS:
                adjusted = core.reference_windows(raw, alignment['video_start_utc_ns'], reference.window_start_s, reference.window_end_s, shift)
                rr = adjusted.reference_bpm.to_numpy(float)
                if shift == 0: np.testing.assert_allclose(r, rr, rtol=0, atol=1e-9, equal_nan=True)
                change_shift, _ = window_changes(y, ok, rr, old_y, old_ok, hr.time_s.to_numpy(float), hr.window_end_s.to_numpy(float))
                sensitivity.append(dict(variant=variant, case=case, shift_s=shift,
                    **core.score(y, ok, rr), V28_MAE_bpm=core.score(old_y, old_ok, rr)['MAE_bpm'],
                    old_correct_to_wrong_or_missing_windows=change_shift['old_correct_to_wrong_or_missing_windows'],
                    new_large_error_introduced_windows=change_shift['new_large_error_introduced_windows']))
            sensitivities[case] = sensitivity
        pooled, baseline = aggregate(rows), aggregate(baseline_rows)
        assert pooled['Nplanned'] == pooled['Nref'] == baseline['Nplanned'] == 309
        old_saved = json.loads((V28/'evaluation_summary.json').read_text())['pooled']
        for key in ('MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct'):
            np.testing.assert_allclose(baseline[key], old_saved[key], rtol=0, atol=1e-9)
        groups = {key: pool_errors([d[key] for d in differences]) for key in ('common_new', 'common_V28', 'newly_covered', 'lost_V28')}
        changes = {key: sum(row[key] for row in rows) for key in rows[0] if key.endswith('_windows') and not isinstance(rows[0][key], dict)}
        for name in ('first20_center', 'first20_window_end'):
            changes[name] = {key: sum(row[name][key] for row in rows) for key in rows[0][name]}
        for name in ('jumps', 'V28_jumps'):
            changes[name] = dict(adjacent_valid_pairs=sum(row[name]['adjacent_valid_pairs'] for row in rows),
                large_jump_count=sum(row[name]['large_jump_count'] for row in rows),
                max_jump_bpm=max(row[name]['max_jump_bpm'] for row in rows))
        checks = promotion(pooled, baseline, rows, groups)
        report = dict(variant=variant, pooled=pooled, cases=rows, baseline_V28=dict(pooled=baseline, cases=baseline_rows),
            pooled_coverage_groups=groups, pooled_window_changes=changes,
            promotion_checks=checks, promotion_pass=all(checks.values()),
            near_all_within_5bpm_target_met=all(r['target_P5_95_met'] for r in rows),
            scope='Previously inspected development clips; overlapping windows; estimated Polar alignment, not hardware synchronization or held-out validation',
            waveform_interpretation='May include measured convex mixtures. Coverage is not morphology quality; no synchronous PPG shape ground truth or demonstrated SNR gain.')
        reports[variant] = report
        report['_paired_tables'], report['_sensitivities'] = paired_tables, sensitivities
    verify_hashes(hashes)
    outputs = {}
    for variant, report in reports.items():
        pairs, sensitivity = report.pop('_paired_tables'), report.pop('_sensitivities')
        for row in report['cases']:
            case = row['case']; directory = ROOT/variant/case/'evaluation'; directory.mkdir()
            pairs[case].to_csv(directory/'paired_windows.csv', index=False)
            pd.DataFrame(sensitivity[case]).to_csv(directory/'alignment_sensitivity.csv', index=False)
            write_new(directory/'metrics.json', row)
        write_new(ROOT/variant/'evaluation_summary.json', report)
        pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in report['cases']]).to_csv(ROOT/variant/'metrics.csv', index=False)
        for path in (ROOT/variant).rglob('*'):
            if path.is_file() and (path.parent.name == 'evaluation' or path.name in ('evaluation_summary.json', 'metrics.csv')):
                outputs[str(path)] = sha(path)
    verify_hashes(hashes)
    final = dict(created_utc=datetime.now(timezone.utc).isoformat(), variants=reports,
        input_hashes=hashes, evaluation_output_hashes=outputs, all_variants_complete_before_scoring=True,
        fixed_shifts_reported_without_selection=list(SHIFTS), evaluation_freeze_sha256=sha(FREEZE))
    write_new(ROOT/'evaluation_summary.json', final)
    print(json.dumps(clean({name: dict(pooled=r['pooled'], changes=r['pooled_window_changes'], promotion_checks=r['promotion_checks'], promotion_pass=r['promotion_pass'])
                            for name, r in reports.items()}), ensure_ascii=False, indent=2))
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else evaluate()
