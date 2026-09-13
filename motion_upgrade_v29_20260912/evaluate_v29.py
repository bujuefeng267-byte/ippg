"""Fixed native 6 s and common 10 s evaluations; no reference-guided inference.

Native 6 s accuracy is not directly comparable to V28's 10 s accuracy. The
separate common10s table holds the 309 window definitions and reference means
fixed. Availability times describe an offline recording, not online latency.
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
V28 = P/'results/data1_6_v28_20260912/direct_guard'
ROOT = P/'results/data1_6_v29_20260912'
CORE = P/'batch_analysis_20260911/evaluate_batch.py'
CASES = tuple(f'data{i}' for i in range(1, 7))
PROFILES = ('short6', 'short6_relaxed')
SHIFTS = (-5, -2, -1, 0, 1, 2, 5)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def core_module():
    spec = importlib.util.spec_from_file_location('v29_reference_core', CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bools(series):
    assert series.notna().all() and series.isin([True, False, 0, 1]).all(), 'Explicit finite masks required'
    return series.to_numpy(bool)


def freeze_evaluation_inputs():
    paths = {CORE, B/'inputs_manifest.json', V28/'evaluation_summary.json'}
    for case in CASES:
        alignment_path = B/case/'evaluation/alignment.json'
        alignment = json.loads(alignment_path.read_text())
        paths.update([alignment_path, Path(alignment['reference_path']),
            B/case/'evaluation/paired_windows.csv', B/case/'inference/summary.json',
            B/case/'inference/frame_trace.csv', B/case/'inference/frame_trace.json',
            V28/case/'waveform.csv', V28/case/'heart_rate.csv', V28/case/'summary.json'])
    return {str(path): sha(path) for path in sorted(paths, key=str)}


def verify_hashes(hashes):
    assert hashes and isinstance(hashes, dict)
    for path, expected in hashes.items():
        assert sha(path) == expected, 'Changed bound source/input: '+str(path)


def spans(mask):
    mask = np.asarray(mask, bool)
    change = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def intervals(mask, fps):
    return [dict(start_frame=int(a), stop_frame_exclusive=int(b), start_s=a/fps,
                 end_s_exclusive=b/fps, duration_s=(b-a)/fps) for a, b in spans(mask)]


def waveform_availability(finite, fps):
    finite = np.asarray(finite, bool)
    assert finite.ndim == 1 and len(finite) > 0 and np.isfinite(fps) and fps > 0
    n = len(finite)
    valid_indices = np.flatnonzero(finite)
    gaps = spans(~finite)
    first, last = (int(valid_indices[0]), int(valid_indices[-1])) if finite.any() else (n, -1)
    leading = first/fps
    trailing = (n-1-last)/fps if finite.any() else 0.
    missing = (n-int(finite.sum()))/fps
    return dict(frames=n, fps=fps, total_duration_s=n/fps, finite_frames=int(finite.sum()),
        finite_duration_s=int(finite.sum())/fps, waveform_coverage_pct=100*finite.mean(),
        first_finite_time_s=first/fps if finite.any() else None,
        last_finite_time_s=last/fps if finite.any() else None,
        leading_missing_s=leading, trailing_missing_s=trailing,
        internal_missing_s=missing-leading-trailing, missing_duration_s=missing,
        longest_missing_s=max((b-a)/fps for a, b in gaps) if gaps else 0.,
        missing_runs=len(gaps), no_finite_output=not bool(finite.any()),
        missing_intervals=intervals(~finite, fps),
        timing_interpretation='Offline source timestamps; first finite sample is not processing latency or an online availability guarantee',
        all_missing_convention='If no sample is finite, all missing time is assigned to leading_missing_s; trailing/internal are zero')


def hr_availability(hr, fps):
    accepted = bools(hr.accepted)
    hop_s = round(fps)/fps
    indices = np.flatnonzero(accepted)
    gaps = spans(~accepted)
    first = hr.iloc[int(indices[0])] if len(indices) else None
    return dict(first_accepted_center_s=float(first.time_s) if first is not None else None,
        first_accepted_window_start_s=float(first.window_start_s) if first is not None else None,
        first_accepted_window_end_s=float(first.window_end_s) if first is not None else None,
        Nplanned=len(hr), Noutput=int(accepted.sum()), rejected_updates=int((~accepted).sum()),
        rejected_update_duration_s=int((~accepted).sum())*hop_s,
        longest_rejected_run_updates=max((b-a for a, b in gaps), default=0),
        longest_rejected_update_duration_s=max((b-a for a, b in gaps), default=0)*hop_s,
        rejected_runs=[dict(first_window_index=int(a), stop_window_index_exclusive=int(b),
            first_center_s=float(hr.iloc[a].time_s), last_center_s=float(hr.iloc[b-1].time_s),
            unavailable_updates=int(b-a), nominal_duration_s=(b-a)*hop_s) for a, b in gaps],
        status_counts=hr.status.value_counts().to_dict(),
        duration_definition='Rejected update counts times the actual frame-rounded hop; excludes unplanned leading/trailing margins and is not union of analysis-window durations',
        timing_interpretation='Window end is required input time for that analysis window, not measured online latency; the estimator is offline')


def match_centers(native_centers, old_centers, tolerance=1e-8):
    """Match actual floating timestamps, never round them to integer seconds."""
    native, old = np.asarray(native_centers, float), np.asarray(old_centers, float)
    assert np.isfinite(native).all() and np.isfinite(old).all()
    assert (np.diff(native) > 0).all() and (np.diff(old) > 0).all()
    result = np.full(len(native), -1, int)
    for i, value in enumerate(native):
        index = int(np.searchsorted(old, value))
        nearby = [j for j in (index-1, index) if 0 <= j < len(old) and abs(old[j]-value) <= tolerance]
        assert len(nearby) <= 1, 'Ambiguous time match'
        if nearby:
            result[i] = nearby[0]
    assert len(set(result[result >= 0])) == int((result >= 0).sum())
    return result


def validate_hr(hr, frames, fps, window_s, finite):
    window, hop = round(window_s*fps), round(fps)
    starts = np.arange(0, frames-window+1, hop)
    assert len(hr) == len(starts)
    for column, expected in [('time_s', (starts+window/2)/fps),
                             ('window_start_s', starts/fps), ('window_end_s', (starts+window)/fps)]:
        np.testing.assert_allclose(hr[column], expected, rtol=0, atol=1e-8)
    accepted, y = bools(hr.accepted), hr.ridge_bpm.to_numpy(float)
    assert np.isnan(y[~accepted]).all() and np.isfinite(y[accepted]).all()
    assert all(np.asarray(finite)[a:a+window].all() for a in starts[accepted]), 'Accepted HR crosses missing waveform'
    for index in np.flatnonzero(accepted):
        candidates = json.loads(hr.iloc[index].evidence_candidates_json)
        assert any(any(abs(float(bpm)-y[index]) <= 1e-9 for bpm in row['supported_bpm']) for row in candidates)
    return y, accepted


def pool(rows):
    n, nr = sum(x['Nvalid'] for x in rows), sum(x['Nref'] for x in rows)
    n5 = sum(x['Nwithin5'] for x in rows)
    return dict(Nplanned=sum(x['Nplanned'] for x in rows), Noutput=sum(x['Noutput'] for x in rows),
        Nref=nr, Nvalid=n, Nwithin5=n5,
        MAE_bpm=sum(x['Nvalid']*x['MAE_bpm'] for x in rows if x['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(x['Nvalid']*x['RMSE_bpm']**2 for x in rows if x['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*n5/n if n else np.nan, R5_all_reference_pct=100*n5/nr if nr else np.nan,
        hr_output_coverage_pct=100*sum(x['Noutput'] for x in rows)/sum(x['Nplanned'] for x in rows) if rows else np.nan)


def error_group(y, reference, mask):
    eligible = np.asarray(mask, bool) & np.isfinite(reference) & (reference > 0) & np.isfinite(y)
    errors = abs(y[eligible]-reference[eligible])
    return dict(Nvalid=int(eligible.sum()), Nwithin5=int((errors <= 5).sum()),
        MAE_bpm=float(errors.mean()) if len(errors) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        P5_valid_pct=100*float((errors <= 5).mean()) if len(errors) else np.nan)


def comparison(y, ok, reference, old_y, old_ok):
    common, new, lost = ok & old_ok, ok & ~old_ok, ~ok & old_ok
    return dict(common_output_windows=int(common.sum()), new_output_windows=int(new.sum()),
        lost_output_windows=int(lost.sum()),
        common_new=error_group(y, reference, common), common_V28=error_group(old_y, reference, common),
        newly_covered=error_group(y, reference, new), lost_V28=error_group(old_y, reference, lost)), (common, new, lost)


def deltas(new, old):
    return {('delta_'+key): new[key]-old[key] for key in (
        'MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct', 'hr_output_coverage_pct')}


def pair_table(hr, reference):
    paired = hr[['time_s', 'window_start_s', 'window_end_s', 'ridge_bpm', 'accepted', 'status']].copy()
    paired.insert(0, 'window_index', np.arange(len(paired)))
    paired = pd.concat([paired.reset_index(drop=True), reference.reset_index(drop=True)], axis=1)
    paired['error_bpm'] = np.where(bools(paired.accepted) & bools(paired.reference_valid),
                                   paired.ridge_bpm-paired.reference_bpm, np.nan)
    paired['abs_error_bpm'] = abs(paired.error_bpm)
    return paired


def evaluate():
    core = core_module()
    protocol_path = ROOT/'protocol_before_run.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = sha(protocol_path)
    assert protocol['evaluation_inputs'] == freeze_evaluation_inputs()
    verify_hashes(protocol['evaluation_hashes'])
    assert str(Path(__file__).resolve()) in protocol['evaluation_hashes']
    sources = {str(Path(name) if Path(name).is_absolute() else HERE/name): value
               for name, value in protocol['source_hashes'].items()}
    verify_hashes(sources)
    assert set(protocol['profiles']) == set(PROFILES)
    assert not (ROOT/'evaluation_summary.json').exists(), 'Preserve completed evaluations'
    hashes = {str(protocol_path): protocol_hash, **protocol['evaluation_inputs'], **protocol['evaluation_hashes'], **sources}
    # Complete both fixed profiles and all six cases before reading any new score.
    for profile in PROFILES:
        for case in CASES:
            directory = ROOT/profile/case
            summary = json.loads((directory/'summary.json').read_text())
            assert summary['status'] == 'complete' and summary['reference_used'] is False
            assert summary['source_hashes'] == protocol['source_hashes']
            assert summary['protocol_sha256'] == protocol_hash and summary['case'] == case
            assert summary.get('profile', summary.get('variant')) == profile
            assert summary['input_hashes'] == protocol['input_hashes'][case]
            verify_hashes(summary['input_hashes'])
            for filename in ('waveform.csv', 'heart_rate.csv', 'heart_rate_10s.csv'):
                assert summary['output_hashes'][filename] == sha(directory/filename)
                hashes[str(directory/filename)] = sha(directory/filename)
            hashes[str(directory/'summary.json')] = sha(directory/'summary.json')
    baseline_rows, profiles = [], {}
    all_native, all_common, all_availability = [], [], []
    for profile in PROFILES:
        native_rows, common_rows, availability_rows, differences = [], [], [], []
        native_groups, common_groups = {}, {}
        for case in CASES:
            directory = ROOT/profile/case
            out = directory/'evaluation'
            assert not out.exists(), 'Preserve partial/completed evaluation artifacts'
            meta = json.loads((B/case/'inference/summary.json').read_text())
            fps, frames = float(meta['fps']), int(meta['frames'])
            wave, old_wave = pd.read_csv(directory/'waveform.csv'), pd.read_csv(V28/case/'waveform.csv')
            hr, common_hr = pd.read_csv(directory/'heart_rate.csv'), pd.read_csv(directory/'heart_rate_10s.csv')
            old_hr = pd.read_csv(V28/case/'heart_rate.csv')
            assert len(wave) == len(old_wave) == frames
            np.testing.assert_allclose(wave.time_s, np.arange(frames)/fps, rtol=0, atol=1e-8)
            np.testing.assert_allclose(wave.time_s, old_wave.time_s, rtol=0, atol=1e-8)
            finite, old_finite = np.isfinite(wave.base.to_numpy(float)), np.isfinite(old_wave.base.to_numpy(float))
            assert not np.isinf(wave.base).any()
            np.testing.assert_array_equal(finite, bools(wave.covered))
            assert not (bools(wave.observed) & ~finite).any() and not (bools(wave.interpolated) & ~finite).any()
            y6, ok6 = validate_hr(hr, frames, fps, 6, finite)
            y10, ok10 = validate_hr(common_hr, frames, fps, 10, finite)
            old_y, old_ok = validate_hr(old_hr, frames, fps, 10, old_finite)
            alignment = json.loads((B/case/'evaluation/alignment.json').read_text())
            raw = pd.read_csv(alignment['reference_path'])
            if profile == PROFILES[0]:
                reference_dir = ROOT/'reference'
                reference_dir.mkdir(exist_ok=True)
                timeline_path = reference_dir/(case+'.csv')
                assert not timeline_path.exists(), 'Preserve aligned reference timeline'
                timeline = pd.DataFrame(dict(
                    time_s=(raw.host_utc_ns.to_numpy(np.int64)-np.int64(alignment['video_start_utc_ns'])).astype(float)/1e9,
                    reference_bpm=raw.hr_bpm.to_numpy(float)))
                timeline = timeline.loc[np.isfinite(timeline.reference_bpm) & (timeline.reference_bpm > 0)]
                timeline.to_csv(timeline_path, index=False)
            ref6 = core.reference_windows(raw, alignment['video_start_utc_ns'], hr.window_start_s, hr.window_end_s, 0)
            ref10 = core.reference_windows(raw, alignment['video_start_utc_ns'], common_hr.window_start_s, common_hr.window_end_s, 0)
            frozen_reference = pd.read_csv(B/case/'evaluation/paired_windows.csv')
            np.testing.assert_allclose(ref10.reference_bpm, frozen_reference.reference_bpm, rtol=0, atol=1e-9, equal_nan=True)
            r6, r10 = ref6.reference_bpm.to_numpy(float), ref10.reference_bpm.to_numpy(float)
            m6, m10 = core.score(y6, ok6, r6), core.score(y10, ok10, r10)
            old_m = core.score(old_y, old_ok, r10)
            wave_av, old_wave_av = waveform_availability(finite, fps), waveform_availability(old_finite, fps)
            hr_av6, hr_av10, old_hr_av = hr_availability(hr, fps), hr_availability(common_hr, fps), hr_availability(old_hr, fps)
            nrow = dict(profile=profile, case=case, window_s=6, **m6,
                **{k: v for k, v in hr_av6.items() if k.startswith('first_')},
                **{k: wave_av[k] for k in ('waveform_coverage_pct', 'first_finite_time_s', 'missing_duration_s', 'longest_missing_s')})
            diff, masks = comparison(y10, ok10, r10, old_y, old_ok)
            crow = dict(profile=profile, case=case, window_s=10, **m10, **deltas(m10, old_m),
                common_windows=diff['common_output_windows'], new_windows=diff['new_output_windows'], lost_windows=diff['lost_output_windows'],
                common_MAE_bpm=diff['common_new']['MAE_bpm'], common_V28_MAE_bpm=diff['common_V28']['MAE_bpm'],
                new_MAE_bpm=diff['newly_covered']['MAE_bpm'], new_Nwithin5=diff['newly_covered']['Nwithin5'],
                new_P5_valid_pct=diff['newly_covered']['P5_valid_pct'], lost_V28_MAE_bpm=diff['lost_V28']['MAE_bpm'])
            native_rows.append(nrow); common_rows.append(crow); differences.append(dict(case=case, **diff))
            if profile == PROFILES[0]:
                baseline_rows.append(dict(case=case, **old_m, waveform_availability=old_wave_av, hr_availability=old_hr_av))
            matched = match_centers(hr.time_s, old_hr.time_s)
            old_at_native = np.zeros(len(hr), bool)
            old_at_native[matched >= 0] = old_ok[matched[matched >= 0]]
            groups = dict(shared_center_both_output=ok6 & (matched >= 0) & old_at_native,
                shared_center_new_output=ok6 & (matched >= 0) & ~old_at_native,
                native_only_plan_output=ok6 & (matched < 0))
            native_groups[case] = {k: error_group(y6, r6, mask) for k, mask in groups.items()}
            native_groups[case].update(native_plan_only_windows=int((matched < 0).sum()),
                shared_center_windows=int((matched >= 0).sum()),
                interpretation='Native 6 s estimates scored against 6 s reference means; V28 10 s acceptance is used only to classify added coverage at matching actual centers')
            common_groups[case] = diff
            transitions = pd.crosstab(pd.Series(old_hr.status, name='V28_status'), pd.Series(common_hr.status, name='V29_common10s_status'))
            new_wave, lost_wave = finite & ~old_finite, ~finite & old_finite
            availability = dict(profile=profile, case=case, waveform=wave_av, V28_waveform=old_wave_av,
                native6s_hr=hr_av6, common10s_hr=hr_av10, V28_hr=old_hr_av,
                new_waveform_seconds=int(new_wave.sum())/fps, lost_waveform_seconds=int(lost_wave.sum())/fps,
                new_waveform_intervals=intervals(new_wave, fps), lost_waveform_intervals=intervals(lost_wave, fps))
            availability_rows.append(availability)
            out.mkdir()
            paired6, paired10 = pair_table(hr, ref6), pair_table(common_hr, ref10)
            paired6['V28_matching_10s_window_index'] = matched
            paired6['V28_10s_accepted_at_same_center'] = old_at_native
            for label, mask in groups.items():
                paired6[label] = mask
            paired10['V28_bpm'], paired10['V28_accepted'] = old_y, old_ok
            paired10['common_output'], paired10['new_output'], paired10['lost_output'] = masks
            paired6.to_csv(out/'paired_native6s.csv', index=False)
            paired10.to_csv(out/'paired_common10s.csv', index=False)
            paired10.loc[masks[1]].to_csv(out/'newly_covered_common10s.csv', index=False)
            transitions.to_csv(out/'status_transitions_common10s.csv')
            write(out/'waveform_availability.json', availability)
            write(out/'metrics.json', dict(native6s=nrow, common10s=crow, native_coverage_groups=native_groups[case], common_coverage_groups=diff))
            sensitivity = []
            for plan, table, yy, kk in [('native6s', hr, y6, ok6), ('common10s', common_hr, y10, ok10)]:
                for shift in SHIFTS:
                    shifted = core.reference_windows(raw, alignment['video_start_utc_ns'], table.window_start_s, table.window_end_s, shift)
                    sensitivity.append(dict(profile=profile, case=case, plan=plan, shift_s=shift,
                        **core.score(yy, kk, shifted.reference_bpm.to_numpy(float))))
            pd.DataFrame(sensitivity).to_csv(out/'alignment_sensitivity.csv', index=False)
        native_pooled, common_pooled, old_pooled = pool(native_rows), pool(common_rows), pool(baseline_rows)
        assert native_pooled['Nplanned'] == 333 and common_pooled['Nplanned'] == old_pooled['Nplanned'] == 309
        # Pool each common/new/lost group by its own number of reference-paired outputs.
        grouped = {}
        for key in ('common_new', 'common_V28', 'newly_covered', 'lost_V28'):
            entries = [x[key] for x in differences]
            total = sum(x['Nvalid'] for x in entries)
            grouped[key] = dict(Nvalid=total, Nwithin5=sum(x['Nwithin5'] for x in entries),
                MAE_bpm=sum(x['Nvalid']*x['MAE_bpm'] for x in entries if x['Nvalid'])/total if total else np.nan,
                P5_valid_pct=100*sum(x['Nwithin5'] for x in entries)/total if total else np.nan)
        profiles[profile] = dict(native6s=dict(pooled=native_pooled, cases=native_rows, coverage_groups=native_groups),
            common10s=dict(pooled=common_pooled, cases=common_rows, vs_V28=deltas(common_pooled, old_pooled), coverage_groups=grouped),
            availability=availability_rows,
            waveform_time_coverage_pct=100*sum(x['waveform']['finite_duration_s'] for x in availability_rows)/sum(x['waveform']['total_duration_s'] for x in availability_rows),
            data6_detail=dict(native6s=next(x for x in native_rows if x['case'] == 'data6'),
                common10s=next(x for x in common_rows if x['case'] == 'data6'),
                availability=next(x for x in availability_rows if x['case'] == 'data6'),
                native_coverage_groups=native_groups['data6'], common_coverage_groups=common_groups['data6']))
        all_native += native_rows; all_common += common_rows; all_availability += availability_rows
    verify_hashes(hashes)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), profiles=profiles,
        baseline_V28=dict(pooled=pool(baseline_rows), cases=baseline_rows), input_hashes=hashes,
        reference_timelines={case:str(ROOT/'reference'/(case+'.csv')) for case in CASES},
        comparison_rules=['Native6s and V28 have different windows and reference means; their accuracy rates are not improvement deltas.',
            'Common10s uses the same saved final waveform with original strict 10 s evidence gates, the 309 original plans and unchanged original reference means.',
            'New coverage accuracy is shown separately from common output accuracy. Missing outputs remain failures in all-reference success.',
            'First finite and accepted timestamps describe offline result locations. No measured real-time latency is claimed.',
            'Original Polar offsets are estimated from archived timestamps, not hardware verified; all declared shifts are reported without selecting an offset.',
            'Previously inspected development videos and overlapping windows do not establish held-out accuracy or independent-sample confidence intervals.'])
    write(ROOT/'evaluation_summary.json', report)
    pd.DataFrame(all_native).to_csv(ROOT/'metrics_native6s.csv', index=False)
    pd.DataFrame(all_common).to_csv(ROOT/'metrics_common10s.csv', index=False)
    write(ROOT/'availability_summary.json', all_availability)
    print(json.dumps(clean(dict(profiles={k: dict(native6s=v['native6s']['pooled'], common10s=v['common10s']['pooled'],
        common10s_vs_V28=v['common10s']['vs_V28'], data6_native=v['data6_detail']['native6s'],
        data6_common=v['data6_detail']['common10s']) for k, v in profiles.items()}, baseline_V28=pool(baseline_rows))), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-inputs', action='store_true')
    args = parser.parse_args()
    if args.freeze_inputs:
        print(json.dumps(freeze_evaluation_inputs(), ensure_ascii=False, indent=2))
    else:
        evaluate()
