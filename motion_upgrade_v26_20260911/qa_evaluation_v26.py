"""Independent CSV/reference arithmetic QA; imports no inference or evaluator code.

Only new QA receipts are written. All reference offsets remain fixed, and none
of the computed metrics is available to an inference routine in this script.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT/'results/data1_6_20260911'
V25 = PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
ROOT = PROJECT/'results/data1_6_v26_20260911'
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
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def bools(series):
    assert series.notna().all() and series.isin([True, False, 0, 1]).all()
    return series.to_numpy(bool)


def metrics(prediction, accepted, reference):
    prediction, reference = np.asarray(prediction, float), np.asarray(reference, float)
    accepted = np.asarray(accepted, bool)
    output = accepted & np.isfinite(prediction)
    eligible = np.isfinite(reference) & (reference > 0)
    use = output & eligible
    error = prediction[use]-reference[use]
    absolute = abs(error)
    n, no, nr, nv = len(prediction), int(output.sum()), int(eligible.sum()), int(use.sum())
    n5, n10 = int((absolute <= 5).sum()), int((absolute <= 10).sum())
    bias = float(error.mean()) if nv else np.nan
    sd = float(error.std(ddof=1)) if nv > 1 else np.nan
    return dict(Nplanned=n, Noutput=no, Nref=nr, Nvalid=nv, Nwithin5=n5, Nwithin10=n10,
        hr_output_coverage_pct=100*no/n if n else np.nan,
        paired_reference_coverage_pct=100*nv/nr if nr else np.nan,
        MAE_bpm=float(absolute.mean()) if nv else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(error*error))) if nv else np.nan,
        Bias_bpm=bias, MAPE_pct=float(np.mean(absolute/reference[use])*100) if nv else np.nan,
        P5_valid_pct=100*n5/nv if nv else np.nan, P10_valid_pct=100*n10/nv if nv else np.nan,
        R5_all_reference_pct=100*n5/nr if nr else np.nan,
        R10_all_reference_pct=100*n10/nr if nr else np.nan,
        P95_abs_error_bpm=float(np.quantile(absolute, .95)) if nv else np.nan,
        Max_abs_error_bpm=float(absolute.max()) if nv else np.nan,
        LoA_lower_descriptive_bpm=bias-1.96*sd, LoA_upper_descriptive_bpm=bias+1.96*sd)


def references(raw, origin, starts, stops, shift):
    # Epoch subtraction precedes float conversion: never round ~1e18 UTC ns.
    utc = raw.host_utc_ns.to_numpy(np.int64)
    assert np.all(np.diff(utc) > 0)
    times = (utc-np.int64(origin)).astype(float)*1e-9-float(shift)
    values = raw.hr_bpm.to_numpy(float)
    keep = np.isfinite(values) & (values > 0)
    times, values = times[keep], values[keep]
    assert len(times) > 1
    tail = np.median(np.diff(times))
    result = []
    for start, stop in zip(starts, stops):
        take = (times >= start) & (times < stop)
        points = times[take]
        maxgap = np.diff(np.r_[start, points, stop]).max()
        valid = start >= times[0] and stop <= times[-1]+tail and len(points) >= 2 and maxgap <= 2
        result.append(float(values[take].mean()) if valid else np.nan)
    return np.asarray(result)


def pooled(rows):
    n = sum(r['Nvalid'] for r in rows)
    nr = sum(r['Nref'] for r in rows)
    nc = sum(r['common_windows'] for r in rows)
    return dict(Nref=nr, Nvalid=n, Nwithin5=sum(r['Nwithin5'] for r in rows),
        MAE_bpm=sum(r['Nvalid']*r['MAE_bpm'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['Nvalid']*r['RMSE_bpm']**2 for r in rows if r['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*sum(r['Nwithin5'] for r in rows)/n if n else np.nan,
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in rows)/nr if nr else np.nan,
        HR_coverage_pct=100*sum(r['Noutput'] for r in rows)/sum(r['Nplanned'] for r in rows),
        common_windows=nc,
        common_MAE_bpm=sum(r['common_windows']*r['common_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan,
        common_V25_MAE_bpm=sum(r['common_windows']*r['common_V25_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['log_projection_baseline', 'log_projection_guarded', 'neural_efficientphys'])
    parser.add_argument('--output', type=Path, default=ROOT/'qa_evaluation_v26.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed QA; choose a new receipt path.')
    errors, input_hashes, totals, detail = [], {}, {'numeric_comparisons': 0, 'maximum_absolute_difference': 0.}, []

    def compare(actual, expected, label):
        if isinstance(expected, (bool, str)):
            if actual != expected:
                errors.append(label+': discrete mismatch')
            return
        a = np.nan if actual is None else float(actual)
        e = np.nan if expected is None else float(expected)
        totals['numeric_comparisons'] += 1
        if np.isnan(a) and np.isnan(e):
            return
        if np.isfinite(a) and np.isfinite(e):
            delta = abs(a-e)
            totals['maximum_absolute_difference'] = max(totals['maximum_absolute_difference'], delta)
            if np.isclose(a, e, rtol=1e-10, atol=1e-9):
                return
        errors.append(f'{label}: saved={actual}, independently calculated={expected}')

    def compare_dict(saved, expected, label):
        for key, value in expected.items():
            if key not in saved:
                errors.append(label+'/'+key+': missing key')
            else:
                compare(saved[key], value, label+'/'+key)

    def bind(path):
        path = Path(path)
        input_hashes[str(path)] = sha(path)

    # Meaningful edge cases: missing estimates are failures in R5, never in P5.
    sanity = metrics([70., np.nan, 86.], [True, False, True], [70., 80., 80.])
    assert sanity['Nvalid'] == 2 and sanity['Nwithin5'] == 1
    assert sanity['P5_valid_pct'] == 50 and sanity['R5_all_reference_pct'] == 100/3
    assert np.isnan(metrics([np.nan], [False], [np.nan])['MAE_bpm'])
    baseline_rows = []
    for i in range(1, 7):
        case = f'data{i}'
        old = pd.read_csv(V25/case/'fusion_heart_rate.csv')
        reference = pd.read_csv(BASE/case/'evaluation/paired_windows.csv').reference_bpm.to_numpy(float)
        baseline_rows.append(metrics(old.ridge_bpm, bools(old.accepted), reference))
    total_valid = sum(r['Nvalid'] for r in baseline_rows)
    prior_pooled = dict(MAE_bpm=sum(r['Nvalid']*r['MAE_bpm'] for r in baseline_rows)/total_valid,
        RMSE_bpm=np.sqrt(sum(r['Nvalid']*r['RMSE_bpm']**2 for r in baseline_rows)/total_valid),
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in baseline_rows)/sum(r['Nref'] for r in baseline_rows))
    baseline_report_path = V25/'stage_evaluation.json'
    compare_dict(json.loads(baseline_report_path.read_text())['pooled'], prior_pooled, 'V25 pooled recomputation')
    bind(baseline_report_path)

    for variant in args.variants:
        directory = ROOT/variant
        report = json.loads((directory/'evaluation_summary.json').read_text())
        saved_csv = pd.read_csv(directory/'metrics.csv').set_index('case')
        rows = []
        assert len(report['cases']) == 6 and len(saved_csv) == 6
        for i in range(1, 7):
            case = f'data{i}'
            folder = directory/case
            hr = pd.read_csv(folder/'heart_rate.csv')
            wave = pd.read_csv(folder/'waveform.csv')
            old = pd.read_csv(V25/case/'fusion_heart_rate.csv')
            old_wave = pd.read_csv(V25/case/'fusion_waveform.csv')
            ref = pd.read_csv(BASE/case/'evaluation/paired_windows.csv')
            meta = json.loads((BASE/case/'inference/summary.json').read_text())
            alignment = json.loads((BASE/case/'evaluation/alignment.json').read_text())
            raw = pd.read_csv(alignment['reference_path'])
            assert sha(alignment['reference_path']) == alignment['reference_sha256']
            fps, nframes = float(meta['fps']), int(meta['frames'])
            assert nframes == len(wave) == len(old_wave)
            width, hop = round(10*fps), round(fps)
            starts = np.arange(0, nframes-width+1, hop)
            np.testing.assert_allclose(wave.time_s, np.arange(nframes)/fps, atol=1e-8, rtol=0)
            np.testing.assert_allclose(ref.time_s, (starts+width/2)/fps, atol=1e-8, rtol=0)
            np.testing.assert_allclose(ref.window_start_s, starts/fps, atol=1e-8, rtol=0)
            np.testing.assert_allclose(ref.window_end_s, (starts+width)/fps, atol=1e-8, rtol=0)
            np.testing.assert_allclose(hr.time_s, ref.time_s, atol=1e-8, rtol=0)
            np.testing.assert_allclose(old.time_s, ref.time_s, atol=1e-8, rtol=0)
            r = references(raw, alignment['video_start_utc_ns'], ref.window_start_s, ref.window_end_s, 0)
            np.testing.assert_allclose(r, ref.reference_bpm, atol=1e-9, rtol=0, equal_nan=True)
            eligible = np.isfinite(r) & (r > 0)
            np.testing.assert_array_equal(eligible, bools(ref.reference_valid))
            ok, prior_ok = bools(hr.accepted), bools(old.accepted)
            y, oy = hr.ridge_bpm.to_numpy(float), old.ridge_bpm.to_numpy(float)
            assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
            finite, old_finite = np.isfinite(wave.base.to_numpy(float)), np.isfinite(old_wave.base.to_numpy(float))
            assert all(finite[a:a+width].all() for a in starts[ok])
            if 'covered' in wave:
                np.testing.assert_array_equal(finite, bools(wave.covered))
            common, new, lost = ok & prior_ok & eligible, ok & ~prior_ok & eligible, ~ok & prior_ok & eligible
            m, oldm = metrics(y, ok, r), metrics(oy, prior_ok, r)
            row = dict(**m, V25_MAE_bpm=oldm['MAE_bpm'], delta_MAE_bpm=m['MAE_bpm']-oldm['MAE_bpm'],
                delta_P5_pp=m['P5_valid_pct']-oldm['P5_valid_pct'],
                delta_R5_pp=m['R5_all_reference_pct']-oldm['R5_all_reference_pct'],
                delta_hr_coverage_pp=m['hr_output_coverage_pct']-oldm['hr_output_coverage_pct'],
                waveform_coverage_pct=100*finite.mean(), delta_waveform_coverage_pp=100*(finite.mean()-old_finite.mean()),
                common_windows=int(common.sum()), new_windows=int(new.sum()), lost_windows=int(lost.sum()),
                common_MAE_bpm=float(abs(y[common]-r[common]).mean()) if common.any() else np.nan,
                common_V25_MAE_bpm=float(abs(oy[common]-r[common]).mean()) if common.any() else np.nan,
                new_MAE_bpm=float(abs(y[new]-r[new]).mean()) if new.any() else np.nan,
                lost_V25_MAE_bpm=float(abs(oy[lost]-r[lost]).mean()) if lost.any() else np.nan,
                mean_HR_bpm=float(y[ok].mean()) if ok.any() else np.nan,
                target_P5_95_met=bool(m['P5_valid_pct'] >= 95))
            saved = json.loads((folder/'evaluation/metrics.json').read_text())
            compare_dict(saved, row, variant+'/'+case+'/metrics')
            compare_dict(saved_csv.loc[case].to_dict(), row, variant+'/'+case+'/csv')
            report_case = next(x for x in report['cases'] if x['case'] == case)
            compare_dict(report_case, row, variant+'/'+case+'/summary')
            assert saved['status_counts'] == hr.status.value_counts().to_dict()
            paired = pd.read_csv(folder/'evaluation/paired_windows.csv')
            assert len(paired) == len(ref)
            for key, expected in [('accepted',ok), ('V25_accepted',prior_ok), ('common',common), ('new',new), ('lost',lost)]:
                np.testing.assert_array_equal(bools(paired[key]), expected)
            for key, expected in [('estimated_bpm',y), ('V25_bpm',oy), ('reference_bpm',r), ('error_bpm',np.where(ok, y-r, np.nan)), ('abs_error_bpm',np.where(ok, abs(y-r), np.nan))]:
                np.testing.assert_allclose(paired[key], expected, rtol=0, atol=1e-9, equal_nan=True)
            sensitivity = pd.read_csv(folder/'evaluation/alignment_sensitivity.csv').set_index('shift_s')
            assert sorted(sensitivity.index) == list(SHIFTS)
            for shift in SHIFTS:
                rs = references(raw, alignment['video_start_utc_ns'], ref.window_start_s, ref.window_end_s, shift)
                compare_dict(sensitivity.loc[shift].to_dict(), metrics(y,ok,rs), variant+'/'+case+f'/shift{shift}')
            rows.append(row)
            detail.append(dict(variant=variant, case=case, independently_recomputed=row,
                source_reference='Polar device HR notifications; estimated archived-mtime alignment',
                reference_sync_hardware_verified=False))
            paths = [folder/'heart_rate.csv',folder/'waveform.csv',folder/'summary.json',
                folder/'evaluation/metrics.json',folder/'evaluation/paired_windows.csv',folder/'evaluation/alignment_sensitivity.csv',
                V25/case/'fusion_heart_rate.csv',V25/case/'fusion_waveform.csv',
                BASE/case/'evaluation/paired_windows.csv',BASE/case/'evaluation/alignment.json',
                BASE/case/'inference/summary.json',Path(alignment['reference_path'])]
            for path in paths:
                bind(path)
        agg = pooled(rows)
        compare_dict(report['pooled'], agg, variant+'/pooled')
        checks = dict(pooled_MAE=agg['MAE_bpm'] < prior_pooled['MAE_bpm'],
            pooled_R5_gain=agg['R5_all_reference_pct'] >= prior_pooled['R5_all_reference_pct']+5,
            each_MAE=all(r['delta_MAE_bpm'] <= 3 for r in rows),
            each_P5=all(r['delta_P5_pp'] >= -3 for r in rows),
            each_R5=all(r['delta_R5_pp'] >= -3 for r in rows),
            each_HR_coverage=all(r['delta_hr_coverage_pp'] >= -3 for r in rows),
            each_waveform_coverage=all(r['delta_waveform_coverage_pp'] >= -3 for r in rows),
            common_MAE=agg['common_MAE_bpm'] < agg['common_V25_MAE_bpm'])
        compare_dict(report['promotion_checks'], checks, variant+'/gates')
        compare(report['promotion_pass'], all(checks.values()), variant+'/promotion_pass')
        compare(report['near_all_within_5bpm_target_met'], all(r['target_P5_95_met'] for r in rows), variant+'/near_all')
        for path, expected in report['input_hashes'].items():
            assert sha(path) == expected, 'Evaluator-recorded input hash changed: '+path
        for name in ['evaluation_summary.json','metrics.csv','protocol_before_run.json','protocol_before_inference.json','runs.json','run_summary.json']:
            if (directory/name).exists():
                bind(directory/name)
        print(json.dumps(clean(dict(variant=variant, pooled=agg, promotion_pass=all(checks.values()), near_all=all(r['target_P5_95_met'] for r in rows))),ensure_ascii=False),flush=True)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors, errors=errors,
        independent_code_sha256=sha(__file__),variants=args.variants,checked_case_variants=len(detail),
        checked_shift_rows=len(detail)*7,**totals,case_results=detail,input_hashes=input_hashes,
        evidence_timing='These QA source/baseline/core/reference hashes are post-run supplemental binding, not retroactive pre-run freezing.',
        review=['Independent arithmetic implementation; no import of evaluate_v26, reference_core, or production models.',
            'Epoch nanoseconds subtracted as int64 before floating conversion; all seven declared shifts independently recomputed without choosing the best.',
            'P5 uses paired accepted outputs; R5 uses all reference-eligible planned windows; missing outputs are not successes.',
            'Promotion guardrails represent partial development improvement and do not imply the per-case P5>=95% target was achieved.',
            'Waveform finite coverage is availability, not proof of physiological morphology recovery.',
            'Previously inspected recordings and overlapping windows are development evidence, not held-out accuracy or independent sample confidence intervals.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(passed=result['passed'],**totals,errors=errors,receipt=str(args.output))),ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
