"""Independently recompute V30 tables/guards from saved CSVs; no model imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v30_20260912'
B = P/'results/data1_6_20260911'
V28 = P/'results/data1_6_v28_20260912/direct_guard'
VARIANTS = ('motion_full_spectrum', 'cdf_v28')
CASES = tuple(f'data{i}' for i in range(1, 7))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(y, accepted, reference):
    ref_valid = np.isfinite(reference) & (reference > 0)
    valid = accepted & np.isfinite(y) & ref_valid
    error = y[valid]-reference[valid]
    n5 = int(np.count_nonzero(abs(error) <= 5))
    return dict(Nplanned=len(y), Noutput=int(np.count_nonzero(accepted)), Nref=int(np.count_nonzero(ref_valid)),
        Nvalid=len(error), Nwithin5=n5, MAE_bpm=float(np.mean(abs(error))) if len(error) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(error*error))) if len(error) else np.nan,
        P5_valid_pct=100*n5/len(error) if len(error) else np.nan,
        R5_all_reference_pct=100*n5/ref_valid.sum() if ref_valid.any() else np.nan,
        hr_output_coverage_pct=100*accepted.mean())


def number(value): return np.nan if value is None else float(value)


def check_metrics(actual, reported):
    for key, expected in actual.items():
        np.testing.assert_allclose(number(reported[key]), number(expected), rtol=0, atol=1e-9, equal_nan=True,
                                   err_msg='Metric mismatch: '+key)


def main():
    target = R/'qa_evaluation_v30.json'
    if target.exists(): raise FileExistsError('Preserve completed QA')
    source = R/'evaluation_summary.json'
    report = json.loads(source.read_text())
    assert set(report['variants']) == set(VARIANTS)
    for group in ('input_hashes', 'evaluation_output_hashes'):
        for path, digest in report[group].items(): assert sha(path) == digest, 'Bound artifact changed: '+path
    rows = []
    for variant in VARIANTS:
        result = report['variants'][variant]
        output, old_output, refs, masks, old_masks = [], [], [], [], []
        finite_seconds, total_seconds = 0., 0.
        deltas, case_receipts = [], []
        for case in CASES:
            directory = R/variant/case
            paired = pd.read_csv(directory/'evaluation/paired_windows.csv')
            hr = pd.read_csv(directory/'heart_rate.csv')
            old = pd.read_csv(V28/case/'heart_rate.csv')
            reference = pd.read_csv(B/case/'evaluation/paired_windows.csv')
            wave = pd.read_csv(directory/'waveform.csv')
            old_wave = pd.read_csv(V28/case/'waveform.csv')
            meta = json.loads((B/case/'inference/summary.json').read_text())
            fps = float(meta['fps'])
            y, oy, r = hr.ridge_bpm.to_numpy(float), old.ridge_bpm.to_numpy(float), reference.reference_bpm.to_numpy(float)
            ok, ook = hr.accepted.to_numpy(bool), old.accepted.to_numpy(bool)
            np.testing.assert_allclose(paired.reference_bpm, r, rtol=0, atol=0, equal_nan=True)
            np.testing.assert_allclose(paired.estimated_bpm, y, rtol=0, atol=0, equal_nan=True)
            assert np.array_equal(paired.accepted, ok)
            common, new, lost = ok & ook, ok & ~ook, ~ok & ook
            for name, mask in [('common_output', common), ('new_output', new), ('lost_output', lost)]:
                assert np.array_equal(paired[name], mask)
            metrics, old_metrics = metric(y, ok, r), metric(oy, ook, r)
            row = next(x for x in result['cases'] if x['case'] == case)
            baseline = next(x for x in result['baseline_V28']['cases'] if x['case'] == case)
            check_metrics(metrics, row); check_metrics(old_metrics, baseline)
            expected_error = np.where(ok & np.isfinite(r) & (r > 0), y-r, np.nan)
            np.testing.assert_allclose(paired.error_bpm, expected_error, rtol=0, atol=1e-9, equal_nan=True)
            finite, old_finite = np.isfinite(wave.base), np.isfinite(old_wave.base)
            assert np.array_equal(wave.covered, finite)
            wave_cov, old_wave_cov = 100*float(finite.mean()), 100*float(old_finite.mean())
            np.testing.assert_allclose(row['waveform_coverage_pct'], wave_cov, rtol=0, atol=1e-9)
            delta = dict(delta_MAE_bpm=metrics['MAE_bpm']-old_metrics['MAE_bpm'],
                delta_P5_pp=metrics['P5_valid_pct']-old_metrics['P5_valid_pct'],
                delta_R5_pp=metrics['R5_all_reference_pct']-old_metrics['R5_all_reference_pct'],
                delta_hr_coverage_pp=metrics['hr_output_coverage_pct']-old_metrics['hr_output_coverage_pct'],
                delta_waveform_coverage_pp=wave_cov-old_wave_cov)
            check_metrics(delta, row); deltas.append(delta)
            finite_seconds += int(finite.sum())/fps; total_seconds += len(wave)/fps
            output.append(y); old_output.append(oy); refs.append(r); masks.append(ok); old_masks.append(ook)
            case_receipts.append(dict(case=case, Nplanned=len(y), Noutput=int(ok.sum()),
                paired_sha256=sha(directory/'evaluation/paired_windows.csv'), passed=True))
        y, oy, r, ok, ook = [np.concatenate(x) for x in (output, old_output, refs, masks, old_masks)]
        pooled, baseline = metric(y, ok, r), metric(oy, ook, r)
        assert pooled['Nplanned'] == pooled['Nref'] == 309
        check_metrics({('HR_coverage_pct' if k == 'hr_output_coverage_pct' else k): v for k, v in pooled.items()}, result['pooled'])
        check_metrics({('HR_coverage_pct' if k == 'hr_output_coverage_pct' else k): v for k, v in baseline.items()}, result['baseline_V28']['pooled'])
        old_report = json.loads((V28/'evaluation_summary.json').read_text())['pooled']
        for key in ('MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct'):
            np.testing.assert_allclose(baseline[key], old_report[key], rtol=0, atol=1e-9)
        np.testing.assert_allclose(result['pooled']['waveform_time_coverage_pct'], 100*finite_seconds/total_seconds, rtol=0, atol=1e-9)
        common = ok & ook & np.isfinite(r) & (r > 0)
        common_mae = float(np.mean(abs(y[common]-r[common]))) if common.any() else np.nan
        old_common_mae = float(np.mean(abs(oy[common]-r[common]))) if common.any() else np.nan
        checks = dict(pooled_MAE=bool(pooled['MAE_bpm'] < baseline['MAE_bpm']),
            pooled_R5_gain=bool(pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']+5),
            each_MAE=all(d['delta_MAE_bpm'] <= 3 for d in deltas),
            each_P5=all(d['delta_P5_pp'] >= -3 for d in deltas),
            each_R5=all(d['delta_R5_pp'] >= -3 for d in deltas),
            each_HR_coverage=all(d['delta_hr_coverage_pp'] >= -3 for d in deltas),
            each_waveform_coverage=all(d['delta_waveform_coverage_pp'] >= -3 for d in deltas),
            common_MAE=bool(common_mae < old_common_mae))
        assert checks == result['promotion_checks']
        assert all(checks.values()) == result['promotion_pass']
        rows.append(dict(variant=variant, cases=case_receipts, baseline_V28_reproduced=True,
            all_metrics_recomputed=True, time_weighted_waveform_coverage_verified=True,
            promotion_checks=checks, promotion_pass=all(checks.values()), passed=True))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluator_report=str(source),
        evaluator_report_sha256=sha(source), qa_source_sha256=sha(Path(__file__)),
        model_imported=False, inference_run=False, variants=rows, passed=all(row['passed'] for row in rows))
    with target.open('x') as stream: json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
