"""Independent CSV-only V31 score/protection audit; no inference imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v31_20260912'
B = P/'results/data1_6_20260911'
V28 = P/'results/data1_6_v28_20260912/direct_guard'
VARIANTS = ('protected_cdf_motion', 'protected_cdf_consensus')
CASES = tuple(f'data{i}' for i in range(1, 7))
TOL = 1e-9


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, ok, r):
    rv = np.isfinite(r) & (r > 0); valid = ok & np.isfinite(y) & rv
    e = y[valid]-r[valid]; n5 = int(np.count_nonzero(abs(e) <= 5))
    return dict(Nplanned=len(y), Noutput=int(ok.sum()), Nref=int(rv.sum()), Nvalid=len(e), Nwithin5=n5,
        MAE_bpm=float(np.mean(abs(e))) if len(e) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(e*e))) if len(e) else np.nan,
        P5_valid_pct=100*n5/len(e) if len(e) else np.nan,
        R5_all_reference_pct=100*n5/rv.sum() if rv.any() else np.nan,
        hr_output_coverage_pct=100*float(ok.mean()))


def check(actual, reported):
    for key, value in actual.items():
        expected = reported[key]
        np.testing.assert_allclose(np.nan if value is None else value,
            np.nan if expected is None else expected, rtol=0, atol=TOL, equal_nan=True, err_msg=key)


def jumps(y, ok):
    selected = [abs(y[i]-y[i-1]) for i in range(1, len(y)) if ok[i] and ok[i-1]]
    return dict(adjacent_valid_pairs=len(selected), large_jump_count=sum(x > 12 for x in selected),
        max_jump_bpm=max(selected, default=0.))


def main():
    source, target = R/'evaluation_summary.json', R/'qa_evaluation_v31.json'
    assert not target.exists(), 'Preserve existing QA'
    report = json.loads(source.read_text())
    assert set(report['variants']) == set(VARIANTS)
    assert report['all_variants_complete_before_scoring'] is True
    for group in ('input_hashes', 'evaluation_output_hashes'):
        for path, digest in report[group].items(): assert sha(path) == digest, 'Bound artifact changed: '+path
    receipts = []
    for variant in VARIANTS:
        result = report['variants'][variant]
        streams = [[] for _ in range(5)]
        seconds, finite_seconds, cases, deltas = 0., 0., [], []
        damage_total = new_large_total = 0
        all_jump_counts_safe = True
        for case in CASES:
            directory = R/variant/case
            hr, old = pd.read_csv(directory/'heart_rate.csv'), pd.read_csv(V28/case/'heart_rate.csv')
            wave, old_wave = pd.read_csv(directory/'waveform.csv'), pd.read_csv(V28/case/'waveform.csv')
            decisions = pd.read_csv(directory/'routing_decisions.csv')
            reference = pd.read_csv(B/case/'evaluation/paired_windows.csv')
            paired = pd.read_csv(directory/'evaluation/paired_windows.csv')
            fps = float(json.loads((B/case/'inference/summary.json').read_text())['fps'])
            y, oy, r = hr.ridge_bpm.to_numpy(float), old.ridge_bpm.to_numpy(float), reference.reference_bpm.to_numpy(float)
            ok, ook = hr.accepted.to_numpy(bool), old.accepted.to_numpy(bool)
            row = next(x for x in result['cases'] if x['case'] == case)
            baseline = next(x for x in result['baseline_V28']['cases'] if x['case'] == case)
            metrics, old_metrics = score(y, ok, r), score(oy, ook, r)
            check(metrics, row); check(old_metrics, baseline)
            np.testing.assert_array_equal(ok, ook)
            for name, values in [('estimated_bpm', y), ('V28_bpm', oy), ('reference_bpm', r)]:
                np.testing.assert_allclose(paired[name], values, rtol=0, atol=0, equal_nan=True)
            rv = np.isfinite(r) & (r > 0)
            oe, ne = abs(oy-r), abs(y-r)
            old_correct = ook & rv & (oe <= 5)
            old_large, new_large = ook & rv & (oe >= 20), ok & rv & (ne >= 20)
            flags = dict(old_correct=old_correct,
                old_correct_to_wrong_or_missing=old_correct & (~ok | ~rv | (ne > 5)),
                old_large_error=old_large, new_large_error=new_large,
                new_large_error_introduced=new_large & ~old_large,
                old_large_error_reduced=old_large & ok & rv & (ne < oe-TOL),
                old_large_error_below_20=old_large & ok & rv & (ne < 20),
                old_large_error_within_5=old_large & ok & rv & (ne <= 5))
            check({key+'_windows': int(values.sum()) for key, values in flags.items()}, row)
            for key, values in flags.items(): np.testing.assert_array_equal(paired[key], values)
            damage_total += int(flags['old_correct_to_wrong_or_missing'].sum())
            new_large_total += int(flags['new_large_error_introduced'].sum())
            for name, mask in [('first20_center', (hr.time_s >= 0) & (hr.time_s < 20)),
                               ('first20_window_end', hr.window_end_s <= 20)]:
                check(dict(planned_windows=int(mask.sum()), V28_output_windows=int((mask & ook).sum()),
                    output_windows=int((mask & ok).sum()), V28_large_error_windows=int((mask & old_large).sum()),
                    large_error_windows=int((mask & new_large).sum()),
                    newly_large_error_windows=int((mask & flags['new_large_error_introduced']).sum())), row[name])
            nj, oj = jumps(y, ok), jumps(oy, ook)
            check(nj, row['jumps']); check(oj, row['V28_jumps'])
            all_jump_counts_safe &= nj['large_jump_count'] <= oj['large_jump_count']
            assert nj['max_jump_bpm'] <= max(12., oj['max_jump_bpm'])+TOL
            width, hop = round(10*fps), round(fps)
            starts = np.arange(0, len(wave)-width+1, hop)
            eligible = decisions.eligible.to_numpy(bool)
            protected = np.zeros(len(wave), bool); planned = protected.copy()
            for i, start in enumerate(starts):
                planned[start:start+width] = True
                if not eligible[i]: protected[start:start+width] = True
            finite, ofinite = np.isfinite(wave.base), np.isfinite(old_wave.base)
            protected |= ~planned | ~ofinite
            np.testing.assert_array_equal(decisions.protected_window, ~eligible)
            np.testing.assert_array_equal(wave.protected_sample, protected)
            np.testing.assert_array_equal(finite, ofinite)
            for field in ('time_s', 'base', 'covered', 'observed', 'interpolated'):
                np.testing.assert_allclose(wave.loc[protected, field], old_wave.loc[protected, field], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_allclose(y[~eligible], oy[~eligible], rtol=0, atol=0, equal_nan=True)
            wc = 100*float(finite.mean()); owc = 100*float(ofinite.mean())
            delta = dict(delta_MAE_bpm=metrics['MAE_bpm']-old_metrics['MAE_bpm'],
                delta_P5_pp=metrics['P5_valid_pct']-old_metrics['P5_valid_pct'],
                delta_R5_pp=metrics['R5_all_reference_pct']-old_metrics['R5_all_reference_pct'],
                delta_hr_coverage_pp=metrics['hr_output_coverage_pct']-old_metrics['hr_output_coverage_pct'],
                delta_waveform_coverage_pp=wc-owc)
            check(delta, row); deltas.append(delta)
            seconds += len(wave)/fps; finite_seconds += int(finite.sum())/fps
            for stack, values in zip(streams, (y, oy, r, ok, ook)): stack.append(values)
            sensitivity = pd.read_csv(directory/'evaluation/alignment_sensitivity.csv')
            assert list(sensitivity.shift_s) == [-5, -2, -1, 0, 1, 2, 5]
            cases.append(dict(case=case, Nplanned=len(y), Noutput=int(ok.sum()),
                original_correct_damaged=int(flags['old_correct_to_wrong_or_missing'].sum()),
                newly_large_errors=int(flags['new_large_error_introduced'].sum()),
                protected_samples=int(protected.sum()), metrics_verified=True, protection_verified=True))
        y, oy, r, ok, ook = [np.concatenate(values) for values in streams]
        pooled, baseline = score(y, ok, r), score(oy, ook, r)
        for values, reported in [(pooled, result['pooled']), (baseline, result['baseline_V28']['pooled'])]:
            check({('HR_coverage_pct' if key == 'hr_output_coverage_pct' else key): value for key, value in values.items()}, reported)
        assert pooled['Nplanned'] == pooled['Nref'] == 309
        saved_baseline = json.loads((V28/'evaluation_summary.json').read_text())['pooled']
        check({key: baseline[key] for key in ('MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct')}, saved_baseline)
        check(dict(waveform_time_coverage_pct=100*finite_seconds/seconds), result['pooled'])
        common = ok & ook & np.isfinite(r) & (r > 0)
        cm, ocm = float(abs(y[common]-r[common]).mean()), float(abs(oy[common]-r[common]).mean())
        checks = dict(no_original_correct_damage=damage_total == 0, no_new_large_errors=new_large_total == 0,
            each_MAE=all(d['delta_MAE_bpm'] <= TOL for d in deltas),
            each_P5=all(d['delta_P5_pp'] >= -TOL for d in deltas),
            each_R5=all(d['delta_R5_pp'] >= -TOL for d in deltas),
            each_HR_coverage=all(d['delta_hr_coverage_pp'] >= -TOL for d in deltas),
            each_waveform_coverage=all(d['delta_waveform_coverage_pp'] >= -TOL for d in deltas),
            each_large_jump_count=bool(all_jump_counts_safe), pooled_MAE=pooled['MAE_bpm'] <= baseline['MAE_bpm']+TOL,
            pooled_P5=pooled['P5_valid_pct'] >= baseline['P5_valid_pct']-TOL,
            pooled_R5=pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']-TOL,
            common_MAE=cm <= ocm+TOL, substantive_gain=(pooled['MAE_bpm'] <= baseline['MAE_bpm']-1+TOL or
                pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']+1-TOL))
        assert checks == result['promotion_checks']
        assert all(checks.values()) == result['promotion_pass']
        receipts.append(dict(variant=variant, cases=cases, baseline_reproduced=True,
            all_metrics_recomputed=True, promotion_checks=checks,
            promotion_pass=all(checks.values()), passed=True))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluator_report=str(source),
        evaluator_report_sha256=sha(source), qa_source_sha256=sha(Path(__file__)),
        model_imported=False, inference_run=False, variants=receipts, passed=True)
    with target.open('x') as stream: json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
