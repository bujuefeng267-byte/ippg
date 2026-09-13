"""Read-only output replay for the two final bounded V26 experiments.

Writes only two new QA receipts. No paired/reference/evaluation values are read.
Production replay is followed by independent sample/mask reconstruction; neither
the inference files nor their historical protocols are modified.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from conservative_component_router_v26 import route_components
from component_harmonics_v26 import fuse_components, ComponentConfig
from evidence_hr import estimate_evidence
from qa_component_evidence_v26 import component, correlation


HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT/'results/data1_6_20260911'
V25 = PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
ROOT = PROJECT/'results/data1_6_v26_20260911'
ROIS = ('forehead', 'left_cheek', 'right_cheek')
HASHES = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(path, expected=None):
    path = Path(path)
    value = sha(path)
    if expected is not None:
        assert value == expected, f'Frozen hash mismatch: {path}'
    if str(path) in HASHES:
        assert HASHES[str(path)] == value, f'Input changed during audit: {path}'
    HASHES[str(path)] = value


def read_json(path):
    bind(path)
    return json.loads(Path(path).read_text())


def read_csv(path):
    bind(path)
    return pd.read_csv(path)


def compare_json(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_json(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare_json(a, b)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, equal_nan=True)
    else:
        assert actual == expected


def compare_table(actual, expected):
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    maximum = 0.
    for column in expected:
        a, b = actual[column], expected[column]
        if column.endswith('_json'):
            for av, bv in zip(a, b):
                compare_json(json.loads(av), json.loads(bv))
        elif pd.api.types.is_bool_dtype(b):
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
        elif pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            np.testing.assert_allclose(av, bv, atol=1e-9, rtol=0, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                maximum = max(maximum, float(abs(av[finite]-bv[finite]).max()))
        else:
            np.testing.assert_array_equal(a.fillna('<NA>').to_numpy(), b.fillna('<NA>').to_numpy())
    return maximum


def readout(wave, trace, fps, saved, source, proposals=None):
    expected = estimate_evidence(wave.base.to_numpy(float), pd.DataFrame({'rgb_valid': wave.observed}),
                                wave.interpolated.to_numpy(bool), fps, motion_trace=trace)
    expected['raw_spectral_peak_bpm'] = expected.spectral_peak_bpm
    expected.loc[~expected.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    starts = np.arange(0, len(trace)-round(10*fps)+1, round(fps))
    expected['window_start_s'] = starts/fps
    expected['window_end_s'] = (starts+round(10*fps))/fps
    if proposals is not None:
        expected['component_proposal_bpm'] = proposals.proposal_bpm
    expected['hr_source'] = source
    maximum = compare_table(saved, expected)
    assert saved.loc[~saved.accepted, ['spectral_peak_bpm', 'ridge_bpm']].isna().all().all()
    # Every published HR must retain actual waveform spectral support.
    for _, row in saved.loc[saved.accepted].iterrows():
        records = json.loads(row.evidence_candidates_json)
        assert any(row.ridge_bpm in item['supported_bpm'] for item in records)
    return dict(planned_windows=len(saved), accepted_windows=int(saved.accepted.sum()),
                maximum_absolute_hr_table_difference=maximum, saved_waveform_HR_replay=True,
                rejected_main_HR_NaN=True, accepted_HR_has_measured_spectral_candidate=True)


def protocol_for(variant):
    directory = ROOT/variant
    protocol = read_json(directory/'protocol_before_run.json')
    assert protocol['reference_used'] is False and protocol['post_diagnosis_development'] is True
    for name, expected in protocol['source_hashes'].items():
        bind(HERE/name, expected)
    runs = read_json(directory/'runs.json')
    assert len(runs) == 6
    return protocol


def base_case(case):
    directory = BASE/case/'inference'
    trace = read_csv(directory/'frame_trace.csv')
    meta = read_json(directory/'frame_trace.json')
    bind(directory/'frame_trace.csv', meta['trace_sha256'])
    return directory, trace, float(meta['fps'])


def independent_component_samples(channels, trace, roi_wave, proposals, records, fps, cfg):
    """Independent detrend/complex FFT/ROI medoid/source mask/OLA replay."""
    n = len(trace)
    width, hop = round(10*fps), round(fps)
    starts = np.arange(0, n-width+1, hop)
    by_window = {i: {} for i in range(len(starts))}
    for record in records:
        by_window[record['window_index']].setdefault(record['channel'], []).append(record)
    total, weight = np.zeros(n), np.zeros(n)
    observed, interpolated = np.ones(n, bool), np.zeros(n, bool)
    taper = np.hanning(width+2)[1:-1]
    for wi, a in enumerate(starts):
        b = a+width
        row = proposals.iloc[wi]
        stored_keys = json.loads(row.channels)
        assert len(stored_keys) == len({key.split('/')[1] for key in stored_keys})
        if not row.proposal_supported:
            assert not row.generated and np.isnan(row.proposal_bpm) and stored_keys == []
            continue
        target = float(row.proposal_bpm)
        best = {roi: (-np.inf, None) for roi in ROIS}
        for key, candidates in sorted(by_window[wi].items()):
            branch, roi, method = key.split('/')
            assert method in ('pos', 'chrom') and branch in ('baseline', 'tracked') and roi in ROIS
            assert np.isfinite(channels[key][a:b]).all()
            assert roi_wave[f'{roi}_observed'].iloc[a:b].mean() >= cfg['min_observed']
            assert trace[f'{roi}_quality'].iloc[a:b].mean() >= cfg['min_quality']
            if branch == 'tracked':
                sources = trace[f'{roi}_pixel_source'].iloc[a:b]
                assert (sources == 'tracked_ratio').mean() >= cfg['min_tracked_fraction']
                assert sources.isin(['baseline_reset', 'numerical_reset']).mean() <= cfg['max_reset_fraction']
            score = max((item['evidence_score']-.025*(target-item['bpm'])**2
                         for item in candidates if target in item['supported_bpm']), default=-np.inf)
            if score > best[roi][0]:
                best[roi] = (score, key)
        assert sum(value[1] is not None for value in best.values()) >= cfg['min_rois']
        waves, keys = [], []
        for roi in ROIS:
            key = best[roi][1]
            if key is None:
                continue
            values = component(channels[key][a:b], fps, target, cfg)
            if values.std() < 1e-8:
                continue
            waves.append(values/values.std())
            keys.append(key)
        used, generated = [], False
        if len(waves) >= cfg['min_rois']:
            correlations = np.array([[correlation(x, y) for y in waves] for x in waves])
            anchor = int(np.argmax(abs(correlations).sum(1)))
            used = [j for j in range(len(waves)) if abs(correlations[anchor, j]) >= cfg['min_component_correlation']]
            generated = len(used) >= cfg['min_rois']
            if generated:
                used_keys = [keys[j] for j in used]
                assert used_keys == stored_keys
                values = np.mean([waves[j]*(1 if correlations[anchor, j] >= 0 else -1) for j in used], axis=0)
                existing = weight[a:b] > 0
                if existing.sum() >= round(fps) and correlation(values[existing], total[a:b][existing]/weight[a:b][existing]) < 0:
                    values = -values
                total[a:b] += values*taper
                weight[a:b] += taper
                for key in used_keys:
                    roi = key.split('/')[1]
                    observed[a:b] &= roi_wave[f'{roi}_observed'].iloc[a:b].to_numpy(bool)
                    interpolated[a:b] |= roi_wave[f'{roi}_interpolated'].iloc[a:b].to_numpy(bool)
        assert bool(row.generated) == generated
        assert int(row.contributing_rois) == (len(used) if generated else 0)
    covered = weight > 0
    values = np.full(n, np.nan)
    values[covered] = total[covered]/weight[covered]
    return pd.DataFrame(dict(time_s=trace.time_s, base=values, covered=covered,
                             observed=observed & covered, interpolated=interpolated & covered))


def audit_harmonics():
    old_source = (HERE/'component_fusion_v26.py').read_text()
    new_source = (HERE/'component_harmonics_v26.py').read_text()
    assert old_source.replace('from motion_evidence import motion_evidence',
        'from motion_harmonic_evidence_v26 import motion_evidence_harmonics as motion_evidence').rstrip() == new_source.rstrip()
    bind(HERE/'component_fusion_v26.py')
    bind(HERE/'qa_component_evidence_v26.py')
    protocol = protocol_for('component_harmonics')
    checked = []
    for case in [f'data{i}' for i in range(1, 7)]:
        original, trace, fps = base_case(case)
        directory = ROOT/'component_harmonics'/case
        for name, expected in protocol['input_hashes'][case].items():
            bind(original/name, expected)
        summary = read_json(directory/'summary.json')
        assert summary['source_hashes'] == protocol['source_hashes'] and summary['reference_used'] is False
        assert summary['hr_from_saved_waveform'] is True
        baseline = read_csv(original/'baseline_roi_waveforms.csv')
        tracked = read_csv(original/'roi_waveforms.csv')
        channels = {}
        for branch, table in [('baseline', baseline), ('tracked', tracked)]:
            np.testing.assert_allclose(table.time_s, trace.time_s, rtol=0, atol=1e-8)
            for roi in ROIS:
                for suffix in ('observed', 'interpolated'):
                    np.testing.assert_array_equal(table[f'{roi}_{suffix}'], baseline[f'{roi}_{suffix}'])
                for method in ('pos', 'chrom'):
                    channels[f'{branch}/{roi}/{method}'] = table[f'{roi}_{method}'].to_numpy(float)
        saved = read_csv(directory/'waveform.csv')
        proposals = read_csv(directory/'component_proposals.csv')
        records = read_json(directory/'all_roi_candidates.json')
        production_wave, production_proposals, production_records = fuse_components(
            channels, trace, baseline, fps, config=ComponentConfig(**protocol['component_config']))
        production_delta = compare_table(saved, production_wave)
        compare_table(proposals, production_proposals)
        compare_json(records, production_records)
        independent = independent_component_samples(channels, trace, baseline, proposals, records,
                                                     fps, protocol['component_config'])
        independent_delta = compare_table(saved, independent)
        hr = readout(saved, trace, fps, read_csv(directory/'heart_rate.csv'),
                     'saved_measured_component_offline', proposals)
        item = dict(case=case, frames=len(trace), covered_frames=int(saved.covered.sum()),
                    generated_component_windows=int(proposals.generated.sum()),
                    production_waveform_max_abs_difference=production_delta,
                    independent_FFT_OLA_max_abs_difference=independent_delta,
                    candidate_records_and_DP_proposals_replayed=True,
                    physical_ROI_votes_and_source_masks_verified=True, **hr)
        checked.append(item)
        print(json.dumps(dict(variant='component_harmonics', **item)), flush=True)
    return dict(cases=checked, only_motion_provider_import_changed=True,
                source_diff_ignores_only_trailing_blank_lines=True,
                actual_measured_complex_Fourier_components_verified=True,
                source_masks_exact=True, saved_waveform_HR_replay=True)


def audit_conservative():
    protocol = protocol_for('conservative_components')
    checked = []
    for case in [f'data{i}' for i in range(1, 7)]:
        _, trace, fps = base_case(case)
        directory = ROOT/'conservative_components'/case
        for path, expected in protocol['input_hashes'][case].items():
            bind(path, expected)
        summary = read_json(directory/'summary.json')
        assert summary['source_hashes'] == protocol['source_hashes'] and summary['reference_used'] is False
        assert summary['hr_from_saved_waveform'] is True
        old = read_csv(V25/case/'fusion_waveform.csv')
        old_hr = read_csv(V25/case/'fusion_heart_rate.csv')
        candidate = read_csv(ROOT/'component_harmonics'/case/'waveform.csv')
        proposals = read_csv(ROOT/'component_harmonics'/case/'component_proposals.csv')
        saved = read_csv(directory/'waveform.csv')
        decisions = read_csv(directory/'routing_decisions.csv')
        production_wave, production_decisions = route_components(old, old_hr, candidate, proposals, trace, fps)
        production_delta = compare_table(saved, production_wave)
        compare_table(decisions, production_decisions)
        n, width, step = len(trace), round(10*fps), round(fps)
        starts = np.arange(0, n-width+1, step)
        denominator, numerator = np.zeros(n), np.zeros(n)
        taper = np.hanning(width+2)[1:-1]
        for wi, a in enumerate(starts):
            b = a+width
            denominator[a:b] += taper
            row = decisions.iloc[wi]
            if row.route_eligible:
                assert bool(old_hr.accepted.iloc[wi]) and np.isfinite(old_hr.ridge_bpm.iloc[wi])
                assert bool(proposals.generated.iloc[wi]) and bool(proposals.proposal_supported.iloc[wi])
                assert np.isfinite(candidate.base.iloc[a:b]).all() and np.isfinite(old.base.iloc[a:b]).all()
                rois = {key.split('/')[1] for key in json.loads(proposals.channels.iloc[wi])}
                assert len(rois) >= 2 and len(rois) == int(proposals.contributing_rois.iloc[wi])
                assert row.motion_available and row.old_motion_risk >= .50
                assert row.old_motion_risk-row.candidate_motion_risk >= .30
                numerator[a:b] += taper
        alpha = np.divide(numerator, denominator, out=np.zeros(n), where=denominator > 0)
        covered = np.isfinite(old.base.to_numpy(float))
        alpha[~covered] = 0
        np.testing.assert_allclose(saved.candidate_weight, alpha, atol=1e-12, rtol=0)
        use_candidate, use_old = alpha > 0, covered & (alpha < 1)
        expected = old.base.to_numpy(float).copy()
        expected[use_candidate] = (expected[use_candidate]*(1-alpha[use_candidate]) +
                                   candidate.base.to_numpy(float)[use_candidate]*alpha[use_candidate])
        np.testing.assert_allclose(saved.base, expected, atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(saved.base), covered)
        np.testing.assert_array_equal(saved.covered, covered)
        # The router's unchanged in-memory values are exact copies. A second
        # decimal CSV write/read can differ by one ULP; record this explicitly
        # while keeping the original finite/NaN and discrete masks exact.
        np.testing.assert_array_equal(production_wave.base.to_numpy()[~use_candidate],
                                      old.base.to_numpy()[~use_candidate])
        np.testing.assert_allclose(saved.base.to_numpy()[~use_candidate],
                                   old.base.to_numpy()[~use_candidate],
                                   atol=1e-12, rtol=0, equal_nan=True)
        untouched_finite = covered & ~use_candidate
        untouched_error = abs(saved.base.to_numpy()[untouched_finite]-old.base.to_numpy()[untouched_finite])
        observed = ((~use_old | old.observed.to_numpy(bool)) &
                    (~use_candidate | candidate.observed.to_numpy(bool))) & covered
        interpolated = ((use_old & old.interpolated.to_numpy(bool)) |
                        (use_candidate & candidate.interpolated.to_numpy(bool))) & covered
        np.testing.assert_array_equal(saved.observed, observed)
        np.testing.assert_array_equal(saved.interpolated, interpolated)
        source = np.full(n, 'missing', dtype='<U23')
        source[covered] = 'old_waveform'
        source[use_candidate & use_old] = 'old_component_mixture'
        source[use_candidate & ~use_old] = 'component_waveform'
        np.testing.assert_array_equal(saved.source, source)
        final_hr = read_csv(directory/'heart_rate.csv')
        hr = readout(saved, trace, fps, final_hr, 'saved_conservative_component_waveform_offline')
        np.testing.assert_array_equal(final_hr.accepted, old_hr.accepted)
        changed = covered & (saved.base.to_numpy(float) != old.base.to_numpy(float))
        item = dict(case=case, frames=n, covered_frames=int(covered.sum()),
                    eligible_route_windows=int(decisions.route_eligible.sum()),
                    samples_with_positive_candidate_weight=int(use_candidate.sum()),
                    actual_changed_waveform_samples=int(changed.sum()),
                    unrouted_values_exact_before_CSV=True,
                    unrouted_finite_CSV_roundtrip_changed_values=int((untouched_error > 0).sum()),
                    unrouted_finite_CSV_roundtrip_max_abs_difference=float(untouched_error.max()) if len(untouched_error) else 0.,
                    production_waveform_max_abs_difference=production_delta,
                    independent_convex_mix_max_abs_difference=float(abs(saved.base.to_numpy()[covered]-expected[covered]).max()) if covered.any() else 0,
                    original_finite_mask_exact=True, original_HR_accepted_window_mask_exact=True,
                    sampling_masks_and_source_exact=True, decisions_and_weight_rule_verified=True, **hr)
        checked.append(item)
        print(json.dumps(dict(variant='conservative_components', **item)), flush=True)
    return dict(cases=checked, original_finite_mask_exact=True,
                original_HR_accepted_window_mask_exact=True,
                total_original_and_final_accepted_windows=sum(x['accepted_windows'] for x in checked),
                total_planned_windows=sum(x['planned_windows'] for x in checked),
                independent_Hann_convex_mixture_and_provenance_verified=True,
                saved_waveform_HR_replay=True)


def main():
    paths = {
        'component_harmonics': ROOT/'qa_component_harmonics_signal_evidence_v26.json',
        'conservative_components': ROOT/'qa_conservative_signal_evidence_v26.json',
    }
    for path in paths.values():
        assert not path.exists(), 'Never overwrite historical QA receipts'
    conservative = audit_conservative()
    harmonic = audit_harmonics()
    for path, expected in HASHES.items():
        assert sha(path) == expected, f'Source/output changed during audit: {path}'
    for name, checks in [('conservative_components', conservative), ('component_harmonics', harmonic)]:
        result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, errors=[],
                      variant=name, qa_source_sha256=sha(__file__), reference_files_read=False,
                      inference_sources_and_outputs_unchanged=True, source_hashes=HASHES,
                      accuracy_or_promotion_claim=False,
                      limitations=[
                          'This audit establishes computation and real optical source consistency, not physiological waveform fidelity or clinical validity.',
                          'Both revisions are post-diagnosis development candidates; audit success does not imply promotion success.',
                          'Candidate components involve adaptive filtering and normalization; their shape/amplitude is not recovered contact PPG morphology.',
                          'Production decision/HR replay supplements the independent coefficient/convex mixture calculations; it is not an independent physiological reference.',
                      ], **checks)
        paths[name].write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(dict(passed=True, variant=name, receipt=str(paths[name]))), flush=True)


if __name__ == '__main__':
    main()
