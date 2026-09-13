"""Frozen, reference-blind V22/V24 regression scoring of saved output files.

This evaluator imports only the hash-pinned previous evaluator's pure scoring
helpers, never inference code. It does not retime references, rerun inference,
tune thresholds, choose a model, or promote an executable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V22 = ROOT / 'rppg_motion_v22'
CORE_SHA = '691a6c594c926892fab94fe69e197a391f39d27b22b3b28eaf972bc7aa7f733e'
COMPAT_CRITERIA_SHA = '234b52a58868329105992236f7affdae24d0c745b3f21f4b62d7ba79c0ab8a68'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if sha(V22 / 'evaluate_v22.py') != CORE_SHA:
    raise RuntimeError('Frozen evaluation dependency changed')
spec = importlib.util.spec_from_file_location('frozen_v22_scoring', V22 / 'evaluate_v22.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

CASES = core.CASES
BASELINES = ('trimmed_gap10', 'trimmed_gap15')
CANDIDATES = ('bounded_tracking_gap10', 'bounded_tracking_gap15',
              'guarded_fusion_gap10', 'guarded_fusion_gap15')
VERSIONS = BASELINES + CANDIDATES
VARIANTS = ('fusion', 'pos', 'chrom')
ESTIMATORS = core.ESTIMATORS
GEOMETRY = ('face_detected', 'flow_available', 'source', 'time_s',
            'face_x0', 'face_y0', 'face_x1', 'face_y1', 'motion_x', 'motion_y',
            'rgb_valid', 'forehead_valid', 'left_cheek_valid', 'right_cheek_valid')
ROIS = ('forehead', 'left_cheek', 'right_cheek')
ANCHOR = {'anchor_hz': 0.15, 'anchor_order': 4}
GUARD = {'min_tracked_fraction': 0.60, 'max_reset_fraction': 0.05,
         'switch_score_ratio': 1.15, 'max_branch_disagreement_bpm': 12.0}
HUMAN_CASES = tuple(c for c in CASES if c != 'synthetic72')
REFERENCE_CASES = ('ubfc', 'data1')
METRIC_COLUMNS = ('Nplanned', 'Noutput', 'Nref', 'Nvalid', 'MAE_bpm', 'RMSE_bpm',
                  'Bias_bpm', 'P5_valid_pct', 'R5_all_reference_pct')


def load_version(case, version, directory, evidence):
    summary = evidence.json(directory / 'summary.json')
    metadata = evidence.json(directory / 'frame_trace.json')
    trace = evidence.csv(directory / 'frame_trace.csv')
    frames, fps = int(summary['frames']), float(summary['fps'])
    assert frames == len(trace) == metadata['n_frames']
    assert sha(directory / 'frame_trace.csv') == metadata['trace_sha256']
    np.testing.assert_allclose(trace.time_s, np.arange(frames) / fps, rtol=0, atol=1e-8)
    config = summary['config']
    assert config['window'] == 10 and config['step'] == 1
    assert (config['min_bpm'], config['max_bpm']) == (42, 210)
    assert config['max_gap'] == (.1 if version.endswith('gap10') else .15)
    assert summary['reference_identity'] is None and config['reference_ubfc'] is None
    window, step = round(10 * fps), round(fps)
    starts = np.arange(0, frames - window + 1, step)
    centers = (starts + window / 2) / fps
    data = dict(directory=directory, summary=summary, metadata=metadata, trace=trace,
                frames=frames, fps=fps, starts=starts, centers=centers, window=window, step=step,
                variants={})
    for variant in VARIANTS:
        hr = evidence.csv(directory / f'{variant}_heart_rate.csv')
        wave = evidence.csv(directory / f'{variant}_waveform.csv')
        assert len(wave) == frames and len(hr) == len(starts)
        np.testing.assert_allclose(wave.time_s, trace.time_s, rtol=0, atol=1e-8)
        np.testing.assert_allclose(hr.time_s, centers, rtol=0, atol=1e-8)
        accepted = core.bools(hr.accepted, f'{case}/{version}/{variant}/accepted')
        for column in ESTIMATORS.values():
            assert np.isfinite(hr.loc[accepted, column]).all()
            if variant == 'fusion' or column == 'ridge_bpm':
                assert hr.loc[~accepted, column].isna().all(), 'Rejected final HR must remain missing'
            else:
                # Frozen shared POS/CHROM readout retains its pre-gate local
                # peak for diffuse-spectrum diagnostics; it is NOT output HR.
                diagnostic = ~accepted & hr[column].notna().to_numpy()
                assert hr.loc[diagnostic, 'status'].eq('diffuse_spectrum').all()
        finite = np.isfinite(wave.base.to_numpy(float))
        finite_windows = np.array([finite[a:a + window].all() for a in starts])
        assert np.all(finite_windows[accepted]), 'Accepted HR requires a finite saved waveform segment'
        assert summary['variants'][variant]['accepted_windows'] == int(accepted.sum())
        np.testing.assert_allclose(summary['variants'][variant]['finite_waveform_fraction'], finite.mean(), atol=1e-12)
        if variant == 'fusion':
            np.testing.assert_array_equal(core.bools(wave.covered, 'covered'), finite)
            assert not np.any(core.bools(wave.observed, 'observed') & ~finite)
            assert not np.any(core.bools(wave.interpolated, 'interpolated') & ~finite)
            generated = core.bools(hr.waveform_generated, 'waveform_generated')
            np.testing.assert_array_equal(core.bools(hr.neighbor_covered, 'neighbor_covered'),
                                          finite_windows & ~generated)
        else:
            generated = None
        data['variants'][variant] = dict(hr=hr, wave=wave, fps=fps, frames=frames, starts=starts,
            centers=centers, window=window, step=step, accepted=accepted,
            generated=generated, finite_samples=finite, finite_windows=finite_windows)
    return data


def compare_column(a, b, label, discrete=False):
    """Missingness and discrete identities are exact; numeric serialization is bounded."""
    assert len(a) == len(b), label
    np.testing.assert_array_equal(a.isna(), b.isna(), err_msg=label + '/missingness')
    exact = (a.to_numpy() == b.to_numpy()) | (a.isna().to_numpy() & b.isna().to_numpy())
    if discrete or not pd.api.types.is_numeric_dtype(a):
        np.testing.assert_array_equal(a.fillna('<NA>').to_numpy(), b.fillna('<NA>').to_numpy(), err_msg=label)
        maximum = None
    else:
        np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-9,
                                   equal_nan=True, err_msg=label)
        both = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
        maximum = float(np.max(np.abs(a.to_numpy(float)[both]-b.to_numpy(float)[both]))) if both.any() else 0.
    return dict(exact_differing_frames=int((~exact).sum()), max_abs_difference=maximum,
                differing_frames_beyond_serialization_tolerance=0, numerically_verified=True)


def verify_geometry(case, baseline, candidate, before, after):
    rows = []
    colors = tuple('rgb') + tuple(f'{roi}_{c}' for roi in ROIS for c in 'rgb')
    for column in ('frame',) + GEOMETRY + colors:
        target = 'baseline_' + column if candidate in CANDIDATES and column in colors else column
        a, b = before['trace'][column], after['trace'][target]
        discrete = column in ('frame', 'source', 'face_detected', 'flow_available', 'rgb_valid') or column.endswith('_valid')
        row = compare_column(a, b, f'{case}/{candidate}/{target}', discrete)
        rows.append(dict(case=case, before=baseline, after=candidate, field=column,
                         candidate_field=target, **row))
    return rows


def load_case(case, versions, args, evidence, frozen22, frozen24):
    data = {}
    for version in versions:
        directory = (args.baseline_root if version in BASELINES else args.validation_root) / version / case
        data[version] = load_version(case, version, directory, evidence)
    baseline = data['trimmed_gap10']
    geometry_rows = []
    for version, item in data.items():
        assert item['frames'] == baseline['frames']
        np.testing.assert_allclose(item['fps'], baseline['fps'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(item['centers'], baseline['centers'], rtol=0, atol=1e-8)
        for key in ('video', 'size', 'mtime_ns', 'max_seconds'):
            assert item['summary']['identity'][key] == baseline['summary']['identity'][key], (case, version, key)
        assert item['summary']['fusion_config'] == baseline['summary']['fusion_config']
        expected = frozen22['source_hashes'] if version in BASELINES else frozen24['source_hashes']
        assert item['summary']['source_hashes'] == expected, f'Unfrozen inference source: {case}/{version}'
        geometry_rows.extend(verify_geometry(case, 'trimmed_gap10', version, baseline, item))
        if version in CANDIDATES:
            expected_mode = version.rsplit('_gap', 1)[0]
            assert item['summary']['config']['algorithm_mode'] == expected_mode
            assert item['summary']['config']['pixel_mode'] == 'tracking_screened'
            assert item['summary']['anchor_config'] == ANCHOR
            assert item['summary']['guard_config'] == (GUARD if expected_mode == 'guarded_fusion' else None)
            assert item['summary']['extraction_path'] in (
                'validated_V23_paired_increments', 'validated_V24_frontend_cache')
            for name, value in ANCHOR.items():
                np.testing.assert_allclose(item['trace'][name], value, rtol=0, atol=1e-12)
    for before, after in [('trimmed_gap10', 'trimmed_gap15')] + [
            ('bounded_tracking_gap10', v) for v in CANDIDATES if v != 'bounded_tracking_gap10']:
        if before in data and after in data:
            a, b = data[before]['trace'], data[after]['trace']
            assert list(a) == list(b)
            for column in a:
                discrete = pd.api.types.is_bool_dtype(a[column]) or pd.api.types.is_integer_dtype(a[column])
                compare_column(a[column], b[column], f'{case}/{before}/{after}/{column}', discrete)
    if CANDIDATES[0] in data:
        validate_reused_increments(case, data[CANDIDATES[0]], evidence, frozen24)
    return data, geometry_rows


def continuity_row(data):
    row = core.continuity(data['accepted'], data['starts'], data['window'], data['fps'])
    rejected = row['longest_rejected_run_windows']
    row['longest_rejected_grid_run_s'] = rejected * data['step'] / data['fps']
    row['longest_rejected_center_span_s'] = max(0, rejected - 1) * data['step'] / data['fps']
    return row


def waveform_row(data, values, mask):
    finite = data['finite_samples']
    observed = core.bools(data['wave'].observed, 'wave observed') & finite
    interpolated = core.bools(data['wave'].interpolated, 'wave interpolated') & finite
    spans = core.spans(finite)
    missing = core.spans(~finite)
    internal = [(a, b) for a, b in missing if a > 0 and b < len(finite)]
    selected = values[mask & np.isfinite(values)]
    row = dict(Nplanned=len(data['accepted']), Naccepted=int(data['accepted'].sum()),
        Nfinite_waveform_windows=int(data['finite_windows'].sum()), Nscored=len(selected),
        finite_sample_count=int(finite.sum()), finite_sample_pct=100 * finite.mean(),
        finite_sample_duration_s=float(finite.sum() / data['fps']),
        longest_finite_run_s=max((b-a for a, b in spans), default=0) / data['fps'],
        longest_missing_run_s=max((b-a for a, b in missing), default=0) / data['fps'],
        longest_internal_missing_run_s=max((b-a for a, b in internal), default=0) / data['fps'],
        finite_runs=len(spans), observed_finite_samples=int(observed.sum()),
        observed_all_samples_pct=100 * observed.mean(), interpolated_samples=int(interpolated.sum()),
        interpolated_all_samples_pct=100 * interpolated.mean(),
        interpolated_covered_samples_pct=100 * interpolated.sum() / finite.sum() if finite.any() else None,
        snr_mean_db=float(selected.mean()) if len(selected) else None,
        snr_median_db=float(np.median(selected)) if len(selected) else None,
        peak_concentration_median_accepted=float(data['hr'].loc[data['accepted'], 'peak_concentration'].median())
            if data['accepted'].any() else None,
        r_wave=None, morphology_reference_status='no_qualified_synchronized_waveform_truth')
    if data['generated'] is not None:
        row.update(waveform_generated_count=int(data['generated'].sum()),
            waveform_generated_pct_of_planned=100 * float(data['generated'].mean()),
            waveform_generated_but_HR_rejected_count=int((data['generated'] & ~data['accepted']).sum()),
            neighbor_covered_count=int(core.bools(data['hr'].neighbor_covered, 'neighbor_covered').sum()),
            accepted_neighbor_covered_count=int((core.bools(data['hr'].neighbor_covered, 'neighbor_covered') & data['accepted']).sum()))
    return row


def pair_rows(case, variant, estimator, before, after, data, reference, shift):
    aa, bb = data[before]['variants'][variant], data[after]['variants'][variant]
    both = aa['accepted'] & bb['accepted']
    added = ~aa['accepted'] & bb['accepted']
    lost = aa['accepted'] & ~bb['accepted']
    common = dict(case=case, variant=variant, estimator=estimator, before=before, after=after,
        reference_kind=core.REFERENCE_KIND.get(case, 'none'), shift_s=shift,
        Ncommon_output=int(both.sum()), Nadded_output=int(added.sum()), Nlost_output=int(lost.sum()))
    rows = []
    for scope, version, mask in [('common_before', before, both), ('common_after', after, both),
                                 ('added_after', after, added), ('lost_before', before, lost)]:
        pred = data[version]['variants'][variant]['hr'][ESTIMATORS[estimator]].to_numpy(float)
        rows.append(dict(**common, scope=scope, version=version, **core.metrics(pred, mask, reference)))
    return rows


def audit_baseline(case, version, estimator, row, old_metrics):
    prior = old_metrics[old_metrics.case.eq(case) & old_metrics.version.eq(version)
                        & old_metrics.estimator.eq(estimator)].iloc[0]
    for key in METRIC_COLUMNS:
        np.testing.assert_allclose(row[key] if row[key] is not None else np.nan, prior[key],
                                   rtol=0, atol=1e-9, equal_nan=True,
                                   err_msg=f'Frozen V22 mismatch: {case}/{version}/{estimator}/{key}')


def same_gap_baseline(candidate):
    assert candidate in CANDIDATES
    return 'trimmed_' + candidate.rsplit('_', 1)[1]


def exact_data1_center_ns(alignment, start, window, shift):
    # The original source is 10,000,000 / 333,333 fps. Use frame ratios,
    # never round an epoch-valued float and never independently re-zero series.
    from fractions import Fraction
    seconds = Fraction(int(2*start+window), 2) * Fraction(
        alignment['fps_denominator'], alignment['fps_numerator'])
    return int(alignment['video_start_utc_ns']) + round(seconds * 1_000_000_000) + int(shift)*1_000_000_000


def assert_candidate_manifest(frozen, plan):
    """No imported inference code or V23 source assumptions."""
    assert plan['primary_candidate'] == 'guarded_fusion_gap10'
    for name in CANDIDATES:
        row = frozen['candidates'][name]
        config = plan['bounded_candidate_round']['candidate_mapping'][name]
        assert row['algorithm_mode'] == name.rsplit('_gap', 1)[0]
        assert row['max_gap_s'] == config['max_gap_s']
        assert row['anchor_config'] == ANCHOR
        assert row['guard_config'] == (GUARD if name.startswith('guarded_') else None)
        assert config['parameters']['anchor_config'] == ANCHOR
        assert config['parameters']['guard_config'] == row['guard_config']


def local_evidence_path(value):
    value = str(value).replace('\\', '/')
    if value.startswith('/mnt/c/') and platform.system() == 'Windows':
        return Path('C:/' + value[len('/mnt/c/'):])
    if len(value) > 2 and value[1:3] == ':/' and platform.system() != 'Windows':
        return Path('/mnt/' + value[0].lower() + '/' + value[3:])
    return Path(value)


def validate_reused_increments(case, item, evidence, frozen):
    """Confirm the saved old increments, not old integrated colors, were reused."""
    old_dir = ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/case
    old_meta = evidence.json(old_dir/'frame_trace.json')
    old = evidence.csv(old_dir/'frame_trace.csv')
    assert sha(old_dir/'frame_trace.csv') == old_meta['trace_sha256']
    expected = {name: frozen['source_hashes'][name] for name in
                ('motion_frontend.py','baseline_frontend.py','pixel_tracking.py','legacy_motion.py','analyze_rppg.py')}
    assert old_meta['identity']['frontend_hashes'] == expected
    for key in ('video','size','mtime_ns','max_seconds','rgb_aggregation','pixel_mode','pixel_config'):
        assert old_meta['identity'][key] == item['summary']['identity'][key], (case,key)
    if 'tracking_caches' in frozen:
        assert frozen['tracking_caches'][case]['sha256'] == old_meta['trace_sha256']
    compare_column(old.time_s, item['trace'].time_s, case+'/reused time')
    for roi in ROIS:
        for color in 'rgb':
            col = f'{roi}_pixel_log_delta_{color}'
            compare_column(old[col], item['trace'][col], case+'/'+col)
            compare_column(old[f'{roi}_{color}'], item['trace'][f'unanchored_{roi}_{color}'],
                           case+f'/unanchored_{roi}_{color}')
        compare_column(old[f'{roi}_pixel_source'], item['trace'][f'{roi}_unanchored_pixel_source'],
                       case+f'/unanchored_{roi}_pixel_source', True)
    assert item['summary']['extraction_path'] == 'validated_V23_paired_increments'


def audit_mechanism(case, version, item, evidence, tables, all_data):
    """Rebuild sample provenance and count true ROIs from positive diagnostics."""
    directory = item['directory']
    trace = item['trace']
    d = item['variants']['fusion']
    roi_wave = evidence.csv(directory/'roi_waveforms.csv')
    proposals = evidence.csv(directory/'fusion_proposals.csv')
    diagnostics = evidence.csv(directory/'fusion_diagnostics.csv')
    n, window, starts = item['frames'], item['window'], item['starts']
    assert len(roi_wave) == n and len(proposals) == len(starts)
    np.testing.assert_allclose(roi_wave.time_s, trace.time_s, rtol=0, atol=1e-8)
    np.testing.assert_allclose(proposals.time_s, item['centers'], rtol=0, atol=1e-8)
    weights = pd.to_numeric(diagnostics.waveform_weight, errors='raise').to_numpy(float)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    assert set(diagnostics.roi).issubset(set(ROIS)), 'Correlated branches are not additional ROIs'
    assert (diagnostics.channel == diagnostics.roi + '_' + diagnostics.method).all()
    used = diagnostics.loc[weights > 0].copy()
    generated = np.zeros(len(starts), bool)
    counts = np.zeros(len(starts), int)
    covered, observed, interpolated = np.zeros(n,bool), np.ones(n,bool), np.zeros(n,bool)
    for index, group in used.groupby('window_index', sort=True):
        assert np.isfinite(index) and index == int(index) and 0 <= index < len(starts)
        i = int(index)
        rois = list(group.roi.unique())
        assert 2 <= len(rois) <= 3 and bool(proposals.proposal_accepted.iloc[i])
        if 'branch' in group:
            assert group.branch.nunique() == 1, 'Only the routed branch can contribute in this window'
        assert int(proposals.waveform_roi_count.iloc[i]) == len(rois)
        generated[i], counts[i] = True, len(rois)
        a, b = starts[i], starts[i] + window
        covered[a:b] = True
        for roi in rois:
            observed[a:b] &= core.bools(trace[f'{roi}_valid'].iloc[a:b], roi+'/valid')
            interpolated[a:b] |= core.bools(roi_wave[f'{roi}_interpolated'].iloc[a:b], roi+'/filled')
    observed &= covered
    interpolated &= covered
    for field, calculated in [('covered',covered),('observed',observed),('interpolated',interpolated)]:
        np.testing.assert_array_equal(calculated, core.bools(d['wave'][field],field))
    np.testing.assert_array_equal(covered, d['finite_samples'])
    np.testing.assert_array_equal(generated, d['generated'])
    np.testing.assert_array_equal(generated, core.bools(proposals.waveform_generated,'proposal generated'))
    np.testing.assert_array_equal(counts, d['hr'].waveform_roi_count.to_numpy(int))
    np.testing.assert_array_equal(core.bools(d['hr'].neighbor_covered,'neighbor'), d['finite_windows'] & ~generated)
    tables['provenance_audit'].append(dict(case=case,version=version,passed=True,
        true_ROI_limit=3,generated_windows=int(generated.sum()),covered_samples=int(covered.sum()),
        observed_samples=int(observed.sum()),interpolated_samples=int(interpolated.sum()),
        extraction_path=item['summary']['extraction_path']))
    for (_,channel), group in diagnostics.groupby(['window_index','channel'],sort=False):
        assert group.channel_status.nunique(dropna=False) == 1, 'Duplicate candidates disagree about channel status'
    unique_channels = diagnostics.drop_duplicates(['window_index','channel'])
    for status, count in unique_channels.channel_status.value_counts(dropna=False).items():
        tables['diagnostic_status_counts'].append(dict(case=case,version=version,scope='unique_window_channel',
            status=str(status),count=int(count),denominator=len(unique_channels)))
    for scope, column, frame in [('final_HR','status',d['hr']),('proposal','proposal_status',proposals)]:
        for value,count in frame[column].value_counts(dropna=False).items():
            tables['fallback_audit'].append(dict(case=case,version=version,scope=scope,reason=str(value),
                count=int(count),denominator=len(frame),pct=100*int(count)/len(frame)))
    if not version.startswith('guarded_'):
        return
    routing = evidence.csv(directory/'branch_routing.csv')
    branch_tables = {b:evidence.csv(directory/f'{b}_branch_proposals.csv') for b in ('baseline','tracked')}
    # The unselected branch's final diagnostic weights were intentionally zeroed.
    # Its matching independently saved branch retains original actual contributors.
    gap = version.rsplit('_',1)[1]
    siblings={'baseline':all_data['trimmed_'+gap], 'tracked':all_data['bounded_tracking_'+gap]}
    original_diagnostics={}
    rename={'proposal_spectral_peak_bpm':'spectral_peak_bpm','consensus_bpm':'ridge_bpm',
            'proposal_accepted':'accepted','proposal_status':'status'}
    for branch,sibling in siblings.items():
        original=evidence.csv(sibling['directory']/'fusion_proposals.csv').rename(columns=rename)
        actual=branch_tables[branch]
        assert len(actual)==len(original)
        for col in actual:
            discrete=pd.api.types.is_bool_dtype(original[col]) or pd.api.types.is_integer_dtype(original[col])
            compare_column(original[col],actual[col],f'{case}/{version}/{branch}/proposal/{col}',discrete)
        sibling_roi=evidence.csv(sibling['directory']/'roi_waveforms.csv')
        branch_roi=roi_wave if branch=='tracked' else evidence.csv(directory/'baseline_roi_waveforms.csv')
        assert list(sibling_roi)==list(branch_roi)
        for col in sibling_roi:
            compare_column(sibling_roi[col],branch_roi[col],f'{case}/{version}/{branch}/ROI/{col}',
                           col.endswith('_observed') or col.endswith('_interpolated'))
        branch_wave=evidence.csv(directory/f'{branch}_branch_waveform.csv')
        for col in ('time_s','base'):
            compare_column(sibling['variants']['fusion']['wave'][col],branch_wave[col],
                           f'{case}/{version}/{branch}/wave/{col}')
        original_diagnostics[branch]=evidence.csv(sibling['directory']/'fusion_diagnostics.csv')
    base_roi = evidence.csv(directory/'baseline_roi_waveforms.csv')
    for roi in ROIS:
        for tag in ('observed','interpolated'):
            np.testing.assert_array_equal(core.bools(base_roi[f'{roi}_{tag}'],tag),core.bools(roi_wave[f'{roi}_{tag}'],tag))
    assert len(routing) == len(starts)
    np.testing.assert_array_equal(routing.window_index, np.arange(len(starts)))
    np.testing.assert_allclose(routing.time_s,item['centers'],rtol=0,atol=1e-8)
    for i,a in enumerate(starts):
        fractions = [float(trace[f'{roi}_pixel_source'].iloc[a:a+window].eq('tracked_ratio').mean()) for roi in ROIS]
        resets = [float(trace[f'{roi}_pixel_source'].iloc[a:a+window].isin(['baseline_reset','numerical_reset']).mean()) for roi in ROIS]
        original_used={branch:diag.loc[diag.window_index.eq(i) & diag.waveform_weight.gt(0)]
                       for branch,diag in original_diagnostics.items()}
        contributing_rois=set(original_used['tracked'].roi)
        tracking_ok = sum(roi in contributing_rois and f >= GUARD['min_tracked_fraction'] and
                          r <= GUARD['max_reset_fraction'] for roi,f,r in zip(ROIS,fractions,resets)) >= 2
        b,t = (branch_tables[x].iloc[i] for x in ('baseline','tracked'))
        bs = float(b.quality_proxy)*(1-.25*float(b.motion_overlap))*(1-.5*float(b.runner_up_ratio))
        ts = float(t.quality_proxy)*(1-.25*float(t.motion_overlap))*(1-.5*float(t.runner_up_ratio))
        bg,tg = (bool(row.accepted) and int(row.waveform_roi_count)>=2 for row in (b,t))
        assert bg == bool(len(original_used['baseline'])) and tg == bool(len(original_used['tracked']))
        if not tg or not tracking_ok:
            selected,reason = ('baseline','tracking_quality_fallback') if bg else ('none','no_qualified_branch')
        elif not bg:
            selected,reason = 'tracked','baseline_unavailable'
        elif abs(float(t.ridge_bpm)-float(b.ridge_bpm)) > GUARD['max_branch_disagreement_bpm']:
            selected,reason = 'baseline','frequency_disagreement_fallback'
        elif ts >= GUARD['switch_score_ratio']*bs:
            selected,reason = 'tracked','stronger_qualified_evidence'
        else:
            selected,reason = 'baseline','baseline_evidence_retained'
        actual = routing.iloc[i]
        assert actual.selected_branch == selected and actual.selection_reason == reason
        assert bool(actual.tracking_eligible) == tracking_ok
        assert bool(actual.baseline_generated) == bg and bool(actual.tracked_generated) == tg
        np.testing.assert_allclose([actual.baseline_score,actual.tracked_score,actual.minimum_tracked_fraction,
            actual.second_largest_tracked_fraction,actual.maximum_reset_fraction],
            [bs,ts,min(fractions),sorted(fractions)[1],max(resets)],rtol=0,atol=1e-9,equal_nan=True)
        group = used.loc[used.window_index.eq(i)]
        assert set(group.branch) == ({selected} if selected != 'none' else set())
        tables['routing_windows'].append(dict(case=case,version=version,**actual.to_dict(),
            reconstructed_selection_matches=True,final_HR_accepted=bool(d['accepted'][i]),
            actual_ROI_count=int(counts[i])))
    assert {str(k):int(v) for k,v in routing.selected_branch.value_counts().items()} == item['summary']['branch_selection_counts']
    for reason,count in routing.selection_reason.value_counts().items():
        tables['fallback_audit'].append(dict(case=case,version=version,scope='guard_branch_choice',reason=reason,
            count=int(count),denominator=len(routing),pct=100*int(count)/len(routing)))


def check_receipt(path, evidence, args, frozen):
    if not path.exists():
        return {'passed': False, 'reason': 'Saved waveform replay QA receipt is missing'}
    receipt = evidence.json(path)
    reasons = []
    if receipt.get('source_hashes') != frozen['source_hashes']:
        reasons.append('Replay receipt must bind this V24 source manifest, never a V23 receipt')
    for key, expected in [('passed', True), ('checked_cases', 6), ('checked_modes', 4)]:
        if receipt.get(key) != expected:
            reasons.append(f'{key} must equal {expected}')
    if set(receipt.get('checked_variants', [])) != set(VARIANTS):
        reasons.append('All three waveform variants must be replayed')
    if set(receipt.get('checked_estimators', [])) != set(ESTIMATORS):
        reasons.append('Both HR estimators must be replayed')
    hashes = receipt.get('hr_sha256', {})
    expected_keys = {f'{mode}/{case}/{variant}_heart_rate.csv'
                     for mode, case, variant in itertools.product(CANDIDATES, CASES, VARIANTS)}
    if set(hashes) != expected_keys:
        reasons.append('Receipt must hash all 72 candidate HR output files')
    else:
        for key, expected in hashes.items():
            if sha(evidence.register(args.validation_root / key)) != expected:
                reasons.append(f'Replayed HR changed: {key}')
    return dict(passed=not reasons, reasons=reasons, receipt=receipt)


def source_balanced_errors(rows):
    """Equal source weights, including every declared capture or returning unavailable."""
    if len(rows) != len(REFERENCE_CASES) or any(row['Nvalid'] <= 0 for row in rows):
        return dict(source_balanced_MAE_bpm=None, source_balanced_RMSE_bpm=None,
                    mean_R5_all_reference_pct=None)
    if any(row[k] is None or not np.isfinite(row[k]) for row in rows
           for k in ('MAE_bpm','RMSE_bpm','R5_all_reference_pct')):
        return dict(source_balanced_MAE_bpm=None, source_balanced_RMSE_bpm=None,
                    mean_R5_all_reference_pct=None)
    return dict(source_balanced_MAE_bpm=float(np.mean([r['MAE_bpm'] for r in rows])),
                source_balanced_RMSE_bpm=float(np.sqrt(np.mean([r['RMSE_bpm']**2 for r in rows]))),
                mean_R5_all_reference_pct=float(np.mean([r['R5_all_reference_pct'] for r in rows])))


def v24_gate(tables, plan, baseline, candidate, engineering_pass):
    """Frozen same-gap regression rule; candidate identity is never selected here."""
    checks = []
    tolerance = plan['V24_upgrade_gate']['noninferiority_numerical_tolerance']
    assert tolerance == 1e-6

    def one(table, **filters):
        found = [row for row in tables[table] if all(row.get(k) == value for k,value in filters.items())]
        assert len(found) == 1, (table, filters, len(found))
        return found[0]

    def check(name, passed, **detail):
        checks.append(dict(check=name,passed=bool(passed),**detail))

    def compare(name, before, after, lower_better=True, strict=False):
        valid = before is not None and after is not None and np.isfinite(before) and np.isfinite(after)
        delta = float(after-before) if valid else None
        gain = (-delta if lower_better else delta) if valid else None
        passed = valid and (gain > tolerance if strict else gain >= -tolerance)
        check(name,passed,before=before,after=after,delta=delta,strict_improvement_required=strict)

    def error_row(case, estimator, scope, side):
        filters=dict(case=case,variant='fusion',estimator=estimator)
        if scope == 'own':
            return one('metrics',version=baseline if side=='before' else candidate,**filters)
        return one('paired_metrics',before=baseline,after=candidate,scope='common_'+side,**filters)

    check('engineering_integrity_replay_provenance_and_routing',engineering_pass)
    for case in REFERENCE_CASES + ('synthetic72',):
        for estimator in ESTIMATORS:
            for scope in ('own','common'):
                a,b = (error_row(case,estimator,scope,side) for side in ('before','after'))
                check(f'{case}/{estimator}/{scope}/nonempty_reference_and_output',
                      a['Nref']>0 and b['Nref']==a['Nref'] and a['Nvalid']>0 and b['Nvalid']>0,
                      Nref=a['Nref'],before_Nvalid=a['Nvalid'],after_Nvalid=b['Nvalid'])
                for metric in ('MAE_bpm','RMSE_bpm'):
                    compare(f'{case}/{estimator}/{scope}/{metric}',a[metric],b[metric])
                compare(f'{case}/{estimator}/{scope}/R5',a['R5_all_reference_pct'],b['R5_all_reference_pct'],False)
    for estimator in ESTIMATORS:
        for scope in ('own','common'):
            macro = {}
            for side,version in [('before',baseline),('after',candidate)]:
                per_case = [error_row(case,estimator,scope,side) for case in REFERENCE_CASES]
                macro[side]=source_balanced_errors(per_case)
                tables['macro_metrics'].append(dict(before=baseline,after=candidate,version=version,side=side,
                    population='human_reference_UBFC_and_estimated_data1',scope=scope,estimator=estimator,
                    Ncaptures=2,weight_per_capture=.5,primary_shift_s=0,
                    source_Nvalid=json.dumps({c:r['Nvalid'] for c,r in zip(REFERENCE_CASES,per_case)}),
                    **macro[side]))
            if estimator=='offline_ridge':
                for metric in ('source_balanced_MAE_bpm','source_balanced_RMSE_bpm'):
                    compare(f'human_reference_macro/{estimator}/{scope}/{metric}',
                            macro['before'][metric],macro['after'][metric],strict=True)
    coverage = {'before': [],'after': []}
    for case in CASES:
        waves = [one('waveform_metrics',case=case,version=v,variant='fusion',scope='own')
                 for v in (baseline,candidate)]
        compare(f'{case}/actual_saved_waveform_coverage',waves[0]['finite_sample_pct'],waves[1]['finite_sample_pct'],False)
        for estimator in ESTIMATORS:
            hr = [one('metrics',case=case,version=v,variant='fusion',estimator=estimator) for v in (baseline,candidate)]
            compare(f'{case}/{estimator}/HR_output_coverage',hr[0]['C_out_pct'],hr[1]['C_out_pct'],False)
            if case in HUMAN_CASES and estimator=='offline_ridge':
                for index,side in enumerate(('before','after')):
                    coverage[side].append((hr[index]['C_out_pct'],waves[index]['finite_sample_pct']))
    for side,version in [('before',baseline),('after',candidate)]:
        assert len(coverage[side])==5
        arr=np.asarray(coverage[side],float)
        coverage[side]=dict(mean_C_out_pct=float(arr[:,0].mean()),mean_finite_sample_pct=float(arr[:,1].mean()))
        tables['macro_metrics'].append(dict(before=baseline,after=candidate,version=version,side=side,
            population='five_human_captures',scope='own',estimator='offline_ridge',Ncaptures=5,
            weight_per_capture=.2,primary_shift_s=0,**coverage[side]))
    for metric in ('mean_C_out_pct','mean_finite_sample_pct'):
        compare('human_coverage_macro/'+metric,coverage['before'][metric],coverage['after'][metric],False,True)
    passed=all(row['passed'] for row in checks)
    return dict(baseline=baseline,candidate=candidate,primary_shift_s=0,
        eligible_for_named_upgrade=passed,performance_gate_passed=passed,
        failed_checks=[row for row in checks if not row['passed']],checks=checks,
        note='Development/regression only. data1 timing estimated. Primary candidate fixed before inference. '
             'SNR/longest gaps are reported and separately checked by V23 compatibility criteria.')


def compatibility_gate(tables, criteria, baseline, candidate, engineering_pass):
    """Apply the frozen rule; never search for a better candidate or shift."""
    checks = []
    tolerance = criteria['noninferiority_numerical_tolerance']

    def check(name, passed, **detail):
        checks.append(dict(check=name, passed=bool(passed), **detail))

    def one(table, **filters):
        found = [row for row in tables[table] if all(row.get(k) == v for k, v in filters.items())]
        assert len(found) == 1, (table, filters, len(found))
        return found[0]

    def no_worse(name, before, after, lower_better=True):
        valid = before is not None and after is not None and np.isfinite(before) and np.isfinite(after)
        delta = after - before if valid else None
        passed = valid and (delta <= tolerance if lower_better else delta >= -tolerance)
        check(name, passed, before=before, after=after, delta=delta)

    check('engineering_integrity_and_replay_QA', engineering_pass)
    for case in criteria['reference_cases']:
        for estimator in criteria['required_estimators']:
            filters = dict(case=case, variant='fusion', estimator=estimator)
            own_a = one('metrics', version=baseline, **filters)
            own_b = one('metrics', version=candidate, **filters)
            pa = one('paired_metrics', before=baseline, after=candidate, scope='common_before', **filters)
            pb = one('paired_metrics', before=baseline, after=candidate, scope='common_after', **filters)
            check(f'{case}/{estimator}/nonempty_common', pa['Nvalid'] > 0 and pb['Nvalid'] > 0)
            for name in ('MAE_bpm', 'RMSE_bpm'):
                no_worse(f'{case}/{estimator}/own/{name}', own_a[name], own_b[name])
                no_worse(f'{case}/{estimator}/common/{name}', pa[name], pb[name])
            no_worse(f'{case}/{estimator}/R5', own_a['R5_all_reference_pct'], own_b['R5_all_reference_pct'], False)
        pair = one('waveform_deltas', case=case, variant='fusion', before=baseline, after=candidate)
        check(f'{case}/nonempty_paired_reference_H1_SNR', pair['Npaired_snr'] > 0)
        no_worse(f'{case}/paired_reference_H1_SNR', 0, pair['mean_paired_delta_snr_db'], False)
    meaningful = []
    for case in CASES:
        a = one('metrics', case=case, version=baseline, variant='fusion', estimator='offline_ridge')
        b = one('metrics', case=case, version=candidate, variant='fusion', estimator='offline_ridge')
        wa = one('waveform_metrics', case=case, version=baseline, variant='fusion', scope='own')
        wb = one('waveform_metrics', case=case, version=candidate, variant='fusion', scope='own')
        ca = one('continuity', case=case, version=baseline, variant='fusion')
        cb = one('continuity', case=case, version=candidate, variant='fusion')
        no_worse(f'{case}/HR_coverage', a['C_out_pct'], b['C_out_pct'], False)
        no_worse(f'{case}/finite_waveform_coverage', wa['finite_sample_pct'], wb['finite_sample_pct'], False)
        no_worse(f'{case}/longest_HR_rejection_grid_s', ca['longest_rejected_grid_run_s'], cb['longest_rejected_grid_run_s'])
        no_worse(f'{case}/longest_waveform_missing_s', wa['longest_missing_run_s'], wb['longest_missing_run_s'])
        if case != 'synthetic72':
            coverage_gain = max(b['C_out_pct'] - a['C_out_pct'], wb['finite_sample_pct'] - wa['finite_sample_pct'])
            if coverage_gain >= 5 - tolerance:
                meaningful.append(dict(case=case, improvement='coverage_percentage_points', value=coverage_gain))
        if case in criteria['reference_cases']:
            for estimator in criteria['required_estimators']:
                aa = one('metrics', case=case, version=baseline, variant='fusion', estimator=estimator)
                bb = one('metrics', case=case, version=candidate, variant='fusion', estimator=estimator)
                if aa['MAE_bpm'] is not None and bb['MAE_bpm'] is not None:
                    gain = aa['MAE_bpm'] - bb['MAE_bpm']
                    if gain >= 1 - tolerance:
                        meaningful.append(dict(case=case, estimator=estimator, improvement='MAE_bpm', value=gain))
    for estimator in criteria['required_estimators']:
        row = one('metrics', case='synthetic72', version=candidate, variant='fusion', estimator=estimator)
        check(f'synthetic72/{estimator}/MAE_at_most_1bpm',
              row['MAE_bpm'] is not None and row['MAE_bpm'] <= 1 + tolerance, value=row['MAE_bpm'])
    check('at_least_one_frozen_meaningful_improvement', bool(meaningful), improvements=meaningful)
    return dict(baseline=baseline, candidate=candidate, primary_shift_s=0,
        eligible_for_named_upgrade=all(row['passed'] for row in checks),
        failed_checks=[row for row in checks if not row['passed']], checks=checks,
        note='Development/regression gate only. data1 timing is estimated. Evaluator does not install or switch defaults.')


def self_check():
    result = core.self_check()
    data = dict(accepted=np.array([True, False, False, True]), starts=np.arange(4) * 30,
                window=300, fps=30., step=30)
    c = continuity_row(data)
    assert c['longest_rejected_grid_run_s'] == 2
    assert c['longest_rejected_center_span_s'] == 1
    # Retain original reference denominators on added/lost/common subsets.
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    added = core.metrics([101, np.nan, 120, np.nan], ~a & b, [100] * 4)
    lost = core.metrics([101, 102, np.nan, np.nan], a & ~b, [100] * 4)
    assert added['Nref'] == lost['Nref'] == 4 and added['MAE_bpm'] == 20
    assert lost['R5_all_reference_pct'] == 25
    result['v23_checks'] = ['HR rejection duration uses explicit decision-grid convention',
        'added/lost errors preserve full planned reference denominator']
    macro=source_balanced_errors([
        dict(Nvalid=1,MAE_bpm=3.,RMSE_bpm=3.,R5_all_reference_pct=50.),
        dict(Nvalid=100,MAE_bpm=4.,RMSE_bpm=4.,R5_all_reference_pct=0.)])
    np.testing.assert_allclose([macro['source_balanced_MAE_bpm'],macro['source_balanced_RMSE_bpm']],
                              [3.5,np.sqrt(12.5)],rtol=0,atol=1e-12)
    assert source_balanced_errors([dict(Nvalid=0),dict(Nvalid=5)])['source_balanced_MAE_bpm'] is None
    alignment=dict(video_start_utc_ns=1788939171166718500,fps_numerator=10000000,fps_denominator=333333)
    assert exact_data1_center_ns(alignment,0,300,2)==1788939171166718500+4_999_995_000+2_000_000_000
    assert exact_data1_center_ns(alignment,30,300,-5)==1788939171166718500+5_999_994_000-5_000_000_000
    fixture={'metrics':[],'paired_metrics':[],'waveform_metrics':[],'macro_metrics':[]}
    baseline,candidate='trimmed_gap10','guarded_fusion_gap10'
    for case in CASES:
        for side,version in [('before',baseline),('after',candidate)]:
            fixture['waveform_metrics'].append(dict(case=case,version=version,variant='fusion',scope='own',
                finite_sample_pct=60. if case=='synthetic72' else 40. if side=='before' else 50.))
            for estimator in ESTIMATORS:
                error=0. if case=='synthetic72' else 2. if estimator=='local_peak' else 10. if side=='before' else 9.
                ref=np.full(10,72. if case=='synthetic72' else 100.)
                mask=np.arange(10)<(5 if side=='before' or case=='synthetic72' else 6)
                values=ref+error
                fixture['metrics'].append(dict(case=case,version=version,variant='fusion',estimator=estimator,
                    **core.metrics(values,mask,ref)))
                fixture['paired_metrics'].append(dict(case=case,before=baseline,after=candidate,variant='fusion',
                    estimator=estimator,scope='common_'+side,**core.metrics(values,np.arange(10)<5,ref)))
    plan={'V24_upgrade_gate':{'noninferiority_numerical_tolerance':1e-6}}
    gate=v24_gate(fixture,plan,baseline,candidate,True)
    assert gate['performance_gate_passed'], gate['failed_checks']
    for row in fixture['metrics']:
        if row['case']=='ubfc' and row['version']==candidate and row['estimator']=='local_peak':
            row['RMSE_bpm']=2.01
    guard=v24_gate(fixture,plan,baseline,candidate,True)
    assert not guard['performance_gate_passed']
    assert any(x['check']=='ubfc/local_peak/own/RMSE_bpm' for x in guard['failed_checks'])
    result['v24_checks']=['Equal-capture MAE and mean-MSE-then-root RMSE, not pooled windows',
        'Missing reference capture never dropped from macro denominator',
        'Integer frame-ratio UTC centers and shift sign',
        'Gate permits unchanged zero-error synthetic but requires real macro gains',
        'Local-peak noninferiority guards reject regression despite primary gains']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=V22 / 'validation')
    parser.add_argument('--validation-root', type=Path, default=HERE / 'validation')
    parser.add_argument('--output', type=Path, default=HERE / 'evaluation')
    parser.add_argument('--baseline-only', action='store_true', help='Reproduce both V22 baselines before candidate results exist')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    qa = self_check()
    if args.self_check:
        print(json.dumps(qa)); return
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing evidence is never overwritten')
    evidence = core.Evidence()
    evidence.register(Path(__file__))
    evidence.register(V22 / 'evaluate_v22.py')
    criteria_path = HERE / 'evaluation_protocol.json'
    criteria = evidence.json(criteria_path)
    criteria_sha = sha(criteria_path)
    compat_path = ROOT / 'rppg_motion_v23/evaluation_upgrade_criteria.json'
    assert sha(compat_path) == COMPAT_CRITERIA_SHA
    compat_criteria = evidence.json(compat_path)
    alignment = evidence.json(ROOT / 'rppg_data1_20260909/alignment_protocol.json')
    assert alignment['video_start_utc_ns'] == 1788939171166718500
    assert alignment['sensitivity_shifts_s'] == [-5, -2, -1, 0, 1, 2, 5]
    frozen22 = evidence.json(args.baseline_root / 'protocol_before_validation.json')
    assert sha(ROOT / 'rppg_data1_20260909/alignment_protocol.json') == frozen22['data1_alignment_protocol_sha256']
    for filename, expected in frozen22['source_hashes'].items():
        assert sha(evidence.register(V22 / filename)) == expected
    frozen24, replay_qa, runs, tests = None, {'passed': False, 'reason': 'baseline-only precheck'}, [], {}
    implementation_freeze, guard_qa = None, {'passed':False,'reason':'baseline-only precheck'}
    versions = BASELINES if args.baseline_only else VERSIONS
    if not args.baseline_only:
        implementation_freeze=evidence.json(HERE/'evaluation_implementation_freeze.json')
        assert implementation_freeze['evaluator_sha256']==sha(Path(__file__))
        assert implementation_freeze['evaluation_protocol_sha256']==criteria_sha
        assert implementation_freeze['frozen_v22_evaluator_sha256']==CORE_SHA
        assert implementation_freeze['new_candidate_accuracy_scored_before_freeze'] is False
        frozen24 = evidence.json(args.validation_root / 'protocol_before_validation.json')
        assert set(frozen24['cases']) == set(CASES)
        assert frozen24['data1_alignment_protocol_sha256'] == frozen22['data1_alignment_protocol_sha256']
        assert criteria['parameters_frozen'] is True, 'V24 evaluation protocol is not frozen'
        assert frozen24['evaluation_protocol_sha256'] == criteria_sha
        assert criteria['primary_candidate'] == 'guarded_fusion_gap10'
        assert criteria['continuity_candidate'] == 'guarded_fusion_gap15'
        assert len(frozen24['source_hashes']) == 11
        assert frozen24['source_hashes'] == criteria['source_hashes']
        assert set(frozen24['candidates']) == set(CANDIDATES)
        assert_candidate_manifest(frozen24, criteria)
        for filename, expected in frozen24['source_hashes'].items():
            assert sha(evidence.register(HERE / filename)) == expected
        for case in CASES:
            for key in ('sha256', 'bytes'):
                assert frozen24['cases'][case][key] == frozen22['cases'][case][key]
        runs = evidence.json(args.validation_root / 'runs.json')
        assert len(runs) == 24 and all(row['exit_code'] == 0 for row in runs)
        assert {(row['case'], row['mode']) for row in runs} == set(itertools.product(CASES, CANDIDATES))
        tests = evidence.json(args.validation_root / 'tests.json')
        assert tests['failures'] == tests['errors'] == 0 and tests['run'] > 0
        assert tests['source_hashes'] == frozen24['source_hashes']
        replay_qa = check_receipt(args.validation_root / 'waveform_replay_qa.json', evidence, args, frozen24)
        guard_qa=evidence.json(args.validation_root/'guard_routing_qa.json')
        assert guard_qa['passed'] is True and not guard_qa['errors']
        assert guard_qa['guard_config']==GUARD
        assert (guard_qa['checked_frontend_outputs'],guard_qa['checked_guarded_outputs'],
                guard_qa['checked_routing_windows'])==(24,12,1194)
        assert guard_qa['reference_read'] is False and guard_qa['production_guard_imported'] is False
        for path,expected in guard_qa['input_sha256'].items():
            assert sha(evidence.register(local_evidence_path(path)))==expected
    prior_metrics = evidence.csv(V22 / 'evaluation/metrics.csv')
    prior_wave = evidence.csv(V22 / 'evaluation/waveform_metrics.csv')
    tables = {name: [] for name in ('metrics', 'common_metrics', 'paired_metrics', 'deltas', 'continuity',
        'waveform_metrics', 'waveform_deltas', 'waveform_windows', 'aligned_windows', 'reference_windows',
        'sensitivity', 'sensitivity_common', 'sensitivity_pairs', 'waveform_sensitivity', 'waveform_pair_sensitivity',
        'geometry_audit', 'frontend_metrics', 'pixel_source_metrics', 'baseline_reproduction',
        'provenance_audit', 'fallback_audit', 'routing_windows', 'diagnostic_status_counts', 'macro_metrics')}
    case_metadata = {}
    pairs = list(itertools.combinations(versions, 2))
    for case in CASES:
        data, geometry_rows = load_case(case, versions, args, evidence, frozen22, frozen24)
        tables['geometry_audit'].extend(geometry_rows)
        base = data['trimmed_gap10']
        for version, item in data.items():
            if version in CANDIDATES:
                audit_mechanism(case, version, item, evidence, tables, data)
            trace = item['trace']
            front = dict(case=case, version=version, frames=item['frames'])
            for col in ('face_detected', 'flow_available', 'rgb_valid', 'forehead_valid', 'left_cheek_valid', 'right_cheek_valid'):
                mask = core.bools(trace[col], col)
                front[col + '_count'] = int(mask.sum())
                front[col + '_pct'] = 100 * mask.mean()
            tables['frontend_metrics'].append(front)
            for column in trace:
                if column.endswith('_pixel_source'):
                    counts = trace[column].fillna('<missing_field>').astype(str).value_counts()
                    roi = column.removesuffix('_pixel_source')
                    stage = 'unanchored_V23' if roi.endswith('_unanchored') else 'anchored_V24'
                    roi = roi.removesuffix('_unanchored')
                    assert roi in ROIS
                    for source, count in counts.items():
                        tables['pixel_source_metrics'].append(dict(case=case, version=version,
                            roi=roi,signal_stage=stage,source=source, frames=item['frames'],
                            count=int(count), all_video_frames_pct=100 * int(count) / item['frames']))
            for variant in VARIANTS:
                tables['continuity'].append(dict(case=case, version=version, variant=variant,
                                                **continuity_row(item['variants'][variant])))
        shifts = alignment['sensitivity_shifts_s'] if case == 'data1' else [0]
        for shift in shifts:
            reference, ref_detail = core.reference_for(case, base['centers'], shift, evidence, alignment)
            for i, center in enumerate(base['centers']):
                tables['reference_windows'].append(dict(case=case, shift_s=shift, window_index=i, time_s=center,
                    reference_bpm=reference[i], reference_kind=core.REFERENCE_KIND.get(case, 'none'),
                    reference_eligible=bool(np.isfinite(reference[i])), **ref_detail[i],
                    video_window_center_utc_ns=exact_data1_center_ns(alignment, base['starts'][i], base['window'], shift)
                        if case == 'data1' else None))
            for variant in VARIANTS:
                common = np.logical_and.reduce([data[v]['variants'][variant]['accepted'] for v in versions])
                own_rows = {}
                scores = {v: core.wave_scores(data[v]['variants'][variant], reference) for v in versions}
                common_snr = np.logical_and.reduce([np.isfinite(scores[v]) for v in versions])
                for version, item in data.items():
                    d = item['variants'][variant]
                    for estimator, column in ESTIMATORS.items():
                        keys = dict(case=case, version=version, variant=variant, estimator=estimator,
                            branch=f'{version}/{variant}/{estimator}', shift_s=shift,
                            reference_kind=core.REFERENCE_KIND.get(case, 'none'))
                        prediction = d['hr'][column].to_numpy(float)
                        own = dict(**keys, scope='own', **core.metrics(prediction, d['accepted'], reference))
                        shared = dict(**keys, scope='strict_common_all_versions_same_variant',
                                      **core.metrics(prediction, common, reference))
                        own_rows[(version, estimator)] = own
                        if case == 'data1':
                            tables['sensitivity'].append(own)
                            tables['sensitivity_common'].append(shared)
                        if shift == 0:
                            tables['metrics'].append(own)
                            tables['common_metrics'].append(shared)
                            if variant == 'fusion' and version in BASELINES:
                                audit_baseline(case, version, estimator, own, prior_metrics)
                                tables['baseline_reproduction'].append(dict(case=case, version=version,
                                    estimator=estimator, metrics_reproduced=True))
                            for i, center in enumerate(base['centers']):
                                valid = bool(d['accepted'][i] and np.isfinite(reference[i]))
                                tables['aligned_windows'].append(dict(**keys, window_index=i, time_s=center,
                                    actual_window_start_s=base['starts'][i] / base['fps'],
                                    actual_window_end_s=(base['starts'][i] + base['window']) / base['fps'],
                                    prediction_bpm=prediction[i] if d['accepted'][i] else np.nan,
                                    diagnostic_raw_peak_bpm=prediction[i] if not d['accepted'][i] else np.nan,
                                    reference_bpm=reference[i],
                                    valid_output=bool(d['accepted'][i]), paired_valid=valid,
                                    all_versions_common_output=bool(common[i]),
                                    error_bpm=prediction[i] - reference[i] if valid else np.nan,
                                    status=d['hr'].status.iloc[i]))
                    for scope, mask in [('own', np.isfinite(scores[version])),
                                        ('strict_common_all_versions_same_variant', common_snr)]:
                        row = dict(case=case, version=version, variant=variant, scope=scope, shift_s=shift,
                            reference_kind=core.REFERENCE_KIND.get(case, 'none'), Nref=int(np.isfinite(reference).sum()),
                            **waveform_row(d, scores[version], mask))
                        if case == 'data1':
                            tables['waveform_sensitivity'].append(row)
                        if shift == 0:
                            tables['waveform_metrics'].append(row)
                            if variant == 'fusion' and scope == 'own' and version in BASELINES:
                                old = prior_wave[prior_wave.case.eq(case) & prior_wave.version.eq(version)
                                                 & prior_wave.scope.eq('own')].iloc[0]
                                for key in ('snr_mean_db', 'snr_median_db', 'finite_sample_pct'):
                                    np.testing.assert_allclose(row[key] if row[key] is not None else np.nan,
                                        old[key], rtol=0, atol=1e-8, equal_nan=True)
                    if shift == 0:
                        for i, center in enumerate(base['centers']):
                            tables['waveform_windows'].append(dict(case=case, version=version, variant=variant,
                                window_index=i, time_s=center, accepted=bool(d['accepted'][i]),
                                finite_segment=bool(d['finite_windows'][i]), reference_bpm=reference[i],
                                snr_ref_h1_db=scores[version][i], all_versions_common_snr=bool(common_snr[i]),
                                waveform_generated=bool(d['generated'][i]) if d['generated'] is not None else None))
                for before, after in pairs:
                    for estimator in ESTIMATORS:
                        paired = pair_rows(case, variant, estimator, before, after, data, reference, shift)
                        if case == 'data1':
                            tables['sensitivity_pairs'].extend(paired)
                        if shift == 0:
                            tables['paired_metrics'].extend(paired)
                            own_a, own_b = own_rows[(before, estimator)], own_rows[(after, estimator)]
                            pa, pb, added, lost = paired
                            assert own_b['Nwithin5'] - own_a['Nwithin5'] == \
                                pb['Nwithin5'] - pa['Nwithin5'] + added['Nwithin5'] - lost['Nwithin5']
                            delta = dict(case=case, variant=variant, estimator=estimator, before=before, after=after,
                                Ncommon_output=pa['Ncommon_output'], Nadded_output=pa['Nadded_output'],
                                Nlost_output=pa['Nlost_output'])
                            for key in ('MAE_bpm', 'RMSE_bpm', 'Bias_bpm', 'P5_valid_pct', 'R5_all_reference_pct', 'C_out_pct'):
                                delta['own_delta_' + key] = core.difference(own_b[key], own_a[key])
                                delta['common_delta_' + key] = core.difference(pb[key], pa[key])
                            tables['deltas'].append(delta)
                    mask = np.isfinite(scores[before]) & np.isfinite(scores[after])
                    changes = scores[after][mask] - scores[before][mask]
                    row = dict(case=case, variant=variant, before=before, after=after, shift_s=shift,
                        Npaired_snr=len(changes), mean_paired_delta_snr_db=float(changes.mean()) if len(changes) else None,
                        median_paired_delta_snr_db=float(np.median(changes)) if len(changes) else None,
                        positive_delta_pct=100 * float(np.mean(changes > 1e-9)) if len(changes) else None)
                    if case == 'data1':
                        tables['waveform_pair_sensitivity'].append(row)
                    if shift == 0:
                        tables['waveform_deltas'].append(row)
        case_metadata[case] = dict(frames=base['frames'], fps=base['fps'], duration_s=base['frames'] / base['fps'],
            actual_window_s=base['window'] / base['fps'], actual_step_s=base['step'] / base['fps'],
            Nplanned=len(base['centers']), directories={v: str(d['directory']) for v, d in data.items()},
            reference_kind=core.REFERENCE_KIND.get(case, 'none'))
    assert len(tables['metrics']) == len(CASES) * len(versions) * len(VARIANTS) * len(ESTIMATORS)
    assert len(tables['baseline_reproduction']) == 24
    assert len(tables['sensitivity']) == 7 * len(versions) * len(VARIANTS) * len(ESTIMATORS)
    gate = {'evaluated': False, 'reason': 'baseline-only precheck'}
    compat_gate = dict(gate)
    if not args.baseline_only:
        engineering_pass = replay_qa['passed'] and guard_qa['passed'] and all(x['passed'] for x in tables['provenance_audit'])
        comparisons = {candidate: v24_gate(tables, criteria, same_gap_baseline(candidate),
                                          candidate, engineering_pass) for candidate in CANDIDATES}
        gate = dict(evaluated=True, default=comparisons[criteria['primary_candidate']],
                    separate_continuity_option=comparisons[criteria['continuity_candidate']],
                    all_predefined_candidates=comparisons,
                    candidate_choice='Fixed before inference; no best-score promotion')
        compat_gate = dict(evaluated=True, original_criteria_sha256=COMPAT_CRITERIA_SHA,
            engineering_adaptation='V24 manifest and declared architecture replace V23 unchanged-backend assumptions.',
            all_predefined_candidates={candidate: compatibility_gate(tables, compat_criteria,
                same_gap_baseline(candidate), candidate, engineering_pass) for candidate in CANDIDATES})
    evidence.verify()
    args.output.mkdir(parents=True, exist_ok=False)
    output_manifest = {}
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        integer_times = [col for col in frame if col.endswith('_utc_ns')]
        for col in integer_times:
            frame[col] = pd.array([row.get(col) for row in rows], dtype='Int64')
        path = args.output / f'{name}.csv'
        frame.to_csv(path, index=False)
        if len(frame.columns):
            reread = pd.read_csv(path, dtype={col: 'Int64' for col in integer_times})
            assert frame.shape == reread.shape and list(frame) == list(reread)
            np.testing.assert_array_equal(frame.isna(), reread.isna())
            for col in integer_times:
                pd.testing.assert_series_equal(frame[col], reread[col], check_names=False)
        output_manifest[path.name] = dict(rows=len(frame), sha256=sha(path), columns=list(frame))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluator_sha256=sha(Path(__file__)),
        frozen_v22_evaluator_sha256=CORE_SHA, evaluation_protocol_sha256=criteria_sha,
        v23_compatibility_criteria_sha256=COMPAT_CRITERIA_SHA,
        implementation_freeze=implementation_freeze,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        cases=list(CASES), versions=list(versions), variants=list(VARIANTS), case_metadata=case_metadata,
        primary_variant='fusion', primary_shift_s=0, alignment_protocol=alignment,
        source_protocol_v22=frozen22, source_protocol_v24=frozen24, run_records=runs,
        evaluation_protocol=criteria, upgrade_gate=gate, v23_compatibility_gate=compat_gate,
        QA=dict(self_check=qa, baseline_metrics_reproduced=True, baseline_SNR_reproduced=True,
            baseline_RGB_geometry_flags_and_sources_verified=True,
            four_candidates_share_frontend_with_1e_9_serialization_tolerance=True,
            tests=tests, waveform_replay=replay_qa,
            provenance_from_positive_diagnostics_verified=True if not args.baseline_only else None,
            reference_free_guard_routing_verified=True if not args.baseline_only else None, input_hashes_unchanged=True,
            independent_guard_routing=guard_qa,
            CSV_shape_missingness_and_integer_timestamp_roundtrip=True),
        definitions=dict(
            error='Predicted minus reference BPM. Own windows differ; paired common/new/lost retain full Nplanned and Nref denominators.',
            P5='100 * Nwithin5 / Nvalid; R5 = 100 * Nwithin5 / Nref. No reference produces NA.',
            primary_HR='Final HR read from saved waveform output; offline_ridge and local_peak both reported and guarded.',
            rejected_diagnostics='Fusion rejected HR columns stay missing. POS/CHROM local-peak column may retain a pre-gate diffuse-spectrum diagnostic; accepted=False always excludes it from coverage and scoring, and aligned output separates that diagnostic from prediction.',
            SNR='Frozen V22 reference-H1 periodic Hann full window, nfft>=8192, .7-3.5Hz, reference H1 +/- .1Hz versus remaining band; accepted HR, finite full waveform and eligible reference required.',
            common='Strict common across all versions is descriptive; upgrade uses pairwise common against its same-gap baseline.',
            waveform_coverage='Finite saved base samples / all original video frames. It includes marked interpolation and neighboring-window contributions and is not physiological accuracy.',
            interruption='HR longest rejected grid run is rejected decision count * actual step duration; first-to-last rejected-center span is also reported. Waveform longest missing run is missing sample count / FPS. Both include boundary missing runs.',
            support='Accepted HR support intervals are unioned; overlapping ten-second windows are never summed into validated duration.',
            proxies='Peak concentration is an engineering frequency concentration proxy, not a probability of correctness.',
            frontend='V24 baseline_* RGB and unchanged geometry/masks are checked against V22. Anchored RGB is intentionally changed. All four V24 modes share the same frontend. Discrete values/missingness exact, numbers atol 1e-9. Actual source path disclosed; V23 paired increments are reused, not a new pixel extraction.',
            macro='Human reference MAE is the equal mean of UBFC/data1 MAEs; source-balanced RMSE=sqrt(mean(per-case RMSE squared)). Human coverage equals the mean of five per-capture percentages. Synthetic is separate.',
            LoA='Bias +/- 1.96 sample SD, descriptive only; highly overlapping windows are not independent.'),
        limitations=[
            'All six sources were previously inspected development/regression data; not held-out generalization evidence.',
            'Five human recordings plus one synthetic generator do not establish six independent participants.',
            'UBFC uses paired device HR, not ECG. data1 absolute synchronization remains estimated.',
            'All seven data1 shifts are disclosed without selecting a best shift or fitting timing.',
            'No-reference videos have no measurable HR accuracy. Synthetic 72 bpm is an engineering control.',
            'No qualified waveform morphology reference; r_wave is NA. Coverage or reference-H1 SNR does not establish morphology recovery.'],
        input_manifest=evidence.manifest, output_manifest=output_manifest)
    (args.output / 'summary.json').write_text(json.dumps(core.clean(report), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'upgrade_gate.json').write_text(json.dumps(core.clean(gate), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'v23_compatibility_gate.json').write_text(json.dumps(core.clean(compat_gate),
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    primary = pd.DataFrame(tables['metrics'])
    print(primary[primary.variant.eq('fusion') & primary.estimator.eq('offline_ridge')]
          [['case', 'version', 'Noutput', 'Nplanned', 'MAE_bpm', 'RMSE_bpm', 'R5_all_reference_pct']].to_string(index=False))
    print(json.dumps(dict(evaluated=gate['evaluated'],
        candidates={k:v['eligible_for_named_upgrade'] for k,v in gate.get('all_predefined_candidates',{}).items()})))
    print(f'Completed: {args.output}')


if __name__ == '__main__':
    main()
