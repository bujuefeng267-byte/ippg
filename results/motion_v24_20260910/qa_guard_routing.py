"""Independent CSV-only V24 branch routing and frontend identity audit.

No production module, physiological reference or video is imported/read. Since
final diagnostics zero unselected weights, original tracked contributions come
from the same-gap bounded counterpart; original baseline contributions come
from V22. Corresponding branch proposals/waves are checked before using them.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALIDATION = HERE/'validation'
CASES = ('user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1')
MODES = ('bounded_tracking_gap10', 'bounded_tracking_gap15', 'guarded_fusion_gap10', 'guarded_fusion_gap15')
ROIS = ('forehead', 'left_cheek', 'right_cheek')
GUARD = dict(min_tracked_fraction=.60, max_reset_fraction=.05,
             switch_score_ratio=1.15, max_branch_disagreement_bpm=12.)
EVIDENCE = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def register(path):
    path = Path(path)
    now = sha(path)
    previous = EVIDENCE.setdefault(str(path), now)
    assert previous == now, f'Input changed: {path}'
    return path


def csv(path):
    return pd.read_csv(register(path))


def js(path):
    return json.loads(register(path).read_text(encoding='utf-8'))


def bools(series):
    assert not series.isna().any() and np.isin(series.to_numpy(), [True, False, 0, 1]).all(), series.name
    return series.to_numpy(bool)


def numeric(actual, expected, label, tolerance=1e-8):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape, f'{label}: shape mismatch'
    assert np.array_equal(np.isfinite(actual), np.isfinite(expected)), f'{label}: missingness differs'
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True, err_msg=label)


def columns_equal(a, b, columns, context):
    assert len(a) == len(b), f'{context}: row count'
    for key in columns:
        if pd.api.types.is_numeric_dtype(a[key]) or pd.api.types.is_bool_dtype(a[key]):
            numeric(a[key], b[key], f'{context}.{key}')
        else:
            np.testing.assert_array_equal(a[key].fillna('<missing>'), b[key].fillna('<missing>'), err_msg=f'{context}.{key}')


def canonical(table):
    return table.rename(columns={'proposal_accepted': 'accepted', 'proposal_status': 'status',
                                 'proposal_spectral_peak_bpm': 'spectral_peak_bpm', 'consensus_bpm': 'ridge_bpm'})


def audit_frontend(mode, case, frozen):
    directory = VALIDATION/mode/case
    trace = csv(directory/'frame_trace.csv')
    summary = js(directory/'summary.json')
    original = csv(ROOT/'rppg_motion_v22/validation/trimmed_gap10'/case/'frame_trace.csv')
    paired = csv(ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/case/'frame_trace.csv')
    assert summary['source_hashes'] == frozen['source_hashes']
    assert summary['config']['reference_ubfc'] is None and summary['reference_identity'] is None
    assert summary['identity']['pixel_mode'] == 'tracking_screened'
    assert summary['anchor_config'] == dict(anchor_hz=.15, anchor_order=4)
    assert summary['identity']['pixel_config'] == frozen['candidates'][mode]['pixel_config']
    varying = {'r', 'g', 'b'} | {f'{roi}_{c}' for roi, c in itertools.product(ROIS, 'rgb')}
    columns_equal(trace, original, [c for c in original if c not in varying], f'{mode}/{case}: V22 geometry/masks')
    for col in varying:
        numeric(trace[f'baseline_{col}'], original[col], f'{mode}/{case}: baseline {col}')
        numeric(trace[f'unanchored_{col}'], paired[col], f'{mode}/{case}: original V23 relative {col}')
    pixel_fields = [c for c in paired if '_pixel_' in c and not c.endswith('_pixel_source')]
    columns_equal(trace, paired, pixel_fields, f'{mode}/{case}: paired increments/three-step diagnostics')
    for roi in ROIS:
        np.testing.assert_array_equal(trace[f'{roi}_unanchored_pixel_source'], paired[f'{roi}_pixel_source'])
        actual = np.isfinite(trace[[f'{roi}_{c}' for c in 'rgb']]).all(1)
        np.testing.assert_array_equal(actual, bools(original[f'{roi}_valid']))
        assert set(trace[f'{roi}_pixel_source']) <= {'tracked_ratio', 'baseline_reset', 'baseline_ratio_fallback', 'missing', 'numerical_reset'}
    # Four modes share reconstruction, so differences must be confined to
    # subsequent waveform processing, not the underlying observations.
    common = csv(VALIDATION/'bounded_tracking_gap10'/case/'frame_trace.csv')
    assert list(trace) == list(common)
    columns_equal(trace, common, list(trace), f'{mode}/{case}: shared V24 reconstruction')
    np.testing.assert_array_equal(trace.frame, np.arange(len(trace)))
    numeric(trace.time_s, np.arange(len(trace))/summary['fps'], 'Original frame times', 1e-7)
    return trace, summary, dict(mode=mode, case=case, frames=len(trace), passed=True,
        v22_fields_unchanged=len(original.columns), paired_fields_unchanged=len(pixel_fields)+len(ROIS))


def route_from_evidence(base, tracked, baseline_generated, tracked_generated, qualified):
    def score(row):
        return float(row.quality_proxy)*(1-.25*float(row.motion_overlap))*(1-.5*float(row.runner_up_ratio))
    bs, ts = score(base), score(tracked)
    if not (tracked_generated and qualified):
        selected, reason = ('baseline', 'tracking_quality_fallback') if baseline_generated else ('none', 'no_qualified_branch')
    elif not baseline_generated:
        selected, reason = 'tracked', 'baseline_unavailable'
    elif abs(float(tracked.ridge_bpm)-float(base.ridge_bpm)) > GUARD['max_branch_disagreement_bpm']:
        selected, reason = 'baseline', 'frequency_disagreement_fallback'
    elif ts >= GUARD['switch_score_ratio']*bs:
        selected, reason = 'tracked', 'stronger_qualified_evidence'
    else:
        selected, reason = 'baseline', 'baseline_evidence_retained'
    return selected, reason, bs, ts


def audit_routing(mode, case, trace, summary):
    directory = VALIDATION/mode/case
    gap = mode.rsplit('_', 1)[1]
    bounded = VALIDATION/f'bounded_tracking_{gap}'/case
    baseline = ROOT/f'rppg_motion_v22/validation/trimmed_{gap}'/case
    assert summary['guard_config'] == GUARD
    routing = csv(directory/'branch_routing.csv')
    final_proposals = canonical(csv(directory/'fusion_proposals.csv'))
    final_diagnostics = csv(directory/'fusion_diagnostics.csv')
    proposals, original_diagnostics = {}, {}
    for branch, counterpart in [('baseline', baseline), ('tracked', bounded)]:
        proposals[branch] = csv(directory/f'{branch}_branch_proposals.csv')
        counterpart_proposals = canonical(csv(counterpart/'fusion_proposals.csv'))
        columns_equal(proposals[branch], counterpart_proposals, list(proposals[branch]), f'{mode}/{case}: {branch} original proposals')
        branch_wave = csv(directory/f'{branch}_branch_waveform.csv')
        counterpart_wave = csv(counterpart/'fusion_waveform.csv')
        columns_equal(branch_wave, counterpart_wave, ['time_s', 'base'], f'{mode}/{case}: {branch} original waveform')
        original_diagnostics[branch] = csv(counterpart/'fusion_diagnostics.csv')
        original_diagnostics[branch].rename(columns={'proposal_accepted': 'output_accepted'}, inplace=True)
    for saved_name, counterpart in [('roi_waveforms.csv', bounded), ('baseline_roi_waveforms.csv', baseline)]:
        local = csv(directory/saved_name)
        other = csv(counterpart/'roi_waveforms.csv')
        columns_equal(local, other, list(local), f'{mode}/{case}: branch ROI waveform evidence')
    fps = float(summary['fps'])
    window, step = round(summary['config']['window']*fps), round(summary['config']['step']*fps)
    starts = list(range(0, len(trace)-window+1, step))
    np.testing.assert_array_equal(routing.window_index, np.arange(len(starts)))
    numeric(routing.time_s, (np.asarray(starts)+window/2)/fps, 'Routing centers')
    assert len(final_proposals) == len(starts)
    assert set(final_diagnostics.branch.unique()) <= {'baseline', 'tracked'}
    for wi, start in enumerate(starts):
        stop = start+window
        generated, contributors, diags = {}, {}, {}
        for branch in ('baseline', 'tracked'):
            d = original_diagnostics[branch].loc[original_diagnostics[branch].window_index == wi].reset_index(drop=True)
            weights = d.waveform_weight.to_numpy(float)
            assert np.isfinite(weights).all() and (weights >= 0).all()
            contributors[branch] = set(d.loc[weights > 0, 'roi'])
            assert contributors[branch] <= set(ROIS)
            generated[branch] = len(contributors[branch]) >= 2
            p = proposals[branch].iloc[wi]
            assert generated[branch] == (bool(p.accepted) and p.waveform_roi_count >= 2)
            if generated[branch]:
                assert p.waveform_roi_count == len(contributors[branch]), 'Methods/branches must not duplicate anatomical ROIs'
            diags[branch] = d
        fractions, resets = {}, {}
        for roi in ROIS:
            source = trace[f'{roi}_pixel_source'].iloc[start:stop]
            fractions[roi] = float((source == 'tracked_ratio').mean())
            resets[roi] = float(source.isin(['baseline_reset', 'numerical_reset']).mean())
        usable = {roi for roi in contributors['tracked'] if fractions[roi] >= GUARD['min_tracked_fraction']
                  and resets[roi] <= GUARD['max_reset_fraction']}
        qualified = len(usable) >= 2
        b, t = proposals['baseline'].iloc[wi], proposals['tracked'].iloc[wi]
        selected, reason, bs, ts = route_from_evidence(b, t, generated['baseline'], generated['tracked'], qualified)
        record = routing.iloc[wi]
        assert record.selected_branch == selected and record.selection_reason == reason, f'{mode}/{case}/{wi}: branch decision'
        assert bool(record.baseline_generated) == generated['baseline']
        assert bool(record.tracked_generated) == generated['tracked']
        assert bool(record.tracking_eligible) == qualified
        for field, value in dict(baseline_score=bs, tracked_score=ts, baseline_bpm=b.ridge_bpm,
                tracked_bpm=t.ridge_bpm, minimum_tracked_fraction=min(fractions.values()),
                second_largest_tracked_fraction=sorted(fractions.values())[1], maximum_reset_fraction=max(resets.values()),
                baseline_motion_overlap=b.motion_overlap, tracked_motion_overlap=t.motion_overlap).items():
            numeric(record[field], value, f'{mode}/{case}/{wi}: routing {field}')
        positive_rois = set()
        for branch in ('baseline', 'tracked'):
            actual = final_diagnostics.loc[(final_diagnostics.window_index == wi) & (final_diagnostics.branch == branch)].reset_index(drop=True)
            original = diags[branch]
            check = actual.copy().rename(columns={'proposal_accepted': 'output_accepted'})
            assert list(check.original_channel) == list(original.channel)
            assert list(check.channel) == list(check.roi+'_'+branch+'_'+original.method)
            assert list(check.method) == list(branch+'_'+original.method)
            unchanged = [c for c in original if c not in {'channel', 'method', 'waveform_weight', 'output_accepted'}]
            columns_equal(check, original, unchanged, f'{mode}/{case}/{wi}: {branch} preserved diagnostics')
            if branch == selected:
                numeric(check.waveform_weight, original.waveform_weight, 'Selected actual contributor weights')
                np.testing.assert_array_equal(bools(check.output_accepted), bools(original.output_accepted))
                positive_rois |= set(check.loc[check.waveform_weight > 0, 'roi'])
            else:
                assert (check.waveform_weight == 0).all(), 'Unselected branch contributes waveform'
                assert not bools(check.output_accepted).any(), 'Unselected branch marked accepted'
        final = final_proposals.iloc[wi]
        if selected == 'none':
            assert not bool(final.accepted) and final.waveform_roi_count == 0
            assert not positive_rois and np.isnan(final.ridge_bpm)
        else:
            assert bool(final.accepted)
            assert final.waveform_roi_count == len(positive_rois) and 2 <= len(positive_rois) <= 3
            chosen = proposals[selected].iloc[wi]
            for field in ('time_s', 'ridge_bpm', 'spectral_peak_bpm', 'roi_count', 'quality_proxy', 'waveform_roi_count'):
                numeric(final[field], chosen[field], f'{mode}/{case}/{wi}: selected proposal {field}')
    return dict(mode=mode, case=case, windows=len(starts), passed=True,
                selected_branch_counts={str(k): int(v) for k, v in routing.selected_branch.value_counts().items()},
                selection_reason_counts={str(k): int(v) for k, v in routing.selection_reason.value_counts().items()})


def main():
    frozen = js(VALIDATION/'protocol_before_validation.json')
    runs = js(VALIDATION/'runs.json')
    assert len(runs) == 24 and all(r['exit_code'] == 0 for r in runs), 'Inference round is incomplete'
    assert {(r['mode'], r['case']) for r in runs} == set(itertools.product(MODES, CASES))
    v23 = js(ROOT/'rppg_motion_v23/validation/protocol_before_validation.json')
    unchanged_sources = ('pixel_tracking.py', 'baseline_frontend.py', 'motion_frontend.py',
                         'legacy_motion.py', 'analyze_rppg.py', 'stable_groups.py', 'waveform_hr.py')
    for name in unchanged_sources:
        assert frozen['source_hashes'][name] == v23['source_hashes'][name], f'Three-step/backend source changed: {name}'
    for name, expected in frozen['source_hashes'].items():
        assert sha(register(HERE/name)) == expected, f'Frozen source changed: {name}'
    frontends, routing, errors = [], [], []
    for mode, case in itertools.product(MODES, CASES):
        try:
            trace, summary, record = audit_frontend(mode, case, frozen)
            frontends.append(record)
            if mode.startswith('guarded_fusion'):
                assert frozen['candidates'][mode]['guard_config'] == GUARD
                routing.append(audit_routing(mode, case, trace, summary))
            print(f'PASS {mode}/{case}', flush=True)
        except Exception as error:
            errors.append(dict(mode=mode, case=case, error=f'{type(error).__name__}: {error}'))
            print(f'FAIL {mode}/{case}: {error}', flush=True)
    for path, expected in EVIDENCE.items():
        if sha(path) != expected:
            errors.append(dict(error='Evidence changed during audit', path=path))
    result = dict(passed=not errors and len(frontends) == 24 and len(routing) == 12,
        checked_utc=datetime.now(timezone.utc).isoformat(), checked_cases=6, checked_modes=4,
        checked_frontend_outputs=len(frontends), checked_guarded_outputs=len(routing),
        checked_routing_windows=sum(x['windows'] for x in routing), guard_config=GUARD,
        production_guard_imported=False, reference_read=False, video_inference_rerun=False,
        unchanged_source_modules=list(unchanged_sources), frontend_checks=frontends, routing_checks=routing,
        input_sha256=EVIDENCE, script_sha256=sha(__file__), errors=errors,
        original_unselected_weight_evidence='Same-gap bounded_tracking diagnostics for tracked branch and V22 diagnostics for baseline branch, after checking branch proposal and waveform equivalence.')
    output = VALIDATION/'guard_routing_qa.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checked_frontends=len(frontends),
                         checked_routing_windows=result['checked_routing_windows'], errors=errors), ensure_ascii=False), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
