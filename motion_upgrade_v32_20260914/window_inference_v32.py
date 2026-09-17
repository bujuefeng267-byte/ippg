"""Reference-free V32 inference from separately saved, measured 10 s windows.

The continuous waveform.csv is an unmodified V28 context artifact. New HRs
come ONLY from the individually hashed files named in windows_manifest.csv.
Overlapping windows are separate observations, not a reconstructed continuous
PPG waveform. No reference, participant identity or video selector is read.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

import proposal_evidence_v31 as proposals_v31
from evidence_hr import DEFAULT_CONFIG, score_candidates
from legacy_motion import estimate as legacy_estimate, runs
from motion_evidence import motion_evidence
from motion_harmonic_evidence_v26 import motion_evidence_harmonics
from protected_inference_v31 import jump_info, save_json
from protected_waveform_router import ProtectionError, _plan, _waveform, constrained_evidence_path


PARAMETERS = dict(window_s=10., step_s=1., min_bpm=42., max_bpm=210.)
GRID = np.arange(42., 210.01, 1.)
FIELDS = ('time_s', 'base', 'covered', 'observed', 'interpolated')
SOURCE_OLD, SOURCE_CDF = 'original_v28', 'independent_cdf'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol_description():
    return dict(**PARAMETERS, reference_used=False, offline=True,
        variants={'window_only': False, 'motion_guard': True},
        proposal='Unchanged V31 raw_consensus admission on original physical ROI channels.',
        state_policy='Measured candidates within 3 bpm of proposed CDF HR; every permitted state revalidated against original physical ROI support and inherited admission evidence.',
        motion_policy='Main variant requires available extended-band direct motion; candidate risk <0.50 and <= original risk + 1e-9 at both proposed and final HR. No half/double relation grants admission.',
        output_policy='Each complete 10 s window owns its saved measured segment. Continuous waveform.csv is byte-identical V28 context, not the source of replacement HR.',
        preservation='Original rejects remain missing; unselected accepted HRs are measured-support-verified V28 anchors. Candidate gate failure reverts that window only.',
        jump_policy='No increase in adjacent accepted >12 bpm jumps; maximum <= max(12, original maximum). Withdraw selected endpoints of newly excessive edges only; full fallback if necessary.',
        waveform_morphology_validated=False, continuous_waveform_improved=False,
        diagnostic_ablation_recommendable=False)


def _original_roi_inputs(trace, fps):
    # Isolated seam for controlled synthetic tests; production always derives
    # these channels from the bound original trace, never from CDF/reference.
    from analyze_components_v26 import validate_component_trace, prepare_roi_channels
    validate_component_trace(trace, fps)
    return prepare_roi_channels(trace, fps)


def _direct_profile(evidence):
    available = bool(evidence['available'])
    if not available:
        return False, np.full(len(GRID), np.nan)
    profile = np.asarray(evidence['profile_direct'], float)
    if (profile.shape != GRID.shape or not np.isfinite(profile).all() or
            np.any((profile < 0) | (profile > 1))):
        raise ValueError('Invalid inherited direct-motion profile')
    return True, profile


def _motion_pass(available, old_risk, candidate_risk):
    return bool(available and np.isfinite([old_risk, candidate_risk]).all() and
                candidate_risk < .50 and candidate_risk <= old_risk + 1e-9)


def _state_evidence(trace, fps, start, width, old_bpm, original):
    """Revalidate all grid states, scoring each original ROI channel once.

    Qualification and physical-region vote counting exactly follow the V31
    proposal gates. POS/CHROM or tracked/baseline are not independent regions.
    """
    channels, observed, qualities, sources = original
    roi_config = proposals_v31.ROI_CONFIG
    support = {roi: np.zeros(len(GRID), bool) for roi in proposals_v31.ROIS}
    consensus_methods = {}
    stop = start + width
    for branch in proposals_v31.BRANCHES:
        for roi in proposals_v31.ROIS:
            finite_quality = float(np.mean(qualities[roi][start:stop])) >= roi_config.min_quality
            sufficient_observed = float(np.mean(observed[roi][start:stop])) >= roi_config.min_observed
            source = sources[roi][start:stop]
            tracked = float(np.mean(source == 'tracked_ratio')) >= roi_config.min_tracked_fraction
            resets = float(np.mean(np.isin(source, ['baseline_reset', 'numerical_reset']))) <= roi_config.max_reset_fraction
            qualified = finite_quality and sufficient_observed and (branch != 'tracked' or (tracked and resets))
            for method in proposals_v31.METHODS:
                key = f'{branch}/{roi}/{method}'
                signal = np.asarray(channels[key][start:stop], float)
                candidates = []
                if qualified and np.isfinite(signal).all():
                    candidates, _ = score_candidates(signal, fps, GRID, motion=None)
                if candidates:
                    strongest = min(candidates, key=lambda item: (-item['relative_power'], item['bpm']))
                    for candidate in candidates:
                        for bpm, power in zip(candidate['supported_bpm'], candidate['supported_relative_power']):
                            if power >= DEFAULT_CONFIG.min_relative_power:
                                support[roi] |= GRID == bpm
                    if branch == 'baseline':
                        consensus_methods[(roi, method)] = ((np.abs(GRID - strongest['bpm']) <= DEFAULT_CONFIG.candidate_radius_bpm) &
                            (abs(strongest['bpm'] - old_bpm) > DEFAULT_CONFIG.peak_distance_bpm))
                elif branch == 'baseline':
                    consensus_methods[(roi, method)] = np.zeros(len(GRID), bool)
    support_count = sum(support.values())
    consensus_count = sum(np.logical_and.reduce([
        consensus_methods[(roi, method)] for method in proposals_v31.METHODS]) for roi in proposals_v31.ROIS)
    available, profile = _direct_profile(motion_evidence_harmonics(trace, int(start), int(stop), fps, GRID))
    old_risk = float(np.interp(old_bpm, GRID, profile)) if available else np.nan
    direct = ((old_risk >= proposals_v31.MIN_OLD_MOTION_RISK) &
              (old_risk - profile >= proposals_v31.MIN_MOTION_RISK_DECREASE)) if available else np.zeros(len(GRID), bool)
    inherited = (support_count >= roi_config.min_rois) & (direct | (consensus_count >= roi_config.min_rois))
    inherited &= np.abs(GRID - old_bpm) > DEFAULT_CONFIG.peak_distance_bpm
    guard = np.array([_motion_pass(available, old_risk, risk) for risk in profile])
    return dict(available=available, profile=profile, old_risk=old_risk,
                support_count=support_count, consensus_count=consensus_count,
                inherited_pass=inherited, motion_pass=guard)


def _write_segments(out, text_sources, numeric_sources, hashes, starts, width, fps, selected):
    rows = []
    old = numeric_sources[SOURCE_OLD]
    for i, start in enumerate(starts):
        stop = start + width
        source = SOURCE_CDF if selected[i] else SOURCE_OLD
        file = f'window_waveforms/window_{i:04d}.csv'
        # Keep source decimal spelling, NaN blanks and Boolean spelling. The
        # same parser then produces exact doubles/masks for old and new files.
        segment = text_sources[source].iloc[start:stop][list(FIELDS)].copy()
        segment.insert(0, 'sample_index', np.arange(start, stop))
        segment.to_csv(out/file, index=False)
        reread = pd.read_csv(out/file)
        expected = numeric_sources[source].iloc[start:stop]
        for field in FIELDS:
            np.testing.assert_array_equal(reread[field], expected[field])
        modified = np.zeros(width, bool)
        if selected[i]:
            original = old.iloc[start:stop]
            for field in ('base', 'covered', 'observed', 'interpolated'):
                x, y = np.asarray(reread[field]), np.asarray(original[field])
                modified |= ~((x == y) | (pd.isna(x) & pd.isna(y)))
        rows.append(dict(window_index=i, window_start_s=start/fps, window_end_s=stop/fps,
            time_s=(start+width/2)/fps, waveform_source=source, waveform_file=file,
            sha256=sha256(out/file), source_waveform_sha256=hashes[source],
            candidate_actual_modified=bool(modified.any()), modified_samples=int(modified.sum())))
    return pd.DataFrame(rows)


def _readout(out, manifest, trace, fps, old_hr, selected, proposal, original, evidence_cache, motion_guard):
    _, starts, width, step, old_ok, old_bpm = _plan(trace, fps, old_hr)
    rows, all_candidates, emissions, failed = [], [], [], []
    for i, start in enumerate(starts):
        entry = manifest.iloc[i]
        path = out/entry.waveform_file
        if sha256(path) != entry.sha256:
            raise ProtectionError('Saved window hash mismatch', [i])
        saved = pd.read_csv(path)
        if not np.array_equal(saved.sample_index.to_numpy(), np.arange(start, start+width)):
            raise ProtectionError('Saved sample indices do not follow the window plan', [i])
        values, masks = _waveform(saved, trace.time_s.to_numpy()[start:start+width], 'Saved individual window')
        table = legacy_estimate(values, pd.DataFrame({'rgb_valid': masks['observed']}),
                                masks['interpolated'], fps, **PARAMETERS)
        if len(table) != 1:
            raise ProtectionError('Saved waveform is not one complete 10 s window', [i])
        row = table.iloc[0].to_dict()
        row.update(window_index=i, time_s=entry.time_s, window_start_s=entry.window_start_s,
            window_end_s=entry.window_end_s, waveform_source=entry.waveform_source,
            waveform_file=entry.waveform_file, waveform_sha256=entry.sha256,
            legacy_gate_passed=bool(row['accepted']), legacy_ridge_bpm=row['ridge_bpm'],
            original_v28_accepted=bool(old_ok[i]), original_v28_ridge_bpm=old_bpm[i],
            eligible=bool(selected[i]), protected_anchor_verified=False,
            evidence_candidate_count=0, evidence_candidates_json='[]', evidence_reacquired=False,
            evidence_score=np.nan, evidence_selected_relative_power=np.nan,
            evidence_selected_peak_bpm=np.nan, readout_failure='')
        candidates, scores = [], np.full(len(GRID), -np.inf)
        if not old_ok[i]:
            row.update(accepted=False, status=(old_hr.status.iloc[i] if 'status' in old_hr else 'original_v28_rejected'))
        elif not row['accepted']:
            if not selected[i]:
                raise ProtectionError('Unchanged V28 accepted window fails actual saved-window quality gates', [i])
            row['readout_failure'] = row['status']
            failed.append(i)
        else:
            motion = motion_evidence(trace, int(start), int(start+width), fps, GRID)
            candidates, scores = score_candidates(values, fps, GRID, motion=motion)
            row.update(evidence_candidate_count=len(candidates),
                       evidence_candidates_json=json.dumps(candidates, allow_nan=False, separators=(',', ':')))
            if selected[i]:
                if i not in evidence_cache:
                    evidence_cache[i] = _state_evidence(trace, fps, start, width, old_bpm[i], original)
                evidence = evidence_cache[i]
                allowed = evidence['inherited_pass'] & (np.abs(GRID-proposal.candidate_bpm.iloc[i]) <= DEFAULT_CONFIG.candidate_radius_bpm)
                if motion_guard:
                    allowed &= evidence['motion_pass']
                scores[~allowed] = -np.inf
                if not np.isfinite(scores).any():
                    row.update(accepted=False, status='no_safe_supported_candidate', readout_failure='no_safe_supported_candidate')
                    failed.append(i)
            else:
                state = np.flatnonzero(GRID == old_bpm[i])
                if len(state) != 1 or not np.isfinite(scores[state[0]]):
                    raise ProtectionError('Unchanged V28 anchor has no actual saved-window spectral support', [i])
                row['protected_anchor_verified'] = True
        row['ridge_bpm'] = np.nan
        rows.append(row)
        all_candidates.append(candidates)
        emissions.append(scores)
    hr = pd.DataFrame(rows)
    for a, b in runs(hr.accepted.to_numpy(bool)):
        anchors = np.where(~selected[a:b], old_bpm[a:b], np.nan)
        path, reacquired = constrained_evidence_path(emissions[a:b], GRID, step/fps, anchors)
        for i, state in enumerate(path, a):
            frequency = float(GRID[state])
            if not np.isfinite(emissions[i][state]):
                raise ProtectionError('DP selected a forbidden or unsupported measured state', [i])
            if selected[i]:
                e = evidence_cache[i]
                safe = e['inherited_pass'][state] and (abs(frequency-proposal.candidate_bpm.iloc[i]) <= DEFAULT_CONFIG.candidate_radius_bpm)
                safe = safe and (not motion_guard or _motion_pass(e['available'], e['old_risk'], e['profile'][state]))
                if not safe:
                    raise ProtectionError('Final selected frequency violates original ROI or motion evidence', [i])
            candidate = max((item for item in all_candidates[i] if frequency in item['supported_bpm']),
                key=lambda item: item['evidence_score']-DEFAULT_CONFIG.candidate_offset_penalty*(item['bpm']-frequency)**2)
            support_index = candidate['supported_bpm'].index(frequency)
            hr.loc[i, ['ridge_bpm', 'evidence_score', 'evidence_selected_relative_power', 'evidence_selected_peak_bpm', 'evidence_reacquired']] = [
                frequency, emissions[i][state], candidate['supported_relative_power'][support_index], candidate['bpm'], bool(reacquired[i-a])]
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['ridge_bpm', 'spectral_peak_bpm']] = np.nan
    hr['hr_source'] = 'individual_saved_measured_window_V28_gates_constrained_evidence_DP'
    return hr, failed


def withdraw_selected_endpoints(selected, failed):
    """Withdraw explicitly failed selected windows, never overlapping samples."""
    selected = np.asarray(selected, bool)
    result = selected.copy()
    indices = np.asarray(sorted(set(int(i) for i in failed)), int)
    if len(indices):
        if np.any((indices < 0) | (indices >= len(selected))):
            raise ValueError('Failed window index outside plan')
        result[indices] = False
    if np.array_equal(result, selected) and selected.any():
        result[:] = False
    return result


def infer_windows(old_wave_path, old_hr_path, candidate_wave_path, candidate_hr_path,
                  trace, fps, out, *, motion_guard=True):
    """Return (heart_rate, routing_decisions, preservation_audit), saving proof.

    Out may contain a caller protocol/candidate directory. Existing core
    artifacts cause failure; no previous result is overwritten. False is the
    predeclared diagnostic ablation, never a recommended runtime mode.
    """
    if type(motion_guard) is not bool:
        raise ValueError('motion_guard must be an explicit Boolean')
    out = Path(out)
    targets = ('window_waveforms', 'waveform.csv', 'heart_rate.csv', 'windows_manifest.csv',
               'proposal_evidence.csv', 'routing_decisions.csv', 'guard_history.json', 'preservation_audit.json')
    for name in targets:
        if (out/name).exists():
            raise FileExistsError(out/name)
    paths = {SOURCE_OLD: Path(old_wave_path), SOURCE_CDF: Path(candidate_wave_path)}
    source_hashes = {source: sha256(path) for source, path in paths.items()}
    hr_hashes = {str(path): sha256(path) for path in (old_hr_path, candidate_hr_path)}
    numeric = {source: pd.read_csv(path) for source, path in paths.items()}
    text = {source: pd.read_csv(path, dtype=str, keep_default_na=False) for source, path in paths.items()}
    old_hr, candidate_hr = pd.read_csv(old_hr_path), pd.read_csv(candidate_hr_path)
    times, starts, width, _, old_ok, old_bpm = _plan(trace, fps, old_hr)
    if not len(starts):
        raise ValueError('At least one complete original 10 s window is required')
    for source in paths:
        _waveform(numeric[source], times, source)
    channels, tables = _original_roi_inputs(trace, fps)
    original = proposals_v31._roi_inputs(trace, fps, channels, tables)
    proposal = proposals_v31.build_proposals(numeric[SOURCE_OLD], old_hr, numeric[SOURCE_CDF],
        candidate_hr, trace, fps, roi_channels=channels, roi_tables=tables,
        config=proposals_v31.ProposalConfig(mode='raw_consensus'))
    initial = proposal.proposal_eligible.to_numpy(bool)
    selected = initial.copy()
    motion_initial = np.zeros(len(proposal), bool)
    for i in np.flatnonzero(initial):
        p = proposal.iloc[i]
        motion_initial[i] = _motion_pass(p.motion_available, p.old_direct_motion_risk, p.candidate_direct_motion_risk)
    if motion_guard:
        selected &= motion_initial
    admitted = selected.copy()
    out.mkdir(parents=True, exist_ok=True)
    (out/'window_waveforms').mkdir()
    proposal.to_csv(out/'proposal_evidence.csv', index=False)
    old_jumps = jump_info(old_hr)
    history, evidence_cache = [], {}
    for iteration in range(int(admitted.sum())+2):
        manifest = _write_segments(out, text, numeric, source_hashes, starts, width, fps, selected)
        ineffective = selected & ~manifest.candidate_actual_modified.to_numpy(bool)
        if ineffective.any():
            history.append(dict(iteration=iteration, status='withdraw_identical_candidate', failed_windows=np.flatnonzero(ineffective).tolist()))
            selected[ineffective] = False
            continue
        hr, failed = _readout(out, manifest, trace, fps, old_hr, selected, proposal, original, evidence_cache, motion_guard)
        assert not np.any(hr.accepted.to_numpy(bool) & ~old_ok)
        lost = np.flatnonzero(old_ok & ~hr.accepted.to_numpy(bool))
        failed = set(failed) | set(lost.tolist())
        new_jumps = jump_info(hr)
        excessive = (new_jumps['count'] > old_jumps['count'] or
                     new_jumps['maximum'] > max(12., old_jumps['maximum'])+1e-9)
        bad_edges = []
        if excessive:
            bad_edges = np.flatnonzero((new_jumps['delta'] > 12.+1e-9) &
                                      (new_jumps['delta'] > old_jumps['delta']+1e-9)).tolist()
            for edge in bad_edges:
                failed.update(index for index in (edge, edge+1) if selected[index])
        record = dict(iteration=iteration, selected_windows=np.flatnonzero(selected).tolist(),
            quality_failures=lost.tolist(), failed_windows=sorted(failed), excessive_edges=bad_edges,
            large_jump_count=new_jumps['count'], maximum_jump_bpm=new_jumps['maximum'],
            excess_jump_guard=bool(excessive))
        history.append(record)
        if not len(lost) and not excessive and not failed:
            record['status'] = 'accepted'
            break
        if not selected.any():
            raise ProtectionError('Complete V28 fallback failed to reproduce its quality and jumps', sorted(failed))
        record['status'] = 'withdraw_selected_windows_and_recompute'
        revised = withdraw_selected_endpoints(selected, failed)
        record['withdrawn_windows'] = np.flatnonzero(selected & ~revised).tolist()
        selected = revised
    else:
        raise ProtectionError('Monotone per-window rollback failed to terminate')
    # The final delivery is itself reread and rescored, rather than copying
    # candidate HR numbers or trusting an in-memory path after file writes.
    final_hr, failures = _readout(out, manifest, trace, fps, old_hr, selected, proposal, original, evidence_cache, motion_guard)
    if failures:
        raise ProtectionError('Final saved-window revalidation failed', failures)
    np.testing.assert_array_equal(final_hr.accepted, old_hr.accepted)
    np.testing.assert_array_equal(final_hr.ridge_bpm, hr.ridge_bpm)
    np.testing.assert_array_equal(final_hr.ridge_bpm[~selected], old_hr.ridge_bpm[~selected])
    decisions = proposal.copy(deep=True)
    decisions['initial_eligible'] = initial
    decisions['initial_motion_guard_pass'] = motion_initial
    decisions['admitted_after_initial_motion_guard'] = admitted
    decisions['eligible'] = selected
    decisions['withdrawn_by_runtime_guard'] = admitted & ~selected
    decisions['rejected_by_initial_motion_guard'] = initial & ~admitted
    decisions['candidate_actual_modified'] = manifest.candidate_actual_modified.to_numpy(bool)
    decisions['modified_samples'] = manifest.modified_samples.to_numpy(int)
    decisions['final_bpm'] = final_hr.ridge_bpm.to_numpy(float)
    decisions['final_motion_available'] = False
    decisions['final_old_direct_motion_risk'] = np.nan
    decisions['final_candidate_direct_motion_risk'] = np.nan
    decisions['final_motion_guard_pass'] = False
    decisions['final_physical_roi_support_count'] = 0
    decisions['final_raw_consensus_roi_count'] = 0
    decisions['waveform_source'] = manifest.waveform_source.to_numpy()
    decisions['waveform_file'] = manifest.waveform_file.to_numpy()
    for i in np.flatnonzero(selected):
        e = evidence_cache[i]
        state = int(np.flatnonzero(GRID == final_hr.ridge_bpm.iloc[i])[0])
        decisions.loc[i, ['final_motion_available', 'final_old_direct_motion_risk',
            'final_candidate_direct_motion_risk', 'final_motion_guard_pass',
            'final_physical_roi_support_count', 'final_raw_consensus_roi_count']] = [
                e['available'], e['old_risk'], e['profile'][state],
                _motion_pass(e['available'], e['old_risk'], e['profile'][state]),
                int(e['support_count'][state]), int(e['consensus_count'][state])]
    for source, path in paths.items():
        if sha256(path) != source_hashes[source]:
            raise ProtectionError('Input waveform changed during inference')
    for path, expected in hr_hashes.items():
        if sha256(path) != expected:
            raise ProtectionError('Input HR changed during inference')
    shutil.copyfile(old_wave_path, out/'waveform.csv')
    if sha256(out/'waveform.csv') != source_hashes[SOURCE_OLD]:
        raise ProtectionError('Continuous V28 context bytes changed')
    final_hr.to_csv(out/'heart_rate.csv', index=False)
    manifest.to_csv(out/'windows_manifest.csv', index=False)
    decisions.to_csv(out/'routing_decisions.csv', index=False)
    save_json(out/'guard_history.json', history)
    audit = dict(reference_used=False, offline=True, motion_guard=motion_guard,
        variant='motion_guard' if motion_guard else 'window_only',
        initial_eligible_windows=int(initial.sum()), admitted_after_initial_motion_guard=int(admitted.sum()),
        final_eligible_windows=int(selected.sum()), rejected_by_initial_motion_guard=int((initial & ~admitted).sum()),
        withdrawn_windows=int((admitted & ~selected).sum()), guard_attempts=len(history),
        planned_windows=len(manifest), accepted_windows=int(final_hr.accepted.sum()),
        modified_window_samples=int(manifest.modified_samples.sum()),
        modified_sample_count_includes_window_overlap=True,
        original_accepted_mask_preserved=True, unselected_window_samples_and_HR_exact=True,
        old_jump_count=old_jumps['count'], new_jump_count=jump_info(final_hr)['count'],
        old_maximum_jump_bpm=old_jumps['maximum'], new_maximum_jump_bpm=jump_info(final_hr)['maximum'],
        windows_manifest_sha256=sha256(out/'windows_manifest.csv'),
        continuous_context_sha256=sha256(out/'waveform.csv'), continuous_context_original_V28_byte_identical=True,
        continuous_context_is_replacement_HR_source=False, continuous_waveform_improved=False,
        individual_windows_morphology_validated=False, source_waveform_hashes=source_hashes,
        source_hr_hashes=hr_hashes, protocol=protocol_description())
    save_json(out/'preservation_audit.json', audit)
    return final_hr, decisions, audit
