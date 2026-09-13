"""Reference-free local waveform edits with exact V28 window protection.

Eligibility is supplied by the caller, never inferred from a reference HR.
Every noneligible complete 10 s window protects its entire sample interval.
An eligible window can therefore have no editable samples. Positive Hann
weights are additionally tapered inside each remaining editable core, so no
overlap/add weight can escape protection. Old gaps are never filled.

Readout recomputes evidence from the saved final waveform. Protected accepted
windows constrain the original evidence DP to the original V28 frequency,
after verifying that frequency is still supported by measured spectral power.
This is offline constrained inference, not HR interpolation or a reference.
"""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from evidence_hr import DEFAULT_CONFIG, evidence_path, score_candidates
from legacy_motion import estimate as legacy_estimate, runs


PARAMETERS = dict(window_s=10., step_s=1., min_bpm=42., max_bpm=210.)
PROTECTION_CONFIG = dict(
    **PARAMETERS, noneligible_union_protection=True,
    old_rejected_windows_protected=True, original_nan_mask_preserved=True,
    overlap_weight='positive_Hann_eligible_sum_over_all_complete_windows',
    core_weight='positive_Hann_on_each_editable_contiguous_core',
    protected_hr='original_V28_state_constrained_after_actual_spectral_revalidation',
    original_evidence_config=asdict(DEFAULT_CONFIG))


class ProtectionError(RuntimeError):
    """A required protected-state invariant failed; do not publish this output."""

    def __init__(self, message, window_indices=None):
        super().__init__(message)
        self.window_indices = ([] if window_indices is None else
                               [int(index) for index in window_indices])


def _numeric(table, name):
    if name not in table:
        raise ValueError(f'Missing required column: {name}')
    return pd.to_numeric(table[name], errors='coerce').to_numpy(float)


def _flags(values, name, length):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != length:
        raise ValueError(f'{name} must match its complete ordered sample/window plan')
    if values.dtype.kind not in 'biuf':
        raise ValueError(f'{name} requires explicit Boolean or numeric 0/1 flags')
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f'{name} requires explicit Boolean or numeric 0/1 flags')
    return values.astype(bool)


def _waveform(table, times, name):
    if len(table) != len(times) or not np.allclose(
            _numeric(table, 'time_s'), times, atol=1e-8, rtol=0):
        raise ValueError(f'{name} waveform must follow the trace sample clock')
    values = _numeric(table, 'base')
    if np.isinf(values).any():
        raise ValueError(f'{name} may contain finite samples or NaN, not infinity')
    masks = {field: _flags(_numeric(table, field), f'{name}.{field}', len(times))
             for field in ('covered', 'observed', 'interpolated')}
    if not np.array_equal(masks['covered'], np.isfinite(values)):
        raise ValueError(f'{name}.covered must equal finite waveform support')
    if any((masks[field] & ~masks['covered']).any() for field in ('observed', 'interpolated')):
        raise ValueError(f'{name} observation/interpolation cannot extend outside support')
    return values, masks


def _plan(trace, fps, old_hr):
    if not np.isfinite(fps) or fps <= PARAMETERS['max_bpm']/30:
        raise ValueError('FPS must support the frozen 42-210 bpm grid')
    times = _numeric(trace, 'time_s')
    if not np.isfinite(times).all() or not np.allclose(
            times, np.arange(len(times))/fps, atol=1e-8, rtol=0):
        raise ValueError('Trace requires its original zero-origin uniform sample clock')
    window, step = round(PARAMETERS['window_s']*fps), round(PARAMETERS['step_s']*fps)
    starts = np.arange(0, max(0, len(times)-window+1), step, dtype=int)
    centers = (starts+window/2)/fps
    if len(old_hr) != len(starts) or not np.allclose(
            _numeric(old_hr, 'time_s'), centers, atol=1e-8, rtol=0):
        raise ValueError('Original HR must follow the complete ordered 10 s / 1 s plan')
    for field, expected in dict(window_index=np.arange(len(starts)),
                                window_start_s=starts/fps,
                                window_end_s=(starts+window)/fps).items():
        if field in old_hr and not np.allclose(_numeric(old_hr, field), expected,
                                              atol=1e-8, rtol=0):
            raise ValueError(f'Original HR {field} contradicts the declared plan')
    accepted = _flags(_numeric(old_hr, 'accepted'), 'old_hr.accepted', len(starts))
    ridge = _numeric(old_hr, 'ridge_bpm')
    if not np.isfinite(ridge[accepted]).all():
        raise ValueError('Every accepted original HR needs its finite measured ridge')
    if np.isfinite(ridge[~accepted]).any():
        raise ValueError('Rejected original HR must remain missing, not held or interpolated')
    return times, starts, window, step, accepted, ridge


def route_protected_waveform(old_wave, old_hr, candidate_wave, trace, fps, eligible):
    """Return (waveform, decisions); inputs are read only and clocks are strict.

    eligible is a Boolean vector on the full original HR plan. Old rejects or
    candidate gaps demote an eligibility request and protect that whole window.
    Other evidence/ROI requirements belong to the caller's frozen eligibility
    policy. Existing V28 columns remain exact outside changed samples; V31
    diagnostics have a v31_ prefix except the explicit protected_sample flag.
    """
    times, starts, window, _, accepted, _ = _plan(trace, fps, old_hr)
    old, old_masks = _waveform(old_wave, times, 'Original V28')
    candidate, candidate_masks = _waveform(candidate_wave, times, 'Independent candidate')
    requested = _flags(eligible, 'eligible', len(starts))
    permitted = requested & accepted
    reasons = np.where(requested, 'eligible', 'not_eligible').astype(object)
    reasons[requested & ~accepted] = 'original_hr_rejected'
    for i, a in enumerate(starts):
        if permitted[i] and not np.isfinite(old[a:a+window]).all():
            raise ProtectionError(f'Accepted original window {i} contains a waveform gap', [i])
        if permitted[i] and not np.isfinite(candidate[a:a+window]).all():
            permitted[i] = False
            reasons[i] = 'candidate_window_gap'

    n = len(times)
    total, eligible_weight = np.zeros(n), np.zeros(n)
    protected = np.zeros(n, bool)
    taper = np.hanning(window+2)[1:-1]
    for i, a in enumerate(starts):
        total[a:a+window] += taper
        if permitted[i]:
            eligible_weight[a:a+window] += taper
        else:
            protected[a:a+window] = True
    # Outside every full window and original gaps are also immutable.
    protected |= (total == 0) | ~old_masks['covered']
    editable = (eligible_weight > 0) & ~protected
    core_taper = np.zeros(n)
    for a, b in runs(editable):
        core_taper[a:b] = np.hanning(b-a+2)[1:-1]
    alpha = np.divide(eligible_weight, total, out=np.zeros(n), where=total > 0)
    alpha = np.clip(alpha, 0., 1.) * core_taper
    alpha[protected] = 0.
    use_candidate = alpha > 0
    if not np.isfinite(candidate[use_candidate]).all():
        raise ProtectionError('Editable candidate support contains a gap')
    values = old.copy()
    # Equal saved samples need no floating-point arithmetic, preserving bits.
    mix = use_candidate & (old.view(np.uint64) != candidate.view(np.uint64))
    values[mix] = (1-alpha[mix])*old[mix] + alpha[mix]*candidate[mix]
    use_old = old_masks['covered'] & (alpha < 1.)
    observed = old_masks['observed'].copy()
    interpolated = old_masks['interpolated'].copy()
    observed[use_candidate] = ((~use_old | old_masks['observed']) &
                                candidate_masks['observed'])[use_candidate]
    interpolated[use_candidate] = ((use_old & old_masks['interpolated']) |
                                    candidate_masks['interpolated'])[use_candidate]
    if not np.array_equal(np.isfinite(values), old_masks['covered']):
        raise ProtectionError('Routing changed the original finite/NaN mask')
    for result, original in ((values, old), (observed, old_masks['observed']),
                             (interpolated, old_masks['interpolated'])):
        if result[protected].tobytes() != original[protected].tobytes():
            raise ProtectionError('A protected sample changed')
    modified = ((values.view(np.uint64) != old.view(np.uint64)) |
                (observed != old_masks['observed']) |
                (interpolated != old_masks['interpolated']))
    waveform = old_wave.copy(deep=True)
    waveform['base'] = values
    # Preserve original dtypes for 0/1 provenance fields.
    for field, values_out in dict(observed=observed, interpolated=interpolated).items():
        waveform[field] = values_out.astype(old_wave[field].dtype)
    waveform['protected_sample'] = protected
    waveform['v31_candidate_weight'] = alpha
    waveform['v31_modified_sample'] = modified
    source = np.full(n, 'original_v28', dtype='<U28')
    source[~old_masks['covered']] = 'missing'
    source[use_candidate & use_old] = 'v28_cdf_measured_mixture'
    source[use_candidate & ~use_old] = 'independent_cdf_waveform'
    waveform['v31_source'] = source
    # Existing source labels describe V28 only where its actual samples survive.
    if 'source' in waveform:
        waveform['source'] = waveform['source'].astype(object)
        waveform.loc[use_candidate, 'source'] = source[use_candidate]
    decisions = pd.DataFrame(dict(window_index=np.arange(len(starts)),
        time_s=(starts+window/2)/fps, window_start_s=starts/fps,
        window_end_s=(starts+window)/fps, requested_eligible=requested,
        eligible=permitted, protected_window=~permitted, reason=reasons,
        editable_samples_in_window=[int(editable[a:a+window].sum()) for a in starts],
        modified_samples_in_window=[int(modified[a:a+window].sum()) for a in starts],
        effective_changed_window=[bool(modified[a:a+window].any()) for a in starts]))
    if decisions.loc[decisions.protected_window, 'effective_changed_window'].any():
        raise ProtectionError('A protected complete window changed')
    waveform.attrs.update(reference_used=False, offline=True,
        waveform_kind='measured_convex_mixture_morphology_unvalidated',
        hr_must_be_read_from_saved_waveform=True, protection_config=PROTECTION_CONFIG)
    decisions.attrs.update(reference_used=False, protection_config=PROTECTION_CONFIG)
    return waveform, decisions


def constrained_evidence_path(emissions, grid, step_s, anchors):
    """Original DP with singleton measured-state constraints; NaN means free.

    An anchor never creates spectral support: its emission must already be
    finite. Every nonanchored row retains the original candidate emissions.
    """
    values, grid = np.asarray(emissions, float), np.asarray(grid, float)
    anchors = np.asarray(anchors, float)
    if values.ndim != 2 or values.shape[1] != len(grid) or anchors.shape != (len(values),):
        raise ValueError('Emission, grid and anchor shapes must agree')
    if np.isinf(anchors).any():
        raise ValueError('Anchors must be finite measured grid states or NaN')
    constrained = values.copy()
    for i in np.flatnonzero(np.isfinite(anchors)):
        state = np.flatnonzero(grid == anchors[i])
        if len(state) != 1 or not np.isfinite(values[i, state[0]]):
            raise ProtectionError(f'Original V28 anchor lacks actual spectral support at row {i}', [i])
        score = values[i, state[0]]
        constrained[i] = -np.inf
        constrained[i, state[0]] = score
    path, reacquired = evidence_path(constrained, grid, step_s, DEFAULT_CONFIG)
    fixed = np.isfinite(anchors)
    if not np.array_equal(grid[path[fixed]], anchors[fixed]):
        raise ProtectionError('Constrained DP changed an original protected state')
    return path, reacquired


def read_saved_protected_hr(path, trace, fps, old_hr, eligible, *, motion_provider=None):
    """Recompute HR from disk with original V28 gates and protected anchors.

    Returns explicit rejected eligible rows on failed legacy gates/candidates,
    enabling the caller to revert the affected editable core and retry. A
    protected gate/anchor failure raises ProtectionError. It never manufactures
    a supported or accepted HR. Reads use round-trip floating-point precision.
    """
    saved = pd.read_csv(path, float_precision='round_trip')
    times, starts, window, step, old_accepted, old_ridge = _plan(trace, fps, old_hr)
    values, masks = _waveform(saved, times, 'Saved final')
    eligible = _flags(eligible, 'eligible', len(starts)) & old_accepted
    protected = ~eligible
    table = legacy_estimate(values, pd.DataFrame({'rgb_valid': masks['observed']}),
                            masks['interpolated'], fps, **PARAMETERS)
    table['accepted'] = table.accepted.astype(bool)
    table['legacy_ridge_bpm'] = table.ridge_bpm.astype(float)
    table['original_v28_accepted'] = old_accepted
    table['original_v28_ridge_bpm'] = old_ridge
    table['eligible'] = eligible
    table['protected_window'] = protected
    table['protected_anchor_verified'] = False
    table['legacy_gate_passed'] = table.accepted
    table['readout_failure'] = ''
    for field in ('evidence_local_bpm', 'evidence_score', 'evidence_runner_up_margin',
                  'evidence_selected_relative_power', 'evidence_selected_peak_bpm',
                  'evidence_selected_periodic_support', 'evidence_selected_motion_overlap'):
        table[field] = np.nan
    table['evidence_candidate_count'] = 0
    table['evidence_candidates_json'] = '[]'
    table['evidence_reacquired'] = False
    table['evidence_tracker_state'] = 'unavailable'
    table['evidence_motion_available'] = False
    # Old rejected windows remain rejected regardless of any new local score.
    table.loc[~old_accepted, 'accepted'] = False
    if 'status' in old_hr:
        table.loc[~old_accepted, 'status'] = old_hr.loc[~old_accepted, 'status'].to_numpy()
    else:
        table.loc[~old_accepted, 'status'] = 'original_v28_rejected'
    bad_protected = protected & old_accepted & ~table.accepted.to_numpy(bool)
    if bad_protected.any():
        raise ProtectionError(f'Protected V28 accepted gates failed: {np.flatnonzero(bad_protected).tolist()}',
                              np.flatnonzero(bad_protected))
    failed = eligible & ~table.accepted.to_numpy(bool)
    table.loc[failed, 'readout_failure'] = table.loc[failed, 'status'].to_numpy()
    if motion_provider is None:
        from motion_evidence import motion_evidence
        motion_provider = motion_evidence
    grid = np.arange(PARAMETERS['min_bpm'], PARAMETERS['max_bpm']+.01, 1.)
    all_candidates, emissions = [], []
    for i, a in enumerate(starts):
        candidates, scores = [], np.full(len(grid), -np.inf)
        if table.accepted.iloc[i]:
            motion = motion_provider(trace, int(a), int(a+window), fps, grid)
            candidates, scores = score_candidates(values[a:a+window], fps, grid,
                                                   motion, DEFAULT_CONFIG)
            if not candidates:
                if protected[i]:
                    raise ProtectionError(f'Protected V28 window {i} has no supported candidate', [i])
                table.loc[i, ['accepted', 'status', 'readout_failure']] = [
                    False, 'no_supported_candidate', 'no_supported_candidate']
            else:
                first = candidates[0]
                table.loc[i, 'evidence_local_bpm'] = first['bpm']
                table.loc[i, 'evidence_candidate_count'] = len(candidates)
                table.loc[i, 'evidence_motion_available'] = first['motion_available']
                table.loc[i, 'evidence_candidates_json'] = json.dumps(
                    candidates, allow_nan=False, separators=(',', ':'))
                if len(candidates) > 1:
                    table.loc[i, 'evidence_runner_up_margin'] = first['evidence_score']-candidates[1]['evidence_score']
                if protected[i]:
                    state = np.flatnonzero(grid == old_ridge[i])
                    if len(state) != 1 or not np.isfinite(scores[state[0]]):
                        raise ProtectionError(f'Original V28 anchor lacks actual spectral support at window {i}', [i])
                    table.loc[i, 'protected_anchor_verified'] = True
        all_candidates.append(candidates)
        emissions.append(scores)
    table['ridge_bpm'] = np.nan
    for a, b in runs(table.accepted.to_numpy(bool)):
        anchors = np.where(protected[a:b], old_ridge[a:b], np.nan)
        path, reacquired = constrained_evidence_path(emissions[a:b], grid, step/fps, anchors)
        table.loc[a:b-1, 'ridge_bpm'] = grid[path]
        table.loc[a:b-1, 'evidence_reacquired'] = reacquired
        for i, state in enumerate(path, a):
            frequency = float(grid[state])
            # Membership tests actual energetic support, not just peak radius.
            candidate = max((item for item in all_candidates[i]
                             if frequency in item['supported_bpm']),
                            key=lambda item: item['evidence_score']-
                            DEFAULT_CONFIG.candidate_offset_penalty*(item['bpm']-frequency)**2)
            support_index = candidate['supported_bpm'].index(frequency)
            table.loc[i, 'evidence_score'] = emissions[i][state]
            table.loc[i, 'evidence_selected_relative_power'] = candidate['supported_relative_power'][support_index]
            table.loc[i, 'evidence_selected_peak_bpm'] = candidate['bpm']
            table.loc[i, 'evidence_selected_periodic_support'] = candidate['periodic_support']
            table.loc[i, 'evidence_selected_motion_overlap'] = candidate['motion_overlap']
            table.loc[i, 'evidence_tracker_state'] = ('acquired' if i == a else
                'reacquired' if bool(table.loc[i, 'evidence_reacquired']) else 'tracked')
    protected_accepted = protected & old_accepted
    if not np.array_equal(table.ridge_bpm.to_numpy()[protected_accepted], old_ridge[protected_accepted]):
        raise ProtectionError('Final readout changed a protected V28 HR state',
            np.flatnonzero(protected_accepted & (table.ridge_bpm.to_numpy() != old_ridge)))
    if table.accepted.to_numpy(bool)[~old_accepted].any():
        raise ProtectionError('Final readout filled an original rejected HR window')
    table['raw_spectral_peak_bpm'] = table.spectral_peak_bpm
    table.loc[~table.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    table['window_start_s'], table['window_end_s'] = starts/fps, (starts+window)/fps
    table['hr_source'] = 'saved_final_waveform_V28_evidence_DP_with_verified_V28_anchors'
    table.attrs.update(reference_used=False, offline=True,
        protection_config=PROTECTION_CONFIG, waveform_is_offline=True,
        hr_tracker='offline_measured_evidence_DP_with_verified_original_V28_constraints',
        coverage_policy='old_rejects_remain_rejected; eligible_failures_explicit_no_filling')
    return table
