"""V29 window-parameterized V28 direct-motion guarded saved-waveform router.

This bounded V28 candidate leaves the V25 waveform untouched unless a
complete declared window has an accepted old HR and a generated multi-ROI component
whose soft motion risk is substantially lower. Rules are fixed before this
revision's real-video evaluation. No reference, video identity, or desired HR
is an input. The sole decision change from V26 also requires direct motion
evidence at the old BPM to reach the SAME existing 0.50 risk threshold.
The original harmonic-risk and risk-decrease gates remain unchanged.
Harmonic motion evidence is a cue, not proof of an artifact.

Eligible windows contribute their complete positive Hann tapers. Their sum /
the sum of ALL complete planned-window tapers is the samplewise replacement
weight. The output is a convex mixture of the two saved samples, without new
filtering, phase changes, amplitude normalization, oscillators or HR filling.
Old waveform gaps are always retained. HR must be re-estimated independently
after the caller saves and reloads the returned waveform.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from motion_harmonic_evidence_v26 import motion_evidence_harmonics


MIN_OLD_MOTION_RISK = 0.50
MIN_MOTION_RISK_DECREASE = 0.30
MIN_PHYSICAL_ROIS = 2
MIN_BPM = 42.0
MAX_BPM = 210.0
PHYSICAL_ROIS = frozenset(('forehead', 'left_cheek', 'right_cheek'))


def _numeric(table, column):
    if column not in table:
        raise ValueError(f'Missing required column: {column}')
    return pd.to_numeric(table[column], errors='coerce').to_numpy(float)


def _mask(table, column):
    values = _numeric(table, column)
    if not np.isfinite(values).all() or not np.isin(values, [0., 1.]).all():
        raise ValueError(f'{column} must contain explicit Boolean / 0-or-1 flags')
    return values.astype(bool)


def _waveform(table, times, name):
    if len(table) != len(times) or not np.allclose(
            _numeric(table, 'time_s'), times, atol=1e-8, rtol=0):
        raise ValueError(f'{name} waveform must match the trace sample clock')
    values = _numeric(table, 'base')
    if np.isinf(values).any():
        raise ValueError(f'{name} waveform may contain finite values or NaN, not infinity')
    covered = _mask(table, 'covered')
    observed = _mask(table, 'observed')
    interpolated = _mask(table, 'interpolated')
    if not np.array_equal(covered, np.isfinite(values)):
        raise ValueError(f'{name} covered flags must equal its finite waveform mask')
    if (observed & ~covered).any() or (interpolated & ~covered).any():
        raise ValueError(f'{name} sampling flags cannot extend beyond covered samples')
    return values, covered, observed, interpolated


def _plan_table(table, centers, starts, window, fps, name):
    if len(table) != len(centers) or not np.allclose(
            _numeric(table, 'time_s'), centers, atol=1e-8, rtol=0):
        raise ValueError(f'{name} must match the complete ordered declared window plan')
    optional = {'window_index': np.arange(len(starts)), 'window_start_s': starts / fps,
                'window_end_s': (starts + window) / fps}
    for column, expected in optional.items():
        if column in table and not np.allclose(_numeric(table, column), expected,
                                               atol=1e-8, rtol=0):
            raise ValueError(f'{name} {column} contradicts the fixed window plan')


def _physical_roi_support(row):
    """Cross-check contributor identities; repeated methods do not add votes."""
    try:
        channels = json.loads(row['channels'])
        if not isinstance(channels, list) or not all(isinstance(x, str) for x in channels):
            return [], False
        parts = [channel.split('/') for channel in channels]
        if any(len(part) != 3 or part[1] not in PHYSICAL_ROIS for part in parts):
            return [], False
        rois = sorted({part[1] for part in parts})
        count = float(row['contributing_rois'])
        return rois, bool(np.isfinite(count) and count == len(rois) and
                          len(rois) >= MIN_PHYSICAL_ROIS)
    except (TypeError, ValueError, KeyError):
        return [], False


def route_components(old_wave, old_hr, candidate_wave, proposals, trace, fps,
                     *, window_s=6.0, step_s=1.0):
    """Return (waveform_df, decisions_df) without calculating a new HR.

    Wave tables require time_s/base/covered/observed/interpolated. old_hr requires
    time_s/accepted/ridge_bpm. Component proposals require time_s/proposal_bpm/
    proposal_supported/generated/contributing_rois/channels (JSON source keys
    branch/physical_ROI/method). All window rows must follow the complete fixed
    plan. Inputs are read only; additional reference/identity columns are ignored.

    Output also records per-sample candidate_weight and source for exact replay.
    Observation is ALL actual positive-weight sources; interpolation is ANY.
    A finite output's old source mask is never expanded or contracted.
    """
    if (not np.isfinite(window_s) or window_s < 6 or
            not np.isfinite(step_s) or step_s <= 0):
        raise ValueError('Use windows >=6 seconds and a positive step')
    if not np.isfinite(fps) or fps <= 0 or MAX_BPM >= fps * 30:
        raise ValueError('FPS must support the fixed 42-210 BPM candidate grid')
    times = _numeric(trace, 'time_s')
    n = len(times)
    if not np.isfinite(times).all() or not np.allclose(
            times, np.arange(n) / fps, atol=1e-8, rtol=0):
        raise ValueError('Trace must use the original uniform zero-origin sample clock')
    old, covered, old_observed, old_interpolated = _waveform(old_wave, times, 'Old')
    candidate, _, candidate_observed, candidate_interpolated = _waveform(
        candidate_wave, times, 'Candidate')
    window, step = round(window_s * fps), round(step_s * fps)
    if step < 1:
        raise ValueError('Step must span at least one frame')
    starts = np.arange(0, max(0, n - window + 1), step, dtype=int)
    centers = (starts + window / 2) / fps
    _plan_table(old_hr, centers, starts, window, fps, 'Old HR')
    _plan_table(proposals, centers, starts, window, fps, 'Component proposals')
    accepted = _mask(old_hr, 'accepted')
    old_bpm = _numeric(old_hr, 'ridge_bpm')
    proposed_bpm = _numeric(proposals, 'proposal_bpm')
    supported = _mask(proposals, 'proposal_supported')
    generated = _mask(proposals, 'generated')
    if 'channels' not in proposals or 'contributing_rois' not in proposals:
        raise ValueError('Actual component contributor identities are required')

    total = np.zeros(n)
    eligible_weight = np.zeros(n)
    taper = np.hanning(window + 2)[1:-1]
    grid = np.arange(MIN_BPM, MAX_BPM + .01, 1.)
    rows = []
    for wi, a in enumerate(starts):
        b = a + window
        # All complete planned windows belong in the denominator, including
        # rejected old HR / candidate windows. This provides gradual transitions.
        total[a:b] += taper
        rois, roi_supported = _physical_roi_support(proposals.iloc[wi])
        row = dict(window_index=wi, time_s=centers[wi], window_start_s=a / fps,
                   window_end_s=b / fps, old_accepted=bool(accepted[wi]),
                   old_bpm=float(old_bpm[wi]), candidate_bpm=float(proposed_bpm[wi]),
                   candidate_generated=bool(generated[wi]),
                   candidate_proposal_supported=bool(supported[wi]),
                   contributing_rois=len(rois), contributing_roi_names=json.dumps(rois),
                   old_window_finite=bool(np.isfinite(old[a:b]).all()),
                   candidate_window_finite=bool(np.isfinite(candidate[a:b]).all()),
                   motion_available=False, motion_status='not_evaluated',
                   motion_speed_face_per_s=np.nan, motion_strength=np.nan,
                   motion_reliability=np.nan, old_motion_risk=np.nan,
                   old_motion_risk_direct=np.nan, old_motion_risk_double=np.nan,
                   old_motion_risk_half=np.nan, candidate_motion_risk_direct=np.nan,
                   candidate_motion_risk_double=np.nan, candidate_motion_risk_half=np.nan,
                   candidate_motion_risk=np.nan, motion_risk_decrease=np.nan,
                   route_eligible=False, reason='old_hr_not_accepted')
        if not accepted[wi]:
            pass
        elif not np.isfinite(old_bpm[wi]):
            row['reason'] = 'old_hr_not_finite'
        elif not row['old_window_finite']:
            row['reason'] = 'old_waveform_gap'
        elif not supported[wi] or not generated[wi]:
            row['reason'] = 'candidate_not_generated_and_supported'
        elif not roi_supported:
            row['reason'] = 'candidate_requires_two_distinct_physical_rois'
        elif not row['candidate_window_finite']:
            row['reason'] = 'candidate_waveform_gap'
        elif not (MIN_BPM <= old_bpm[wi] <= MAX_BPM and
                  np.isfinite(proposed_bpm[wi]) and MIN_BPM <= proposed_bpm[wi] <= MAX_BPM):
            row['reason'] = 'frequency_outside_frozen_grid'
        else:
            motion = motion_evidence_harmonics(trace, int(a), int(b), fps, grid)
            row.update(motion_available=bool(motion['available']),
                       motion_status=motion['status'],
                       motion_speed_face_per_s=motion['speed_face_per_s'],
                       motion_strength=motion['strength'],
                       motion_reliability=motion['reliability'])
            if not motion['available']:
                row['reason'] = 'motion_evidence_unavailable'
            else:
                old_risk, new_risk = np.interp([old_bpm[wi], proposed_bpm[wi]],
                                              grid, motion['profile'])
                decrease = float(old_risk - new_risk)
                row.update(old_motion_risk=float(old_risk),
                           candidate_motion_risk=float(new_risk), motion_risk_decrease=decrease)
                for relation in ('direct', 'double', 'half'):
                    relation_profile = motion.get('profile_' + relation)
                    relation_risks = (np.interp([old_bpm[wi], proposed_bpm[wi]],
                                               grid, relation_profile)
                                      if relation_profile is not None else [np.nan, np.nan])
                    row['old_motion_risk_' + relation] = float(relation_risks[0])
                    row['candidate_motion_risk_' + relation] = float(relation_risks[1])
                if old_risk < MIN_OLD_MOTION_RISK:
                    row['reason'] = 'old_motion_risk_below_threshold'
                elif decrease < MIN_MOTION_RISK_DECREASE:
                    row['reason'] = 'insufficient_motion_risk_decrease'
                elif not (row['old_motion_risk_direct'] >= MIN_OLD_MOTION_RISK):
                    row['reason'] = 'direct_motion_evidence_insufficient'
                else:
                    row.update(route_eligible=True, reason='lower_motion_risk_component')
                    eligible_weight[a:b] += taper
        rows.append(row)

    alpha = np.divide(eligible_weight, total, out=np.zeros(n), where=total > 0)
    alpha = np.clip(alpha, 0., 1.)
    alpha[~covered] = 0.
    use_candidate = alpha > 0
    if not np.isfinite(candidate[use_candidate]).all():
        raise AssertionError('An eligible full-window contribution cannot cross a candidate gap')
    # Copy first to preserve all non-routed finite values and NaNs exactly.
    output = old.copy()
    output[use_candidate] = ((1 - alpha[use_candidate]) * old[use_candidate] +
                             alpha[use_candidate] * candidate[use_candidate])
    use_old = covered & (alpha < 1.)
    observed = ((~use_old | old_observed) &
                (~use_candidate | candidate_observed) & covered)
    interpolated = ((use_old & old_interpolated) |
                    (use_candidate & candidate_interpolated)) & covered
    if not np.array_equal(np.isfinite(output), covered):
        raise AssertionError('Routing must preserve the exact original finite/NaN mask')
    source = np.full(n, 'missing', dtype='<U23')
    source[covered] = 'old_waveform'
    source[use_candidate & use_old] = 'old_component_mixture'
    source[use_candidate & ~use_old] = 'component_waveform'
    waveform = pd.DataFrame(dict(time_s=times.copy(), base=output,
                                covered=covered.copy(), observed=observed,
                                interpolated=interpolated, candidate_weight=alpha,
                                source=source))
    waveform.attrs.update(reference_used=False, waveform_is_offline=True,
                          hr_must_be_reestimated_from_saved_waveform=True,
                          waveform_kind='measured_waveform_convex_mixture_not_morphology_validation')
    decisions = pd.DataFrame(rows)
    decisions.attrs.update(reference_used=False, min_old_motion_risk=MIN_OLD_MOTION_RISK,
                           min_old_direct_motion_risk=MIN_OLD_MOTION_RISK,
                           routing_is_subset_of_v26=True,
                           min_motion_risk_decrease=MIN_MOTION_RISK_DECREASE,
                           weight_denominator='all_complete_planned_windows')
    return waveform, decisions
