"""Reference-free offline HR competition on the *saved* fused waveform.

This is a transparent classical estimator, not a reproduced pretrained model.
All reported frequencies have local spectral energy in the actual waveform.
Harmonics are supporting evidence for an existing candidate, never an arithmetic
replacement for a peak. Motion is soft evidence, not proof that a peak is false.
The inherited sampling/flatness/concentration gates remain unchanged. Missing
windows reset the path and never receive held or interpolated HR.

Defaults below are fixed before human-reference evaluation. Tests use only
controlled numerical waveforms. Score/margin fields are not probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

from legacy_motion import estimate as legacy_estimate, runs


@dataclass(frozen=True)
class EvidenceConfig:
    min_relative_power: float = .05
    min_prominence: float = .02
    peak_distance_bpm: float = 6.
    candidate_radius_bpm: float = 3.
    max_candidates: int = 8
    spectral_weight: float = .35
    periodic_weight: float = 1.10
    harmonic_weight: float = 1.00
    motion_weight: float = 1.25
    third_harmonic_weight: float = .35
    transition_penalty: float = .05
    max_rate_bpm_per_s: float = 12.
    reacquisition_cost: float = 3.
    candidate_offset_penalty: float = .025

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.min_relative_power >= 1 or self.min_prominence >= 1:
            raise ValueError('Relative power and prominence must be below one')
        if not isinstance(self.max_candidates, int) or self.max_candidates < 2:
            raise ValueError('At least two integer candidates are required')


DEFAULT_CONFIG = EvidenceConfig()


def _period_correlation(segment, fps, bpm):
    """Fractional-period correlation; interpolation is diagnostic only."""
    lag = 60. * fps / bpm
    index = np.arange(int(np.ceil(lag)), len(segment), dtype=float)
    if len(index) < 4:
        return 0.
    left = segment[index.astype(int)]
    right = np.interp(index-lag, np.arange(len(segment)), segment)
    left, right = left-left.mean(), right-right.mean()
    denominator = np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.clip(np.dot(left, right)/denominator, -1, 1)) if denominator > 1e-15 else 0.


def _motion_values(evidence, grid):
    if not evidence or not bool(evidence.get('available', False)):
        return np.zeros(len(grid)), None, 0., False
    profile = np.asarray(evidence['profile'], dtype=float)
    if profile.shape != grid.shape or not np.isfinite(profile).all():
        raise ValueError('Motion evidence must match the finite BPM grid')
    strength = float(evidence.get('strength', 0.))
    reliability = float(evidence.get('reliability', 0.))
    if (np.any((profile < 0) | (profile > 1)) or
            not np.isfinite([strength, reliability]).all() or
            not 0 <= strength <= 1 or not 0 <= reliability <= 1):
        raise ValueError('Motion profile, strength and reliability must be in [0, 1]')
    return profile, strength, reliability, True


def score_candidates(segment, fps, grid, motion=None, config=DEFAULT_CONFIG):
    """Return candidate diagnostics and supported-grid emissions.

No candidate is manufactured at half or double a peak. Period and harmonic
evidence are evaluated only after local peaks pass the energy threshold.
"""
    x, grid = np.asarray(segment, float), np.asarray(grid, float)
    if (x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all() or
            grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or
            not np.allclose(np.diff(grid), 1., atol=1e-10, rtol=0) or
            not np.isfinite(fps) or fps <= 0 or grid[0] <= 0 or grid[-1] >= fps*30):
        raise ValueError('Finite waveform, positive FPS and a one-BPM grid required')
    frequency, full_power = welch(x, fs=fps, nperseg=len(x),
        nfft=max(2048, len(x)), detrend='constant')
    power = np.interp(grid/60, frequency, full_power)
    maximum = float(np.max(power))
    emissions = np.full(len(grid), -np.inf)
    if maximum <= 0 or np.std(x) < 1e-8:
        return [], emissions
    relative = power/maximum
    peaks, properties = find_peaks(np.r_[0., relative, 0.],
        height=config.min_relative_power, prominence=config.min_prominence,
        distance=max(1, round(config.peak_distance_bpm)))
    ranked = sorted([(int(k)-1, float(prom)) for k, prom in
        zip(peaks, properties['prominences']) if 0 <= int(k)-1 < len(grid)],
        key=lambda item: (-relative[item[0]], grid[item[0]]))[:config.max_candidates]
    profile, strength, reliability, available = _motion_values(motion, grid)
    candidates = []
    for k, prominence in ranked:
        bpm = float(grid[k])
        full_period = _period_correlation(x, fps, bpm)
        half = len(x)//2
        split_period = .5*(_period_correlation(x[:half], fps, bpm)+
                           _period_correlation(x[half:], fps, bpm))
        periodic = max(0., .75*full_period+.25*split_period)
        # Exact h*f PSD support, not a search for arbitrary neighbouring peaks.
        harmonic2 = float(np.interp(2*bpm/60, frequency, full_power, right=0.)/maximum)
        harmonic3 = float(np.interp(3*bpm/60, frequency, full_power, right=0.)/maximum)
        harmonic = float(np.clip(np.sqrt(relative[k]*max(0., harmonic2)) +
            config.third_harmonic_weight*np.sqrt(relative[k]*max(0., harmonic3)), 0., 1.))
        # Provider profile already incorporates strength and reliability.
        overlap = float(profile[k])
        score = (config.spectral_weight*np.log(max(relative[k], 1e-30)) +
                 config.periodic_weight*periodic + config.harmonic_weight*harmonic -
                 config.motion_weight*overlap)
        support = (np.abs(grid-bpm) <= config.candidate_radius_bpm) & (
            relative >= config.min_relative_power)
        emissions[support] = np.maximum(emissions[support], score -
            config.candidate_offset_penalty*(grid[support]-bpm)**2)
        candidates.append(dict(bpm=bpm, relative_power=float(relative[k]),
            peak_concentration=float(power[np.abs(grid-bpm) <= 6].sum()/power.sum()),
            prominence=prominence, period_correlation=full_period,
            split_period_correlation=split_period, periodic_support=periodic,
            harmonic2_relative_power=harmonic2, harmonic3_relative_power=harmonic3,
            harmonic_support=harmonic, motion_profile=float(profile[k]),
            motion_strength=strength, motion_reliability=reliability,
            motion_overlap=overlap, motion_available=available, evidence_score=float(score),
            supported_bpm=grid[support].tolist(),
            supported_relative_power=relative[support].tolist()))
    candidates.sort(key=lambda row: (-row['evidence_score'], row['bpm']))
    return candidates, emissions


def evidence_path(emissions, grid, step_s, config=DEFAULT_CONFIG):
    """DP with normal bounded transitions plus explicitly costed reacquisition.

The jump route costs 3 evidence-score units per event and still requires an
energetically supported destination. It is not an HR interpolation route.
"""
    values, grid = np.asarray(emissions, float), np.asarray(grid, float)
    if values.ndim != 2 or values.shape[1] != len(grid):
        raise ValueError('Emission shape and grid must agree')
    if len(values) == 0:
        return np.array([], int), np.array([], bool)
    if (np.isnan(values).any() or np.isposinf(values).any() or
            not np.isfinite(values).any(axis=1).all() or not np.isfinite(step_s) or step_s <= 0):
        raise ValueError('Each path row requires supported evidence and positive step')
    delta = grid[:, None]-grid[None, :]
    normal = np.abs(delta) <= config.max_rate_bpm_per_s*step_s + 1e-9
    normal_score = np.where(normal, -config.transition_penalty*(delta/step_s)**2, -np.inf)
    # Reacquisition is a separate route at every distance, avoiding a cheaper
    # 13-bpm jump than a 12-bpm ordinary transition at an arbitrary boundary.
    transition = np.maximum(normal_score, -config.reacquisition_cost)
    score = values[0].copy()
    back = np.zeros(values.shape, int)
    for i in range(1, len(values)):
        alternatives = score[:, None]+transition
        back[i] = np.argmax(alternatives, axis=0)
        score = alternatives[back[i], np.arange(len(grid))]+values[i]
    path = np.zeros(len(values), int)
    path[-1] = int(np.argmax(score))
    for i in range(len(path)-1, 0, -1):
        path[i-1] = back[i, path[i]]
    reacquired = np.r_[False, normal_score[path[:-1], path[1:]] < -config.reacquisition_cost]
    return path, reacquired


def estimate_evidence(signal, trace, interpolated, fps, window_s=10, step_s=1,
                      min_bpm=42, max_bpm=210, *, motion_trace=None,
                      motion_provider=None, config=DEFAULT_CONFIG):
    """Legacy validity gates + evidence DP, without waveform/coverage changes."""
    values = np.asarray(signal, float)
    table = legacy_estimate(values, trace, interpolated, fps, window_s, step_s,
                            min_bpm, max_bpm)
    table['accepted'] = table.accepted.astype(bool)
    table['legacy_ridge_bpm'] = table.ridge_bpm.astype(float)
    table['evidence_local_bpm'] = np.nan
    table['evidence_score'] = np.nan
    table['evidence_runner_up_margin'] = np.nan
    table['evidence_selected_relative_power'] = np.nan
    table['evidence_selected_peak_bpm'] = np.nan
    table['evidence_selected_periodic_support'] = np.nan
    table['evidence_selected_motion_overlap'] = np.nan
    table['evidence_candidate_count'] = 0
    table['evidence_candidates_json'] = '[]'
    table['evidence_reacquired'] = False
    table['evidence_tracker_state'] = 'unavailable'
    table['evidence_motion_available'] = False
    grid = np.arange(min_bpm, max_bpm+.01, 1.)
    window, step = round(window_s*fps), round(step_s*fps)
    emissions, all_candidates = [], []
    if motion_trace is not None and len(motion_trace) != len(values):
        raise ValueError('Motion trace must match saved waveform length')
    if motion_trace is not None and motion_provider is None:
        from motion_evidence import motion_evidence
        motion_provider = motion_evidence
    for index, accepted in enumerate(table.accepted):
        if not accepted:
            emissions.append(np.full(len(grid), -np.inf))
            all_candidates.append([])
            continue
        start, stop = index*step, index*step+window
        motion = (motion_provider(motion_trace, start, stop, fps, grid)
                  if motion_trace is not None else None)
        candidates, score = score_candidates(values[start:stop], fps, grid, motion, config)
        # The accepted global peak always meets these looser candidate gates.
        # A failure is explicit, never silently interpolated or hidden.
        if not candidates:
            table.loc[index, ['accepted', 'status']] = [False, 'no_supported_candidate']
        else:
            first = candidates[0]
            table.loc[index, 'evidence_local_bpm'] = first['bpm']
            table.loc[index, 'evidence_candidate_count'] = len(candidates)
            table.loc[index, 'evidence_motion_available'] = first['motion_available']
            table.loc[index, 'evidence_candidates_json'] = json.dumps(candidates,
                ensure_ascii=True, allow_nan=False, separators=(',', ':'))
            if len(candidates) > 1:
                table.loc[index, 'evidence_runner_up_margin'] = (
                    first['evidence_score']-candidates[1]['evidence_score'])
        emissions.append(score)
        all_candidates.append(candidates)
    table['ridge_bpm'] = np.nan
    for start, stop in runs(table.accepted.to_numpy(bool)):
        path, reacquired = evidence_path(emissions[start:stop], grid, step/fps, config)
        table.loc[start:stop-1, 'ridge_bpm'] = grid[path]
        table.loc[start:stop-1, 'evidence_reacquired'] = reacquired
        for i, state in enumerate(path, start):
            candidate = max((row for row in all_candidates[i]
                if abs(row['bpm']-grid[state]) <= config.candidate_radius_bpm),
                key=lambda row: row['evidence_score'] -
                    config.candidate_offset_penalty*(row['bpm']-grid[state])**2)
            table.loc[i, 'evidence_score'] = emissions[i][state]
            supported = candidate['supported_bpm'].index(float(grid[state]))
            table.loc[i, 'evidence_selected_relative_power'] = candidate['supported_relative_power'][supported]
            table.loc[i, 'evidence_selected_peak_bpm'] = candidate['bpm']
            table.loc[i, 'evidence_selected_periodic_support'] = candidate['periodic_support']
            table.loc[i, 'evidence_selected_motion_overlap'] = candidate['motion_overlap']
            table.loc[i, 'evidence_tracker_state'] = ('acquired' if i == start else
                'reacquired' if bool(table.loc[i, 'evidence_reacquired']) else 'tracked')
    table.loc[~table.accepted, 'ridge_bpm'] = np.nan
    table.attrs.update(hr_tracker='offline_evidence_DP_on_actual_fused_waveform',
        evidence_config=asdict(config), reference_used=False, waveform_is_offline=True,
        gates='unchanged legacy sampling, flatness and concentration gates',
        coverage_policy='no HR interpolation, no extra confidence rejection')
    return table
