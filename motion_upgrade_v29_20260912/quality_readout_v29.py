"""Explicit sampling gates with the unchanged V25 candidate scoring and offline DP.

The 6 s boundary uses rounded frame counts, including 30.00003/180.0018 FPS.
No global monkeypatch, waveform filling, reference values or new HR candidates.
"""
from dataclasses import asdict, dataclass
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from legacy_motion import runs, ridge_path
from evidence_hr import DEFAULT_CONFIG, score_candidates, evidence_path

@dataclass(frozen=True)
class SamplingConfig:
    min_observed: float = .90
    max_interpolated: float = .10
    min_concentration: float = .12

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in asdict(self).values()):
            raise ValueError('Sampling thresholds must be finite fractions in [0,1]')

def sampling_estimate(signal, trace, interpolated, fps, window_s=10, step_s=1,
             min_bpm=42, max_bpm=210, *, sampling=SamplingConfig()):
    if (not np.isfinite(fps) or fps <= 0 or not np.isfinite(window_s) or window_s < 6 or
            not np.isfinite(step_s) or step_s <= 0 or not 0 < min_bpm < max_bpm < fps*30):
        raise ValueError('Invalid sampling window, FPS, step or BPM range')
    signal, interpolated = np.asarray(signal, float), np.asarray(interpolated, bool)
    if signal.ndim != 1 or len(trace) != len(signal) or len(interpolated) != len(signal):
        raise ValueError('Signal and sampling flags must use the same clock')
    window, step = round(window_s * fps), round(step_s * fps)
    if window < round(6 * fps) or step < 1:
        raise ValueError("Use windows >= 6 seconds and a positive step")
    rows, powers = [], []
    grid = np.arange(min_bpm, max_bpm + 0.01, 1.0)
    for a in range(0, len(signal) - window + 1, step):
        b = a + window
        segment = signal[a:b]
        valid_fraction = float(trace.rgb_valid.iloc[a:b].mean())
        filled_fraction = float(interpolated[a:b].mean())
        reason = "candidate_only"
        p = np.zeros(len(grid))
        peak, concentration = np.nan, np.nan
        if not np.isfinite(segment).all():
            reason = "gap_or_filter_edge"
        elif valid_fraction < sampling.min_observed or filled_fraction > sampling.max_interpolated:
            reason = "insufficient_observed_rgb"
        elif np.std(segment) < 1e-8:
            reason = "flat_signal"
        else:
            f, power = welch(segment, fs=fps, nperseg=window, nfft=max(2048, window), detrend="constant")
            p = np.interp(grid / 60, f, power)
            k = np.argmax(p)
            peak = grid[k]
            concentration = p[np.abs(grid - peak) <= 6].sum() / max(p.sum(), 1e-30)
            if concentration < sampling.min_concentration:
                reason = "diffuse_spectrum"
        rows.append(dict(time_s=(a + window / 2) / fps,
                         observed_fraction=valid_fraction, interpolated_fraction=filled_fraction,
                         spectral_peak_bpm=peak, peak_concentration=concentration,
                         accepted=reason == "candidate_only", status=reason, ridge_bpm=np.nan))
        powers.append(p)
    table = pd.DataFrame(rows, columns=["time_s", "observed_fraction", "interpolated_fraction",
                                      "spectral_peak_bpm", "peak_concentration", "accepted", "status", "ridge_bpm"])
    for a, b in runs(table.accepted.to_numpy(bool)):
        path = ridge_path(powers[a:b], grid, step / fps)
        table.loc[a:b - 1, "ridge_bpm"] = grid[path]
    return table


def estimate_evidence(signal, trace, interpolated, fps, window_s=10, step_s=1,
                      min_bpm=42, max_bpm=210, *, motion_trace=None,
                      motion_provider=None, config=DEFAULT_CONFIG, sampling=SamplingConfig()):
    """Legacy validity gates + evidence DP, without waveform/coverage changes."""
    values = np.asarray(signal, float)
    table = sampling_estimate(values, trace, interpolated, fps, window_s, step_s,
                            min_bpm, max_bpm, sampling=sampling)
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
