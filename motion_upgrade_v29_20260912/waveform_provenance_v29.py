"""Preserve contributor provenance on an explicitly parameterized window plan.

Spectral consensus proposes contributors; it is not the final HR. Sampling
markers describe the ROIs used in overlap fusion, not recovered morphology or
the full temporal support of the upstream zero-phase filters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def fused_provenance(wave, trace, roi_wave, proposals, diagnostics, fps,
                            window_s=6, step_s=1, min_bpm=42, max_bpm=210):
    """Return per-sample provenance only, without estimating HR or reading truth.

Only positive waveform weights establish contributors. Observation is ALL used
ROIs/overlapping contributions; interpolation is ANY used contribution. Samples
without a documented source are masked, never authorized by a finite value.
The production CLI checks that this mask equals its saved waveform's mask.
"""
    values = np.asarray(wave, dtype=float)
    n = len(trace)
    if values.ndim != 1 or len(values) != n or len(roi_wave) != n:
        raise ValueError('Waveform and provenance inputs must have equal length')
    trace_times = trace.time_s.to_numpy(float)
    if (not np.isfinite(trace_times).all() or not np.allclose(
            roi_wave.time_s.to_numpy(float), trace_times, rtol=0, atol=1e-8)):
        raise ValueError('ROI waveform and trace timestamps must agree')
    if not (np.isfinite(fps) and fps > 0 and np.isfinite(window_s) and
            window_s >= 6 and np.isfinite(step_s) and step_s > 0 and
            0 < min_bpm < max_bpm < fps * 30):
        raise ValueError('Invalid FPS, window, step or HR range')
    window, step = round(window_s * fps), round(step_s * fps)
    if window < round(6 * fps) or step < 1:
        raise ValueError('Use a >=6 second window and positive frame step')
    starts = np.arange(0, max(0, n-window+1), step, dtype=int)
    centers = (starts + window/2) / fps
    if len(proposals) != len(starts) or not np.allclose(
            proposals.time_s.to_numpy(float), centers, rtol=0, atol=1e-8):
        raise ValueError('Proposals must match the complete ordered window plan')
    for column, expected in [('window_start_s', starts/fps),
                              ('window_end_s', (starts+window)/fps)]:
        if not np.allclose(proposals[column].to_numpy(float), expected,
                           rtol=0, atol=1e-8):
            raise ValueError('Proposal boundaries do not match the window plan')

    covered = np.zeros(n, bool)
    observed = np.ones(n, bool)
    interpolated = np.zeros(n, bool)
    generated = np.zeros(len(starts), bool)
    used_counts = np.zeros(len(starts), int)
    weights = pd.to_numeric(diagnostics.waveform_weight, errors='coerce').to_numpy(float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Waveform contribution weights must be finite and nonnegative')
    used = diagnostics.loc[weights > 0].copy()
    indices = pd.to_numeric(used.window_index, errors='coerce').to_numpy(float)
    if (not np.isfinite(indices).all() or (indices != np.floor(indices)).any() or
            (indices < 0).any() or (indices >= len(starts)).any()):
        raise ValueError('Invalid contributor window index')
    used['window_index'] = indices.astype(int)
    for index, group in used.groupby('window_index', sort=True):
        if not np.isfinite(index) or index != int(index) or not 0 <= index < len(starts):
            raise ValueError('Invalid contributor window index')
        wi = int(index)
        rois = sorted(group.roi.unique())
        if len(rois) < 2 or not bool(proposals.iloc[wi].accepted):
            raise ValueError('Waveform contributors require an accepted multi-ROI proposal')
        if 'waveform_generated' in proposals and not bool(proposals.iloc[wi].waveform_generated):
            raise ValueError('Positive contributors contradict waveform_generated')
        if int(proposals.iloc[wi].waveform_roi_count) != len(rois):
            raise ValueError('Waveform ROI count contradicts its actual contributors')
        start, stop = starts[wi], starts[wi] + window
        generated[wi], used_counts[wi] = True, len(rois)
        covered[start:stop] = True  # fusion Hann taper is positive at both ends
        for roi in rois:
            valid = pd.to_numeric(trace[f'{roi}_valid'], errors='coerce').to_numpy(float)
            filled = pd.to_numeric(roi_wave[f'{roi}_interpolated'], errors='coerce').to_numpy(float)
            if not np.isfinite(filled).all() or not np.isin(filled, [0, 1]).all():
                raise ValueError('Contributor interpolation flags must be Boolean')
            observed[start:stop] &= np.isfinite(valid[start:stop]) & (valid[start:stop] == 1)
            interpolated[start:stop] |= filled[start:stop].astype(bool)
    observed &= covered
    interpolated &= covered
    provenance = pd.DataFrame(dict(time_s=trace.time_s.to_numpy(float),
                                    covered=covered, observed=observed,
                                    interpolated=interpolated))
    return provenance
