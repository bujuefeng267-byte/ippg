"""Wang 2017 color-distortion filtering of copied RGB traces, before POS/CHROM.

This independent implementation uses a symmetric real FFT representation,
fixed 10 s complete windows and a 1 s hop within each valid segment. Positive
Hann overlap weights never turn observed RGB into missing data. Short segments
and uncovered segment tails retain their actual input RGB. No gap is bridged.
The result is an offline derived view, never a replacement frontend cache.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

ROIS = ('forehead', 'left_cheek', 'right_cheek')


@dataclass(frozen=True)
class CDFConfig:
    window_s: float = 10.
    step_s: float = 1.
    min_bpm: float = 42.
    max_bpm: float = 210.

    def __post_init__(self):
        if asdict(self) != dict(window_s=10., step_s=1., min_bpm=42., max_bpm=210.):
            raise ValueError('Only the prospectively fixed 10s/1s/42-210bpm CDF is supported')


DEFAULT_CONFIG = CDFConfig()


def spans(mask):
    change = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def cdf_window(rgb, fps, config=DEFAULT_CONFIG):
    """Return filtered Nx3 RGB, one real weight per rFFT bin, frequencies.

    S = [-1,2,-1]/sqrt(6) @ FFT(AC/DC RGB); W = |S|² / sum|RGB_FFT|².
    The same nonnegative W multiplies all three channels. irFFT supplies the
    conjugate negative-frequency half, avoiding an asymmetric filter.
    """
    x = np.asarray(rgb, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) != round(config.window_s*fps):
        raise ValueError('CDF window must contain the exact frame-rounded 10 s RGB samples')
    if not np.isfinite(fps) or fps <= config.max_bpm/30 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError('CDF requires positive finite RGB and sufficient frame rate')
    mean = x.mean(axis=0)
    normalized = x/mean-1.
    spectrum = np.fft.rfft(normalized, axis=0)
    direction = np.array([-1., 2., -1.])/np.sqrt(6.)
    projected = spectrum @ direction
    energy = np.sum(abs(spectrum)**2, axis=1)
    weights = np.divide(abs(projected)**2, energy, out=np.zeros(len(energy)), where=energy > 0)
    weights = np.clip(weights, 0., 1.)
    frequencies = np.fft.rfftfreq(len(x), d=1./fps)
    weights[(frequencies < config.min_bpm/60.) | (frequencies > config.max_bpm/60.)] = 0.
    filtered = (np.fft.irfft(spectrum*weights[:, None], n=len(x), axis=0)+1.)*mean
    return filtered, weights, frequencies


def preprocess_rgb(rgb, fps, config=DEFAULT_CONFIG):
    """Filter valid runs independently, retaining every original missing entry."""
    x = np.asarray(rgb, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(fps) or fps <= config.max_bpm/30:
        raise ValueError('An Nx3 RGB trace with sufficient FPS is required')
    if np.isinf(x).any():
        raise ValueError('RGB may contain missing NaN entries but no infinity')
    valid = np.isfinite(x).all(axis=1)
    if (x[valid] <= 0).any():
        raise ValueError('Observed RGB must be strictly positive; invalid RGB is not silently repaired')
    n, width, hop = len(x), round(config.window_s*fps), round(config.step_s*fps)
    output, numerator, denominator = x.copy(), np.zeros_like(x), np.zeros(n)
    processed_weight = np.zeros(n)
    status = np.full(n, 'missing_input', dtype='<U38')
    status[valid] = 'original_uncovered_tail'
    taper = np.hanning(width+2)[1:-1]
    assert np.all(taper > 0)
    rows = []
    for segment_id, (a, b) in enumerate(spans(valid)):
        if b-a < width:
            status[a:b] = 'original_short_segment'
            rows.append(dict(segment_id=segment_id, start_frame=int(a), stop_frame_exclusive=int(b),
                status='original_short_segment', window=False, finite_samples=int(b-a)))
            continue
        for start in range(int(a), int(b)-width+1, hop):
            stop = start+width
            filtered, weights, _ = cdf_window(x[start:stop], fps, config)
            applied = np.isfinite(filtered).all() and (filtered > 0).all()
            # Preserve physical positivity by falling back to actual RGB for the
            # whole window, never clip or synthesize individual samples.
            selected = filtered if applied else x[start:stop]
            numerator[start:stop] += taper[:, None]*selected
            denominator[start:stop] += taper
            if applied:
                processed_weight[start:stop] += taper
            rows.append(dict(segment_id=segment_id, start_frame=start, stop_frame_exclusive=stop,
                status='cdf_applied' if applied else 'original_nonpositive_or_nonfinite_window',
                window=True, finite_samples=width, minimum_weight=float(weights.min()),
                maximum_weight=float(weights.max())))
    covered = denominator > 0
    output[covered] = numerator[covered]/denominator[covered, None]
    fraction = np.divide(processed_weight, denominator, out=np.zeros(n), where=covered)
    status[covered & (fraction > 0)] = 'cdf_applied'
    status[covered & (fraction == 0)] = 'original_numeric_window_fallback'
    assert np.array_equal(np.isnan(output), np.isnan(x))
    assert np.array_equal(np.isfinite(output).all(axis=1), valid)
    assert (output[valid] > 0).all()
    return output, pd.DataFrame(dict(status=status, cdf_weight_fraction=fraction)), rows


def preprocess_trace(trace, fps, config=DEFAULT_CONFIG):
    """Copy only six RGB streams; preserve sampling, geometry and quality data.

    The tracked stream is the existing anchored RGB, not a second integration
    of pixel increments. Inherited source labels describe acquisition. New cdf
    fields explicitly describe subsequent filtering of the derived RGB view.
    """
    if len(trace) == 0:
        raise ValueError('A complete nonempty original trace is required')
    np.testing.assert_array_equal(trace.frame, np.arange(len(trace)))
    np.testing.assert_allclose(trace.time_s, np.arange(len(trace))/fps, rtol=0, atol=1e-8)
    out = trace.copy(deep=True)
    changes, windows, summaries = set(), [], {}
    for branch in ('baseline', 'tracked'):
        prefix = 'baseline_' if branch == 'baseline' else ''
        for roi in ROIS:
            columns = [f'{prefix}{roi}_{channel}' for channel in 'rgb']
            original = trace[columns].to_numpy(float)
            valid = pd.to_numeric(trace[f'{roi}_valid'], errors='raise').to_numpy(float)
            if not np.isfinite(valid).all() or not np.isin(valid, [0., 1.]).all():
                raise ValueError('Original ROI observations require Boolean masks')
            np.testing.assert_array_equal(np.isfinite(original).all(axis=1), valid.astype(bool))
            filtered, diagnostics, records = preprocess_rgb(original, fps, config)
            out[columns] = filtered
            changes.update(columns)
            key = branch+'/'+roi
            out[f'cdf_{branch}_{roi}_status'] = diagnostics.status
            out[f'cdf_{branch}_{roi}_weight_fraction'] = diagnostics.cdf_weight_fraction
            summaries[key] = dict(observed_samples=int(valid.sum()),
                status_counts=diagnostics.status.value_counts().to_dict(),
                complete_windows=sum(bool(row['window']) for row in records),
                applied_windows=sum(row['status'] == 'cdf_applied' for row in records))
            windows.extend([dict(branch=branch, roi=roi, **row) for row in records])
        # Aggregates are only compatibility fields for inherited validators;
        # actual pulse extraction still processes each physical ROI separately.
        stack = np.stack([out[[f'{prefix}{roi}_{c}' for c in 'rgb']].to_numpy(float) for roi in ROIS])
        available = np.isfinite(stack).all(axis=2)
        count = available.sum(axis=0)
        merged = np.full((len(trace), 3), np.nan)
        np.divide(np.where(available[:, :, None], stack, 0.).sum(axis=0), count[:, None],
                  out=merged, where=count[:, None] > 0)
        columns = [prefix+c for c in 'rgb']
        old_mask = np.isfinite(trace[columns]).all(axis=1).to_numpy(bool)
        if np.any(old_mask & (count == 0)):
            raise ValueError('Original aggregate has no corresponding valid ROI')
        merged[~old_mask] = np.nan
        out[columns] = merged
        changes.update(columns)
    for column in trace.columns:
        if column not in changes:
            pd.testing.assert_series_equal(out[column], trace[column], check_exact=True)
    out['cdf_view_kind'] = 'derived_cdf_after_verified_anchored_rgb_not_frontend_cache'
    out.attrs.update(trace.attrs, cdf_config=asdict(config), reference_used=False, offline=True)
    summary = dict(config=asdict(config), streams=summaries, changed_original_columns=sorted(changes),
        observation_masks_preserved=True, per_channel_nan_masks_preserved=True,
        tracking_geometry_and_quality_preserved=True, offline=True, reference_used=False,
        segment_policy='Independent contiguous fully observed RGB runs; no CDF interpolation across NaN',
        fallback_policy='Original RGB for runs shorter than 10s, uncovered <1s tails, or nonpositive/nonfinite filtered windows',
        window_policy='Frame-rounded 10s windows at frame-rounded 1s hop from each valid run start',
        overlap_policy='Normalize positive Hann sum over complete windows, without creating new missing samples',
        source='https://sstuijk.estue.nl/publications/fg17.pdf',
        adaptation='Independent implementation; preserved fixed project band 42-210bpm and segment-aware overlap/fallback')
    return out, summary, pd.DataFrame(windows)
