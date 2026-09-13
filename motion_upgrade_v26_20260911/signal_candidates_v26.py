"""Bounded log-chrominance candidates on observed per-ROI RGB, no reference HR.

These are independent engineering variants named logpos / logchrom, NOT exact
reproductions of POS or CHROM and not trained models. Their colour projections
are motivated by the primary papers DOI 10.1109/TBME.2016.2609282 (2017) and
10.1109/TBME.2013.2266196 (2013). Log common-mode removal, coefficient bounds,
cancellation fallback and explicit overlap normalization are this experiment's
additions. No paper's reported accuracy is transferred to this implementation.

For RGB(t)=scale(t)*colour*exp(pulse(t)*colour_vector), subtracting the per-frame
mean of log RGB cancels scale(t) exactly, including scale at the pulse frequency.
Arbitrary coloured motion, clipping, nonlinear camera processing and a purely
achromatic pulse are not identifiable under this model. A motion-frequency
notch or motion regression would not resolve that ambiguity, so neither is used.

All outputs are actual colour-derived samples. No oscillator, frequency picking,
reference, HR interpolation, spatial ROI merge or relaxation of the downstream
two-ROI requirement exists here. Short-gap filling, filtering and edge trim use
the existing project functions and limits. The pipeline is offline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analyze_rppg import bandpass
from legacy_motion import fill_short_gaps, gap_frame_limit, runs


ROI_NAMES = ('forehead', 'left_cheek', 'right_cheek')
METHODS = ('logpos', 'logchrom')
SOURCES = (
    'https://research.tue.nl/en/publications/algorithmic-principles-of-remote-ppg/',
    'https://pubmed.ncbi.nlm.nih.gov/23744659/',
)


@dataclass(frozen=True)
class SignalCandidateConfig:
    """One fixed parameter set, chosen before evaluation on human recordings."""
    extraction_window_s: float = 1.6
    hop_fraction: float = .5
    min_alpha: float = .25
    max_alpha: float = 4.
    cancellation_ratio: float = .10
    min_identifiable_rank_ratio: float = 1e-8
    edge_s: float = 1.6
    min_run_s: float = 4.
    numerical_epsilon: float = 1e-12

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError('All signal candidate settings must be positive and finite')
        if (self.extraction_window_s < 1. or self.hop_fraction > 1 or
                self.min_alpha > self.max_alpha or self.cancellation_ratio >= 1 or
                self.min_identifiable_rank_ratio >= 1 or
                self.min_run_s < 4. or self.edge_s < 1.6):
            raise ValueError('Invalid bounded extraction or inherited edge/run limits')


DEFAULT_CONFIG = SignalCandidateConfig()


def project_log_window(rgb, method, config=DEFAULT_CONFIG):
    """Project one finite positive RGB block; returns (actual samples, evidence).

Coefficients form a convex signed combination, limiting instantaneous output
    magnitude to the larger input chrominance magnitude. Severe self-cancellation
    of a rank-one colour signal falls back to an observed component. Full-rank
    colour signals may cancel a nuisance while retaining a different direction.
"""
    if method not in METHODS:
        raise ValueError(f'Unknown colour candidate: {method}')
    values = np.asarray(rgb, float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or
            not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError('A finite, positive N-by-3 RGB block is required')
    logged = np.log(values)
    common = logged.mean(axis=1, keepdims=True)
    colour = logged-common
    colour -= colour.mean(axis=0, keepdims=True)
    r, g, b = colour.T
    if method == 'logpos':
        first, second, sign = g-b, -2*r+g+b, 1.
    else:
        first, second, sign = 3*r-2*g, 1.5*r+g-1.5*b, -1.
    first, second = first-first.mean(), second-second.mean()
    first_std, second_std = float(np.std(first)), float(np.std(second))
    strongest_std = max(first_std, second_std)
    if strongest_std <= config.numerical_epsilon:
        return np.zeros(len(values)), dict(status='achromatic_or_flat_unidentifiable',
            alpha_raw=None, alpha=None, first_std=first_std, second_std=second_std,
            output_std=0., cancellation_fraction=None,
            colour_covariance_rank_ratio=0.,
            common_log_std=float(np.std(common)), chromatic_log_std=float(np.std(colour)),
            max_signed_coefficient_l1=1.)
    raw_alpha = first_std/max(second_std, config.numerical_epsilon)
    alpha = float(np.clip(raw_alpha, config.min_alpha, config.max_alpha))
    combined = (first+sign*alpha*second)/(1.+alpha)
    fraction = float(np.std(combined)/strongest_std)
    eigenvalues = np.linalg.eigvalsh(np.array([[np.mean(first*first), np.mean(first*second)],
                                             [np.mean(first*second), np.mean(second*second)]]))
    rank_ratio = float(max(0., eigenvalues[0])/max(eigenvalues[-1], config.numerical_epsilon**2))
    status = 'bounded_log_projection'
    if fraction < config.cancellation_ratio and rank_ratio < config.min_identifiable_rank_ratio:
        # A single colour direction may be pulse, coloured illumination or both.
        # Retain measured evidence instead of declaring perfect cancellation.
        combined = first.copy() if first_std >= second_std else sign*second.copy()
        status = 'colour_degeneracy_measured_component_fallback'
    return combined, dict(status=status, alpha_raw=raw_alpha, alpha=alpha,
        first_std=first_std, second_std=second_std, output_std=float(np.std(combined)),
        cancellation_fraction=fraction, common_log_std=float(np.std(common)),
        colour_covariance_rank_ratio=rank_ratio,
        chromatic_log_std=float(np.std(colour)), max_signed_coefficient_l1=1.)


def _extract_run(rgb, fps, config):
    """Independent overlapping colour transforms with positive finite support."""
    window = max(2, round(config.extraction_window_s*fps))
    hop = max(1, round(window*config.hop_fraction))
    if len(rgb) < window:
        raise ValueError('RGB run is shorter than the colour extraction window')
    starts = list(range(0, len(rgb)-window+1, hop))
    if starts[-1] != len(rgb)-window:
        starts.append(len(rgb)-window)
    taper = np.hanning(window+2)[1:-1]
    sums = {method: np.zeros(len(rgb)) for method in METHODS}
    weights = np.zeros(len(rgb))
    diagnostics = []
    for start in starts:
        stop = start+window
        for method in METHODS:
            signal, evidence = project_log_window(rgb[start:stop], method, config)
            sums[method][start:stop] += signal*taper
            diagnostics.append(dict(method=method, local_start=start, local_stop=stop, **evidence))
        weights[start:stop] += taper
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError('Every generated sample needs a positive finite overlap weight')
    return {key: value/weights for key, value in sums.items()}, diagnostics


DIAGNOSTIC_COLUMNS = ['roi', 'rgb_source', 'method', 'run_start_frame', 'run_end_frame',
    'window_start_frame', 'window_end_frame', 'window_start_s', 'window_end_s',
    'observed_fraction', 'interpolated_fraction', 'status', 'alpha_raw', 'alpha',
    'first_std', 'second_std', 'output_std', 'cancellation_fraction',
    'colour_covariance_rank_ratio', 'common_log_std', 'chromatic_log_std', 'max_signed_coefficient_l1']


def make_candidate_signals(trace, fps, gap_s=.10, min_bpm=42, max_bpm=210, *,
                           rgb_source='active', config=DEFAULT_CONFIG):
    """Return (six real-method signal arrays, ROI waveform table, diagnostics).

Use ``FusionConfig(methods=METHODS)`` with these signals. Call separately for
rgb_source='baseline' and 'active' to maintain original/tracked branch identity.
The active input is whatever RGB the caller explicitly supplies (e.g. anchored
tracking RGB); this function neither creates nor modifies tracking decisions.
"""
    if not isinstance(config, SignalCandidateConfig):
        raise TypeError('config must be SignalCandidateConfig')
    if (not np.isfinite(fps) or fps <= 0 or not np.isfinite(min_bpm) or
            not np.isfinite(max_bpm) or not 0 < min_bpm < max_bpm < fps*30):
        raise ValueError('Invalid FPS or sub-Nyquist physiological band')
    if rgb_source not in ('active', 'baseline'):
        raise ValueError('rgb_source must be active or baseline')
    limit = gap_frame_limit(gap_s, fps)
    n = len(trace)
    if 'time_s' not in trace:
        raise ValueError('The source frame time axis is required')
    times = pd.to_numeric(trace.time_s, errors='coerce').to_numpy(float)
    if not np.isfinite(times).all() or not np.allclose(times, np.arange(n)/fps, atol=1e-7, rtol=0):
        raise ValueError('Trace time axis must match complete ordered frames')
    if 'frame' in trace and not np.array_equal(trace.frame.to_numpy(), np.arange(n)):
        raise ValueError('Nonadjacent frames must not be bridged')
    edge = round(config.edge_s*fps)
    minimum_run = max(round(config.min_run_s*fps), 2*edge+1,
                      round(config.extraction_window_s*fps))
    signals = {}
    table = pd.DataFrame({'time_s': times})
    records = []
    for roi in ROI_NAMES:
        columns = [f'{"baseline_" if rgb_source == "baseline" else ""}{roi}_{c}' for c in 'rgb']
        rgb = (trace[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
               if all(column in trace for column in columns) else np.full((n, 3), np.nan))
        raw_valid = (pd.to_numeric(trace[f'{roi}_valid'], errors='coerce').to_numpy(float)
                     if f'{roi}_valid' in trace else np.full(n, np.nan))
        if np.any(np.isfinite(raw_valid) & ~np.isin(raw_valid, [0., 1.])):
            raise ValueError('ROI validity must be Boolean or missing')
        observed = (raw_valid == 1) & np.isfinite(rgb).all(axis=1) & (rgb > 0).all(axis=1)
        rgb = rgb.copy()
        rgb[~observed] = np.nan
        filled_rgb, interpolated = fill_short_gaps(rgb, limit)
        finite = np.isfinite(filled_rgb).all(axis=1) & (filled_rgb > 0).all(axis=1)
        output = {method: np.full(n, np.nan) for method in METHODS}
        raw_output = {method: np.full(n, np.nan) for method in METHODS}
        for start, stop in runs(finite):
            if stop-start < minimum_run:
                continue
            block, diagnostics = _extract_run(filled_rgb[start:stop], fps, config)
            for method, value in block.items():
                # Identical filtering, normalization and 1.6s edge exclusion as
                # the original extractor. No filter traverses a long RGB gap.
                filtered = bandpass(value, fps, min_bpm, max_bpm)
                raw_output[method][start+edge:stop-edge] = value[edge:-edge]
                output[method][start+edge:stop-edge] = filtered[edge:-edge]
            for row in diagnostics:
                a, b = start+row.pop('local_start'), start+row.pop('local_stop')
                records.append(dict(roi=roi, rgb_source=rgb_source,
                    run_start_frame=start, run_end_frame=stop, window_start_frame=a,
                    window_end_frame=b, window_start_s=a/fps, window_end_s=b/fps,
                    observed_fraction=float(observed[a:b].mean()),
                    interpolated_fraction=float(interpolated[a:b].mean()), **row))
        for method in METHODS:
            key = f'{roi}_{method}'
            signals[key] = output[method]
            table[key] = output[method]
            table[f'{key}_raw'] = raw_output[method]
        table[f'{roi}_observed'] = observed
        table[f'{roi}_interpolated'] = interpolated
    diagnostics = pd.DataFrame(records, columns=DIAGNOSTIC_COLUMNS)
    metadata = dict(config=asdict(config), methods=METHODS, rgb_source=rgb_source,
        reference_used=False, hr_estimated=False, offline=True,
        gap_limit_frames=limit, inherited_gap_policy='bounded interior only; never long gaps or edges',
        inherited_edge_samples=edge, downstream_minimum_rois=2,
        units='Raw log-chrominance and existing bandpass-normalized candidate samples',
        identity='Independent log variants; not original POS/CHROM or published model replication',
        sources=SOURCES, limitation='Cannot identify arbitrary coloured/coincident nuisance or achromatic pulse')
    table.attrs.update(metadata)
    diagnostics.attrs.update(metadata)
    return signals, table, diagnostics
