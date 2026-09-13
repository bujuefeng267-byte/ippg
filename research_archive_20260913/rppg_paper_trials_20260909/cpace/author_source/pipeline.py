"""
rPPG Pipeline
=============

Orchestrates the end-to-end pipeline for any method:

    roi_rgb dict -> seed estimation -> projection -> (optional homodyne)
                 -> per-ROI HR (windowed PSD, zero-padded + parabolic)
                 -> consensus HR + L-R cheek PLV

The same pipeline applies to every method. The only difference between
methods is the projection rule, encapsulated in projections.PROJECTION_METHODS.
This makes the cross-method comparisons in the paper auditable: any reviewer
can verify pipeline parity by reading run_method below.

The headline HR estimator uses 10 s windows with 2 s step. Inside each
window, the PSD is computed with 8x zero-padding and the peak is refined
by parabolic interpolation, giving sub-BPM resolution. Across windows,
the per-window HR estimates are combined by power-weighted median, which
downweights windows where motion dominates the signal power. Across ROIs,
the per-ROI HR estimates are combined by median.
"""

import numpy as np

from projections import PROJECTION_METHODS, REQUIRES_SEED
from homodyne import homodyne_correct
from utils import bandpass
from metrics import (
    estimate_hr_seed_green, windowed_hr, cross_roi_plv_lr_cheek,
)


def run_method(roi_rgb, fs, method='cPACE',
               apply_homodyne=True, hr_seed_hz=None,
               cardiac_band=(0.7, 3.0), eigen_bw=0.30,
               seed_roi='Forehead'):
    """Run a complete rPPG pipeline for a single recording.

    Args:
        roi_rgb: dict mapping ROI name to (r, g, b) tuples of 1D arrays.
                 Expected ROIs: 'Forehead', 'L_Cheek', 'R_Cheek',
                 optionally 'Nose'. cPACE requires at least 2 ROIs.
        fs: sampling rate in Hz.
        method: name from projections.PROJECTION_METHODS
                (GREEN, CHROM, POS, OMIT, LGI, PBV, ICA, cPACE).
        apply_homodyne: whether to apply homodyne envelope correction
                        after the projection step.
        hr_seed_hz: optional heart-rate seed in Hz. If None, estimated
                    from the green-channel PSD peak of the seed ROI. Can be
                    overridden when an external reference signal is available.
        cardiac_band: (lo, hi) in Hz.
        eigen_bw: half-width in Hz of the narrowband filter around the seed
                  (cPACE eigenvector step + homodyne demodulation + PLV).
        seed_roi: which ROI to use for green-PSD seed estimation.

    Returns:
        Dict with method, fs, hr_seed_hz, hr_seed_bpm, apply_homodyne,
        roi_names, hr_per_roi, hr_consensus_bpm, plv_lr_cheek,
        cardiac_signals.
        hr_per_roi values are per-ROI heart rates estimated from 10 s
        windows with 2 s step, zero-padded FFT (8x) and parabolic
        interpolation for sub-BPM resolution, with power-weighted median
        consensus across windows. hr_consensus_bpm is the median across ROIs.
    """
    if method not in PROJECTION_METHODS:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: {sorted(PROJECTION_METHODS.keys())}"
        )

    # Step 1: HR seed (default: green-channel PSD peak from forehead)
    if hr_seed_hz is None:
        hr_seed_hz = estimate_hr_seed_green(
            roi_rgb, fs, seed_roi=seed_roi, cardiac_band=cardiac_band
        )

    # Step 2: apply projection method
    proj_fn = PROJECTION_METHODS[method]
    if method in REQUIRES_SEED:
        cardiac_signals = proj_fn(roi_rgb, fs, hr_seed_hz, bw=eigen_bw)
    else:
        cardiac_signals = proj_fn(roi_rgb, fs)

    # Step 3: optional homodyne envelope correction (per ROI)
    if apply_homodyne:
        cardiac_signals = {
            name: bandpass(homodyne_correct(s, fs, hr_seed_hz, bw=eigen_bw), fs)
            for name, s in cardiac_signals.items()
        }
    # Note: cPACE-without-homodyne is the only variant where the projection
    # output isn't re-bandpassed. That matches the paper's ablation.

    # Step 4: per-ROI HR estimation via windowed PSD with zero-padding
    # (10 s windows, 2 s step, 8x zero-pad, parabolic interpolation,
    #  power-weighted median across windows -- matches paper methods)
    hr_per_roi = {
        name: windowed_hr(s, fs, win_sec=10.0, step_sec=2.0,
                          cardiac_band=cardiac_band)
        for name, s in cardiac_signals.items()
    }
    valid_hrs = [v for v in hr_per_roi.values() if v is not None and not np.isnan(v)]
    hr_consensus_bpm = float(np.median(valid_hrs)) if valid_hrs else np.nan

    # Step 5: L-R cheek PLV
    plv_lr = cross_roi_plv_lr_cheek(cardiac_signals, fs, hr_seed_hz, bw=eigen_bw)

    return {
        'method': method,
        'fs': fs,
        'hr_seed_hz': hr_seed_hz,
        'hr_seed_bpm': hr_seed_hz * 60.0,
        'apply_homodyne': apply_homodyne,
        'roi_names': list(roi_rgb.keys()),
        'hr_per_roi': hr_per_roi,
        'hr_consensus_bpm': hr_consensus_bpm,
        'plv_lr_cheek': plv_lr,
        'cardiac_signals': cardiac_signals,
    }


def run_all_methods(roi_rgb, fs, methods=None, **kwargs):
    """Run multiple methods on the same recording.

    All methods receive the same ROI inputs, the same hr_seed (estimated
    once and shared across methods), and the same HR/PLV estimation
    parameters. This is the pipeline-parity guarantee.

    Args:
        roi_rgb: dict mapping ROI name to (r, g, b) tuples.
        fs: sampling rate in Hz.
        methods: list of method names. Default:
                 GREEN, CHROM, POS, OMIT, LGI, PBV, ICA, cPACE.
        **kwargs: forwarded to run_method (apply_homodyne, eigen_bw, ...).
    """
    if methods is None:
        methods = ['GREEN', 'CHROM', 'POS', 'OMIT', 'LGI', 'PBV', 'ICA', 'cPACE']

    # Estimate seed once, shared across methods
    if kwargs.get('hr_seed_hz') is None:
        kwargs['hr_seed_hz'] = estimate_hr_seed_green(
            roi_rgb, fs,
            seed_roi=kwargs.get('seed_roi', 'Forehead'),
            cardiac_band=kwargs.get('cardiac_band', (0.7, 3.0)),
        )

    results = {}
    for method in methods:
        try:
            results[method] = run_method(roi_rgb, fs, method=method, **kwargs)
        except Exception as e:
            results[method] = {'error': str(e), 'method': method}
    return results
