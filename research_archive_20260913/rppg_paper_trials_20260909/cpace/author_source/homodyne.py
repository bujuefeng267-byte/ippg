"""
Homodyne envelope correction.

Demodulates a 1D cardiac signal at the seed heart-rate frequency, extracts
the envelope via low-pass filter, and flattens amplitude modulation by
dividing by the envelope. Removes respiratory amplitude modulation that
would otherwise dominate downstream eigenvector selection and PSD peak
finding.

The same function is applied (optionally) to all methods after their
projection step, ensuring parity in the post-projection pipeline.
"""

import numpy as np
from scipy import signal as sig
from scipy.signal import hilbert as sp_hilbert

from utils import bandpass


def homodyne_correct(s, fs, hr_seed_hz, bw=0.30,
                     env_smooth_hz=0.30, min_amp_frac=0.05):
    """Apply homodyne envelope correction to a 1D signal.

    Args:
        s:           1D signal (numpy array)
        fs:          sampling rate in Hz
        hr_seed_hz:  heart-rate seed frequency (center of demodulation band)
        bw:          half-width of demodulation band (default 0.30 Hz)
        env_smooth_hz:   low-pass cutoff for envelope smoothing (default 0.30 Hz)
        min_amp_frac:    minimum envelope as fraction of median envelope
                         (default 0.05); prevents division by very small values
    """
    lo = max(0.5, hr_seed_hz - bw)
    hi = min(fs / 2 - 0.1, hr_seed_hz + bw)

    y_bp = bandpass(s, fs, lo, hi)
    z = sp_hilbert(y_bp)
    amp = np.abs(z)
    phase = np.angle(z)

    # Smooth the envelope (low-pass filter)
    nyq = fs / 2
    lo_norm = min(0.99, env_smooth_hz / nyq)
    if lo_norm > 0.01:
        b_f, a_f = sig.butter(2, lo_norm, btype='low')
        amp_smooth = sig.filtfilt(b_f, a_f, amp)
    else:
        amp_smooth = amp.copy()

    amp_smooth = np.maximum(amp_smooth, 1e-10)
    floor = min_amp_frac * np.median(amp_smooth)
    amp_smooth_safe = np.maximum(amp_smooth, floor)

    return (amp / amp_smooth_safe) * np.cos(phase)
