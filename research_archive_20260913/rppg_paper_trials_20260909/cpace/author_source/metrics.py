"""
Heart-rate seed estimation and downstream metrics (HR, PLV).

Heart-rate estimation matches the paper's headline pipeline:
    - Per ROI: 10 s windows with 2 s step.
    - Within each window: zero-padded PSD (8x pad) for sub-bin frequency
      resolution, with parabolic interpolation around the peak for
      continuous-valued peak picking.
    - Across windows: power-weighted median (robust to outlier windows).
    - Across ROIs: median.

PLV is computed on the L_Cheek - R_Cheek pair per the paper convention,
amplitude-weighted, narrowband around the seed HR.
"""

import numpy as np
from scipy import signal as sig
from scipy.signal import hilbert as sp_hilbert
from scipy.signal import detrend

from utils import bandpass


# =========================================================================
# HR SEED ESTIMATION
# =========================================================================

def estimate_hr_seed_green(roi_rgb, fs, seed_roi='Forehead',
                           cardiac_band=(0.7, 3.0)):
    """Estimate heart-rate seed (Hz) from green channel PSD peak.

    Deployment-realistic seed. Uses zero-padded FFT for sub-bin resolution
    on the bandpass-filtered green channel of the seed ROI (forehead by
    default).
    """
    if seed_roi not in roi_rgb:
        seed_roi = list(roi_rgb.keys())[0]
    g = roi_rgb[seed_roi][1]
    g_filt = bandpass(g, fs, lo=cardiac_band[0], hi=cardiac_band[1])
    hr_bpm, _ = _psd_hr_zeropad(g_filt, fs, cardiac_band[0], cardiac_band[1])
    if np.isnan(hr_bpm):
        return 1.2  # fallback ~72 BPM
    return float(hr_bpm / 60.0)


# =========================================================================
# PSD PEAK WITH ZERO-PADDING AND PARABOLIC INTERPOLATION
# =========================================================================

def _psd_hr_zeropad(signal, fs, lo=0.7, hi=3.0, pad_factor=8):
    """PSD peak with zero-padding and parabolic interpolation for sub-bin
    frequency resolution. Returns (hr_bpm, peak_power)."""
    n = len(signal)
    if n < int(2 * fs):
        return np.nan, 0.0
    n_fft = n * pad_factor
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mag = np.abs(np.fft.rfft(signal * np.hanning(n), n=n_fft))
    mag_sq = mag ** 2
    cardiac = (freqs >= lo) & (freqs <= hi)
    if cardiac.sum() < 3:
        return np.nan, 0.0
    cf = freqs[cardiac]
    cm = mag_sq[cardiac]
    peak_idx = int(np.argmax(cm))

    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_idx < len(cm) - 1:
        alpha = np.log(cm[peak_idx - 1] + 1e-20)
        beta  = np.log(cm[peak_idx]     + 1e-20)
        gamma = np.log(cm[peak_idx + 1] + 1e-20)
        denom = (alpha - 2 * beta + gamma)
        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-20 else 0.0
        freq_res = cf[1] - cf[0]
        peak_freq = cf[peak_idx] + delta * freq_res
    else:
        peak_freq = cf[peak_idx]

    return float(peak_freq * 60.0), float(cm[peak_idx])


def _weighted_median(values, weights):
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_w = weights[sorted_idx]
    cumw = np.cumsum(sorted_w)
    cutoff = cumw[-1] * 0.5
    idx = np.searchsorted(cumw, cutoff)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


# =========================================================================
# WINDOWED HR ESTIMATION (HEADLINE)
# =========================================================================

def windowed_hr(signal, fs, win_sec=10.0, step_sec=2.0,
                cardiac_band=(0.7, 3.0), pad_factor=8):
    """Per-window HR with power-weighted median consensus across windows.

    Inside each window, the PSD peak is estimated via zero-padded FFT and
    parabolic interpolation for sub-BPM resolution. Power-weighting across
    windows downweights outlier windows where motion dominates.

    This is the headline HR estimator matching the paper's methods section.

    Args:
        signal:        1D cardiac signal
        fs:            sampling rate in Hz
        win_sec:       window length in seconds (default 10)
        step_sec:      window hop in seconds (default 2)
        cardiac_band:  (lo, hi) in Hz
        pad_factor:    zero-pad multiplier for sub-bin resolution (default 8)

    Returns:
        Heart rate in BPM (float). Returns np.nan if no valid windows.
    """
    n = len(signal)
    win_samp = int(win_sec * fs)
    step_samp = int(step_sec * fs)
    lo, hi = cardiac_band

    if n < win_samp:
        hr, _ = _psd_hr_zeropad(signal, fs, lo, hi, pad_factor)
        return hr

    hrs, weights = [], []
    for start in range(0, n - win_samp + 1, step_samp):
        end = start + win_samp
        chunk = detrend(signal[start:end])
        hr, power = _psd_hr_zeropad(chunk, fs, lo, hi, pad_factor)
        if not np.isnan(hr) and lo * 60 <= hr <= hi * 60:
            hrs.append(hr)
            weights.append(power)

    if not hrs:
        hr, _ = _psd_hr_zeropad(signal, fs, lo, hi, pad_factor)
        return hr

    hrs_arr = np.array(hrs)
    w_arr = np.array(weights)
    if w_arr.sum() > 0:
        return _weighted_median(hrs_arr, w_arr)
    return float(np.median(hrs_arr))


# =========================================================================
# CROSS-ROI PLV (L_Cheek-R_Cheek pair)
# =========================================================================

def cross_roi_plv_lr_cheek(signals, fs, hr_seed_hz, bw=0.30):
    """Amplitude-weighted PLV between L_Cheek and R_Cheek.

    Per the paper, PLV is reported on the symmetric L_Cheek-R_Cheek pair
    to avoid confounds from different ROI counts across datasets.
    """
    if 'L_Cheek' not in signals or 'R_Cheek' not in signals:
        return None

    lo = max(0.5, hr_seed_hz - bw)
    hi = min(fs / 2 - 0.1, hr_seed_hz + bw)
    if lo >= hi:
        return None

    s1 = detrend(signals['L_Cheek'])
    s2 = detrend(signals['R_Cheek'])
    bp1 = bandpass(s1, fs, lo, hi)
    bp2 = bandpass(s2, fs, lo, hi)
    a1, a2 = sp_hilbert(bp1), sp_hilbert(bp2)
    amp1, amp2 = np.abs(a1), np.abs(a2)
    phase_diff = np.unwrap(np.angle(a1)) - np.unwrap(np.angle(a2))
    uv = np.exp(1j * phase_diff)
    w = np.sqrt(amp1 * amp2)
    return float(np.abs(np.sum(w * uv) / (np.sum(w) + 1e-10)))
