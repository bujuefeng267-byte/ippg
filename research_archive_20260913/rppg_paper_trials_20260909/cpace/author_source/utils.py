"""
Shared signal-processing utilities used by all rPPG methods.
"""

import numpy as np
from scipy import signal as sig


def bandpass(x, fs, lo=0.7, hi=3.0, order=4):
    """Zero-phase Butterworth bandpass filter.

    Default cardiac band 0.7-3.0 Hz (42-180 BPM).

    Args:
        x:  1D signal (numpy array)
        fs: sampling rate in Hz
        lo: low cutoff in Hz (default 0.7)
        hi: high cutoff in Hz (default 3.0)
        order: filter order (default 4)
    """
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 0.001)
    hi_n = min(hi / nyq, 0.999)
    b, a = sig.butter(order, [lo_n, hi_n], btype='band')
    return sig.filtfilt(b, a, x)
