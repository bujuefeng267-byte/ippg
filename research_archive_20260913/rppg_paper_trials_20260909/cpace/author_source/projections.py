"""
rPPG projection methods.

All methods share the same signature:

    method_signal(roi_rgb, fs, **kwargs) -> dict of ROI -> 1D signal

This makes the pipeline code identical for every method: take the dict,
optionally apply homodyne (also dict-in/dict-out), then compute HR and
PLV on the resulting signals.

Methods included:

    GREEN  Raw green channel baseline (Verkruysse et al. 2008)
    CHROM  de Haan & Jeanne (2013)
    POS    Wang et al. (2017)
    OMIT   Casado & Lopez (2023), Face2PPG
    LGI    Pilz et al. (2018), Local Group Invariance
    PBV    de Haan & van Leest (2014), Blood Volume Pulse
    ICA    Poh et al. (2010), Independent Component Analysis
    cPACE  q-perpendicular projection + eigenvector selection +
           cross-ROI PLV-based v1/v2 selection (this work)
"""

import numpy as np
from scipy.signal import detrend
from scipy.linalg import eigh

from utils import bandpass


# =========================================================================
# BASELINE: GREEN
# =========================================================================

def green_signal(roi_rgb, fs):
    """GREEN baseline: raw bandpass-filtered green channel per ROI."""
    return {name: bandpass(g, fs) for name, (r, g, b) in roi_rgb.items()}


# =========================================================================
# BASELINE: CHROM (de Haan & Jeanne 2013)
# =========================================================================

def chrom_signal(roi_rgb, fs, win_sec=1.6):
    """CHROM: chrominance projection with Hanning-windowed overlap-add."""
    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        signals[name] = _chrom_one_roi(r, g, b, fs, win_sec)
    return signals


def _chrom_one_roi(r, g, b, fs, win_sec):
    n = len(r)
    win_samp = max(3, int(win_sec * fs))

    if n < win_samp:
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn, gn, bn = r / rm, g / gm, b / bm
        x1 = 3 * rn - 2 * gn
        x2 = 1.5 * rn + gn - 1.5 * bn
        alpha = np.std(x1) / (np.std(x2) + 1e-10)
        return bandpass(x1 - alpha * x2, fs)

    step = win_samp // 2
    output = np.zeros(n)
    weight = np.zeros(n)
    window = np.hanning(win_samp)

    for start in range(0, n - win_samp + 1, step):
        end = start + win_samp
        rw, gw, bw = r[start:end], g[start:end], b[start:end]
        rm, gm, bm = rw.mean() + 1e-10, gw.mean() + 1e-10, bw.mean() + 1e-10
        rn, gn, bn = rw / rm, gw / gm, bw / bm
        x1 = 3 * rn - 2 * gn
        x2 = 1.5 * rn + gn - 1.5 * bn
        alpha = np.std(x1) / (np.std(x2) + 1e-10)
        output[start:end] += (x1 - alpha * x2) * window
        weight[start:end] += window

    weight[weight < 1e-10] = 1e-10
    return bandpass(output / weight, fs)


# =========================================================================
# BASELINE: POS (Wang et al. 2017)
# =========================================================================

def pos_signal(roi_rgb, fs, win_sec=1.6):
    """POS: plane-orthogonal-to-skin chrominance projection with overlap-add."""
    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        signals[name] = _pos_one_roi(r, g, b, fs, win_sec)
    return signals


def _pos_one_roi(r, g, b, fs, win_sec):
    n = len(r)
    win_samp = max(3, int(win_sec * fs))

    if n < win_samp:
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn, gn, bn = r / rm, g / gm, b / bm
        s1 = gn - bn
        s2 = -2 * rn + gn + bn
        alpha = np.std(s1) / (np.std(s2) + 1e-10)
        return bandpass(s1 + alpha * s2, fs)

    step = win_samp // 2
    output = np.zeros(n)
    weight = np.zeros(n)
    window = np.hanning(win_samp)

    for start in range(0, n - win_samp + 1, step):
        end = start + win_samp
        rw, gw, bw = r[start:end], g[start:end], b[start:end]
        rm, gm, bm = rw.mean() + 1e-10, gw.mean() + 1e-10, bw.mean() + 1e-10
        rn, gn, bn = rw / rm, gw / gm, bw / bm
        s1 = gn - bn
        s2 = -2 * rn + gn + bn
        alpha = np.std(s1) / (np.std(s2) + 1e-10)
        output[start:end] += (s1 + alpha * s2) * window
        weight[start:end] += window

    weight[weight < 1e-10] = 1e-10
    return bandpass(output / weight, fs)


# =========================================================================
# BASELINE: OMIT / Face2PPG (Casado & Lopez 2023)
# =========================================================================

def omit_signal(roi_rgb, fs):
    """OMIT: QR-based orthogonal projection.

    Q[:, 0] captures the dominant variance direction; Q[:, 1] is the
    orthogonal residual, used as the BVP estimate per the Face2PPG paper.
    """
    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn = detrend(r / rm)
        gn = detrend(g / gm)
        bn = detrend(b / bm)
        X = np.vstack([rn, gn, bn])
        Q, _ = np.linalg.qr(X.T)
        bvp = Q[:, 1]
        bvp = bvp * (np.std(gn) / (np.std(bvp) + 1e-10))
        if np.corrcoef(bvp, gn)[0, 1] > 0:
            bvp = -bvp
        signals[name] = bvp
    return signals


# =========================================================================
# BASELINE: LGI (Pilz et al. 2018)
# =========================================================================

def lgi_signal(roi_rgb, fs):
    """LGI: SVD-based orthogonal projection.

    Project orthogonal to the dominant left singular vector u1, then take
    the green-like row of the residual. Methodologically similar to OMIT
    but uses SVD vs QR.
    """
    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn = detrend(r / rm)
        gn = detrend(g / gm)
        bn = detrend(b / bm)
        X = np.vstack([rn, gn, bn])

        U, _, _ = np.linalg.svd(X, full_matrices=False)
        u1 = U[:, 0]
        P = np.eye(3) - np.outer(u1, u1)
        Y = P @ X

        bvp = Y[1, :]
        if np.corrcoef(bvp, gn)[0, 1] > 0:
            bvp = -bvp
        signals[name] = bvp
    return signals


# =========================================================================
# BASELINE: PBV (de Haan & van Leest 2014)
# =========================================================================

_PBV_VECTOR = np.array([0.3434, 0.7095, 0.6156])
"""Canonical blood-volume-pulse signature from rPPG-Toolbox."""


def pbv_signal(roi_rgb, fs, pbv=None):
    """PBV: Wiener-style projection along the blood-volume-pulse signature.

    Solves C w = pbv where C = X X^T / T, then projects X onto
    w / (w . pbv).
    """
    if pbv is None:
        pbv = _PBV_VECTOR
    pbv = np.asarray(pbv, dtype=float)
    pbv = pbv / (np.linalg.norm(pbv) + 1e-10)

    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn = detrend(r / rm)
        gn = detrend(g / gm)
        bn = detrend(b / bm)
        X = np.vstack([rn, gn, bn])

        C = X @ X.T / X.shape[1]
        C = C + 1e-8 * np.trace(C) / 3.0 * np.eye(3)

        try:
            w = np.linalg.solve(C, pbv)
        except np.linalg.LinAlgError:
            w = pbv.copy()

        denom = float(w @ pbv)
        if abs(denom) > 1e-10:
            w = w / denom

        bvp = w @ X
        if np.corrcoef(bvp, gn)[0, 1] > 0:
            bvp = -bvp
        signals[name] = bvp
    return signals


# =========================================================================
# BASELINE: ICA (Poh et al. 2010)
# =========================================================================

def ica_signal(roi_rgb, fs, lo=0.7, hi=3.0):
    """ICA: FastICA on three RGB channels.

    Selects the source with the highest cardiac-band spectral peak (more
    robust than Poh's original "always second component" heuristic).
    Requires scikit-learn.
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        raise ImportError(
            "ICA requires scikit-learn. Install: pip install scikit-learn"
        )

    signals = {}
    for name, (r, g, b) in roi_rgb.items():
        rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
        rn = detrend(r / rm)
        gn = detrend(g / gm)
        bn = detrend(b / bm)
        X = np.vstack([rn, gn, bn]).T  # (T, 3) for sklearn

        try:
            ica = FastICA(n_components=3, random_state=0,
                          max_iter=500, tol=1e-4, whiten='unit-variance')
            S = ica.fit_transform(X)
        except (TypeError, ValueError):
            # Older sklearn versions
            try:
                ica = FastICA(n_components=3, random_state=0,
                              max_iter=500, tol=1e-4)
                S = ica.fit_transform(X)
            except Exception:
                signals[name] = gn
                continue

        n_fft = S.shape[0]
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        band = (freqs >= lo) & (freqs <= hi)
        if band.sum() < 3:
            signals[name] = S[:, 0]
            continue

        win = np.hanning(n_fft)
        best_score, best_idx = -np.inf, 0
        for i in range(S.shape[1]):
            mag = np.abs(np.fft.rfft(S[:, i] * win))
            score = float(mag[band].max())
            if score > best_score:
                best_score, best_idx = score, i

        bvp = S[:, best_idx]
        if np.corrcoef(bvp, gn)[0, 1] > 0:
            bvp = -bvp
        signals[name] = bvp
    return signals


# =========================================================================
# cPACE — this paper
# =========================================================================

def cpace_signal(roi_rgb, fs, hr_seed_hz, bw=0.30, apply_coherence_gate=False,
                 gate_alpha=2.0):
    """cPACE projection.

    Per ROI:
        1. Per-channel temporal normalization and detrend.
        2. Project to q-perpendicular subspace: P = I - q_hat q_hat^T,
           where q_hat is the unit vector along the temporal-mean RGB.
           This removes the isochromatic skin reflectance direction so the
           remaining variance is in the chromatic (absorption) plane.
        3. Narrowband filter (hr_seed +/- bw Hz) on the projected signal.
        4. Compute the 3x3 covariance and the top two eigenvectors v1, v2.

    The eigenvector selection (v1 vs v2) is done jointly across ROIs by
    picking whichever gives higher mean cross-ROI PLV. This is a defensive
    measure for windows where motion variance exceeds cardiac variance,
    in which case v1 tracks motion and v2 tracks cardiac.

    Optionally, an amplitude-only cross-ROI coherence gate can be applied
    after eigenvector selection. The gate weights samples by an instantaneous
    cross-ROI phase coherence measure; samples where the per-ROI phases
    disagree (e.g., due to a transient motion burst) are amplitude-attenuated.
    The gate is OFF by default; turn it on with apply_coherence_gate=True.

    Args:
        roi_rgb:    dict of ROI -> (r, g, b) tuples (length >= 2)
        fs:         sampling rate in Hz
        hr_seed_hz: heart-rate seed in Hz
        bw:         eigen-window half-width around seed (default 0.30 Hz)
        apply_coherence_gate: optional amplitude-only gate keyed on cross-ROI
                              phase coherence (default False)
        gate_alpha: sharpness of the coherence gate (default 2.0)

    Returns:
        Dict of ROI -> 1D signal.
    """
    if len(roi_rgb) < 2:
        raise ValueError(
            "cPACE requires at least 2 ROIs for cross-ROI PLV-based "
            "eigenvector selection."
        )

    # Stage 1: compute v1 and v2 candidate signals per ROI
    candidates = {}
    for name, (r, g, b) in roi_rgb.items():
        candidates[name] = _cpace_candidates_one_roi(r, g, b, fs, hr_seed_hz, bw)

    # Stage 2: select v1 or v2 based on cross-ROI PLV
    chosen = _select_best_eigenvector(candidates, fs, hr_seed_hz)

    # Stage 3 (optional): cross-ROI coherence gate
    if apply_coherence_gate:
        lo = max(0.5, hr_seed_hz - bw)
        hi = min(fs / 2 - 0.1, hr_seed_hz + bw)
        chosen = _crossroi_coherence_gate(chosen, fs, lo, hi, gate_alpha)

    return chosen


def _cpace_candidates_one_roi(r, g, b, fs, hr_seed_hz, bw):
    """Return both v1 and v2 candidate projected signals for one ROI."""
    n = len(r)
    rm, gm, bm = r.mean() + 1e-10, g.mean() + 1e-10, b.mean() + 1e-10
    rn = detrend(r / rm - 1.0)
    gn = detrend(g / gm - 1.0)
    bn = detrend(b / bm - 1.0)
    X_raw = np.vstack([rn, gn, bn])

    # q-perpendicular projection
    d = np.array([rm, gm, bm], dtype=float)
    d = d / (np.linalg.norm(d) + 1e-10)
    P = np.eye(3) - np.outer(d, d)
    X_chrom = P @ X_raw

    # Narrowband filter around seed
    lo = max(0.5, hr_seed_hz - bw)
    hi = min(fs / 2 - 0.1, hr_seed_hz + bw)
    X_bp = np.vstack([bandpass(X_chrom[i], fs, lo, hi) for i in range(3)])

    # 3x3 covariance, top two eigenvectors
    cov = X_bp @ X_bp.T / n
    _, evecs = eigh(cov)
    v1 = evecs[:, -1]
    v2 = evecs[:, -2]

    # Sign convention: invert if green coefficient is positive
    if v1[1] > 0:
        v1 = -v1
    if v2[1] > 0:
        v2 = -v2

    return {'v1': v1 @ X_chrom, 'v2': v2 @ X_chrom}


def _select_best_eigenvector(candidates, fs, hr_seed_hz, plv_bw=0.25):
    """Pick v1 or v2 based on which yields higher mean cross-ROI PLV.

    Defensive against windows where motion variance exceeds cardiac
    variance, in which case v1 tracks motion and v2 tracks cardiac.
    """
    # Local import to avoid circular dependency with metrics.py
    from scipy.signal import hilbert as sp_hilbert

    roi_names = list(candidates.keys())
    pairs = [(n1, n2) for i, n1 in enumerate(roi_names) for n2 in roi_names[i + 1:]]

    def mean_plv(key):
        lo = max(0.5, hr_seed_hz - plv_bw)
        hi = min(fs / 2 - 0.1, hr_seed_hz + plv_bw)
        if lo >= hi or not pairs:
            return 0.0
        plvs = []
        for n1, n2 in pairs:
            s1 = detrend(candidates[n1][key])
            s2 = detrend(candidates[n2][key])
            bp1 = bandpass(s1, fs, lo, hi)
            bp2 = bandpass(s2, fs, lo, hi)
            a1, a2 = sp_hilbert(bp1), sp_hilbert(bp2)
            amp1, amp2 = np.abs(a1), np.abs(a2)
            phase_diff = np.unwrap(np.angle(a1)) - np.unwrap(np.angle(a2))
            uv = np.exp(1j * phase_diff)
            w = np.sqrt(amp1 * amp2)
            plvs.append(float(np.abs(np.sum(w * uv) / (np.sum(w) + 1e-10))))
        return np.mean(plvs) if plvs else 0.0

    plv_v1 = mean_plv('v1')
    plv_v2 = mean_plv('v2')
    chosen_key = 'v1' if plv_v1 >= plv_v2 else 'v2'
    return {name: candidates[name][chosen_key] for name in roi_names}


def _crossroi_coherence_gate(signals_dict, fs, lo, hi, gate_alpha=2.0):
    """Amplitude-only gate driven by instantaneous cross-ROI phase coherence.

    For each sample, computes the instantaneous phase coherence across all
    ROIs (Kuramoto order parameter R_t = |mean(exp(i*phi_roi))|). Samples
    where the per-ROI phases agree (R_t near 1) are passed through; samples
    where they disagree (R_t near 0) are amplitude-attenuated. The gate is
    smoothed over a 0.3 s kernel to avoid sample-level chatter.

    Args:
        signals_dict: dict of ROI -> 1D signal
        fs:           sampling rate
        lo, hi:       cardiac-band cutoffs in Hz for Hilbert phase extraction
        gate_alpha:   sharpness of the gate (default 2.0).
                      Larger alpha = sharper gate.

    Returns:
        Dict of ROI -> gated signal.
    """
    from scipy.signal import hilbert as sp_hilbert

    roi_names = list(signals_dict.keys())
    n = min(len(signals_dict[name]) for name in roi_names)

    if len(roi_names) < 2:
        return {name: signals_dict[name][:n] for name in roi_names}

    # Instantaneous phase per ROI on bandpass-filtered signal
    analytic = {}
    for name in roi_names:
        s_bp = bandpass(signals_dict[name][:n], fs, lo, hi)
        analytic[name] = sp_hilbert(s_bp)

    # Cross-ROI phase coherence (Kuramoto order parameter per sample)
    phases = np.array([np.angle(analytic[name]) for name in roi_names])
    R_t = np.abs(np.mean(np.exp(1j * phases), axis=0))

    # Smooth over 0.3 s
    smooth_win = max(3, int(0.3 * fs))
    kernel = np.ones(smooth_win) / smooth_win
    R_smooth = np.convolve(R_t, kernel, mode='same')

    # Soft gate via logistic-like sharpening
    eps = 1e-8
    R_c = np.clip(R_smooth, eps, 1 - eps)
    w = R_c ** gate_alpha / (R_c ** gate_alpha + (1 - R_c) ** gate_alpha)

    # Apply gate as an amplitude scaling to the original (non-bandpassed) signal
    out = {}
    for name in roi_names:
        s = signals_dict[name][:n]
        a = sp_hilbert(bandpass(s, fs, lo, hi))
        amp = np.abs(a)
        amp_gated = amp * w
        scale = np.ones(n)
        mask = amp > 1e-10
        scale[mask] = amp_gated[mask] / amp[mask]
        out[name] = s * scale

    return out


# =========================================================================
# METHOD REGISTRY
# =========================================================================

PROJECTION_METHODS = {
    'GREEN': green_signal,
    'CHROM': chrom_signal,
    'POS':   pos_signal,
    'OMIT':  omit_signal,
    'LGI':   lgi_signal,
    'PBV':   pbv_signal,
    'ICA':   ica_signal,
    'cPACE': cpace_signal,
}
"""All projection methods. cPACE requires hr_seed_hz; others do not."""

REQUIRES_SEED = {'cPACE'}
"""Methods that require an hr_seed_hz argument."""
