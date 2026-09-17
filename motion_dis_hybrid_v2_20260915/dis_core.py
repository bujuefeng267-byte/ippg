"""Offline Xiao DIS/PBV/POS adaptation; never reads heart-rate references.

Equations and preprocessing adapted from the MIT-licensed author get_rppg.m:
https://github.com/contactless-healthcare/Camera-based-Monitoring-for-Full-Fitness-Cycle
This is a six-local-patch Python adaptation, not the published experiment.
Actual input FPS, 42--210 bpm, bounded missing-data masks and normalized
overlap-add are explicit engineering differences. No rate-targeted filter.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt

REGIONS = tuple(f'{parent}_{half}' for parent in
                ('forehead', 'left_cheek', 'right_cheek') for half in ('a', 'b'))
METHODS = ('pbv', 'dis', 'pos')


@dataclass(frozen=True)
class Config:
    extraction_window_s: float = 5.
    extraction_hop_s: float = 5./60.
    min_bpm: float = 42.
    max_bpm: float = 210.
    max_gap_s: float = .1
    min_observed: float = .9
    selected_regions: int = 5
    pbv: tuple = (.3, .8, .5)


CONFIG = Config()


def protocol():
    return dict(**asdict(CONFIG), methods=METHODS, reference_used=False,
        offline=True, hr_window_s=10., hr_step_s=1.,
        hr_estimator='existing V28 evidence estimator, motion_trace=None for all arms',
        mask='Missing means NaN; only bounded <=0.1s internal input gaps interpolate.',
        coverage='Finite derived samples, not validated physiological waveform quality.',
        region_geometry='Six actual subregions: each original forehead/cheek box split horizontally; seeds in full-resolution source pixels, coordinates mapped to tracking image.',
        motion='Matched-patch dx/dy per source video frame, normalized by full frame height.',
        adaptation=['actual FPS; rounded 5/60s extraction hop', '42--210 bpm fixed full band',
                    'six measured patches, select best five when >=five eligible',
                    'explicit missing/observed/interpolated masks', 'normalized overlap-add',
                    'numerically stable SOS implementation of the same Butterworth filter'],
        unchanged_author_parts=['5-second window', 'PBV signature', 'RGB/motion preprocessing',
            'DIS joint pseudoinverse', 'median-FFT-bin SNR ranking', 'per-window standardization'],
        minimum_fps_not_validated=True, interpolated_video_frames=False,
        default_replacement=False)


def runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_bounded(values, limit):
    x = np.asarray(values, float).copy()
    ob = np.isfinite(x).all(axis=1)
    filled = np.zeros(len(x), bool)
    for a, b in runs(~ob):
        if a > 0 and b < len(x) and b-a <= limit:
            alpha = np.arange(1, b-a+1)[:, None]/(b-a+1)
            x[a:b] = x[a-1] + alpha*(x[b]-x[a-1])
            filled[a:b] = True
    return x, filled


def preprocess(x, fps, config=CONFIG):
    """Author row normalization and zero-phase filters, regions x time."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite region x time array required')
    denominator = x.mean(axis=1, keepdims=True)+1e-6
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError('Degenerate author mean-normalization denominator')
    norm = x/denominator-1.
    norm /= np.std(norm, axis=1, ddof=1).mean()+1e-6
    sos = butter(4, [config.min_bpm/60, config.max_bpm/60], fs=fps, btype='bandpass', output='sos')
    y = sosfiltfilt(sos, norm, axis=1)
    y -= y.mean(axis=1, keepdims=True)
    return filtfilt(np.ones(3)/3., [1.], y, axis=1)


def project(c, m, method, pbv=CONFIG.pbv):
    """Per-region author projection, returning normalized actual waveforms."""
    c, m = np.asarray(c, float), np.asarray(m, float)
    if method not in METHODS or c.ndim != 3 or c.shape[-1] != 3 or m.shape != c.shape[:2]+(2,):
        raise ValueError('Expected matching region x time x RGB/motion channels')
    out = np.full(c.shape[:2], np.nan)
    for i in range(len(c)):
        color, movement = c[i].T, m[i].T+1e-6
        if method == 'dis':
            z = np.vstack([color, movement])
            p = np.r_[pbv, 0., 0.] @ np.linalg.pinv(z@z.T) @ z
        elif method == 'pbv':
            p = np.asarray(pbv) @ np.linalg.pinv(color@color.T) @ color
        else:
            s = np.array([[0., 1., -1.], [-2., 1., 1.]]) @ color
            scale = np.std(s[1], ddof=1)
            if scale < 1e-12:
                continue
            p = s[0] + np.std(s[0], ddof=1)/scale*s[1]
        scale = np.std(p, ddof=1)
        if np.isfinite(p).all() and scale > 1e-12:
            out[i] = (p-p.mean())/scale
    return out


def select_regions(p, count=5):
    eligible = np.flatnonzero(np.isfinite(p).all(axis=1))
    if len(eligible) < count:
        return np.array([], int), np.full(p.shape[1], np.nan)
    waves = p[eligible]
    spectrum = np.abs(np.fft.fft(waves, axis=1)[:, :waves.shape[1]//2-1])
    peaks = np.argmax(spectrum, axis=1)
    center = int(np.floor(np.median(peaks)+.5))
    at_center = spectrum[:, center]
    score = at_center/np.maximum(spectrum.sum(axis=1)-at_center, 1e-30)
    local = np.argsort(-score, kind='stable')[:count]
    h = waves[local].mean(axis=0)
    scale = np.std(h, ddof=1)
    if not np.isfinite(h).all() or scale < 1e-12:
        return np.array([], int), np.full(p.shape[1], np.nan)
    return eligible[local], (h-h.mean())/scale


def extract_all(trace, fps, config=CONFIG, *, return_sources=False):
    if not np.isfinite(fps) or fps <= config.max_bpm/30:
        raise ValueError('Actual sample rate must exceed this band Nyquist limit; no accuracy guarantee')
    n = len(trace)
    np.testing.assert_allclose(trace.time_s, np.arange(n)/fps, rtol=0, atol=1e-8)
    width, hop = round(config.extraction_window_s*fps), max(1, round(config.extraction_hop_s*fps))
    values, observed, filled = [], [], []
    for region in REGIONS:
        flag = trace[region+'_valid']
        if flag.isna().any() or not flag.isin([True, False, 0, 1]).all():
            raise ValueError('Explicit Boolean regional observation flags required')
        x = trace[[region+'_'+c for c in ('r', 'g', 'b', 'dx', 'dy')]].to_numpy(float)
        ob = flag.to_numpy(bool)
        if np.isinf(x).any() or not np.array_equal(ob, np.isfinite(x).all(axis=1)):
            raise ValueError('Regional measurements and observation flag disagree')
        x[~ob] = np.nan
        y, inter = fill_bounded(x, int(np.floor(config.max_gap_s*fps+1e-9)))
        values.append(y); observed.append(ob); filled.append(inter)
    x, observed, filled = np.asarray(values), np.asarray(observed), np.asarray(filled)
    sums = {method: np.zeros(n) for method in METHODS}
    counts = {method: np.zeros(n, int) for method in METHODS}
    ob_all = {method: np.ones(n, bool) for method in METHODS}
    inter_any = {method: np.zeros(n, bool) for method in METHODS}
    diagnostics = []
    region_sums = np.zeros((len(REGIONS), n))
    region_counts = np.zeros((len(REGIONS), n), int)
    region_ob = np.ones((len(REGIONS), n), bool)
    region_inter = np.zeros((len(REGIONS), n), bool)
    for start in range(0, n-width+1, hop):
        stop = start+width
        eligible = np.flatnonzero(np.isfinite(x[:, start:stop]).all(axis=(1, 2)) &
            (observed[:, start:stop].mean(axis=1) >= config.min_observed))
        if len(eligible) < config.selected_regions:
            for method in METHODS:
                diagnostics.append(dict(start_frame=start, stop_frame=stop, method=method,
                    status='fewer_than_five_observed_regions', eligible=len(eligible), selected=''))
            continue
        segment = x[eligible, start:stop]
        try:
            color = np.stack([preprocess(segment[:, :, j], fps, config) for j in range(3)], axis=2)
            motion = np.stack([preprocess(segment[:, :, j], fps, config) for j in range(3, 5)], axis=2)
        except ValueError:
            for method in METHODS:
                diagnostics.append(dict(start_frame=start, stop_frame=stop, method=method,
                    status='normalization_failure', eligible=len(eligible), selected=''))
            continue
        for method in METHODS:
            p = project(color, motion, method, config.pbv)
            if method == 'dis':
                for j, physical in enumerate(eligible):
                    if np.isfinite(p[j]).all():
                        region_sums[physical, start:stop] += p[j]
                        region_counts[physical, start:stop] += 1
                        region_ob[physical, start:stop] &= observed[physical, start:stop]
                        region_inter[physical, start:stop] |= filled[physical, start:stop]
            selected, h = select_regions(p, config.selected_regions)
            chosen = eligible[selected]
            status = 'generated' if len(chosen) else 'degenerate_projection'
            diagnostics.append(dict(start_frame=start, stop_frame=stop, method=method,
                status=status, eligible=len(eligible), selected=';'.join(REGIONS[j] for j in chosen)))
            if len(chosen):
                sums[method][start:stop] += h
                counts[method][start:stop] += 1
                ob_all[method][start:stop] &= observed[chosen, start:stop].all(axis=0)
                inter_any[method][start:stop] |= filled[chosen, start:stop].any(axis=0)
    products = {}
    for method in METHODS:
        covered = counts[method] > 0
        y = np.full(n, np.nan)
        y[covered] = sums[method][covered]/counts[method][covered]
        products[method] = pd.DataFrame(dict(time_s=trace.time_s, base=y, covered=covered,
            observed=covered & ob_all[method], interpolated=covered & inter_any[method],
            overlap_count=counts[method]))
    if return_sources:
        sources = pd.DataFrame({'time_s':trace.time_s})
        for i, region in enumerate(REGIONS):
            cover = region_counts[i] > 0
            values = np.full(n, np.nan)
            values[cover] = region_sums[i,cover]/region_counts[i,cover]
            sources[region+'_base'] = values
            sources[region+'_covered'] = cover
            sources[region+'_observed'] = cover & region_ob[i]
            sources[region+'_interpolated'] = cover & region_inter[i]
        return products, pd.DataFrame(diagnostics), sources
    return products, pd.DataFrame(diagnostics)
