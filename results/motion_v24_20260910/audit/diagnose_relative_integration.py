"""Bounded mechanism audit: V23 saved traces and controlled colour arrays only.

No user video rerun, no reference labels, and no production files are modified.
Synthetic bias scenarios are counterexamples, not diagnoses of a real recording.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfilt, sosfreqz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V23 = ROOT/'rppg_motion_v23'
sys.path.insert(0, str(V23))
from pixel_tracking import paired_log_change, sample_patches, PixelConfig
from legacy_motion import make_waveforms

CASES = ('user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1')
ROIS = ('forehead', 'left_cheek', 'right_cheek')
MODES = ('tracking_only', 'tracking_screened')


def segments(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def saved_trace_audit():
    rows = []
    for mode in MODES:
        for gap in ('gap10', 'gap15'):
            for case in CASES:
                directory = V23/'validation'/f'{mode}_{gap}'/case
                trace = pd.read_csv(directory/'frame_trace.csv')
                summary = json.loads((directory/'summary.json').read_text())
                fps = summary['fps']
                for roi in ROIS:
                    current = trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy(float)
                    base = trace[[f'baseline_{roi}_{c}' for c in 'rgb']].to_numpy(float)
                    valid = np.isfinite(current).all(1) & np.isfinite(base).all(1) & (current > 0).all(1) & (base > 0).all(1)
                    difference = np.full(current.shape, np.nan)
                    difference[valid] = np.log(current[valid]/base[valid])
                    runs = segments(valid)
                    if not runs:
                        continue
                    start, stop = max(runs, key=lambda ab: ab[1]-ab[0])
                    part = difference[start:stop]
                    pairchange = np.diff(part, axis=0)
                    low_fraction = []
                    for c in range(3):
                        if len(part) >= 60:
                            f, power = welch(part[:, c], fs=fps, nperseg=min(len(part), round(10*fps)))
                            denominator = power[(f > 0) & (f <= 3.5)].sum()
                            low_fraction.append(float(power[(f > 0) & (f < .7)].sum()/denominator) if denominator > 1e-24 else None)
                        else:
                            low_fraction.append(None)
                    row = dict(mode=mode, gap=gap, case=case, roi=roi, frames=len(trace), valid_frames=int(valid.sum()),
                               longest_run_frames=stop-start, longest_run_s=(stop-start)/fps,
                               tracked_frames=int((trace[f'{roi}_pixel_source'] == 'tracked_ratio').sum()),
                               fallback_frames=int((trace[f'{roi}_pixel_source'] == 'baseline_ratio_fallback').sum()),
                               reset_frames=int(trace[f'{roi}_pixel_reset'].sum()),
                               screened_frames=int((trace[f'{roi}_pixel_screen_rejected'] > 0).sum()),
                               screened_points=int(trace[f'{roi}_pixel_screen_rejected'].sum()))
                    for i, c in enumerate('rgb'):
                        row[f'{c}_relative_gain_min'] = float(np.exp(np.nanmin(difference[:, i])))
                        row[f'{c}_relative_gain_max'] = float(np.exp(np.nanmax(difference[:, i])))
                        row[f'{c}_log_offset_p95_minus_p05'] = float(np.nanpercentile(difference[:, i], 95)-np.nanpercentile(difference[:, i], 5))
                        row[f'{c}_longest_run_end_minus_start'] = float(part[-1, i]-part[0, i])
                        row[f'{c}_mean_pair_difference_vs_baseline'] = float(pairchange[:, i].mean()) if len(pairchange) else None
                        row[f'{c}_offset_power_below_0p7Hz_fraction'] = low_fraction[i]
                    rows.append(row)
    return pd.DataFrame(rows)


def reconstruct(deltas, log_baseline, anchor_hz=None, fps=30.):
    out = np.empty_like(log_baseline)
    out[0] = log_baseline[0]
    alpha = 0. if anchor_hz is None else 1-np.exp(-2*np.pi*anchor_hz/fps)
    for t in range(1, len(out)):
        prediction = out[t-1]+deltas[t]
        out[t] = (1-alpha)*prediction+alpha*log_baseline[t]
    return out


def reconstruct_lp4(deltas, log_baseline, cutoff=.15, fps=30.):
    integral = log_baseline[0]+np.cumsum(deltas, axis=0)
    sections = butter(4, cutoff, btype='lowpass', fs=fps, output='sos')
    # Initial difference is exactly zero by construction, so zero SOS state
    # represents a correctly initialized segment. This is causal, not filtfilt.
    correction = sosfilt(sections, log_baseline-integral, axis=0)
    return integral+correction


def pulse_readout(rgb, known_pulse, fps):
    trace = pd.DataFrame(rgb, columns=list('rgb'))
    trace['rgb_valid'] = True
    trace['motion_x'] = 0.
    trace['motion_y'] = 0.
    answer = {}
    for method in ('pos', 'chrom'):
        wave, _, _ = make_waveforms(trace, fps, method, .1, 42, 210)
        valid = np.isfinite(wave)
        signal = wave[valid]
        f, power = welch(signal, fs=fps, nperseg=min(len(signal), round(20*fps)), nfft=8192)
        band = (f >= .7) & (f <= 3.5)
        peak = f[band][np.argmax(power[band])]*60
        # Shape correlation is against this known synthetic injected waveform,
        # never against human data. Sign ambiguity is retained in the number.
        answer[method] = dict(peak_bpm=float(peak), correlation=float(np.corrcoef(signal, known_pulse[valid])[0, 1]))
    return answer


def synthetic_audit():
    fps, n = 30., 1800
    t = np.arange(n)/fps
    pulse = np.sin(2*np.pi*1.4*t)
    pulse_log = pulse[:, None]*np.array([.0006, .0020, .00025])
    log_base = np.log(np.array([120., 100., 90.]))+pulse_log
    true_delta = np.diff(log_base, axis=0, prepend=log_base[:1])
    rng = np.random.default_rng(73)
    samples = rng.uniform(80, 140, (100, 3))
    results = {}

    scenarios = {
        'common_pulse_exact_pairs': np.zeros((n, 100, 3)),
        # 0.015% common green ratio error per frame, small enough to survive
        # local outlier rejection, but cumulative by construction.
        'common_small_correspondence_bias': np.broadcast_to([.00003, .00015, -.00002], (n, 100, 3)).copy(),
        # Patch errors average exactly zero before robust trimming, but are
        # asymmetric: 80% +e, 20% -4e. Demonstrates selection/trim bias.
        'asymmetric_zero_mean_patch_error': np.zeros((n, 100, 3)),
        'minority_large_local_artifact': np.zeros((n, 100, 3)),
    }
    asymmetric = scenarios['asymmetric_zero_mean_patch_error']
    asymmetric[:, :, 1] = .00015
    asymmetric[:, :20, 1] = -.0006
    artifact = scenarios['minority_large_local_artifact']
    artifact[:, :20, :] = np.array([.03, .06, -.02])[None, None, :] * np.sin(2*np.pi*2.6*t)[:, None, None]
    for scenario, error in scenarios.items():
        for screened in (False, True):
            deltas, rejected = np.zeros((n, 3)), 0
            for i in range(1, n):
                old = samples.copy()
                new = old*np.exp(true_delta[i]+error[i])
                delta, kept, info = paired_log_change(old, new, screened, PixelConfig())
                if delta is None:
                    raise AssertionError('Controlled legal sample case unexpectedly lacked pairs')
                deltas[i] = delta
                rejected += info['screen_rejected']
            key = f'{scenario}/'+('screened' if screened else 'tracking_only')
            result = dict(screened_points=rejected, variants={})
            for anchor in (None, .1, .15, .25, 'lp4'):
                label = 'lp4_0.15Hz_causal' if anchor == 'lp4' else ('unbounded' if anchor is None else f'anchor_{anchor:g}Hz')
                output = reconstruct_lp4(deltas, log_base, fps=fps) if anchor == 'lp4' else reconstruct(deltas, log_base, anchor, fps)
                difference = output-log_base
                result['variants'][label] = dict(
                    last_rgb_ratio_to_truth=np.exp(difference[-1]).tolist(),
                    max_abs_log_error=np.max(np.abs(difference), axis=0).tolist(),
                    readout=pulse_readout(np.exp(output), pulse, fps))
            results[key] = result

    # Direct sampling counterexample: subpixel correspondence offset on a
    # spatial colour gradient repeats after each seed reset. FB consistency
    # alone could be zero for an equally biased inverse map; this is not an
    # assertion about OpenCV LK on the user's videos.
    yy, xx = np.mgrid[:120, :160]
    image = np.stack([100+.04*xx, 90+.2*xx, 80+.02*xx], axis=2).astype(np.float32)
    px, py = np.meshgrid(np.linspace(30, 120, 10), np.linspace(30, 90, 10))
    points = np.column_stack([px.ravel(), py.ravel()]).astype(np.float32)
    old = sample_patches(image, points, 3)
    new = sample_patches(image, points+[.08, 0], 3)
    delta, _, info = paired_log_change(old, new, True, PixelConfig())
    results['gradient_subpixel_offset_counterexample'] = dict(offset_px=.08,
        actual_motion_px=0., kept_points=info['n_kept'], screened_points=info['screen_rejected'],
        log_delta=delta.tolist(), sixty_second_ratio=np.exp(delta*(n-1)).tolist(),
        caveat='Injected correspondence error; does not establish that real LK has this bias.')
    # Real-track fallback is an increment switch, not a DC correction.
    results['fallback_algebra_counterexample'] = dict(
        virtual_previous=150., baseline_previous=100., baseline_current=101.,
        baseline_ratio_fallback_output=150.*101./100.,
        baseline_current_after_missing_reset=101.,
        interpretation='Existing 1.5x relative offset survives fallback and disappears abruptly at reset.')
    # Counterexample to treating the anchor cutoff as a brick-wall boundary:
    # the baseline branch leaks some motion even above the anchor frequency.
    motion = np.sin(2*np.pi*2.6*t)
    corrupted_baseline = log_base+motion[:, None]*np.array([.02, .08, -.01])
    leakage = {}
    for anchor in (None, .1, .15, .25, 'lp4'):
        label = 'lp4_0.15Hz_causal' if anchor == 'lp4' else ('unbounded' if anchor is None else f'anchor_{anchor:g}Hz')
        output = reconstruct_lp4(true_delta, corrupted_baseline, fps=fps) if anchor == 'lp4' else reconstruct(true_delta, corrupted_baseline, anchor, fps)
        difference = output-log_base
        if anchor == 'lp4':
            _, response = sosfreqz(butter(4, .15, btype='lowpass', fs=fps, output='sos'), worN=[2.6], fs=fps)
            transfer = abs(response[0])
        else:
            z = np.exp(-2j*np.pi*2.6/fps)
            alpha = 0. if anchor is None else 1-np.exp(-2*np.pi*anchor/fps)
            transfer = abs(alpha/(1-(1-alpha)*z)) if anchor else 0.
        leakage[label] = dict(baseline_motion_gain_at_2p6Hz=float(transfer),
            max_green_log_error_after_5s=float(np.max(np.abs(difference[150:, 1]))),
            readout=pulse_readout(np.exp(output), pulse, fps))
    results['anchor_motion_leakage_counterexample'] = leakage
    return results


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    saved_trace_audit().to_csv(HERE/'saved_trace_drift.csv', index=False)
    result = synthetic_audit()
    (HERE/'synthetic_integration_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(saved_roi_traces=72, synthetic_scenarios=len(result), output=str(HERE)), ensure_ascii=False))
    for key, value in result.items():
        if 'variants' in value:
            print(key, json.dumps(value, ensure_ascii=False))
        else:
            print(key, json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
