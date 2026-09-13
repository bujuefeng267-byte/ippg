"""Predeclared synthetic falsification of brightness-driven optical-flow error.

Only flow grayscale representation is changed. V23 paired RGB patches, FB
threshold, local screening, baseline-ratio fallback and reconstruction are used
unchanged. The oracle uses the known synthetic displacement, not real labels.
No human videos, reference data, or frozen production files are modified.
"""
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V23 = HERE.parents[1]/'rppg_motion_v23'
sys.path.insert(0, str(V23))
from pixel_tracking import PixelTracker, PixelConfig
from baseline_frontend import sample_rois, REGION_NAMES
from legacy_motion import mesh_object

FPS, N_FRAMES = 30., 240
SIGMA, FLOOR, MAP_SCALE = 5., 5., 32.
SEEDS = (73023, 73024)
TEXTURES = ('low_texture_gradient', 'skin_texture')
PULSE_AMPLITUDES = (.002, .01)
MOTION_STEPS = (0, 1, 3)
LIGHTING_AMPLITUDES = (0., .08)
FLOW_MODES = ('raw_gray', 'local_normalized_gray', 'oracle_displacement')


def normalized_gray(gray):
    gray = np.asarray(gray, np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), SIGMA)
    second = cv2.GaussianBlur(gray*gray, (0, 0), SIGMA)
    variance = np.maximum(second-mean*mean, 0.)
    normalized = (gray-mean)/np.sqrt(variance+FLOOR*FLOOR)
    return np.clip(np.rint(128+MAP_SCALE*normalized), 0, 255).astype(np.uint8)


class DiagnosticTracker(PixelTracker):
    def __init__(self, flow_mode):
        super().__init__(PixelConfig(), mode='tracking_screened')
        self.flow_mode = flow_mode
        self.expected_dx = 0.
        self.errors_accepted = []
        self.errors_finite = []
        self.accepted_geometry = 0
        self.total_geometry = 0
        self.normal_previous = None

    def _flow(self, gray, points):
        if self.flow_mode == 'oracle_displacement':
            moved = points+np.array([self.expected_dx, 0], np.float32)
            good, fb = np.ones(len(points), bool), np.zeros(len(points))
        elif self.flow_mode == 'local_normalized_gray':
            saved = self.previous_gray
            prior = self.normal_previous if self.normal_previous is not None else normalized_gray(saved)
            current = normalized_gray(gray)
            self.previous_gray = prior
            try:
                moved, good, fb = super()._flow(current, points)
            finally:
                self.previous_gray = saved
            self.normal_previous = current
        else:
            moved, good, fb = super()._flow(gray, points)
        expected = points+np.array([self.expected_dx, 0], np.float32)
        error = np.linalg.norm(moved-expected, axis=1)
        self.errors_finite.extend(error[np.isfinite(error)].tolist())
        self.errors_accepted.extend(error[good & np.isfinite(error)].tolist())
        self.accepted_geometry += int(good.sum())
        self.total_geometry += len(points)
        return moved, good, fb


def make_scene(texture, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:160, :240]
    random = rng.standard_normal((160, 240)).astype(np.float32)
    if texture == 'low_texture_gradient':
        noise = 2*cv2.GaussianBlur(random, (0, 0), 2.)
        ramp = .20*(xx-120)+.04*(yy-80)
    else:
        noise = 12*cv2.GaussianBlur(random, (0, 0), .7)
        ramp = .04*(xx-120)+.02*(yy-80)
    return (np.array([130., 105., 85.])+ramp[:, :, None]*[1., .8, .6]+noise[:, :, None]).astype(np.float32)


def sample_row(image, frame, shift):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = 50+shift, 8., 190+shift, 152.
    corners = np.repeat(np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]]), 25, axis=0)
    row = sample_rois(image, mesh_object(corners, w, h))
    row.update(frame=frame, time_s=frame/FPS, source='known_synthetic_geometry',
               face_x0=x0/w, face_y0=y0/h, face_x1=x1/w, face_y1=y1/h,
               motion_x=0., motion_y=0.)
    return row


def fit_modulation(values, times, valid, *, increments=False):
    sine, cosine = np.sin(2*np.pi*1.4*times), np.cos(2*np.pi*1.4*times)
    light_s, light_c = np.sin(2*np.pi*.12*times), np.cos(2*np.pi*.12*times)
    if increments:
        difference = lambda x: np.r_[np.nan, np.diff(x)]
        design = np.column_stack([difference(sine), difference(cosine), difference(light_s),
                                  difference(light_c), np.ones(len(times)), times-times.mean()])
    else:
        centered = times-times.mean()
        design = np.column_stack([sine, cosine, light_s, light_c,
                                  np.ones(len(times)), centered, centered**2])
    use = np.asarray(valid, bool) & np.isfinite(values) & np.isfinite(design).all(1)
    if use.sum() < 30 or np.linalg.matrix_rank(design[use]) < design.shape[1]:
        return None
    beta = np.linalg.lstsq(design[use], np.asarray(values)[use], rcond=None)[0]
    residual = np.asarray(values)[use]-design[use]@beta
    return dict(n=int(use.sum()), amplitude=float(np.hypot(beta[0], beta[1])),
                in_phase_amplitude=float(beta[0]), phase_rad=float(np.arctan2(beta[1], beta[0])),
                residual_rms=float(np.sqrt(np.mean(residual**2))))


def run_case(texture, seed, amplitude, step, lighting):
    scene = make_scene(texture, seed)
    time_axis = np.arange(N_FRAMES)/FPS
    phase = np.arange(N_FRAMES) % 32
    # Integer triangular translation: adjacent steps have exact magnitude 1
    # or 3 pixels, with reversals, bounded to +/-8 or +/-24 pixels.
    shifts = step*(np.where(phase <= 16, phase, 32-phase)-8)
    trackers = {mode: DiagnosticTracker(mode) for mode in FLOW_MODES}
    output = {mode: [] for mode in FLOW_MODES}
    baseline_g = []
    for i, t in enumerate(time_axis):
        gain = np.exp(amplitude*np.sin(2*np.pi*1.4*t)*np.array([.4, 1., .2])
                      + lighting*np.sin(2*np.pi*.12*t))
        frame = np.clip(np.rint(np.roll(scene, int(shifts[i]), axis=1)*gain), 0, 255).astype(np.uint8)
        baseline = sample_row(frame, i, shifts[i])
        baseline_g.append([baseline[f'{roi}_g'] for roi in REGION_NAMES])
        for mode, tracker in trackers.items():
            tracker.expected_dx = 0. if i == 0 else float(shifts[i]-shifts[i-1])
            output[mode].append(tracker.update(frame, baseline, i))
    modes = {}
    for mode, records in output.items():
        data = pd.DataFrame(records)
        tracker = trackers[mode]
        accepted_error = np.asarray(tracker.errors_accepted)
        all_error = np.asarray(tracker.errors_finite)
        per_roi = []
        for roi in REGION_NAMES:
            source = data[f'{roi}_pixel_source'].to_numpy()
            log_g = np.log(data[f'{roi}_g'].to_numpy(float))
            log_delta = data[f'{roi}_pixel_log_delta_g'].to_numpy(float)
            tracked = source == 'tracked_ratio'
            all_fit = fit_modulation(log_delta, time_axis, np.isfinite(log_delta), increments=True)
            tracked_fit = fit_modulation(log_delta, time_axis, tracked, increments=True)
            reconstruction_fit = fit_modulation(log_g, time_axis, np.isfinite(log_g))
            baseline_fit = fit_modulation(np.log(data[f'baseline_{roi}_g'].to_numpy(float)), time_axis, np.ones(len(data), bool))
            for fit in (all_fit, tracked_fit, reconstruction_fit, baseline_fit):
                if fit:
                    fit['gain_to_injected_amplitude'] = fit['amplitude']/amplitude
                    fit['signed_gain_to_injected_amplitude'] = fit['in_phase_amplitude']/amplitude
            per_roi.append(dict(roi=roi, tracked_frames=int(tracked.sum()),
                fallback_frames=int((source == 'baseline_ratio_fallback').sum()),
                screened_points=int(data[f'{roi}_pixel_screen_rejected'].sum()),
                tracked_fraction_excluding_initial=float(tracked[1:].mean()),
                paired_increment_all_output=all_fit, paired_increment_tracked_only=tracked_fit,
                reconstructed_log_green=reconstruction_fit, baseline_log_green=baseline_fit))
        modes[mode] = dict(
            geometric_fb_acceptance=float(tracker.accepted_geometry/tracker.total_geometry) if tracker.total_geometry else None,
            flow_error_accepted_median_px=float(np.median(accepted_error)) if len(accepted_error) else None,
            flow_error_accepted_p95_px=float(np.percentile(accepted_error, 95)) if len(accepted_error) else None,
            flow_error_all_finite_median_px=float(np.median(all_error)) if len(all_error) else None,
            flow_error_all_finite_p95_px=float(np.percentile(all_error, 95)) if len(all_error) else None,
            per_roi=per_roi)
    return dict(texture=texture, seed=seed, pulse_log_amplitude_g=amplitude,
                known_heart_rate_bpm=84., motion_step_px=step, lighting_log_amplitude=lighting,
                frames=N_FRAMES, fps=FPS, modes=modes)


def summarize(records):
    groups = {}
    for record in records:
        key = record['texture']+'/'+('static' if record['motion_step_px'] == 0 else 'moving')
        for mode in FLOW_MODES:
            group = groups.setdefault(key+'/'+mode, dict(flow_error=[], paired_gain_error=[],
                signed_paired_gain=[], reconstructed_gain_error=[], tracked_fraction=[], geometric_fraction=[]))
            evidence = record['modes'][mode]
            if evidence['flow_error_accepted_median_px'] is not None:
                group['flow_error'].append(evidence['flow_error_accepted_median_px'])
            group['geometric_fraction'].append(evidence['geometric_fb_acceptance'])
            for roi in evidence['per_roi']:
                group['tracked_fraction'].append(roi['tracked_fraction_excluding_initial'])
                fit = roi['paired_increment_tracked_only']
                if fit:
                    group['signed_paired_gain'].append(fit['signed_gain_to_injected_amplitude'])
                    group['paired_gain_error'].append(abs(fit['gain_to_injected_amplitude']-1))
                fit = roi['reconstructed_log_green']
                if fit:
                    group['reconstructed_gain_error'].append(abs(fit['gain_to_injected_amplitude']-1))
    return {key: {name: dict(n=len(values), median=float(np.median(values)) if values else None,
                             p95=float(np.percentile(values, 95)) if values else None)
                  for name, values in group.items()} for key, group in groups.items()}


def main():
    cv2.setNumThreads(2)
    started = time.perf_counter()
    matrix = list(product(TEXTURES, SEEDS, PULSE_AMPLITUDES, MOTION_STEPS, LIGHTING_AMPLITUDES))
    records = []
    for i, params in enumerate(matrix, 1):
        record = run_case(*params)
        records.append(record)
        print(f'{i}/{len(matrix)} '+str(params), flush=True)
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), hypothesis='Brightness constancy may interpret physiological colour variation as displacement; not assumed to explain any real video.',
        comparison='Raw grayscale vs fixed local normalization; known integer-shift oracle is a synthetic control.',
        normalization=dict(gaussian_sigma_px=SIGMA, variance_floor_gray=FLOOR, map_center=128, map_scale=MAP_SCALE),
        matrix=dict(textures=list(TEXTURES), seeds=list(SEEDS), pulse_amplitudes_g=list(PULSE_AMPLITUDES),
                    motion_step_px=list(MOTION_STEPS), lighting_log_amplitudes=list(LIGHTING_AMPLITUDES), frames=N_FRAMES, fps=FPS),
        cases=len(records), references_read=False, real_videos_read=False,
        unchanged='V23 sampled RGB, same 7x7 patches, FB<=0.5, local screening, min_tracks=12, fallback and reconstruction.',
        limitations=['Synthetic texture and modulation, not physiological accuracy validation.',
                     'Static accepted flow error measures false motion; moving error is against exact per-frame displacement.',
                     'Fallback can preserve pulse while optical tracking fails: paired-only amplitude and tracked fraction are reported separately.',
                     'Amplitude is a least-squares 1.4Hz coefficient with known slow-light and polynomial nuisance terms, not a selected spectral peak.',
                     'Parameters and full factorial matrix fixed before results; all cases retained.'],
        runtime_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((V23/name).read_bytes()).hexdigest() for name in ['pixel_tracking.py','baseline_frontend.py','legacy_motion.py']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summaries=summarize(records),results=records)
    destination = HERE/'photometric_flow_diagnosis.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(cases=len(records), output=str(destination), runtime_seconds=result['runtime_seconds'], summaries=result['summaries']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
