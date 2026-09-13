"""Fixed official PURE EfficientPhys, local-only development trial.

Real RGB crops -> anti-aliased 30 Hz -> standardized contiguous segments ->
overlapping official network -> derivative overlap-add -> integrated BVP.
No labels, HR-guided offset selection, training, or weight selection occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT/'results/data1_6_20260911'
OUTPUT = PROJECT/'results/data1_6_v26_20260911/neural_efficientphys'
LOCAL_READOUT = HERE.parent.parent/'rppg_motion_v25'
READOUT = LOCAL_READOUT if LOCAL_READOUT.exists() else PROJECT/'motion_upgrade_v25_20260911'
PARAMETERS = dict(checkpoint='PURE_EfficientPhys.pth', model='EfficientPhys',
    upstream_commit='b7500b848f84ad7f86e277b4612563b69f4f88f9',
    checkpoint_sha256='e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49',
    target_fps=30., crop_size=72, crop_expansion=1.5, max_bbox_gap_s=.1,
    chunk_output_frames=180, chunk_input_frames=181, chunk_stride=90, frame_depth=10,
    normalization='single scalar population mean/std of complete resampled valid segment',
    antialias='scipy resample_poly Kaiser beta5, rational rate denominator<=1000; linear pad',
    timestamp_policy='globally anchored 30Hz actual seconds, no gap-crossing interpolation',
    tail='append final full 181-real-sample clip; no duplicate final frame or discarded full tail',
    derivative_timing='output i describes interval [t_i,t_i+1]; integral assigned to t_i+1',
    overlap='positive 180-sample Hann, fuse raw derivatives before a single segment integral',
    filter='order3 zero-phase Butterworth .7-3.5Hz; segment std normalization',
    edge_mask_s=1.6, min_segment_30hz_frames=181,
    masks='direct original-frame crop provenance; does not describe entire network/filter receptive field',
    hr='frozen V25 evidence DP on saved waveform, original 10s/1s complete plan',
    reference_used=False, training=False, offline=True, benchmark_reproduction=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def runs(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))


def prepare_boxes(trace, fps):
    boxes = trace[['face_x0', 'face_y0', 'face_x1', 'face_y1']].to_numpy(float).copy()
    original = trace.rgb_valid.to_numpy(bool) & np.isfinite(boxes).all(axis=1)
    original &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes[~original] = np.nan
    imputed = np.zeros(len(trace), bool)
    limit = int(np.floor(PARAMETERS['max_bbox_gap_s']*fps+1e-9))
    for a, b in runs(~original):
        if a and b < len(trace) and b-a <= limit:
            fraction = np.arange(1, b-a+1)[:, None]/(b-a+1)
            boxes[a:b] = boxes[a-1]+fraction*(boxes[b]-boxes[a-1])
            imputed[a:b] = True
    return boxes, original, imputed


def crop_bounds(box, width, height):
    x0, y0, x1, y1 = box*np.array([width, height, width, height])
    cx, cy = (x0+x1)/2, (y0+y1)/2
    half_width, half_height = .75*(x1-x0), .75*(y1-y0)
    a, b = max(0, int(np.floor(cx-half_width))), max(0, int(np.floor(cy-half_height)))
    c, d = min(width, int(np.ceil(cx+half_width))), min(height, int(np.ceil(cy+half_height)))
    if c-a < 2 or d-b < 2:
        raise ValueError('Face box is outside the video')
    return a, b, c, d


def read_crops(video, trace, fps):
    boxes, observed, imputed = prepare_boxes(trace, fps)
    valid = np.isfinite(boxes).all(axis=1)
    crops = np.zeros((len(trace), 72, 72, 3), np.uint8)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, atol=1e-6, rtol=1e-6):
        raise ValueError('Video FPS/decoder differs from frozen trace')
    for i in range(len(trace)):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Video ended before frozen frame {i}')
        if valid[i]:
            top_shape = frame.shape
            left, top, right, bottom = crop_bounds(boxes[i], top_shape[1], top_shape[0])
            crops[i] = cv2.cvtColor(cv2.resize(frame[top:bottom, left:right], (72, 72),
                                              interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if i and i % 1000 == 0:
            print(json.dumps(dict(event='decoded', frames=i, expected=len(trace))), flush=True)
    extra, _ = cap.read()
    cap.release()
    if extra:
        raise ValueError('Video contains extra frames beyond frozen trace')
    return crops, valid, observed, imputed


def resample_segment(samples, start_frame, fps, target_fps=30.):
    """One contiguous segment only, exact physical target times, bounded memory."""
    values = np.asarray(samples)
    if values.ndim < 1 or len(values) < 2 or fps < target_fps*.95:
        raise ValueError('At least two frames and near-30Hz or faster source required')
    ratio = Fraction(target_fps/fps).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    source_begin = start_frame/fps
    decimated_n = (len(values)*up+down-1)//down
    if decimated_n < 2:
        return np.array([], int), np.empty((0,)+values.shape[1:], np.float32), dict(
            up=up, down=down, source_start_frame=int(start_frame), source_frames=len(values),
            target_frames=0, status='fewer_than_two_resampled_points')
    decimated_times = source_begin+np.arange(decimated_n)*down/(up*fps)
    last_time = min((start_frame+len(values)-1)/fps, decimated_times[-1])
    first_index = int(np.ceil(source_begin*target_fps-1e-9))
    stop_index = int(np.floor(last_time*target_fps+1e-9))+1
    indices = np.arange(first_index, stop_index)
    times = indices/target_fps
    left = np.clip(np.searchsorted(decimated_times, times, side='right')-1, 0, decimated_n-2)
    alpha = (times-decimated_times[left])/(decimated_times[left+1]-decimated_times[left])
    if np.any(alpha < -1e-7) or np.any(alpha > 1+1e-7):
        raise ValueError('Resampling attempted extrapolation')
    flat = values.reshape(len(values), -1)
    sampled = np.empty((len(indices), flat.shape[1]), np.float32)
    for a in range(0, flat.shape[1], 256):
        block = flat[:, a:a+256].astype(np.float32)
        if up != down:
            block = resample_poly(block, up, down, axis=0, window=('kaiser', 5.), padtype='line')
        sampled[:, a:a+256] = block[left]*(1-alpha[:, None])+block[left+1]*alpha[:, None]
    return indices, sampled.reshape((len(indices),)+values.shape[1:]), dict(
        up=up, down=down, effective_polyphase_hz=fps*up/down,
        target_hz=target_fps, source_start_frame=int(start_frame),
        source_frames=len(values), target_frames=len(indices),
        antialias_applied=up < down, phase_correction='linear interpolation onto global exact time grid')


def chunk_starts(length):
    if length < 181:
        return []
    starts = list(range(0, length-180, 90))
    if starts[-1] != length-181:
        starts.append(length-181)
    return starts


def overlap_infer(crops, predictor):
    """Predictor takes 181 consecutive standardized crops, emits 180 derivatives."""
    n = len(crops)
    mean, std = float(np.mean(crops, dtype=np.float64)), float(np.std(crops, dtype=np.float64))
    if not np.isfinite([mean, std]).all() or std < 1e-8:
        return np.full(max(0, n-1), np.nan), dict(status='flat_crop', clips=0)
    starts = chunk_starts(n)
    numerator, denominator = np.zeros(max(0, n-1)), np.zeros(max(0, n-1))
    taper = np.hanning(182)[1:-1]
    for start in starts:
        block = (crops[start:start+181].astype(np.float32)-mean)/std
        prediction = np.asarray(predictor(block), dtype=float).reshape(-1)
        if prediction.shape != (180,) or not np.isfinite(prediction).all():
            raise ValueError('Official model produced invalid shape/nonfinite values')
        numerator[start:start+180] += prediction*taper
        denominator[start:start+180] += taper
    result = np.full(len(numerator), np.nan)
    result[denominator > 0] = numerator[denominator > 0]/denominator[denominator > 0]
    return result, dict(status='predicted' if starts else 'shorter_than_181_frames',
        clips=len(starts), clip_starts=starts, scalar_mean=mean, scalar_std=std,
        predicted_intervals=int(np.isfinite(result).sum()), uncovered_intervals=int((denominator == 0).sum()))


def reconstruct(derivative):
    derivative = np.asarray(derivative, float)
    if not np.isfinite(derivative).all():
        return np.full(len(derivative)+1, np.nan), np.full(len(derivative)+1, np.nan)
    # The unknown additive initial BVP level is zero, then removed by bandpass.
    integrated = np.r_[0., np.cumsum(derivative)]
    sos = butter(3, [.7, 3.5], fs=30., btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, integrated)
    std = float(np.std(filtered))
    if std < 1e-8:
        return integrated, np.full(len(integrated), np.nan)
    filtered = (filtered-filtered.mean())/(std+1e-9)
    filtered[:48] = np.nan
    filtered[-48:] = np.nan
    return integrated, filtered


def load_model(device='cuda'):
    import torch
    manifest = json.loads((HERE/'upstream_manifest.json').read_text(encoding='utf-8-sig'))
    for row in manifest['files']:
        if sha(HERE/'upstream'/row['local_file']) != row['sha256']:
            raise ValueError('Upstream artifact hash changed')
    path = HERE/'upstream'/PARAMETERS['checkpoint']
    assert sha(path) == PARAMETERS['checkpoint_sha256']
    spec = importlib.util.spec_from_file_location('official_efficientphys', HERE/'upstream/EfficientPhys.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    model = source.EfficientPhys(frame_depth=10, img_size=72)
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in weights.items()):
        raise ValueError('Expected tensor-only state dict')
    if all(key.startswith('module.') for key in weights):
        weights = {key[7:]: value for key, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    def predictor(block):
        x = torch.from_numpy(np.ascontiguousarray(block.transpose(0, 3, 1, 2))).to(device)
        with torch.inference_mode():
            return model(x).reshape(-1).cpu().numpy()
    return model, predictor, dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
        device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()), weights_only=True, strict=True)


def case_run(item, predictor, frozen, runtime, output):
    started = time.perf_counter()
    case = item['case']
    source = BASE/case/'inference'
    meta = json.loads((source/'frame_trace.json').read_text())
    trace = pd.read_csv(source/'frame_trace.csv')
    fps = float(meta['fps'])
    if sha(source/'frame_trace.csv') != meta['trace_sha256'] or len(trace) != meta['n_frames']:
        raise ValueError('Frozen frontend cache identity mismatch')
    np.testing.assert_allclose(trace.time_s, np.arange(len(trace))/fps, atol=1e-8, rtol=0)
    video = Path(item['video']['path'])
    if video.stat().st_size != item['video']['bytes'] or sha(video) != item['video']['sha256']:
        raise ValueError('Input video differs from complete frozen manifest')
    if meta['identity']['video'] != str(video) or meta['identity']['size'] != video.stat().st_size:
        raise ValueError('Video and trace belong to different input')
    print(json.dumps(dict(event='case_start', case=case, fps=fps, frames=len(trace))), flush=True)
    crops, valid, observed, imputed = read_crops(video, trace, fps)
    n30 = int(np.floor((len(trace)-1)/fps*30))+1
    table30 = pd.DataFrame(dict(time_s=np.arange(n30)/30., raw_network_derivative=np.nan,
        integrated_bvp=np.nan, base=np.nan, segment_id=np.full(n30, -1),
        derivative_supported=np.zeros(n30, bool)))
    diagnostics = []
    for sid, (start, stop) in enumerate(runs(valid)):
        if stop-start < 2:
            diagnostics.append(dict(segment_id=sid, original_start=int(start), original_stop=int(stop), status='one_frame'))
            continue
        idx, sampled, sampling = resample_segment(crops[start:stop], int(start), fps)
        if len(idx) < 181:
            diagnostics.append(dict(segment_id=sid, sampling=sampling, status='shorter_than_181_frames'))
            continue
        derivative, neural = overlap_infer(sampled, predictor)
        integrated, filtered = reconstruct(derivative)
        table30.loc[idx[:-1], 'raw_network_derivative'] = derivative
        table30.loc[idx, 'integrated_bvp'] = integrated
        table30.loc[idx, 'base'] = filtered
        table30.loc[idx, 'segment_id'] = sid
        table30.loc[idx[:-1], 'derivative_supported'] = np.isfinite(derivative)
        diagnostics.append(dict(segment_id=sid, sampling=sampling, neural=neural))
        print(json.dumps(dict(event='segment_complete', case=case, segment=sid, frames_30hz=len(idx), clips=neural['clips'])), flush=True)
    del crops
    wave = np.full(len(trace), np.nan)
    for start, stop in runs(np.isfinite(table30.base)):
        times = table30.time_s.to_numpy()[start:stop]
        selected = (trace.time_s.to_numpy() >= times[0]) & (trace.time_s.to_numpy() <= times[-1])
        wave[selected] = np.interp(trace.time_s.to_numpy()[selected], times, table30.base.to_numpy()[start:stop])
    covered = np.isfinite(wave)
    destination = output/case
    destination.mkdir(exist_ok=False)
    saved = pd.DataFrame(dict(time_s=trace.time_s, base=wave, covered=covered,
        observed=observed & covered, interpolated=imputed & covered))
    saved.to_csv(destination/'waveform.csv', index=False)
    table30.to_csv(destination/'network_30hz.csv', index=False)
    pd.DataFrame(dict(time_s=trace.time_s, crop_valid=valid, crop_observed=observed,
                      bbox_interpolated=imputed)).to_csv(destination/'crop_provenance.csv', index=False)
    dump(destination/'segment_diagnostics.json', diagnostics)
    sys.path.insert(0, str(READOUT))
    from evidence_hr import estimate_evidence
    restored = pd.read_csv(destination/'waveform.csv')
    hr = estimate_evidence(restored.base.to_numpy(), pd.DataFrame({'rgb_valid':restored.observed}),
        restored.interpolated.to_numpy(bool), fps, 10, 1, 42, 210, motion_trace=trace)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    hr['window_start_s'] = np.arange(len(hr))*round(fps)/fps
    hr['window_end_s'] = hr.window_start_s+round(10*fps)/fps
    hr['hr_source'] = 'saved_efficientphys_integrated_waveform_offline'
    hr.to_csv(destination/'heart_rate.csv', index=False)
    summary = dict(case=case, method='EfficientPhys_PURE_30Hz_overlap', fps=fps, frames=len(trace),
        parameters=PARAMETERS, runtime=runtime, source_video=str(video), source_video_sha256=item['video']['sha256'],
        source_trace=str(source/'frame_trace.csv'), source_trace_sha256=meta['trace_sha256'],
        code_hashes=frozen, reference_used=False, total_windows=len(hr), accepted_windows=int(hr.accepted.sum()),
        waveform_coverage=float(covered.mean()), elapsed_s=time.perf_counter()-started,
        output_hashes={name:sha(destination/name) for name in ['waveform.csv','network_30hz.csv',
            'heart_rate.csv','crop_provenance.csv','segment_diagnostics.json']},
        limitations=['Adapted dynamic crop, overlap, global 30Hz clock and project readout differ from author benchmark',
            'Raw network predicts derivative; cumulative BVP has arbitrary amplitude/offset and is not clinical morphology',
            'Frame timeline interpolation of 30Hz output does not create additional temporal resolution',
            'Masks describe direct crop support, not full temporal network/filter receptive field',
            'Same six videos previously inspected: development regression, not held-out generalization'])
    dump(destination/'summary.json', summary)
    print(json.dumps(dict(event='case_complete', case=case, accepted=int(hr.accepted.sum()), waveform_coverage=float(covered.mean()))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    if not args.run_all:
        parser.error('Specify --run-all for the six fixed videos')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(2)
    verification = json.loads((HERE/'test_receipt.json').read_text())
    if not verification['passed'] or verification['adapter_sha256'] != sha(__file__):
        raise ValueError('Current adapter lacks passing synthetic/backend tests')
    files = [HERE/'run_trial.py', HERE/'test_trial.py', HERE/'upstream_manifest.json']
    files += sorted((HERE/'upstream').iterdir())
    files += [READOUT/name for name in ('evidence_hr.py','legacy_motion.py','analyze_rppg.py','motion_evidence.py')]
    frozen = {str(path):sha(path) for path in files}
    entries = json.loads((BASE/'inputs_manifest.json').read_text())
    items = [dict(case=row['case'], video=row['video']) for row in entries]
    assert {row['case'] for row in items} == {f'data{i}' for i in range(1,7)}
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output/'protocol_before_inference.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parameters=PARAMETERS, code_hashes=frozen, cases=items,
        input_manifest_sha256=sha(BASE/'inputs_manifest.json'), test_receipt_sha256=sha(HERE/'test_receipt.json'),
        reference_used=False, case_policy='One predeclared checkpoint; no training or reference-based selection'))
    model, predictor, runtime = load_model(args.device)
    summaries = [case_run(item, predictor, frozen, runtime, args.output) for item in items]
    assert all(sha(path) == checksum for path, checksum in frozen.items()), 'Sources changed during inference'
    dump(args.output/'runs.json', [dict(case=row['case'], returncode=0, elapsed_s=row['elapsed_s']) for row in summaries])
    dump(args.output/'run_summary.json', dict(all_six_successful=True, cases=summaries, runtime=runtime))


if __name__ == '__main__':
    main()
