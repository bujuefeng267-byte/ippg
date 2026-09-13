"""Frozen PURE weights on project-tracked real face crops; no reference input.

This is an offline project adapter, not the author's static-Haar benchmark.
Run backend_test.py and test_preprocessing.py first, then --run-all.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ESTIMATOR = HERE.parent / 'cpace/project_estimator'
sys.path.insert(0, str(ESTIMATOR))
from legacy_motion import estimate, fill_short_gaps, gap_frame_limit, runs
from analyze_rppg import bandpass

CASES = ('user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1')
PARAMETERS = {
    'checkpoint': 'PURE_cross_RhythmMamba.pth',
    'training': 'none; frozen author PURE pretraining',
    'crop': 'V22 tracked normalized face box, 1.5 expansion, clamp at frame edges',
    'rgb_valid_required': True, 'max_bbox_gap_s': 0.10,
    'resize': '128x128 RGB, cv2.INTER_AREA',
    'normalization': 'one scalar mean/std(ddof=0) over all real cropped frames of each bounded segment, including discarded tail',
    'clips': '160 frames, nonoverlapping, incomplete segment tail discarded',
    'prediction_normalization': 'per clip torch mean/std correction=1, as author trainer',
    'sample_clock': 'original video fps, no 30 Hz resampling',
    'postprocessing': 'project third-order zero-phase Butterworth 42-210 bpm per contiguous predicted segment',
    'edge_mask_s': 1.6,
    'final_hr': 'legacy saved-waveform 10s/1s local peak and offline DP',
    'reference_used': False, 'offline': True,
    'benchmark_reproduction': False,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def prepare_boxes(trace, fps):
    """Short gaps interpolate geometry only. Video pixels are never synthesized."""
    boxes = trace[['face_x0', 'face_y0', 'face_x1', 'face_y1']].to_numpy(float)
    observed = trace.rgb_valid.to_numpy(bool) & np.isfinite(boxes).all(axis=1)
    observed &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes[~observed] = np.nan
    boxes, interpolated = fill_short_gaps(boxes, gap_frame_limit(.10, fps))
    return boxes, observed, interpolated


def crop_bounds(box, width, height):
    x0, y0, x1, y1 = box * np.array([width, height, width, height])
    cx, cy = (x0+x1)/2, (y0+y1)/2
    hw, hh = .75*(x1-x0), .75*(y1-y0)
    left, top = max(0, int(np.floor(cx-hw))), max(0, int(np.floor(cy-hh)))
    right, bottom = min(width, int(np.ceil(cx+hw))), min(height, int(np.ceil(cy+hh)))
    if right-left < 2 or bottom-top < 2:
        raise ValueError('Invalid or out-of-frame face box')
    return left, top, right, bottom


def segment_statistics(crops):
    # Integer-origin accumulation in float64, with bounded temporary memory.
    total = square = 0.0
    count = crops.size
    for a in range(0, len(crops), 160):
        x = crops[a:a+160].astype(np.float64)
        total += float(x.sum())
        square += float(np.square(x).sum())
    mean = total/count
    std = float(np.sqrt(max(0, square/count-mean*mean)))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError('Flat face input cannot be standardized')
    return mean, std


def read_crops(video, trace, fps):
    boxes, observed, interpolated = prepare_boxes(trace, fps)
    valid = np.isfinite(boxes).all(axis=1)
    crops = np.zeros((len(trace), 128, 128, 3), np.uint8)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    if not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, atol=1e-6, rtol=1e-6):
        raise ValueError('Video and frozen trace fps differ')
    for i in range(len(trace)):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Video ended early at frame {i}')
        if valid[i]:
            h, w = frame.shape[:2]
            left, top, right, bottom = crop_bounds(boxes[i], w, h)
            crop = cv2.resize(frame[top:bottom, left:right], (128, 128), interpolation=cv2.INTER_AREA)
            crops[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    extra, _ = cap.read()
    cap.release()
    if extra:
        raise ValueError('Video contains more frames than frozen trace')
    return crops, valid, observed, interpolated


def infer_crops(model, crops, valid, fps, device):
    import torch
    raw = np.full(len(crops), np.nan)
    diagnostics = []
    with torch.inference_mode():
        for sid, (a, b) in enumerate(runs(valid)):
            used = (b-a)//160*160
            record = dict(segment_index=sid, start_frame=int(a), end_frame_exclusive=int(b),
                          used_frames=int(used), tail_discarded_frames=int(b-a-used))
            if not used:
                record['status'] = 'shorter_than_160_frames'
                diagnostics.append(record)
                continue
            mean, std = segment_statistics(crops[a:b])
            record.update(status='predicted', scalar_mean=mean, scalar_std=std)
            for start in range(a, a+used, 160):
                x = (crops[start:start+160].astype(np.float32)-mean)/std
                x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)[None])).to(device)
                y = model(x)
                if tuple(y.shape) != (1, 160) or not torch.isfinite(y).all():
                    raise ValueError('Unexpected or nonfinite neural prediction')
                ystd = y.std(dim=-1, keepdim=True, correction=1)
                if torch.any(ystd < 1e-8):
                    raise ValueError('Flat neural output cannot be normalized')
                y = (y-y.mean(dim=-1, keepdim=True))/ystd
                raw[start:start+160] = y[0].cpu().numpy()
            diagnostics.append(record)
            print(json.dumps({'event':'segment_predicted', 'segment':sid, 'frames':int(used)}), flush=True)
    return raw, diagnostics


def postprocess(raw, fps):
    wave = np.full(len(raw), np.nan)
    edge = round(1.6*fps)
    for a, b in runs(np.isfinite(raw)):
        if b-a <= 2*edge:
            continue
        filtered = bandpass(raw[a:b], fps, 42, 210)
        wave[a+edge:b-edge] = filtered[edge:-edge]
    return wave


def final_hr(saved, fps):
    hr = estimate(saved.base.to_numpy(float), pd.DataFrame({'rgb_valid': saved.observed.to_numpy(bool)}),
                  saved.interpolated.to_numpy(bool), fps, 10, 1, 42, 210)
    hr['raw_spectral_peak_bpm'] = hr.spectral_peak_bpm
    hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']] = np.nan
    hr['hr_source'] = 'saved_rhythm_waveform_offline'
    return hr


def run_case(case, model, model_meta, frozen, identity, device):
    start = time.perf_counter()
    source = ROOT / 'rppg_motion_v22/validation/trimmed_gap10' / case
    trace = pd.read_csv(source/'frame_trace.csv')
    meta = json.loads((source/'frame_trace.json').read_text(encoding='utf-8'))
    fps = float(meta['fps'])
    if not np.allclose(trace.time_s, np.arange(len(trace))/fps, rtol=0, atol=1e-8):
        raise ValueError('Unexpected trace clock')
    if sha(source/'frame_trace.csv') != meta['trace_sha256']:
        raise ValueError('Frozen trace changed')
    video = Path(identity['video'])
    if sha(video) != identity['sha256']:
        raise ValueError('Video hash differs from frozen V22 identity')
    print(json.dumps({'event':'start_case', 'case':case}), flush=True)
    crops, valid, observed, interpolated = read_crops(video, trace, fps)
    raw, diagnostics = infer_crops(model, crops, valid, fps, device)
    del crops
    wave = postprocess(raw, fps)
    covered = np.isfinite(wave)
    output = HERE / 'results' / case
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(dict(time_s=trace.time_s, base=wave, covered=covered,
                      observed=observed & covered, interpolated=interpolated & covered)).to_csv(output/'waveform.csv', index=False)
    pd.DataFrame(dict(time_s=trace.time_s, raw_network=raw, crop_observed=observed,
                      bbox_interpolated=interpolated, crop_valid=valid)).to_csv(output/'raw_network.csv', index=False)
    saved = pd.read_csv(output/'waveform.csv')
    hr = final_hr(saved, fps)
    hr.to_csv(output/'heart_rate.csv', index=False)
    dump(output/'segment_diagnostics.json', diagnostics)
    summary = dict(case=case, method='RhythmMamba_PURE_project_crop_reference_scan',
                   fps=fps, frames=len(trace), parameters=PARAMETERS, model=model_meta,
                   trace_sha256=sha(source/'frame_trace.csv'), source_trace=str(source/'frame_trace.csv'),
                   source_video_sha256=identity['sha256'], source_video=str(video),
                   reference_used=False, offline=True, code_hashes=frozen,
                   accepted_windows=int(hr.accepted.sum()), total_windows=len(hr),
                   waveform_coverage=float(covered.mean()), status_counts=hr.status.value_counts().to_dict(),
                   wall_seconds=time.perf_counter()-start,
                   limitations=['Project dynamic tracked crops differ from author static first-frame Haar crops',
                                'Normalization spans each bounded segment and includes discarded tail; offline',
                                'Reference scan backend speed does not reproduce author CUDA kernel speed',
                                '160-frame clip tails and long tracking gaps retain missing outputs',
                                'Original fps and project filter/10s HR differ from author nominal FS30 whole-record scoring'])
    summary['output_hashes'] = {name: sha(output/name) for name in
                                ['waveform.csv', 'heart_rate.csv', 'raw_network.csv', 'segment_diagnostics.json']}
    dump(output/'summary.json', summary)
    print(json.dumps({'event':'case_complete', 'case':case, 'hr_windows':summary['accepted_windows'],
                      'waveform_coverage':summary['waveform_coverage']}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-all', action='store_true')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if not args.run_all:
        p.error('Use --run-all')
    if (HERE/'results').exists():
        raise FileExistsError('Trial outputs exist; do not overwrite frozen predictions')
    import torch
    from backend import load_model
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in ['backend_verification.json', 'preprocessing_verification.json']:
        record = json.loads((HERE/name).read_text(encoding='utf-8'))
        if not record.get('passed'):
            raise RuntimeError(f'{name} did not pass')
    files = [HERE/'run_rhythm_trial.py', HERE/'backend.py', HERE/'backend_test.py', HERE/'test_preprocessing.py']
    files += sorted(ESTIMATOR.glob('*.py'))
    frozen = {str(f.relative_to(HERE.parent)):sha(f) for f in files}
    identities = json.loads((ROOT/'rppg_motion_v22/validation/protocol_before_validation.json').read_text(encoding='utf-8'))['cases']
    dump(HERE/'protocol_before_inference.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parameters=PARAMETERS, code_hashes=frozen, cases=identities, reference_used=False))
    model, model_meta = load_model(args.device)
    summaries = {}
    for case in CASES:
        summaries[case] = run_case(case, model, model_meta, frozen, identities[case], args.device)
    if any(sha(f) != frozen[str(f.relative_to(HERE.parent))] for f in files):
        raise RuntimeError('Source changed during inference')
    dump(HERE/'run_summary.json', dict(all_six_successful=True, created_utc=datetime.now(timezone.utc).isoformat(),
                                     parameters=PARAMETERS, model=model_meta, cases=summaries))


if __name__ == '__main__':
    main()
