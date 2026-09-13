#!/usr/bin/env python3
"""Independent open-rppg adapter using the recorded robust frontend boxes.

This uses open-rppg's converted/trained weights, NOT the authors' original
PyTorch RhythmMamba benchmark. Exports raw BVP; never treats its SQI as accuracy.
Run in .venv-neural. Predictions are not generated across long missing ROIs.
"""
import argparse
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import cv2
import numpy as np


def segments(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model', default='RhythmMamba.pure')
    a = p.parse_args()
    import rppg
    metadata = json.loads(a.trace.with_suffix('.json').read_text())
    identity = metadata['identity']
    if (str(a.video.resolve()) != identity['video'] or a.video.stat().st_size != identity['size']
            or a.video.stat().st_mtime_ns != identity['mtime_ns']):
        raise ValueError('Frame trace does not belong to this video')
    fps = metadata['fps']
    with a.trace.open() as f:
        rows = list(csv.DictReader(f))
    boxes = np.array([[float(r[k]) if r[k].strip() else np.nan
                       for k in ('face_x0','face_y0','face_x1','face_y1')] for r in rows])
    observed = np.array([r['rgb_valid'] == 'True' for r in rows])
    boxes[~observed] = np.nan
    interpolated = np.zeros(len(rows), bool)
    for start, stop in segments(~np.isfinite(boxes).all(axis=1)):
        if start > 0 and stop < len(boxes) and stop-start <= round(0.1*fps):
            alpha = np.arange(1,stop-start+1)[:,None]/(stop-start+1)
            boxes[start:stop] = boxes[start-1]+alpha*(boxes[stop]-boxes[start-1])
            interpolated[start:stop] = True
    a.output.mkdir(parents=True, exist_ok=False)
    model = rppg.Model(a.model)
    model.face_detection_threads = 2
    model.face_resampling_threads = 2
    initial_state = copy.deepcopy(model.state)
    print('model', a.model, 'meta', model.meta, flush=True)
    h, w = model.input[1:3]
    faces = np.zeros((len(boxes), h, w, 3), np.uint8)
    valid = np.isfinite(boxes).all(axis=1)
    cap = cv2.VideoCapture(str(a.video))
    try:
        for i in range(len(rows)):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Video ends before frame trace')
            if not valid[i]:
                continue
            ih, iw = frame.shape[:2]
            x0,y0,x1,y1 = boxes[i]
            x0,x1 = np.clip(np.round([x0*iw,x1*iw]),0,iw).astype(int)
            y0,y1 = np.clip(np.round([y0*ih,y1*ih]),0,ih).astype(int)
            if x1-x0 < 10 or y1-y0 < 10:
                valid[i] = False
                continue
            faces[i] = cv2.cvtColor(cv2.resize(frame[y0:y1,x0:x1], (w,h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    prediction = np.full(len(rows), np.nan)
    completed = []
    for start, stop in segments(valid):
        if stop-start < max(round(10*fps), round(model.input[0]*fps/model.fps)):
            continue
        model.state = copy.deepcopy(initial_state)
        print('infer frames', start, stop, flush=True)
        model.process_faces_tensor(faces[start:stop], fps=fps)
        bvp, ts = model.bvp(raw=True)
        bvp, ts = np.asarray(bvp, float), np.asarray(ts, float)
        if len(bvp) < 2 or len(bvp) != len(ts) or not np.isfinite(bvp).all():
            raise ValueError('Invalid/missing model output')
        # Upstream may resample to model fps and repeat timestamps. Average ties.
        unique, inv = np.unique(ts, return_inverse=True)
        bvp = np.bincount(inv, weights=bvp)/np.bincount(inv)
        target = np.arange(stop-start)/fps
        out = np.interp(target, unique, bvp, left=np.nan, right=np.nan)
        edge = round(1.6*fps)
        prediction[start+edge:stop-edge] = out[edge:-edge]
        completed.append(dict(start_frame=int(start),stop_frame=int(stop),statistics=dict(model.statistic)))
    with (a.output/'neural_waveform.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','bvp_raw','observed','interpolated_roi'])
        writer.writerows(zip(np.arange(len(rows))/fps,prediction,observed,interpolated))
    weights=[]
    for path in (Path(rppg.__file__).parent/'weights').rglob('*'):
        if path.is_file():
            weights.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    info=dict(model=a.model,open_rppg_version=version('open-rppg'),fps=fps,
              model_meta=model.meta,trace=str(a.trace),input_identity=identity,
              finite_prediction_fraction=float(np.isfinite(prediction).mean()),
              segments=completed,packaged_weights=weights,
              warning='Independent converted-weight comparison. Not official PyTorch replication. No claim of exercise accuracy.')
    (a.output/'neural_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('completed', len(completed), 'segments', flush=True)


if __name__=='__main__':
    main()
