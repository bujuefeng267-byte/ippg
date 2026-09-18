#!/usr/bin/env python3
"""Camera/RTSP or file replay through the same live frontend and backend."""
import argparse
import os
from backend_registry import MODES, selected_mode
import json
from pathlib import Path
import threading
import time
import cv2
from app import Session, dump


class LatestFrameCapture:
    """Continuously drain a live source into one replaceable frame slot.

    Arrival timestamps are taken by the capture thread, independently of slow
    downstream processing. They are NOT verified hardware exposure timestamps;
    camera/RTSP device and transport buffers can still introduce delay/jitter.
    The capture thread alone owns read/release to avoid releasing a decoder
    concurrently with a blocked read. An unresponsive read may outlive close's
    timeout; this is reported and the daemon exits once that read returns.
    """
    def __init__(self, cap, clock=time.perf_counter):
        self.cap, self.clock = cap, clock
        self.condition = threading.Condition()
        self.stopping = False
        self.finished = False
        self.error = None
        self.pending = None
        self.read_frames = self.overwritten_frames = 0
        try:
            self.driver_buffer_request_accepted = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
        except (AttributeError, cv2.error):
            self.driver_buffer_request_accepted = False
        self.thread = threading.Thread(target=self._capture, name='rppg-live-capture', daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while True:
                with self.condition:
                    if self.stopping:
                        break
                ok, frame = self.cap.read()
                arrival = self.clock()
                if not ok:
                    break
                with self.condition:
                    if self.stopping:
                        break
                    self.read_frames += 1
                    if self.pending is not None:
                        self.overwritten_frames += 1
                    self.pending = (arrival, frame)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                self.cap.release()
            finally:
                with self.condition:
                    self.finished = True
                    self.condition.notify_all()

    def read(self, timeout_s=5.):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.pending is None and not self.finished and not self.stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Live source delivered no frame for five seconds')
                self.condition.wait(remaining)
            if self.error:
                raise RuntimeError(self.error)
            if self.pending is None:
                return None
            latest, self.pending = self.pending, None
            return latest

    def close(self, timeout_s=2.):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout_s)
        return not self.thread.is_alive()

    def snapshot(self):
        with self.condition:
            return dict(capture_read_frames=self.read_frames,
                        capture_overwritten_frames=self.overwritten_frames,
                        capture_queue_capacity=1,
                        capture_pending_frames=int(self.pending is not None),
                        capture_worker_finished=self.finished,
                        driver_buffer_request_accepted=self.driver_buffer_request_accepted,
                        hardware_capture_timestamps_verified=False,
                        device_and_transport_latency_measured=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=MODES,default=selected_mode())
    parser.add_argument('--source',required=True,help='Local video, RTSP URL, or camera:0 (camera visible to this OS)')
    parser.add_argument('--out',required=True,type=Path,help='New output directory')
    parser.add_argument('--max-seconds',type=float,default=None)
    parser.add_argument('--max-width',type=int,default=960)
    parser.add_argument('--realtime',action='store_true',help='Pace a file using its declared FPS; no effect on live sources')
    args=parser.parse_args()
    os.environ['RPPG_REALTIME_MODE']=args.mode
    cv2.setNumThreads(1)
    live=args.source.startswith(('camera:','rtsp://','rtsps://','http://','https://'))
    source=int(args.source.split(':')[1]) if args.source.startswith('camera:') else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError('Cannot open source. For a Windows webcam use the browser entry instead of a WSL camera index.')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not live and not 7 < fps <= 1000:
        cap.release()
        raise ValueError('File replay needs valid declared FPS greater than 7')
    reader=LatestFrameCapture(cap) if live else None
    try:
        session=Session(args.out,args.max_width,label='live_cv2' if live else 'file_stream_replay')
    except BaseException:
        if reader:
            reader.close()
        else:
            cap.release()
        raise
    # Only file replay assumes a CFR frame-index clock; never assign nominal FPS to live frames.
    session.manifest.update(clock='independent_capture_thread_read_completion' if live else 'CFR_frame_index_over_declared_fps',
        declared_source_fps=fps, paced_file_replay=bool(args.realtime and not live))
    if live:
        session.manifest.update(capture_queue_capacity=1, capture_policy='keep_latest_drop_oldest',
            hardware_capture_timestamps_verified=False, device_and_transport_latency_measured=False)
    origin=None
    started=time.perf_counter()
    i=0
    try:
        while True:
            if reader:
                captured=reader.read()
                if captured is None:
                    break
                arrival,frame=captured
            else:
                ok,frame=cap.read()
                arrival=time.perf_counter()
                if not ok:
                    break
            if origin is None:
                origin=arrival
            timestamp=arrival-origin if live else i/fps
            if args.max_seconds is not None and timestamp >= args.max_seconds:
                break
            if args.realtime and not live:
                remaining=started+timestamp-time.perf_counter()
                if remaining>0:
                    time.sleep(remaining)
            session.ingest(frame,timestamp)
            i+=1
            if i%300==0:
                print(json.dumps({'frames':i,'source_s':timestamp,'queue_rows':session.rows.qsize()},ensure_ascii=False),flush=True)
            if session.error:
                raise RuntimeError(session.error)
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.close()
            session.manifest.update(live_capture=reader.snapshot())
        else:
            cap.release()
        summary=session.close()
        elapsed=time.perf_counter()-started
        summary.update(total_wall_s=elapsed, end_to_end_fps=i/elapsed if elapsed else None,
            measured_frames=i, source_duration_s=(i/fps if not live else session.last_time),
            includes_decode_frontend_backend_and_output=True,
            note='Unpaced file throughput is computational capacity, not measured camera-to-display latency.')
        if reader:
            summary.update(live_capture=reader.snapshot(),
                note='Live timestamps measure independent read arrival; hardware exposure timestamps and device/transport latency remain unverified.')
        dump(args.out/'benchmark.json',summary)
        print(json.dumps({k:summary[k] for k in ('measured_frames','end_to_end_fps','total_wall_s','dropped_rows','error')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
