#!/usr/bin/env python3
"""Loopback-only camera bridge and bounded, two-stage live rPPG service."""
from __future__ import annotations
import argparse
from collections import deque
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import secrets
import sys
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from backend_registry import MODES, selected_mode, startup_seconds, mode_description

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('RPPG_PROJECT_ROOT', HERE.parent)).resolve()
# Staging can use the existing frontend/backend without copying or modifying
# them; a packaged release resolves its own sibling modules first.
if not (HERE/'frontend.py').is_file():
    runtime = PROJECT_ROOT/'realtime_rppg_20260918'
    if runtime.is_dir():
        sys.path.append(str(runtime))
MAX_FRAME_PIXELS = 4_000_000
RAW_CONTENT_TYPE = 'application/x-rppg-rgba'
HR_FIELDS = ('window_start_s','window_end_s','time_s','hr_bpm','accepted','status',
    'observed_fraction','interpolated_fraction','source_hz','selected_branch',
    'waveform_coverage_fraction','fusion_status','peak_concentration',
    'local_evidence_bpm','selected_current_evidence','motion_available',
    'emitted_at_source_s','input_delay_s','skipped_update_count','processing_ms',
    'variant','reference_used','event_index','published_elapsed_wall_s',
    'input_queue_lag_s','window_file')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')


def raw_frame_size(width, height):
    """Strict dimensions before a body allocation; reject coerced fractions."""
    if not isinstance(width, str) or not width.isascii() or not width.isdecimal():
        raise ValueError('X-Frame-Width must be a decimal integer')
    if not isinstance(height, str) or not height.isascii() or not height.isdecimal():
        raise ValueError('X-Frame-Height must be a decimal integer')
    w, h = int(width), int(height)
    if not 1 <= w <= 4096 or not 1 <= h <= 4096 or w*h > MAX_FRAME_PIXELS:
        raise ValueError('Raw frame dimensions exceed four megapixels or 4096 pixels per axis')
    return w, h, w*h*4


def decode_frame(body, content_type, width=None, height=None):
    """Decode a canvas RGBA8 byte buffer or the legacy JPEG98 transport.

    RGBA transport exactly preserves browser-canvas RGB bytes, not native
    camera sensor bytes. Canvas drawing, browser color conversion and optional
    resizing already occurred before this boundary. Alpha is intentionally
    discarded because the video canvas represents an opaque image.
    """
    mime = (content_type or 'image/jpeg').split(';',1)[0].strip().lower()
    if mime == RAW_CONTENT_TYPE:
        w, h, length = raw_frame_size(width, height)
        if len(body) != length:
            raise ValueError(f'Raw RGBA length mismatch: expected {length}, received {len(body)}')
        rgba = np.frombuffer(body,np.uint8).reshape(h,w,4)
        return cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR), 'rgba8'
    if mime != 'image/jpeg':
        raise ValueError('Expected application/x-rppg-rgba or image/jpeg')
    if not body.startswith(b'\xff\xd8'):
        raise ValueError('Expected a JPEG frame')
    frame = cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[0]*frame.shape[1] > MAX_FRAME_PIXELS:
        raise ValueError('Expected a JPEG smaller than four megapixels')
    return frame, 'jpeg'


class Session:
    """Frontend never waits for the ten-second-window backend.

    A full queue drops oldest rows, preserving timestamps; the backend therefore
    sees the gap and cannot convert the missed frames into fabricated samples.
    """
    def __init__(self, output, max_width=960, label='browser_camera'):
        from frontend import RealtimeFrontend
        from backend_registry import create_backend as StreamingBackend
        self.id = secrets.token_hex(12)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.frontend = RealtimeFrontend(max_width=max_width)
        self.backend = StreamingBackend()
        self.rows = queue.Queue(maxsize=300)
        self.lock = threading.Lock()
        self.ingest_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active = True
        self.error = None
        self.frames = self.dropped = 0
        self.event_index = 0
        self.events = deque(maxlen=120)
        self.frontend_ms = deque(maxlen=300)
        self.transport_decode_ms = deque(maxlen=300)
        self.transport_counts = {}
        self.transport_bytes = 0
        self.times = deque(maxlen=120)
        self.last_time = None
        self.last_ingest_wall = None
        self.latest = None
        self.start_wall = time.perf_counter()
        self.trace_handle = (self.output/'frame_trace.csv').open('w', newline='', encoding='utf-8')
        self.trace_writer = None
        self.hr_handle = (self.output/'heart_rate.csv').open('w', newline='', encoding='utf-8')
        self.hr_writer = None
        self.ppg_handle = (self.output/'ppg_windows.csv').open('w',newline='',encoding='utf-8')
        self.ppg_writer = csv.DictWriter(self.ppg_handle,fieldnames=('window_index','window_start_s','window_end_s',
            'sample_time_s','bvp_au','covered','observed','interpolated'))
        self.ppg_writer.writeheader()
        self.manifest = dict(schema='realtime_rppg_v1', source=label, max_width=max_width,
            mode='past_only_trailing_window', reference_used_by_inference=False,
            raw_camera_images_saved=False, start_local=datetime.now().isoformat(),
            clock='source_capture_timestamp_seconds', waveform_kind='independent_trailing_windows',
            offline_default_replaced=False)
        self.manifest['backend_configuration'] = self.backend.configuration
        dump(self.output/'session.json', self.manifest)
        self.thread = threading.Thread(target=self._consume, name='rppg-window-worker', daemon=True)
        self.thread.start()

    def ingest(self, bgr, timestamp, *, transport='direct', byte_count=0, decode_ms=0.):
        with self.ingest_lock:
            if not self.active or self.error:
                raise ValueError(self.error or 'Session is stopped')
            if not np.isfinite(timestamp) or timestamp < 0 or (self.last_time is not None and timestamp <= self.last_time):
                raise ValueError('Capture timestamps must be finite and strictly increasing')
            if self.last_time is None and timestamp > .5:
                raise ValueError('First frame timestamp must start near zero')
            started = time.perf_counter()
            row = self.frontend.process(bgr, float(timestamp))
            cost_ms = (time.perf_counter()-started)*1000
            if self.trace_writer is None:
                self.trace_writer = csv.DictWriter(self.trace_handle, fieldnames=list(row))
                self.trace_writer.writeheader()
            self.trace_writer.writerow(row)
            self.trace_handle.flush()
            with self.lock:
                self.last_time = float(timestamp)
                self.last_ingest_wall = time.perf_counter()
                self.frames += 1
                self.times.append(timestamp)
                self.frontend_ms.append(cost_ms)
                self.transport_decode_ms.append(decode_ms)
                self.transport_counts[transport] = self.transport_counts.get(transport,0)+1
                self.transport_bytes += byte_count
            try:
                self.rows.put_nowait(row)
            except queue.Full:
                try:
                    self.rows.get_nowait()
                    self.rows.task_done()
                    self.dropped += 1
                except queue.Empty:
                    pass
                self.rows.put_nowait(row)
            return dict(frontend_ms=cost_ms, queue_rows=self.rows.qsize(),
                        transport=transport, decode_ms=decode_ms, frame_bytes=byte_count)

    def _consume(self):
        try:
            while not self.stop_event.is_set() or not self.rows.empty():
                try:
                    row = self.rows.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    event = self.backend.append(row)
                    if event is not None:
                        self._publish(event)
                finally:
                    self.rows.task_done()
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
            self.stop_event.set()

    def _publish(self, event):
        event = clean(event)
        with self.lock:
            event['published_elapsed_wall_s'] = time.perf_counter()-self.start_wall
            event['input_queue_lag_s'] = max(0., (self.last_time or 0.)-event.get('emitted_at_source_s',event['window_end_s']))
            index = self.event_index
            self.event_index += 1
            event['window_file'] = f'window_{index:05d}.json'
            dump(self.output/event['window_file'], event)
            # CSV contains scalar readout/provenance; JSON preserves exact waveform.
            scalar = {k:event.get(k) for k in HR_FIELDS}
            if self.hr_writer is None:
                self.hr_writer = csv.DictWriter(self.hr_handle, fieldnames=list(scalar))
                self.hr_writer.writeheader()
            self.hr_writer.writerow(scalar)
            self.hr_handle.flush()
            wave=event['waveform']
            for j,t in enumerate(wave['time_s']):
                self.ppg_writer.writerow(dict(window_index=index,window_start_s=event['window_start_s'],
                    window_end_s=event['window_end_s'],sample_time_s=t,bvp_au=wave['values'][j],
                    covered=wave['covered'][j],observed=wave['observed'][j],interpolated=wave['interpolated'][j]))
            self.ppg_handle.flush()
            self.latest = event
            self.events.append({k: event.get(k) for k in ('window_end_s','hr_bpm','accepted','status')})

    def status(self):
        with self.lock:
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0.
            age = None if self.latest is None else max(0., (self.last_time or 0.)-self.latest['window_end_s'])
            return clean(dict(session=self.id, active=self.active, error=self.error,
                mode=selected_mode(), startup_s=startup_seconds(),
                expected_delay_s=self.backend.configuration.get('fixed_delay_s',0.),
                frames=self.frames, source_elapsed_s=self.last_time or 0., input_fps=fps,
                frontend_p95_ms=float(np.percentile(self.frontend_ms,95)) if self.frontend_ms else 0.,
                transport_decode_p95_ms=float(np.percentile(self.transport_decode_ms,95)) if self.transport_decode_ms else 0.,
                transport_counts=dict(self.transport_counts), transport_bytes=self.transport_bytes,
                queue_rows=self.rows.qsize(), dropped_rows=self.dropped,
                latest_age_s=age, input_stale_wall_s=(time.perf_counter()-self.last_ingest_wall if self.last_ingest_wall else None),
                latest=self.latest, history=list(self.events), output=str(self.output)))

    def close(self):
        with self.ingest_lock:
            self.active = False
            self.stop_event.set()
            self.frontend.close()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Backend did not finish; output files are still open')
        self.trace_handle.close()
        self.hr_handle.close()
        self.ppg_handle.close()
        summary = self.status()
        self.manifest.update(summary=summary, end_local=datetime.now().isoformat())
        dump(self.output/'session.json', self.manifest)
        return summary


class Service:
    def __init__(self, output, width):
        self.output, self.width = Path(output), width
        self.token = secrets.token_urlsafe(32)
        self.current = None
        self.control = threading.Lock()

    def start(self):
        with self.control:
            if self.current and self.current.active:
                raise ValueError('A camera session is already active; stop it first')
            name = datetime.now().strftime('%Y%m%d_%H%M%S_')+secrets.token_hex(3)
            self.current = Session(self.output/name, self.width)
            return self.current.status()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        # Browsers can reuse the loopback connection for successive frames.
        # Every response carries Content-Length. Rejections close the socket
        # because a rejected POST may still have an unread request body.
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True

        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            body = json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            if status >= 400:
                self.close_connection = True
                self.send_header('Connection','close')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if self.headers.get('X-RPPG-Token') != service.token:
                return False
            origin = self.headers.get('Origin')
            return not origin or urlparse(origin).netloc == self.headers.get('Host')

        def do_GET(self):
            if self.path == '/':
                host = self.headers.get('Host','').split(':')[0]
                if host not in ('localhost','127.0.0.1'):
                    return self.reply({'error':'Localhost only'},403)
                body = (HERE/'index.html').read_text(encoding='utf-8').replace('__TOKEN__', service.token).replace('__STARTUP__',str(startup_seconds())).replace('__MODE__',mode_description()).encode()
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Frame-Options','DENY')
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/status' and self.authorized():
                self.reply(service.current.status() if service.current else {'active':False, 'startup_s':startup_seconds(), 'mode':selected_mode()})
            else:
                self.reply({'error':'Not found'},404)

        def do_POST(self):
            if not self.authorized():
                return self.reply({'error':'Invalid local session token'},403)
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 <= length <= MAX_FRAME_PIXELS*4:
                    return self.reply({'error':'Frame body too large'},413)
                if self.path == '/api/frame' and self.headers.get('Content-Type','').split(';',1)[0].strip().lower() == RAW_CONTENT_TYPE:
                    _, _, expected = raw_frame_size(self.headers.get('X-Frame-Width'),self.headers.get('X-Frame-Height'))
                    if length != expected:
                        raise ValueError(f'Raw RGBA Content-Length must equal {expected}')
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Truncated request body')
                if self.path == '/api/start':
                    return self.reply(service.start())
                current = service.current
                if current is None or self.headers.get('X-Session') != current.id:
                    return self.reply({'error':'No matching session'},409)
                if self.path == '/api/stop':
                    return self.reply(current.close())
                if self.path == '/api/frame':
                    began = time.perf_counter()
                    frame, transport = decode_frame(body,self.headers.get('Content-Type'),
                        self.headers.get('X-Frame-Width'),self.headers.get('X-Frame-Height'))
                    decode_ms = (time.perf_counter()-began)*1000
                    return self.reply(current.ingest(frame, float(self.headers['X-Capture-Time']),
                        transport=transport,byte_count=length,decode_ms=decode_ms))
                self.reply({'error':'Not found'},404)
            except (ValueError, KeyError) as exc:
                self.reply({'error':str(exc)},400)
            except Exception as exc:
                self.reply({'error':f'{type(exc).__name__}: {exc}'},500)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--mode',choices=MODES,default=selected_mode())
    parser.add_argument('--out',type=Path,default=HERE.parent/'results'/'realtime')
    parser.add_argument('--max-width',type=int,default=960)
    args = parser.parse_args()
    os.environ['RPPG_REALTIME_MODE']=args.mode
    if not 320 <= args.max_width <= 1920:
        parser.error('--max-width must be 320..1920')
    cv2.setNumThreads(1)
    service = Service(args.out,args.max_width)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(service))
    print(f'Open http://127.0.0.1:{args.port} in a desktop browser. Ctrl+C stops the service.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if service.current and service.current.active:
            service.current.close()
        server.server_close()


if __name__ == '__main__':
    main()
