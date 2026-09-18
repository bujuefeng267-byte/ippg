import json
import threading
from collections import deque
import csv
import io
from pathlib import Path
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import numpy as np
from app import clean, handler_for, Session


class OutputTests(unittest.TestCase):
    def test_missing_then_valid_hr_share_csv_schema_and_preserve_missing_wave(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session.__new__(Session)
            s.output=Path(folder)
            s.lock=threading.Lock()
            s.start_wall=time.perf_counter()
            s.last_time=12.
            s.event_index=0
            s.hr_writer=None
            s.hr_handle=io.StringIO()
            s.ppg_handle=io.StringIO()
            s.ppg_writer=csv.DictWriter(s.ppg_handle,fieldnames=('window_index','window_start_s','window_end_s','sample_time_s','bvp_au','covered','observed','interpolated'))
            s.ppg_writer.writeheader()
            s.events=deque(maxlen=120)
            wave=dict(time_s=[0.,1.],values=[None,1.],covered=[False,True],observed=[False,True],interpolated=[False,False])
            event=dict(window_start_s=0.,window_end_s=10.,time_s=5.,hr_bpm=None,accepted=False,status='missing',waveform=wave)
            s._publish(event)
            s._publish(dict(event,hr_bpm=80.,accepted=True,status='candidate_only',local_evidence_bpm=80.,peak_concentration=.5))
            rows=list(csv.DictReader(io.StringIO(s.hr_handle.getvalue())))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['hr_bpm'],'')
            self.assertEqual(rows[1]['hr_bpm'],'80.0')
            samples=list(csv.DictReader(io.StringIO(s.ppg_handle.getvalue())))
            self.assertEqual(samples[0]['bvp_au'],'')
            self.assertEqual(samples[0]['covered'],'False')


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service=SimpleNamespace(token='test-token',current=None)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.service))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_absence_is_json_null(self):
        value=clean({'wave':[np.nan,np.inf,np.float64(1)],'flag':np.bool_(True)})
        self.assertEqual(json.loads(json.dumps(value,allow_nan=False)),{'wave':[None,None,1.],'flag':True})

    def test_public_page_has_local_token(self):
        page=urlopen(self.url).read().decode()
        self.assertIn("const token='test-token'",page)
        self.assertIn('getUserMedia',page)

    def test_no_frame_without_token(self):
        with self.assertRaises(HTTPError) as cm:
            urlopen(Request(self.url+'/api/frame',data=b'x'))
        self.assertEqual(cm.exception.code,403)

    def test_cross_origin_rejected(self):
        with self.assertRaises(HTTPError) as cm:
            urlopen(Request(self.url+'/api/start',data=b'',headers={'X-RPPG-Token':'test-token','Origin':'https://unrelated.invalid'}))
        self.assertEqual(cm.exception.code,403)

    def test_missing_session_rejected(self):
        with self.assertRaises(HTTPError) as cm:
            urlopen(Request(self.url+'/api/frame',data=b'x',headers={'X-RPPG-Token':'test-token'}))
        self.assertEqual(cm.exception.code,409)

    def test_stopped_status(self):
        data=json.load(urlopen(Request(self.url+'/api/status',headers={'X-RPPG-Token':'test-token'})))
        self.assertFalse(data['active'])


if __name__=='__main__':
    unittest.main()
