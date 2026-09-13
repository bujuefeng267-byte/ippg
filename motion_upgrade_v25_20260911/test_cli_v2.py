"""Regression tests for empty reference scoring and fractional-FPS minimum windows."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import analyze_motion_v2 as cli


def synthetic_trace(fps,n):
    t=np.arange(n)/fps
    trace=pd.DataFrame(dict(frame=np.arange(n),time_s=t,r=140+.2*np.sin(2*np.pi*1.2*t),
        g=115+.8*np.sin(2*np.pi*1.2*t),b=100+.1*np.sin(2*np.pi*1.2*t),
        rgb_valid=True,source='mesh',motion_x=0.,motion_y=0.))
    for roi in cli.ROIS:
        for c in 'rgb': trace[f'{roi}_{c}']=trace[c]
        trace[f'{roi}_valid']=True
        trace[f'{roi}_quality']=1.
    return trace


class CLIBoundaryTests(unittest.TestCase):
    def run_case(self,fps,n,window,reference):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);video=root/'identity_only.avi';video.write_bytes(b'fixture')
            out=root/'result'
            args=['analyze_motion_v2.py',str(video),'--output',str(out),'--window',str(window)]
            if reference:
                t=np.arange(0,60,.02);path=root/'ground_truth.txt'
                np.savetxt(path,np.vstack([np.sin(2*np.pi*1.2*t),np.full(len(t),72.),t]))
                args+=['--reference-ubfc',str(path)]
            with patch.object(cli,'extract',return_value=(synthetic_trace(fps,n),fps)), \
                    patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            return json.loads((out/'summary.json').read_text()),{
                name:pd.read_csv(out/f'{name}_heart_rate.csv') for name in ('pos','chrom','fusion')}

    def test_short_video_with_reference_reports_no_scores(self):
        summary,tables=self.run_case(30.,90,10,True)
        for name in tables:
            self.assertEqual(len(tables[name]),0)
            self.assertIn('reference_bpm',tables[name])
            for value in summary['variants'][name]['reference'].values():
                self.assertEqual(value['n'],0)
                self.assertIsNone(value['mae_bpm'])
                self.assertIsNone(value['rmse_bpm'])

    def test_minimum_window_rounds_up_at_fractional_fps(self):
        summary,tables=self.run_case(30.00003000003,450,6,False)
        self.assertEqual(summary['requested_window_s'],6)
        self.assertAlmostEqual(summary['effective_window_s'],181/30.00003000003)
        self.assertGreaterEqual(summary['effective_window_s'],6)
        self.assertEqual(len(tables['pos']),len(tables['fusion']))

if __name__=='__main__':unittest.main()
