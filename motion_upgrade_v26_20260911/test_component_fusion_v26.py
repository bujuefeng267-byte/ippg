import unittest
import numpy as np
import pandas as pd
from component_fusion_v26 import measured_component,fuse_components,ROIS

def fixture(fps=30):
    t=np.arange(24*fps)/fps
    trace=pd.DataFrame(dict(time_s=t,motion_x=np.zeros(len(t)),motion_y=np.zeros(len(t)),face_y0=.2,face_y1=.6))
    rw=pd.DataFrame(dict(time_s=t));channels={}
    for i,roi in enumerate(ROIS):
        trace[f'{roi}_quality']=1.;trace[f'{roi}_pixel_source']='tracked_ratio';trace[f'{roi}_valid']=True
        rw[f'{roi}_observed']=True;rw[f'{roi}_interpolated']=False
        channels[f'baseline/{roi}/pos']=np.sin(2*np.pi*1.4*t+.02*i)
    return channels,trace,rw

class ComponentTests(unittest.TestCase):
    def test_actual_coefficients_and_linearity(self):
        fps=30;t=np.arange(900)/fps;a=np.sin(2*np.pi*1.4*t);b=.5*np.sin(2*np.pi*3*t+.6)
        np.testing.assert_allclose(measured_component(2*a+3*b,fps,84),2*measured_component(a,fps,84)+3*measured_component(b,fps,84),atol=1e-12)
        center=measured_component(a+b,fps,84)[200:-200]
        self.assertGreater(np.corrcoef(center,a[200:-200])[0,1],.99)
    def test_phase_and_no_oscillator(self):
        x=np.zeros(300);np.testing.assert_array_equal(measured_component(x,30,90),x)
        t=np.arange(900)/30;a=np.cos(2*np.pi*1.4*t+.63)
        self.assertGreater(np.corrcoef(measured_component(a,30,84)[200:-200],a[200:-200])[0,1],.99)
    def test_pure_pulse_and_no_truth_dependency(self):
        ch,tr,rw=fixture();w,p,d=fuse_components(ch,tr,rw,30)
        self.assertTrue(p.generated.all());self.assertLess(abs(p.proposal_bpm.median()-84),2)
        altered=tr.assign(reference_bpm=177,polar_hr=192)
        w2,p2,d2=fuse_components(ch,altered,rw,30)
        pd.testing.assert_frame_equal(w,w2);pd.testing.assert_frame_equal(p,p2)
    def test_same_roi_duplicates_are_not_votes(self):
        ch,tr,rw=fixture();one=next(iter(ch.values()));many={f'{branch}/forehead/{method}':one for branch in ('baseline','tracked') for method in ('pos','chrom')}
        w,p,_=fuse_components(many,tr,rw,30);self.assertFalse(p.generated.any());self.assertTrue(w.base.isna().all())
    def test_gap_not_filled(self):
        ch,tr,rw=fixture()
        for key in ch:ch[key][240:360]=np.nan
        w,p,_=fuse_components(ch,tr,rw,30)
        self.assertTrue(w.base.iloc[240:360].isna().all());self.assertFalse(p.proposal_supported.all())
    def test_bad_tracking_cannot_vote(self):
        ch,tr,rw=fixture();ch={k.replace('baseline/','tracked/'):v for k,v in ch.items()}
        for roi in ROIS:tr[f'{roi}_pixel_source']='baseline_reset'
        w,p,_=fuse_components(ch,tr,rw,30);self.assertFalse(p.generated.any())
    def test_anti_phase_can_be_aligned(self):
        ch,tr,rw=fixture();ch['baseline/left_cheek/pos']*=-1
        w,p,_=fuse_components(ch,tr,rw,30);self.assertTrue(p.generated.all());self.assertTrue(np.isfinite(w.base).all())

if __name__=='__main__':unittest.main()
