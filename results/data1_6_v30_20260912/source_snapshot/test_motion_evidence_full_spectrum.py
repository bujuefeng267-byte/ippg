"""Synthetic mechanism and preservation contracts; no real videos or references."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0,str(Path('/home/fengbujue/项目/rppg识别/motion_upgrade_v28_20260912')))
from motion_evidence import motion_evidence as old_motion
from evidence_hr import estimate_evidence
from motion_evidence_full_spectrum import motion_evidence, NORMALIZATION_CONFIG


def trace_for(bpm=20.,fps=30.,seconds=10.,speed=.5):
    t=np.arange(round(fps*seconds))/fps
    v=np.sqrt(2)*speed*np.sin(2*np.pi*bpm/60*t)
    return pd.DataFrame(dict(time_s=t,face_y0=.2,face_y1=.6,
        motion_x=v*.4/fps,motion_y=np.zeros(len(t))))


class FullSpectrumMotionTests(unittest.TestCase):
    def test_only_normalization_changes_out_of_band_risk(self):
        grid=np.arange(42.,211.)
        for bpm in (20.,30.,240.):
            trace=trace_for(bpm)
            old=old_motion(trace,0,len(trace),30.,grid)
            new=motion_evidence(trace,0,len(trace),30.,grid)
            self.assertGreater(old['profile'].max(),.75)
            self.assertLess(new['profile'].max(),.001)
            self.assertTrue((new['profile']<=old['profile']+1e-15).all())
            for key in ('available','observed_fraction','speed_face_per_s','strength','reliability','spectral_coverage','status'):
                self.assertEqual(new[key],old[key])

    def test_genuine_in_band_peak_is_preserved_at_both_sample_rates(self):
        grid=np.arange(42.,211.)
        for fps in (30.00003000003,180.001800018):
            for bpm in (60.,120.):
                trace=trace_for(bpm,fps)
                old=old_motion(trace,0,len(trace),fps,grid)
                new=motion_evidence(trace,0,len(trace),fps,grid)
                self.assertEqual(grid[new['profile'].argmax()],bpm)
                self.assertLess(abs(new['profile'].max()-old['profile'].max()),.001)

    def test_unavailable_and_flat_motion_are_not_reclassified(self):
        grid=np.arange(42.,211.)
        for mode in ('missing','fragmented','flat','steady_translation'):
            trace=trace_for()
            if mode=='missing':trace['motion_x']=np.nan
            elif mode=='fragmented':trace.loc[::30,'motion_x']=np.nan
            elif mode=='flat':trace['motion_x']=0.
            else:trace['motion_x']=.4/30
            old=old_motion(trace,0,len(trace),30.,grid)
            new=motion_evidence(trace,0,len(trace),30.,grid)
            np.testing.assert_array_equal(new['profile'],old['profile'])
            for key in ('available','observed_fraction','spectral_coverage','status'):
                self.assertEqual(new[key],old[key])
        self.assertTrue(NORMALIZATION_CONFIG['include_dc'])
        self.assertEqual(NORMALIZATION_CONFIG['detrend'],'constant')

    def test_two_axes_and_different_block_ffts_are_length_weighted_before_peak(self):
        fps=30.;trace=trace_for(20.,fps,76.)
        trace.loc[2100:2159,['motion_x','motion_y']]=np.nan
        trace.loc[2160:,'motion_x']=np.sqrt(2)*.7*np.sin(2*np.pi*1.5*trace.time_s.iloc[2160:])*.4/fps
        trace.loc[:2099,'motion_y']=np.sqrt(2)*.3*np.sin(2*np.pi*2*trace.time_s.iloc[:2100])*.4/fps
        grid=np.arange(42.,211.)
        actual=motion_evidence(trace,0,len(trace),fps,grid)
        blocks=[]
        for a,b in ((0,2100),(2160,2280)):
            nfft=max(8192,2**int(np.ceil(np.log2(max(4*(b-a),256*fps)))))
            powers=[]
            for axis in ('motion_x','motion_y'):
                x=trace[axis].iloc[a:b].to_numpy()*fps/.4
                f,p=welch(x,fs=fps,window='hann',nperseg=b-a,noverlap=0,nfft=nfft,detrend='constant')
                powers.append(p)
            blocks.append((f,np.sum(powers,axis=0),b-a))
        common=blocks[0][0]
        complete=sum(np.interp(common,f,p)*n for f,p,n in blocks)/sum(n for _,_,n in blocks)
        self.assertAlmostEqual(actual['full_spectrum_peak_power'],complete.max(),places=12)
        expected_query=sum(np.interp(grid/60,f,p)*n for f,p,n in blocks)/sum(n for _,_,n in blocks)
        expected=expected_query/complete.max()*actual['strength']*actual['reliability']
        np.testing.assert_allclose(actual['profile'],expected,rtol=1e-12,atol=1e-14)

    def test_final_readout_keeps_waveform_masks_and_supported_candidates(self):
        fps=30.;trace=trace_for(30.,fps,24.)
        x=np.sin(2*np.pi*108/60*trace.time_s.to_numpy())+.5*np.sin(2*np.pi*54/60*trace.time_s.to_numpy())
        x[330:360]=np.nan
        before=x.copy();observed=pd.DataFrame({'rgb_valid':np.isfinite(x)});filled=np.zeros(len(x),bool)
        old=estimate_evidence(x,observed,filled,fps,motion_trace=trace,motion_provider=old_motion)
        new=estimate_evidence(x,observed,filled,fps,motion_trace=trace,motion_provider=motion_evidence)
        np.testing.assert_array_equal(x,before)
        np.testing.assert_array_equal(old.accepted,new.accepted)
        np.testing.assert_array_equal(old.status,new.status)
        np.testing.assert_array_equal(old.time_s,new.time_s)
        self.assertTrue(new.loc[~new.accepted,'ridge_bpm'].isna().all())
        for a,b in zip(old.evidence_candidates_json,new.evidence_candidates_json):
            clean=lambda s:sorted((r['bpm'],r['supported_bpm'],r['supported_relative_power']) for r in json.loads(s))
            self.assertEqual(clean(a),clean(b))


if __name__=='__main__':
    unittest.main()
