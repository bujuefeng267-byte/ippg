import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from prepare_patch_signals_v27 import (PATCHES,anchor_patches,spatial_inputs,
                                       prepare_signals,read_saved_signals,validate_inputs)

def fixture(fps=30,seconds=18):
    n=round(seconds*fps);t=np.arange(n)/fps
    frame=pd.DataFrame(dict(frame=np.arange(n),time_s=t,motion_x=0.,motion_y=0.,
                            face_x0=.2,face_y0=.1,face_x1=.7,face_y1=.8))
    rows=[]
    for j,name in enumerate(PATCHES):
        pulse=np.sin(2*np.pi*1.4*t+.01*j)
        rgb=np.array([110.,140.,90.])[None,:]+pulse[:,None]*np.array([.8,2.,.3])
        delta=np.zeros_like(rgb);delta[1:]=np.log(rgb[1:]/rgb[:-1])
        source=np.full(n,'tracked_ratio',dtype=object);source[0]='baseline_reset'
        one=pd.DataFrame(dict(frame=np.arange(n),time_s=t,patch_id=name,
                              region=name.rsplit('_',1)[0],valid=True,quality=1.,pixel_source=source,pixel_tracks=25))
        for i,c in enumerate('rgb'):
            one[f'raw_{c}']=rgb[:,i];one[f'tracked_{c}']=rgb[:,i];one[f'pixel_log_delta_{c}']=delta[:,i]
        rows.append(one)
    return pd.concat(rows,ignore_index=True),frame

class PreparationTests(unittest.TestCase):
    def test_common_measured_pulse_survives_anchor_at_both_frame_rates(self):
        for fps in (30,180):
            p,f=fixture(fps,5);a=anchor_patches(p,f,fps)
            np.testing.assert_allclose(a[['anchored_r','anchored_g','anchored_b']],
                                       a[['raw_r','raw_g','raw_b']],atol=1e-10,rtol=0)

    def test_fixed_patch_identity_and_complete_clock_enforced(self):
        p,f=fixture(30,1)
        with self.assertRaises(ValueError):validate_inputs(p.iloc[:-1],f,30)
        bad=p.copy();bad.loc[0,'region']='right_cheek'
        with self.assertRaises(ValueError):validate_inputs(bad,f,30)
        bad=p.copy();bad.loc[0,'raw_r']=np.nan
        with self.assertRaises(ValueError):validate_inputs(bad,f,30)

    def test_one_patch_missing_never_erases_other_histories(self):
        p,f=fixture();bad=(p.patch_id=='forehead_0')&p.frame.between(180,239)
        p.loc[bad,'valid']=False;p.loc[bad,'quality']=0;p.loc[bad,'pixel_source']='missing'
        for kind in ('raw','tracked','pixel_log_delta'):
            p.loc[bad,[f'{kind}_{c}' for c in 'rgb']]=np.nan
        a=anchor_patches(p,f,30)
        gap=a[(a.patch_id=='forehead_0')&a.frame.between(180,239)]
        self.assertTrue(gap.anchored_r.isna().all())
        next_row=a[(a.patch_id=='forehead_0')&(a.frame==240)].iloc[0]
        self.assertEqual(next_row.anchored_source,'baseline_reset')
        untouched=a[a.patch_id=='left_cheek_0']
        self.assertTrue(untouched.anchored_r.notna().all())
        early=spatial_inputs(a,f,'early')['forehead']
        self.assertTrue(early.valid.all());self.assertTrue((early.subpatch_count.iloc[180:240]==3).all())
        self.assertTrue((early.quality.iloc[180:240]==.75).all())

    def test_late_channels_retain_independent_local_perturbation(self):
        p,f=fixture();a=anchor_patches(p,f,30)
        before=spatial_inputs(a,f,'late');changed=a.copy()
        changed.loc[changed.patch_id=='forehead_0','raw_g']+=10
        after=spatial_inputs(changed,f,'late')
        pd.testing.assert_frame_equal(before['forehead_1'],after['forehead_1'])
        e0=spatial_inputs(a,f,'early');e1=spatial_inputs(changed,f,'early')
        np.testing.assert_allclose(e1['forehead'].raw_g-e0['forehead'].raw_g,2.5,atol=1e-12)
        pd.testing.assert_frame_equal(e0['left_cheek'],e1['left_cheek'])

    def test_saved_waveforms_and_masks_replay(self):
        p,f=fixture();a=anchor_patches(p,f,30)
        for mode,expected in [('early',3),('late',12)]:
            data,wave,meta=prepare_signals(a,f,30,mode)
            self.assertEqual(len(data[1]),expected);self.assertEqual(len(data[0]),4*expected)
            self.assertTrue(any(np.isfinite(x).any() for x in data[0].values()))
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'wave.csv';wave.to_csv(path,index=False)
                replay=read_saved_signals(path,meta)
            for key in data[0]:np.testing.assert_allclose(data[0][key],replay[0][key],atol=1e-12,equal_nan=True)
            for key in data[1]:np.testing.assert_array_equal(data[2][key],replay[2][key])

if __name__=='__main__':unittest.main()
