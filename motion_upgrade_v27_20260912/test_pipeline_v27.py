import unittest
import numpy as np
from test_prepare_patch_signals_v27 import fixture
from prepare_patch_signals_v27 import anchor_patches,prepare_signals
from patch_hr_v27 import infer_patch_hr
from run_v27 import readout

class PipelineContracts(unittest.TestCase):
    def test_spatial_and_readout_contracts(self):
        p,frames=fixture(30,24)
        anchored=anchor_patches(p,frames,30)
        for spatial in ('early','late'):
            args,_,_=prepare_signals(anchored,frames,30,spatial)
            for mode in ('median','psd_cluster'):
                wave,hr,_=infer_patch_hr(*args,frames,30,mode=mode)
                self.assertEqual(len(hr),15)
                self.assertTrue(hr.accepted.any())
                self.assertLess(float(np.nanmax(abs(hr.ridge_bpm-84))),2.)
                self.assertEqual(len(wave),len(frames))
                self.assertTrue(np.array_equal(wave.covered,np.isfinite(wave.base)))
                for row in hr[hr.accepted].itertuples():
                    start=round(row.window_start_s*30)
                    self.assertTrue(np.isfinite(wave.base.iloc[start:start+300]).all())
                for motion in (False,True):
                    for temporal in (False,True):
                        secondary=readout(wave,frames,30,motion,temporal)
                        self.assertEqual(len(secondary),len(hr))
                        self.assertTrue(secondary.loc[~secondary.accepted,'ridge_bpm'].isna().all())
                        self.assertTrue(secondary.accepted.any())
                        self.assertLess(float(np.nanmax(abs(secondary.ridge_bpm-84))),2.)

if __name__=='__main__':unittest.main()
