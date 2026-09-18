from pathlib import Path
import os
import sys
import unittest
import numpy as np
from fast_kernels import pos_projection, overlap_count

P=Path(os.environ.get('RPPG_PROJECT_ROOT',str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0,str(P/'motion_upgrade_v28_20260912'))
from analyze_rppg import pos_signal

class FastTests(unittest.TestCase):
    def test_numerical_equivalence_including_constant_and_near_constant(self):
        rng=np.random.default_rng(20260918)
        for fps in (20.,30.,120.,180.):
            for size in (round(1.6*fps),round(10*fps)):
                for amplitude in (0.,1e-8,.5,10.):
                    rgb=np.array([90.,110.,80.])+amplitude*rng.normal(size=(size,3))
                    np.testing.assert_allclose(pos_projection(rgb,fps),pos_signal(rgb,fps),rtol=2e-10,atol=2e-10)

    def test_normalized_projection_matches_explicit_contribution_count(self):
        rng=np.random.default_rng(2)
        rgb=np.array([90.,110.,80.])+rng.normal(size=(300,3))
        expected=pos_signal(rgb,30.)/overlap_count(300,30.)
        np.testing.assert_allclose(pos_projection(rgb,30.,True),expected,rtol=2e-10,atol=2e-10)

    def test_no_missing_colour_fabrication(self):
        x=np.ones((300,3))*100
        x[25]=np.nan
        with self.assertRaises(ValueError):
            pos_projection(x,30.)

if __name__=='__main__':unittest.main()
