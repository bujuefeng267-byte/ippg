"""Ensure the optional continuity setting never creates long-gap observations."""
import unittest
import numpy as np
from legacy_motion import fill_short_gaps, gap_frame_limit

class GapPolicyTests(unittest.TestCase):
    def test_four_frame_dropout_is_explicit_interpolation_only_in_extended_mode(self):
        x=np.column_stack([np.linspace(90,100,100)]*3)
        x[40:44]=np.nan
        short,short_flags=fill_short_gaps(x,gap_frame_limit(.1,30.00003000003))
        extended,extended_flags=fill_short_gaps(x,gap_frame_limit(.15,30.00003000003))
        self.assertTrue(np.isnan(short[40:44]).all())
        self.assertFalse(short_flags.any())
        self.assertTrue(np.isfinite(extended).all())
        np.testing.assert_array_equal(np.flatnonzero(extended_flags),np.arange(40,44))
        np.testing.assert_array_equal(extended[:40],x[:40])
        np.testing.assert_array_equal(extended[44:],x[44:])

    def test_long_or_unbounded_dropouts_remain_missing(self):
        x=np.full((100,3),100.0)
        x[:2]=np.nan;x[40:50]=np.nan;x[-2:]=np.nan
        out,flags=fill_short_gaps(x,5)
        np.testing.assert_array_equal(np.isnan(out),np.isnan(x))
        self.assertFalse(flags.any())

    def test_fractional_fps_never_exceeds_requested_gap_seconds(self):
        for fps in [15.,25.,29.97002997,30.,30.00003000003,59.94,60.]:
            for gap in [.0,.1,.15,.2]:
                n=gap_frame_limit(gap,fps)
                self.assertLessEqual(n/fps,gap+1e-9)
                self.assertGreater((n+1)/fps,gap)

if __name__=='__main__':unittest.main()
