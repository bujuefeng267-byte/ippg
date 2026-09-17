import unittest
import numpy as np
import pandas as pd
from common import canonical


def example():
    return pd.DataFrame({'accepted':[True,False], 'ridge_bpm':[80.,np.nan],
                         'status':['candidate_only','gap'], 'evidence_candidates_json':['[]','[]']})


class InputContractTests(unittest.TestCase):
    def test_missing_windows_are_retained(self):
        hr=example();result=canonical(hr,330,30.)
        self.assertEqual(len(result),2)
        self.assertTrue(np.isnan(result.ridge_bpm.iloc[1]))
        np.testing.assert_allclose(result.time_s,[5.,6.])
        self.assertNotIn('time_s',hr)

    def test_reference_paired_input_rejected(self):
        hr=example();hr['reference_bpm']=[80.,80.]
        with self.assertRaisesRegex(ValueError,'reference-paired'):
            canonical(hr,330,30.)

    def test_clock_or_plan_mismatch_rejected(self):
        hr=example();hr['time_s']=[5.,7.]
        with self.assertRaises(AssertionError):canonical(hr,330,30.)
        with self.assertRaises(ValueError):canonical(example(),360,30.)

    def test_silent_zero_fill_and_missing_accepted_rejected(self):
        hr=example();hr.loc[1,'ridge_bpm']=0.
        with self.assertRaises(ValueError):canonical(hr,330,30.)
        hr=example();hr.loc[0,'ridge_bpm']=np.nan
        with self.assertRaises(ValueError):canonical(hr,330,30.)


if __name__=='__main__':unittest.main()
