import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from protected_inference_v31 import jump_info, withdraw, write_wave_preserving_protected_text


class RuntimeProtectionTests(unittest.TestCase):
    def test_csv_preserves_original_decimal_spelling_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/'old.csv'
            new = Path(tmp)/'new.csv'
            old.write_text('time_s,base,covered,observed,interpolated\n'
                           '0.0000000000000000,0.12345678901234567,True,True,False\n'
                           '0.0333333000000000,-1.9876543210987654,True,True,False\n'
                           '0.0666666000000000,,False,False,False\n', encoding='utf-8')
            values = pd.read_csv(old)
            values['protected_sample'] = [True, False, True]
            values.loc[1, 'base'] = 8.
            write_wave_preserving_protected_text(new, values, old)
            text = pd.read_csv(new, dtype=str, keep_default_na=False)
            self.assertEqual(text.base.iloc[0], '0.12345678901234567')
            self.assertEqual(text.base.iloc[2], '')
            self.assertEqual(pd.read_csv(new).base.iloc[1], 8.)

    def test_jump_counts_do_not_bridge_missing_windows(self):
        hr = pd.DataFrame({'accepted':[True, True, False, True, True],
                           'ridge_bpm':[70., 83., np.nan, 140., 145.]})
        measured = jump_info(hr)
        self.assertEqual(measured['count'], 1)
        self.assertEqual(measured['maximum'], 13.)
        np.testing.assert_array_equal(measured['delta'], [13.,0.,0.,5.])

    def test_withdraw_closes_intersecting_windows_only(self):
        starts = np.arange(30)
        eligible = np.ones(30, bool)
        revised = withdraw(eligible, starts, 10, [15])
        np.testing.assert_array_equal(np.flatnonzero(~revised), np.arange(6,25))
        self.assertFalse(np.any(revised & ~eligible))

    def test_global_fallback_terminates_when_no_local_admission_matches(self):
        starts = np.arange(30)
        eligible = np.zeros(30, bool)
        eligible[0] = True
        self.assertFalse(withdraw(eligible, starts, 10, [29]).any())
        self.assertFalse(withdraw(np.zeros(30,bool), starts, 10, [29]).any())


if __name__ == '__main__':
    unittest.main()
