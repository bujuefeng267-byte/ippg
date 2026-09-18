import json
import os
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd

from retained_backend import StreamingBackend as Original
from retained_fast_backend import StreamingBackend as Fast
from test_retained_backend import rows

P=Path(os.environ.get("RPPG_PROJECT_ROOT",str(Path(__file__).resolve().parent.parent)))
CASEMAP=Path(os.environ['RPPG_TEST_CASEMAP']) if os.environ.get('RPPG_TEST_CASEMAP') else None


def resampled_case(case,offset=0):
    caseinfo=json.loads(CASEMAP.read_text())["cases"][case]
    source=pd.read_csv(caseinfo["trace"])
    source=source[(source.time_s>=offset-.2)&(source.time_s<=offset+22)].copy()
    source['capture_dt_s']=source.time_s.diff()
    b=Original(P)
    b.rows.extend(source.to_dict('records'))
    b.first_time=0.
    return b._resample(offset+22)[0]


def synthetic_gap():
    source=rows(22)
    for i,row in enumerate(source):
        row['capture_dt_s']=1/30 if i else float('nan')
        if 7 <= row['time_s'] < 7.06 or 13 <= row['time_s'] < 13.5:
            for roi in ('forehead','left_cheek','right_cheek'):
                row[f'{roi}_valid']=False
    b=Original(P)
    b.rows.extend(source)
    b.first_time=0.
    return b._resample(22.)[0]


def equivalent(old,new):
    errors=[]
    old_wave,old_hr,old_dec,old_routing=old
    wave,hr,dec,routing=new
    for name in ('covered','observed','interpolated'):
        np.testing.assert_array_equal(wave[name],old_wave[name])
    np.testing.assert_allclose(wave.base,old_wave.base,atol=2e-10,rtol=2e-10,equal_nan=True)
    np.testing.assert_array_equal(hr.accepted,old_hr.accepted)
    np.testing.assert_allclose(hr.ridge_bpm,old_hr.ridge_bpm,atol=0,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(dec.route_eligible,old_dec.route_eligible)
    np.testing.assert_array_equal(routing.selected_branch,old_routing.selected_branch)
    if np.isfinite(wave.base).any():
        return float(np.nanmax(np.abs(wave.base-old_wave.base)))
    return 0.


class FastRetainedTests(unittest.TestCase):
    def setUp(self):
        self.original=Original(P)
        self.fast=Fast(P)

    def test_no_shared_module_mutation(self):
        import analyze_components_v26
        import legacy_motion
        self.assertIs(self.original.components,analyze_components_v26)
        self.assertIsNot(self.fast.components,analyze_components_v26)
        self.assertIs(self.original.components.prepare_roi_channels,analyze_components_v26.prepare_roi_channels)
        self.assertIs(legacy_motion.pos_signal,self.fast._legacy.pos_signal)

    def test_channels_and_masks_all_four_cases_and_synthetic_gap(self):
        inputs={'synthetic_gap':synthetic_gap()}
        if CASEMAP is not None:
            inputs.update({case:resampled_case(case,start) for case,start in [('data1',0),('data5',25),('data6',20),('data8',0)]})
        for name,trace in inputs.items():
            with self.subTest(name=name):
                channels,tables=self.original.components.prepare_roi_channels(trace,30.)
                fast_channels,fast_tables=self.fast.components.prepare_roi_channels(trace,30.)
                self.assertEqual(channels.keys(),fast_channels.keys())
                for key in channels:
                    np.testing.assert_allclose(channels[key],fast_channels[key],atol=2e-10,rtol=2e-10,equal_nan=True)
                for branch in tables:
                    for column in tables[branch].columns:
                        if column.endswith(('observed','interpolated')):
                            np.testing.assert_array_equal(tables[branch][column],fast_tables[branch][column])
                        else:
                            np.testing.assert_allclose(tables[branch][column],fast_tables[branch][column],atol=2e-10,rtol=2e-10,equal_nan=True)

    def test_whole_pipeline_four_cases_and_gap(self):
        receipts=[]
        cases=[('synthetic_gap',0)]
        if CASEMAP is not None:
            cases=[('data1',0),('data5',25),('data6',20),('data8',0)]+cases
        for name,start in cases:
            with self.subTest(name=name):
                trace=synthetic_gap() if name=='synthetic_gap' else resampled_case(name,start)
                started=time.perf_counter()
                old=self.original._pipeline(trace)
                original_s=time.perf_counter()-started
                started=time.perf_counter()
                new=self.fast._pipeline(trace)
                fast_s=time.perf_counter()-started
                delta=equivalent(old,new)
                receipts.append(dict(case=name,start_s=start,window_s=22.,original_s=original_s,
                    accelerated_s=fast_s,waveform_max_abs_difference=delta,accepted_exact=True,hr_exact=True,route_exact=True))
        # Unit tests do not overwrite archived real-data verification receipts.


if __name__=='__main__':
    unittest.main()
