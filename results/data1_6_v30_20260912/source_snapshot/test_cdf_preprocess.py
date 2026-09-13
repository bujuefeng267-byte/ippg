"""Synthetic tests only: CDF physics, missingness, boundaries and copied views."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import cdf_preprocess as cdf


def signals(fps, seconds):
    time = np.arange(round(seconds*fps))/fps
    chromatic = .025*np.sin(2*np.pi*1.4*time)[:,None]*np.array([.3,.8,.3])
    intensity = .20*np.sin(2*np.pi*1.*time)[:,None]*np.ones((1,3))
    return np.array([100.,80.,60.])*(1.+chromatic+intensity)


class CDFTests(unittest.TestCase):
    def test_expected_shape_constant_and_positive_edges_at_30_and_180fps(self):
        for fps in (30.,180.001800018):
            x = np.tile([100.,80.,60.],(round(24*fps),1))
            y,d,rows = cdf.preprocess_rgb(x,fps)
            self.assertEqual(y.shape,x.shape)
            self.assertTrue(np.isfinite(y).all())
            np.testing.assert_allclose(y,x,atol=1e-12,rtol=0)
            self.assertEqual(d.status.iloc[0],'cdf_applied')
            self.assertTrue(any(row['window'] for row in rows))

    def test_soft_color_weights_reduce_distinct_common_intensity_frequency(self):
        fps = 30.
        x = signals(fps,10.)
        filtered,weights,freq = cdf.cdf_window(x,fps)
        self.assertTrue(np.all((weights >= 0)&(weights <= 1)))
        motion,pulse = np.argmin(abs(freq-1.)),np.argmin(abs(freq-1.4))
        self.assertLess(weights[motion],1e-24)
        self.assertAlmostEqual(weights[pulse],(1./6.)/(.3**2+.8**2+.3**2),places=12)
        original_fft = np.fft.rfft(x/x.mean(axis=0)-1.,axis=0)
        filtered_fft = np.fft.rfft(filtered/x.mean(axis=0)-1.,axis=0)
        np.testing.assert_allclose(filtered_fft[pulse],original_fft[pulse]*weights[pulse],atol=1e-12,rtol=0)
        self.assertLess(np.linalg.norm(filtered_fft[motion]),1e-11)

    def test_real_symmetric_filter_preserves_inband_phase_and_rejects_outside(self):
        fps,time = 30.,np.arange(300)/30.
        direction = np.array([-1.,2.,-1.])/np.sqrt(6.)
        x = 100.*(1.+.02*np.sin(2*np.pi*1.4*time+.37)[:,None]*direction+
                  .02*np.sin(2*np.pi*5.*time+.9)[:,None]*direction)
        y,weights,freq = cdf.cdf_window(x,fps)
        expected = 100.*(1.+.02*np.sin(2*np.pi*1.4*time+.37)[:,None]*direction)
        np.testing.assert_allclose(y,expected,atol=1e-11,rtol=0)
        self.assertTrue(np.all(weights[(freq < .7)|(freq > 3.5)] == 0))

    def test_original_nan_entries_and_short_segments_are_unchanged(self):
        x = signals(30.,30.)
        x[100:105] = np.nan
        x[451,1] = np.nan
        original = x.copy()
        y,d,rows = cdf.preprocess_rgb(x,30.)
        np.testing.assert_array_equal(np.isnan(y),np.isnan(original))
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(y[:100],original[:100])
        self.assertTrue((d.status.iloc[:100] == 'original_short_segment').all())
        # A partial missing row retains its original finite entries as well.
        self.assertEqual(y[451,0],original[451,0])
        self.assertEqual(y[451,2],original[451,2])

    def test_uncovered_tail_is_real_original_rgb_not_missing_or_extrapolated(self):
        x = signals(30.,10.5)
        y,d,_ = cdf.preprocess_rgb(x,30.)
        np.testing.assert_array_equal(y[300:],x[300:])
        self.assertTrue((d.status.iloc[300:] == 'original_uncovered_tail').all())
        self.assertTrue(np.isfinite(y).all())

    def test_no_cross_gap_influence_on_other_segment(self):
        x = signals(30.,25.)
        x[330:333] = np.nan
        altered = x.copy(); altered[333:] *= np.array([1.2,.7,1.4])
        first,_,_ = cdf.preprocess_rgb(x,30.)
        second,_,_ = cdf.preprocess_rgb(altered,30.)
        np.testing.assert_array_equal(first[:333],second[:333])

    def test_nonpositive_filtered_window_falls_back_without_clipping(self):
        x = signals(30.,10.)
        def impossible(rgb,fps,config):
            return -np.ones_like(rgb),np.zeros(151),np.fft.rfftfreq(300,1/30.)
        with patch.object(cdf,'cdf_window',side_effect=impossible):
            y,d,rows = cdf.preprocess_rgb(x,30.)
        np.testing.assert_allclose(y,x,atol=1e-13,rtol=0)
        self.assertTrue((d.status == 'original_numeric_window_fallback').all())
        self.assertEqual(rows[0]['status'],'original_nonpositive_or_nonfinite_window')

    def test_derived_trace_preserves_geometry_masks_and_branch_identity(self):
        fps,n = 30.,720
        raw = signals(fps,n/fps)
        trace = pd.DataFrame(dict(frame=np.arange(n),time_s=np.arange(n)/fps,
                                 rgb_valid=True,motion_x=.01,motion_y=.02,pixel_rgb_kind='anchored_paired_relative_colour'))
        for index,roi in enumerate(cdf.ROIS):
            trace[f'{roi}_valid'] = True
            trace[f'{roi}_quality'] = .7+index*.01
            trace[f'{roi}_pixel_source'] = 'tracked_ratio'
            for j,channel in enumerate('rgb'):
                trace[f'baseline_{roi}_{channel}'] = raw[:,j]*(1.+index*.1)
                trace[f'{roi}_{channel}'] = raw[:,j]*(1.+index*.1)*1.02
        for j,channel in enumerate('rgb'):
            trace['baseline_'+channel] = raw[:,j]*1.1
            trace[channel] = raw[:,j]*1.1*1.02
        original = trace.copy(deep=True)
        derived,summary,windows = cdf.preprocess_trace(trace,fps)
        pd.testing.assert_frame_equal(trace,original,check_exact=True)
        self.assertEqual(len(summary['streams']),6)
        self.assertTrue(summary['tracking_geometry_and_quality_preserved'])
        self.assertTrue(len(windows) > 0)
        for roi in cdf.ROIS:
            for suffix in ('valid','quality','pixel_source'):
                pd.testing.assert_series_equal(derived[f'{roi}_{suffix}'],original[f'{roi}_{suffix}'],check_exact=True)
            for channel in 'rgb':
                np.testing.assert_allclose(derived[f'{roi}_{channel}'],derived[f'baseline_{roi}_{channel}']*1.02,atol=1e-11,rtol=0)
        self.assertFalse(np.allclose(derived.baseline_forehead_g,trace.baseline_forehead_g))
        self.assertTrue(derived.cdf_view_kind.str.startswith('derived_cdf').all())

    def test_invalid_input_and_unfrozen_parameters_are_rejected(self):
        with self.assertRaises(ValueError): cdf.CDFConfig(window_s=6.)
        with self.assertRaises(ValueError): cdf.preprocess_rgb(np.zeros((300,3)),30.)
        with self.assertRaises(ValueError): cdf.preprocess_rgb(np.ones((300,2)),30.)
        with self.assertRaises(ValueError): cdf.preprocess_rgb(np.full((300,3),np.inf),30.)
        with self.assertRaises(ValueError): cdf.cdf_window(np.ones((299,3)),30.)


if __name__ == '__main__': unittest.main()
