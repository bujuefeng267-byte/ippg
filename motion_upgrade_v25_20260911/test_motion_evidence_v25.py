"""Synthetic kinematics and pulse preservation tests; no human video/reference."""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from motion_evidence import motion_evidence
from motion_fusion import FusionConfig, fuse_windows, MOTION_COLUMNS

GRID=np.arange(42.,211.)
ROIS=('forehead','left_cheek','right_cheek')


def trajectory(fps=30., seconds=12., amplitude=.08, bpm=90., face_height=.4):
    """Known sinusoidal position in face heights, discretized into measured steps."""
    t=np.arange(round(seconds*fps))/fps
    position=lambda x: amplitude*np.sin(2*np.pi*bpm/60*x)
    dx=(position(t)-position(t-1/fps))*face_height
    trace=pd.DataFrame(dict(time_s=t,motion_x=dx,motion_y=np.zeros(len(t)),
                            face_y0=np.full(len(t),.2),face_y1=np.full(len(t),.2+face_height)))
    for name in ROIS:
        trace[f'{name}_valid']=True;trace[f'{name}_quality']=.95
    return t,trace


def signals(t,bpm=90):
    pulse=np.sin(2*np.pi*bpm/60*t)
    return pulse,{f'{roi}_{method}':pulse.copy() for roi in ROIS for method in ['pos','chrom']}


class MotionEvidenceTests(unittest.TestCase):
    def test_physical_velocity_is_fps_and_face_size_invariant(self):
        evidence=[]
        for fps in (30.,180.):
            for face in (.2,.5):
                _,trace=trajectory(fps=fps,face_height=face)
                result=motion_evidence(trace,0,len(trace),fps,GRID)
                self.assertTrue(result['available'])
                self.assertAlmostEqual(GRID[np.argmax(result['profile'])],90,delta=1)
                theoretical=.08*2*np.pi*1.5/np.sqrt(2)
                self.assertAlmostEqual(result['speed_face_per_s'],theoretical,delta=.003)
                evidence.append(result)
        for result in evidence[1:]:
            self.assertAlmostEqual(result['speed_face_per_s'],evidence[0]['speed_face_per_s'],delta=.003)
            np.testing.assert_allclose(result['profile'],evidence[0]['profile'],atol=.004,rtol=0)

    def test_missing_motion_or_face_scale_is_not_zero_motion_evidence(self):
        _,trace=trajectory()
        for broken in (trace.drop(columns='motion_x'),trace.drop(columns='face_y1'),
                       trace.assign(motion_y=np.nan),trace.assign(face_y1=.2)):
            result=motion_evidence(broken,0,len(broken),30.,GRID)
            self.assertFalse(result['available']);self.assertEqual(result['reliability'],0)
            self.assertTrue(np.isnan(result['speed_face_per_s']))
            self.assertTrue(np.isnan(result['strength']))
            self.assertTrue((result['profile']==0).all())

    def test_bounded_gap_uses_observed_segments_without_inventing_samples(self):
        t,trace=trajectory(seconds=12)
        complete=motion_evidence(trace,0,len(trace),30.,GRID)
        trace.loc[(t>=5)&(t<6),'motion_x']=np.nan
        original=trace.copy(deep=True)
        result=motion_evidence(trace,0,len(trace),30.,GRID)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['observed_fraction'],11/12)
        self.assertAlmostEqual(result['reliability'],11/12)
        self.assertLess(result['profile'].max(),complete['profile'].max())
        self.assertAlmostEqual(GRID[np.argmax(result['profile'])],90,delta=1)
        pd.testing.assert_frame_equal(trace,original)

    def test_fragmented_motion_has_no_supported_spectrum(self):
        t,trace=trajectory()
        trace.loc[(t%1)<.2,'motion_x']=np.nan
        result=motion_evidence(trace,0,len(trace),30.,GRID)
        self.assertGreater(result['observed_fraction'],.5)
        self.assertFalse(result['available']);self.assertEqual(result['spectral_coverage'],0)

    def test_tiny_same_frequency_jitter_does_not_become_strong_interference(self):
        _,tiny=trajectory(amplitude=.0001)
        _,large=trajectory(amplitude=.08)
        low=motion_evidence(tiny,0,len(tiny),30.,GRID)
        high=motion_evidence(large,0,len(large),30.,GRID)
        self.assertTrue(low['available'])
        self.assertLess(low['profile'].max(),1e-4)
        self.assertGreater(high['profile'].max(),.8)

    def test_observed_static_and_constant_translation_are_not_missing(self):
        _,trace=trajectory(amplitude=0)
        for displacement in (0.,.005):
            current=trace.assign(motion_x=displacement)
            result=motion_evidence(current,0,len(current),30.,GRID)
            self.assertTrue(result['available']);self.assertEqual(result['reliability'],1.)
            self.assertAlmostEqual(result['speed_face_per_s'],displacement*30/.4,places=10)
            self.assertTrue((result['profile']==0).all())

    def test_reference_like_columns_are_never_used(self):
        _,trace=trajectory()
        a=motion_evidence(trace,0,len(trace),30.,GRID)
        b=motion_evidence(trace.assign(reference_bpm=-100,ground_truth=999),0,len(trace),30.,GRID)
        for key in a:
            if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
            else:self.assertEqual(a[key],b[key])

    def test_invalid_parameters_fail(self):
        _,trace=trajectory()
        for fps,grid in [(0,GRID),(np.nan,GRID),(5,GRID),(30,GRID[::-1])]:
            with self.assertRaises(ValueError):motion_evidence(trace,0,len(trace),fps,grid)
        with self.assertRaises(ValueError):FusionConfig(motion_evidence_mode='unknown')
        with self.assertRaises(ValueError):FusionConfig(reliable_motion_penalty=1.)


class ReliableFusionTests(unittest.TestCase):
    def test_default_and_explicit_legacy_are_identical(self):
        t,trace=trajectory();_,data=signals(t)
        a=fuse_windows(data,trace,30.)
        b=fuse_windows(data,trace,30.,config=FusionConfig(motion_evidence_mode='legacy'))
        pd.testing.assert_frame_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        pd.testing.assert_frame_equal(a[2],b[2])

    def test_same_frequency_real_pulse_survives_even_large_motion(self):
        t,trace=trajectory(amplitude=.4);pulse,data=signals(t)
        table,wave,diag=fuse_windows(data,trace,30.,config=FusionConfig(motion_evidence_mode='reliable'))
        self.assertTrue(table.accepted.all());self.assertTrue(table.motion_available.all())
        self.assertLess(np.abs(table.ridge_bpm-90).max(),1)
        self.assertGreaterEqual(diag.motion_weight.min(),.25-1e-12)
        self.assertGreater(abs(np.corrcoef(pulse,wave)[0,1]),.999)

    def test_unknown_motion_does_not_cancel_pulse_or_claim_known_motion(self):
        t,trace=trajectory();pulse,data=signals(t)
        trace['motion_x']=np.nan
        table,wave,diag=fuse_windows(data,trace,30.,config=FusionConfig(motion_evidence_mode='reliable'))
        self.assertTrue(table.accepted.all());self.assertFalse(table.motion_available.any())
        self.assertTrue(table.motion_speed_face_per_s.isna().all())
        self.assertTrue(diag.motion_strength.isna().all())
        self.assertGreater(abs(np.corrcoef(pulse,wave)[0,1]),.999)

    def test_motion_competitor_is_softly_ranked_without_removing_broadband_waveform(self):
        t,trace=trajectory(amplitude=.12,bpm=150)
        mixture=np.sin(2*np.pi*90/60*t)+1.15*np.sin(2*np.pi*150/60*t)
        data={f'{roi}_{m}':mixture.copy() for roi in ROIS for m in ['pos','chrom']}
        table,wave,diag=fuse_windows(data,trace,30.,config=FusionConfig(motion_evidence_mode='reliable'))
        self.assertTrue(table.accepted.all())
        self.assertLess(np.abs(table.ridge_bpm-90).max(),1)
        self.assertTrue(diag.candidate_bpm.eq(150).any(), 'Motion competitor remains in diagnostics')
        self.assertTrue((diag.loc[diag.candidate_bpm.eq(150),'motion_weight']>0).all())
        self.assertGreater(abs(np.corrcoef(wave,mixture)[0,1]),.999)
        # This test makes no promise that broadband waveform maximum or final HR
        # becomes 90: only evidence ranking changed, not the waveform contents.

    def test_tiny_jitter_does_not_resolve_equal_competitors(self):
        t,trace=trajectory(amplitude=.0001,bpm=150)
        mixture=np.sin(2*np.pi*90/60*t)+np.sin(2*np.pi*150/60*t)
        data={f'{roi}_{m}':mixture.copy() for roi in ROIS for m in ['pos','chrom']}
        table,wave,diag=fuse_windows(data,trace,30.,config=FusionConfig(motion_evidence_mode='reliable'))
        self.assertFalse(table.accepted.any());self.assertTrue(np.isnan(wave).all())
        self.assertTrue(table.status.eq('ambiguous_consensus').all())


if __name__=='__main__':unittest.main()
