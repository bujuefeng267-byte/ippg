"""Reference-free synthetic counterexamples for the fixed V27 backends.

These do not open videos, prior outputs, labels, or evaluation files. Knowing
the frequencies of generated inputs here tests mathematics, not model tuning
on the six development recordings.
"""
import json
import unittest

import numpy as np
import pandas as pd

from patch_hr_v27 import (DEFAULT_CONFIG, REGIONS, complete_link_clusters,
    fusion_readout_diagnostic, infer_patch_hr, patch_spectrum,
    protocol_description)


def fixture(frequencies=None, fps=30., seconds=12., branch='baseline'):
    if frequencies is None:
        frequencies={f'{region}_{k}':(region,84.) for region in REGIONS for k in range(4)}
    n=round(seconds*fps);t=np.arange(n)/fps
    regions={p:item[0] for p,item in frequencies.items()}
    channels={f'{branch}/{p}/pos':np.sin(2*np.pi*bpm/60*t+.12*i)
        for i,(p,(_,bpm)) in enumerate(frequencies.items())}
    observed={p:np.ones(n,bool) for p in regions}
    interpolated={p:np.zeros(n,bool) for p in regions}
    quality={p:np.ones(n) for p in regions}
    sources={p:np.full(n,'tracked_ratio',object) for p in regions}
    trace=pd.DataFrame({'time_s':t,'motion':np.zeros(n),
        'face_x':np.zeros(n),'face_y':np.zeros(n),'face_w':np.ones(n),'face_h':np.ones(n)})
    return channels,regions,observed,interpolated,quality,sources,trace,fps


def two_regions(a=84.,b=84.,**kwargs):
    return fixture({'a':('forehead',a),'b':('left_cheek',b)},**kwargs)


class PatchHRTests(unittest.TestCase):
    def test_pure_pulse_and_polarity_both_modes(self):
        args=fixture()
        # Deliberate opposite phases must align without changing frequencies.
        for i,key in enumerate(args[0]):
            if i%2:args[0][key]*=-1
        for mode in ('median','psd_cluster'):
            with self.subTest(mode=mode):
                wave,hr,diag=infer_patch_hr(*args,mode=mode)
                self.assertTrue(hr.accepted.all())
                np.testing.assert_allclose(hr.ridge_bpm,84.,atol=1.)
                self.assertTrue(wave.covered.all())
                self.assertTrue(wave.observed.all())
                self.assertFalse(wave.interpolated.any())
                # Different true phase inputs may yield a different common
                # phase. Test the actual frequency subspace, not phase zero.
                theta=2*np.pi*84/60*wave.time_s.to_numpy()
                basis=np.column_stack([np.sin(theta),np.cos(theta),np.ones(len(theta))])
                residual=wave.base.to_numpy()-basis@np.linalg.lstsq(basis,wave.base,rcond=None)[0]
                self.assertLess(np.linalg.norm(residual)/np.linalg.norm(wave.base),1e-10)
                counts=diag[diag.record_type=='contributor'].groupby('window_index').patch_id.nunique()
                self.assertTrue((counts==12).all())

    def test_same_physical_frequency_across_frame_rates(self):
        for bpm in (84.,110.,180.):
            summaries=[]
            for fps in (30.,180.):
                args=two_regions(bpm,bpm,fps=fps,seconds=10.)
                spec=patch_spectrum(args[0]['baseline/a/pos'],fps)
                summaries.append(spec)
                for mode in ('median','psd_cluster'):
                    wave,hr,_=infer_patch_hr(*args,mode=mode)
                    self.assertTrue(hr.accepted.all())
                    self.assertAlmostEqual(hr.ridge_bpm.iloc[0],bpm,delta=1.)
                    self.assertEqual(len(wave),round(10*fps))
            self.assertAlmostEqual(summaries[0]['peak_fwhm_bpm'],summaries[1]['peak_fwhm_bpm'],delta=.03)
            self.assertAlmostEqual(summaries[0]['spectral_score'],summaries[1]['spectral_score'],delta=.002)
            self.assertGreater(float(summaries[0]['shape']@summaries[1]['shape']),.9999)

    def test_method_and_branch_aliases_never_create_physical_votes(self):
        args=two_regions(seconds=10.)
        original=infer_patch_hr(*args)
        for key,value in list(args[0].items()):
            _,patch,_=key.split('/')
            for branch in ('baseline','tracked'):
                for method in ('pos','chrom'):
                    args[0][f'{branch}/{patch}/{method}']=value.copy()
        wave,hr,diag=infer_patch_hr(*args)
        self.assertEqual(hr.eligible_patches.iloc[0],2)
        self.assertEqual(hr.contributing_patches.iloc[0],2)
        self.assertEqual(hr.contributing_regions.iloc[0],2)
        np.testing.assert_allclose(wave.base,original[0].base,atol=0)
        np.testing.assert_allclose(hr.ridge_bpm,original[1].ridge_bpm,atol=0)
        self.assertEqual(len(diag[diag.record_type=='contributor']),2)
        one=fixture({'a':('forehead',84.)},seconds=10.)
        for method in ('pos','chrom'):
            for branch in ('baseline','tracked'):
                one[0][f'{branch}/a/{method}']=one[0]['baseline/a/pos'].copy()
        for mode in ('median','psd_cluster'):
            _,single,_=infer_patch_hr(*one,mode=mode)
            self.assertFalse(single.accepted.any())
            self.assertTrue(single.ridge_bpm.isna().all())

    def test_equal_region_weight_prevents_patch_count_domination(self):
        frequencies={f'f{k}':('forehead',60.) for k in range(4)}
        frequencies.update({'l':('left_cheek',120.),'r':('right_cheek',120.)})
        _,hr,diag=infer_patch_hr(*fixture(frequencies,seconds=10.))
        self.assertEqual(hr.ridge_bpm.iloc[0],120.)
        votes=diag[diag.record_type=='contributor'].groupby('region').coefficient.sum()
        np.testing.assert_allclose(votes,1/3,rtol=0,atol=1e-15)

    def test_even_median_does_not_synthesize_unsupported_frequency(self):
        args=two_regions(60.,120.,seconds=10.)
        wave,hr,_=infer_patch_hr(*args)
        self.assertEqual(hr.ridge_bpm.iloc[0],90.)
        self.assertFalse(hr.estimate_near_contributor_peak.iloc[0])
        self.assertNotIn('spectral_peak_bpm',hr)
        t=wave.time_s.to_numpy();x=wave.base.to_numpy()
        amplitude=lambda bpm:abs(np.dot(x,np.exp(-2j*np.pi*bpm/60*t)))
        self.assertLess(amplitude(90.),.01*min(amplitude(60.),amplitude(120.)))
        selected=np.column_stack([v-v.mean() for v in args[0].values()])
        fit=selected@np.linalg.lstsq(selected,x,rcond=None)[0]
        np.testing.assert_allclose(fit,x,atol=1e-12)

    def test_full_shape_distance_not_only_strongest_frequency(self):
        # Both maxima occur at bin zero, but incompatible secondary spectra.
        shapes=np.array([[1.,.9,0.,0.],[1.,0.,.9,0.],[1.,.89,0.,0.]])
        self.assertTrue((np.argmax(shapes,axis=1)==0).all())
        self.assertEqual(complete_link_clusters(shapes,.25),[[0,2],[1]])

    def test_complete_link_does_not_merge_via_bridge(self):
        theta=np.deg2rad([0,30,60])
        clusters=complete_link_clusters(np.column_stack([np.cos(theta),np.sin(theta)]),.25)
        self.assertEqual(sorted(map(len,clusters)),[1,2])
        self.assertNotIn([0,1,2],clusters)

    def test_local_bad_patch_and_one_region_artifact(self):
        # Every forehead patch is dominated by an artifact. Two remaining
        # physical regions agree on the same pulse; aliases cannot add votes.
        frequencies={f'f{k}':('forehead',144.) for k in range(4)}
        frequencies.update({'l':('left_cheek',84.),'r':('right_cheek',84.)})
        args=fixture(frequencies,seconds=10.)
        _,hr,diag=infer_patch_hr(*args,mode='psd_cluster')
        self.assertTrue(hr.accepted.iloc[0])
        self.assertEqual(hr.ridge_bpm.iloc[0],84.)
        self.assertEqual(hr.contributing_regions.iloc[0],2)
        self.assertEqual(set(diag[diag.record_type=='contributor'].region),{'left_cheek','right_cheek'})
        # A single extra bad patch in an otherwise clean forehead does not
        # displace the coherent three-region PSD cluster.
        clean=fixture(seconds=10.)
        clean[0]['baseline/forehead_0/pos']=np.sin(2*np.pi*144/60*clean[6].time_s.to_numpy())
        for mode in ('median','psd_cluster'):
            self.assertEqual(infer_patch_hr(*clean,mode=mode)[1].ridge_bpm.iloc[0],84.)

    def test_shared_multiregion_artifact_is_not_claimed_recoverable(self):
        # No reference-free spatial method can label common strong periodic
        # content as nonphysiological from these identical spectra alone.
        args=fixture(seconds=10.)
        t=args[6].time_s.to_numpy()
        for key in args[0]:args[0][key]=np.sin(2*np.pi*144/60*t)+.1*np.sin(2*np.pi*84/60*t)
        for mode in ('median','psd_cluster'):
            _,hr,_=infer_patch_hr(*args,mode=mode)
            self.assertTrue(hr.accepted.iloc[0])
            self.assertEqual(hr.ridge_bpm.iloc[0],144.)
        self.assertTrue(protocol_description()['no_primary_motion_penalty'])

    def test_shared_patch_representatives_between_modes(self):
        args=fixture(seconds=10.)
        t=args[6].time_s.to_numpy()
        for key,x in list(args[0].items()):
            patch=key.split('/')[1]
            args[0][f'tracked/{patch}/chrom']=x+.5*np.sin(2*np.pi*153/60*t)
        left=infer_patch_hr(*args,mode='median')[2]
        right=infer_patch_hr(*args,mode='psd_cluster')[2]
        cols=['channel','representative','status','peak_bpm','spectral_score']
        pd.testing.assert_frame_equal(left[left.record_type=='channel'][cols].reset_index(drop=True),
            right[right.record_type=='channel'][cols].reset_index(drop=True))

    def test_long_gaps_are_not_filled_and_output_remains_missing(self):
        args=two_regions(seconds=26.)
        for patch in args[1]:
            args[2][patch][360:420]=False
            args[4][patch][360:420]=0
            args[0][f'baseline/{patch}/pos'][360:420]=np.nan
        for mode in ('median','psd_cluster'):
            wave,hr,_=infer_patch_hr(*args,mode=mode)
            self.assertTrue(wave.base.iloc[360:420].isna().all())
            self.assertFalse(wave.covered.iloc[360:420].any())
            self.assertTrue(hr.loc[~hr.accepted,'ridge_bpm'].isna().all())
            self.assertGreater((~hr.accepted).sum(),0)
            for row in hr[hr.accepted].itertuples():
                a=round(row.window_start_s*args[-1]);b=round(row.window_end_s*args[-1])
                self.assertTrue(np.isfinite(wave.base.iloc[a:b]).all())

    def test_only_verified_short_interior_interpolation_is_allowed(self):
        for gap,expected in ((3,True),(4,False)):
            args=two_regions(seconds=10.)
            for patch in args[1]:
                args[2][patch][100:100+gap]=False
                args[3][patch][100:100+gap]=True
            wave,hr,diag=infer_patch_hr(*args)
            self.assertEqual(bool(hr.accepted.iloc[0]),expected)
            if expected:
                self.assertFalse(wave.observed.iloc[100:103].any())
                self.assertTrue(wave.interpolated.iloc[100:103].all())
            else:
                self.assertTrue(wave.base.isna().all())
                self.assertEqual(set(diag.status),{'unverified_or_long_interpolation'})
        # A four-frame whole-trace gap must not become an allowable one-frame
        # interpolation simply because the evaluated window starts inside it.
        args=two_regions(seconds=11.)
        for patch in args[1]:
            args[2][patch][28:32]=False;args[3][patch][28:32]=True
        _,hr,_=infer_patch_hr(*args)
        self.assertFalse(hr.accepted.any())
        # Explicit fill mask is mandatory even for a finite three-frame gap.
        args=two_regions(seconds=10.)
        for patch in args[1]:args[2][patch][100:103]=False
        self.assertFalse(infer_patch_hr(*args)[1].accepted.any())

    def test_tracking_guards_do_not_disable_baseline(self):
        args=two_regions(branch='tracked',seconds=10.)
        for patch in args[1]:args[5][patch][:]='baseline_reset'
        _,hr,diag=infer_patch_hr(*args)
        self.assertFalse(hr.accepted.any())
        self.assertEqual(set(diag.status),{'tracking_ineligible'})
        for key,x in list(args[0].items()):args[0][key.replace('tracked/','baseline/')]=x.copy()
        _,hr,_=infer_patch_hr(*args)
        self.assertTrue(hr.accepted.all())
        self.assertTrue(all(s.startswith('baseline/') for s in json.loads(hr.selected_channels_json.iloc[0])))

    def test_zero_signal_quality_and_observation_rejection(self):
        args=two_regions(seconds=10.)
        for key in args[0]:args[0][key][:]=0
        for mode in ('median','psd_cluster'):
            wave,hr,diag=infer_patch_hr(*args,mode=mode)
            self.assertTrue(wave.base.isna().all())
            self.assertTrue(hr.ridge_bpm.isna().all())
            self.assertEqual(set(diag.status),{'flat_or_zero_band_signal'})
        args=two_regions(seconds=10.)
        for patch in args[1]:args[4][patch][:]=.19
        self.assertFalse(infer_patch_hr(*args)[1].accepted.any())
        args=two_regions(seconds=10.)
        for patch in args[1]:args[2][patch][::5]=False;args[3][patch][::5]=True
        self.assertFalse(infer_patch_hr(*args)[1].accepted.any())

    def test_extra_reference_columns_are_ignored(self):
        args=fixture(seconds=10.)
        for mode in ('median','psd_cluster'):
            before=infer_patch_hr(*args,mode=mode)
            args[6]['reference_bpm']=42.+np.arange(len(args[6]))%169
            args[6]['host_utc_ns']=np.arange(len(args[6]),dtype=np.int64)+1_800_000_000_000_000_000
            after=infer_patch_hr(*args,mode=mode)
            for a,b in zip(before,after):pd.testing.assert_frame_equal(a,b)

    def test_fractional_fps_and_short_video_schema(self):
        fps=10000000/333333
        args=two_regions(fps=fps,seconds=12.)
        _,hr,_=infer_patch_hr(*args)
        expected=(np.arange(3)*round(fps)+round(10*fps)/2)/fps
        np.testing.assert_allclose(hr.time_s,expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(hr.window_end_s-hr.window_start_s,round(10*fps)/fps,rtol=0,atol=1e-12)
        wave,hr,_=infer_patch_hr(*two_regions(seconds=6.))
        self.assertEqual(len(hr),0)
        self.assertEqual(hr.ridge_bpm.dtype,float)
        self.assertTrue(wave.base.isna().all())

    def test_invalid_masks_and_nonfinite_quality_fail_loudly(self):
        args=two_regions(seconds=10.);args[3]['a'][0]=True
        with self.assertRaises(ValueError):infer_patch_hr(*args)
        args=two_regions(seconds=10.);args[4]['a'][0]=np.nan
        with self.assertRaises(ValueError):infer_patch_hr(*args)

    def test_diagnostic_readout_uses_actual_saved_wave_and_masks_rejection(self):
        args=two_regions(60.,120.,seconds=10.)
        wave,primary,_=infer_patch_hr(*args)
        def estimator(values,trace,filled,fps,window_s,step_s,low,high,motion_trace):
            np.testing.assert_allclose(values,wave.base)
            np.testing.assert_array_equal(trace.rgb_valid,wave.observed)
            np.testing.assert_array_equal(filled,wave.interpolated)
            self.assertEqual((window_s,step_s,low,high),(10.,1.,42.,210.))
            return pd.DataFrame({'accepted':[False],'spectral_peak_bpm':[60.],'ridge_bpm':[60.]})
        diagnostic=fusion_readout_diagnostic(wave,args[6],args[-1],estimator)
        self.assertEqual(primary.ridge_bpm.iloc[0],90.)
        self.assertEqual(diagnostic.raw_spectral_peak_bpm.iloc[0],60.)
        self.assertTrue(diagnostic.ridge_bpm.isna().all())
        self.assertTrue(diagnostic.spectral_peak_bpm.isna().all())


if __name__=='__main__':unittest.main()
