"""Synthetic local-admission tests; no real videos or reference HR are read."""
import unittest
import numpy as np
import pandas as pd
from proposal_evidence_v31 import build_proposals, ProposalConfig, ROIS


def fixture(fps=30., seconds=12., old_bpm=90., new_bpm=132., motion_bpm=None):
    n = round(seconds*fps)
    t = np.arange(n)/fps
    old_signal = np.sin(2*np.pi*old_bpm/60*t)
    new_signal = np.sin(2*np.pi*new_bpm/60*t)
    dx = np.zeros(n) if motion_bpm is None else .4/fps*2.*np.sin(2*np.pi*motion_bpm/60*t)
    trace = pd.DataFrame(dict(frame=np.arange(n),time_s=t,motion_x=dx,motion_y=0.,
        face_x0=.2,face_x1=.6,face_y0=.2,face_y1=.6))
    channels, tables = {}, {branch:pd.DataFrame({'time_s':t}) for branch in ('baseline','tracked')}
    for roi in ROIS:
        trace[roi+'_valid'] = True
        trace[roi+'_quality'] = .8
        trace[roi+'_pixel_source'] = 'tracked_ratio'
        for branch in ('baseline','tracked'):
            tables[branch][roi+'_observed'] = True
            tables[branch][roi+'_interpolated'] = False
            for method in ('pos','chrom'):
                channels[f'{branch}/{roi}/{method}'] = new_signal+.15*old_signal
    def waveform(x):
        return pd.DataFrame(dict(time_s=t,base=x,covered=True,observed=True,interpolated=False))
    width,hop = round(10*fps),round(fps)
    starts = np.arange(0,n-width+1,hop)
    def heart_rate(bpm):
        return pd.DataFrame(dict(time_s=(starts+width/2)/fps,window_start_s=starts/fps,
            window_end_s=(starts+width)/fps,ridge_bpm=bpm,accepted=True))
    return dict(old_wave=waveform(old_signal),old_hr=heart_rate(old_bpm),
        candidate_wave=waveform(new_signal),candidate_hr=heart_rate(new_bpm),trace=trace,fps=fps,
        roi_channels=channels,roi_tables=tables)


def run(inputs, mode='direct_motion'):
    return build_proposals(**inputs,config=ProposalConfig(mode=mode))


class ProposalEvidenceTests(unittest.TestCase):
    def test_direct_motion_and_independent_original_roi_support_admit(self):
        result = run(fixture(motion_bpm=90.))
        self.assertTrue(result.proposal_eligible.all())
        self.assertTrue((result.physical_roi_support_count == 3).all())
        self.assertTrue((result.old_direct_motion_risk >= .5).all())
        self.assertTrue((result.direct_motion_risk_decrease >= .3).all())

    def test_only_double_motion_relation_never_authorizes_direct_mode(self):
        result = run(fixture(motion_bpm=180.))
        self.assertFalse(result.proposal_eligible.any())
        self.assertTrue(result.raw_consensus_pass.all())
        self.assertTrue((result.old_direct_motion_risk < .5).all())

    def test_low_absolute_motion_or_equal_motion_risk_does_not_authorize(self):
        small = fixture(motion_bpm=90.)
        small['trace']['motion_x'] *= .01
        self.assertFalse(run(small).proposal_eligible.any())
        equal = fixture(motion_bpm=90.)
        t = equal['trace'].time_s.to_numpy()
        equal['trace']['motion_x'] += .4/30*2.*np.sin(2*np.pi*132/60*t)
        self.assertFalse(run(equal).proposal_eligible.any())

    def test_declared_raw_mode_can_admit_without_motion_but_direct_cannot(self):
        inputs = fixture()
        self.assertFalse(run(inputs).proposal_eligible.any())
        result = run(inputs,'raw_consensus')
        self.assertTrue(result.proposal_eligible.all())
        self.assertTrue((result.raw_consensus_roi_count == 3).all())
        self.assertTrue(result.reason.eq('original_baseline_POS_CHROM_main_peak_consensus').all())

    def test_raw_mode_is_superset_of_direct_admission_for_identical_inputs(self):
        for motion in (None,90.,180.,132.):
            inputs = fixture(motion_bpm=motion)
            strict,broad = run(inputs),run(inputs,'raw_consensus')
            self.assertFalse((strict.proposal_eligible & ~broad.proposal_eligible).any())

    def test_repeated_branches_and_methods_are_only_one_physical_roi_vote(self):
        inputs = fixture(motion_bpm=90.)
        old = inputs['old_wave'].base.to_numpy()
        for key in inputs['roi_channels']:
            if '/forehead/' not in key:
                inputs['roi_channels'][key] = old
        for mode in ('direct_motion','raw_consensus'):
            result = run(inputs,mode)
            self.assertTrue((result.physical_roi_support_count == 1).all())
            self.assertFalse(result.proposal_eligible.any())

    def test_raw_consensus_requires_both_baseline_methods(self):
        inputs = fixture()
        for roi in ROIS:
            inputs['roi_channels'][f'baseline/{roi}/chrom'] = inputs['old_wave'].base.to_numpy()
        result = run(inputs,'raw_consensus')
        self.assertTrue((result.physical_roi_support_count == 3).all())
        self.assertTrue((result.raw_consensus_roi_count == 0).all())
        self.assertFalse(result.proposal_eligible.any())

    def test_tracked_only_consensus_is_not_baseline_main_peak_consensus(self):
        inputs = fixture()
        for key in inputs['roi_channels']:
            if key.startswith('baseline/'):
                inputs['roi_channels'][key] = inputs['old_wave'].base.to_numpy()
        result = run(inputs,'raw_consensus')
        self.assertTrue((result.physical_roi_support_count == 3).all())
        self.assertFalse(result.raw_consensus_pass.any())
        self.assertFalse(result.proposal_eligible.any())

    def test_strongest_psd_is_not_replaced_by_harmonic_score_ranking(self):
        inputs = fixture(old_bpm=90.,new_bpm=180.)
        x = inputs['candidate_wave'].base.to_numpy()+.4*inputs['old_wave'].base.to_numpy()
        for key in inputs['roi_channels']:
            inputs['roi_channels'][key] = x.copy()
        result = run(inputs,'raw_consensus')
        self.assertTrue(result.raw_consensus_pass.all())
        self.assertTrue(result.proposal_eligible.all())

    def test_original_roi_quality_and_observation_gates_stay_in_effect(self):
        low = fixture(motion_bpm=90.)
        for roi in ROIS[:2]: low['trace'][roi+'_quality'] = .1
        self.assertFalse(run(low,'raw_consensus').proposal_eligible.any())
        missing = fixture(motion_bpm=90.)
        for roi in ROIS:
            missing['trace'].loc[:35,roi+'_valid'] = False
            for table in missing['roi_tables'].values():
                table.loc[:35,roi+'_observed'] = False
                table.loc[:35,roi+'_interpolated'] = True
        self.assertFalse(bool(run(missing,'raw_consensus').iloc[0].proposal_eligible))

    def test_original_tracking_gate_limits_tracked_support(self):
        inputs = fixture(motion_bpm=90.)
        for key in inputs['roi_channels']:
            if key.startswith('baseline/'):
                inputs['roi_channels'][key] = inputs['old_wave'].base.to_numpy()
        for roi in ROIS:
            inputs['trace'][roi+'_pixel_source'] = 'baseline_ratio_fallback'
        result = run(inputs)
        self.assertTrue((result.physical_roi_support_count == 0).all())
        self.assertFalse(result.proposal_eligible.any())

    def test_rejected_hr_or_small_disagreement_is_never_a_proposal(self):
        rejected = fixture(motion_bpm=90.)
        rejected['candidate_hr']['accepted'] = False
        rejected['candidate_hr']['ridge_bpm'] = np.nan
        result = run(rejected,'raw_consensus')
        self.assertFalse(result.proposal_eligible.any())
        self.assertTrue(result.reason.eq('candidate_hr_not_accepted').all())
        close = fixture(new_bpm=96.,motion_bpm=90.)
        result = run(close,'raw_consensus')
        self.assertFalse(result.proposal_eligible.any())
        self.assertTrue(result.reason.eq('no_material_disagreement').all())

    def test_complete_saved_windows_required_without_waveform_repair(self):
        for name in ('old_wave','candidate_wave'):
            inputs = fixture(motion_bpm=90.)
            inputs[name].loc[150,'base'] = np.nan
            inputs[name].loc[150,['covered','observed']] = False
            result = run(inputs,'raw_consensus')
            self.assertFalse(result.proposal_eligible.any())
            self.assertTrue(np.isnan(inputs[name].loc[150,'base']))

    def test_unknown_motion_is_not_reported_as_proof_of_low_motion(self):
        inputs = fixture()
        inputs['trace']['motion_x'] = np.nan
        result = run(inputs)
        self.assertFalse(result.motion_available.any())
        self.assertTrue(result.old_direct_motion_risk.isna().all())
        self.assertFalse(result.proposal_eligible.any())
        self.assertTrue(run(inputs,'raw_consensus').proposal_eligible.all())

    def test_inputs_unchanged_and_extra_reference_identity_columns_ignored(self):
        inputs = fixture(motion_bpm=90.)
        original = {name:inputs[name].copy(deep=True) for name in ('old_wave','old_hr','candidate_wave','candidate_hr','trace')}
        expected = run(inputs,'raw_consensus')
        for name,copy in original.items(): pd.testing.assert_frame_equal(inputs[name],copy,check_exact=True)
        for name in original:
            inputs[name]['reference_bpm'] = 199.
            inputs[name]['case'] = 'arbitrary_identifier'
        actual = run(inputs,'raw_consensus')
        pd.testing.assert_frame_equal(expected,actual,check_exact=True)

    def test_fractional_30_and_180fps_keep_original_window_clocks(self):
        for fps in (30.00003000003,180.001800018):
            inputs = fixture(fps=fps,motion_bpm=90.)
            result = run(inputs)
            np.testing.assert_allclose(result.time_s,inputs['old_hr'].time_s,atol=1e-8,rtol=0)
            self.assertTrue(result.proposal_eligible.all())

    def test_contract_corruption_and_unknown_modes_are_rejected(self):
        with self.assertRaises(ValueError): ProposalConfig(mode='by_video')
        mismatch = fixture(); mismatch['candidate_hr']['time_s'] += .01
        with self.assertRaises(ValueError): run(mismatch)
        flags = fixture(); flags['roi_tables']['baseline'].loc[0,'forehead_observed'] = False
        with self.assertRaises(ValueError): run(flags)
        quality = fixture(); quality['trace']['forehead_quality'] = np.nan
        with self.assertRaises(ValueError): run(quality)


if __name__ == '__main__': unittest.main()
