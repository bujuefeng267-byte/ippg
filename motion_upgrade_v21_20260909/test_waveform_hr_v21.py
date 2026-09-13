"""Independent synthetic regression tests for final-waveform HR/provenance.

No human reference or video is read. The reference-like columns in one test are
deliberately misleading extra metadata. CSV round-trips use StringIO only.
"""
from io import StringIO
import unittest

import numpy as np
import pandas as pd

try:
    from .waveform_hr import estimate_fused_waveform
    from .legacy_motion import estimate as legacy_estimate
except ImportError:
    from waveform_hr import estimate_fused_waveform
    from legacy_motion import estimate as legacy_estimate


FPS = 30.0
ROIS = ('forehead', 'left_cheek', 'right_cheek')
HR_COLUMNS = ['time_s', 'accepted', 'status', 'spectral_peak_bpm', 'ridge_bpm']


def fixture(seconds=24.0, bpm=84.0, proposal_bpm=150.0):
    n = round(seconds * FPS)
    t = np.arange(n) / FPS
    wave = np.sin(2 * np.pi * bpm / 60 * t) + .10*np.sin(4 * np.pi * bpm / 60 * t + .4)
    trace = pd.DataFrame(dict(frame=np.arange(n), time_s=t, source='mesh',
                              rgb_valid=True, r=100., g=110., b=90.,
                              motion_x=0., motion_y=0.))
    roi_wave = pd.DataFrame(dict(time_s=t))
    for roi in ROIS:
        trace[f'{roi}_valid'] = True
        trace[f'{roi}_quality'] = 1.
        roi_wave[f'{roi}_observed'] = True
        roi_wave[f'{roi}_interpolated'] = False
        for method in ['pos', 'chrom']:
            roi_wave[f'{roi}_{method}'] = wave.copy()
    starts = np.arange(0, n-round(10*FPS)+1, round(FPS), dtype=int)
    proposals = pd.DataFrame(dict(time_s=(starts+round(10*FPS)/2)/FPS,
        window_start_s=starts/FPS, window_end_s=(starts+round(10*FPS))/FPS,
        accepted=True, status='accepted_consensus', spectral_peak_bpm=proposal_bpm,
        ridge_bpm=proposal_bpm, roi_count=3, waveform_roi_count=2,
        waveform_generated=True))
    records = []
    for wi, row in proposals.iterrows():
        for roi in ROIS:
            records.append(dict(window_index=wi, time_s=row.time_s, roi=roi,
                method='pos', channel=f'{roi}_pos', candidate_rank=1,
                candidate_bpm=proposal_bpm, candidate_weight=1.,
                selected_consensus=True, output_accepted=True,
                waveform_weight=.5 if roi in ROIS[:2] else 0.))
    diagnostics = pd.DataFrame(records, columns=['window_index','time_s','roi','method',
        'channel','candidate_rank','candidate_bpm','candidate_weight',
        'selected_consensus','output_accepted','waveform_weight'])
    return wave, trace, roi_wave, proposals, diagnostics


def evaluate(case):
    return estimate_fused_waveform(*case, fps=FPS, window_s=10, step_s=1,
                                   min_bpm=42, max_bpm=210)


class FinalWaveformHRTests(unittest.TestCase):
    def assert_core_equal(self, actual, expected):
        expected = expected[HR_COLUMNS].copy()
        expected.loc[~expected.accepted, ['spectral_peak_bpm','ridge_bpm']] = np.nan
        pd.testing.assert_frame_equal(actual[HR_COLUMNS].reset_index(drop=True),
                                      expected.reset_index(drop=True),
                                      check_dtype=False, rtol=0, atol=1e-9)

    def test_wrong_proposal_frequency_cannot_replace_waveform_frequency(self):
        case = fixture(bpm=84, proposal_bpm=150)
        table, provenance = evaluate(case)
        self.assertEqual(len(table), 15)
        self.assertTrue(table.accepted.all())
        self.assertLess(np.max(abs(table.ridge_bpm-84)), 1.)
        self.assertLess(np.max(abs(table.spectral_peak_bpm-84)), 1.)
        self.assertTrue(provenance.covered.all())
        self.assertTrue(provenance.observed.all())
        self.assertFalse(provenance.interpolated.any())
        changed = fixture(bpm=84, proposal_bpm=48)
        changed_table, changed_provenance = evaluate(changed)
        self.assert_core_equal(table, changed_table)
        pd.testing.assert_frame_equal(provenance, changed_provenance)

    def test_zero_weight_roi_cannot_contaminate_sampling_provenance(self):
        base = fixture()
        expected_hr, expected_provenance = evaluate(base)
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        # Selected in spectral consensus, then excluded from waveform fusion.
        self.assertTrue(diagnostics.loc[diagnostics.roi.eq('right_cheek'), 'selected_consensus'].all())
        trace['right_cheek_valid'] = False
        roi_wave['right_cheek_observed'] = False
        roi_wave['right_cheek_interpolated'] = True
        roi_wave[['right_cheek_pos','right_cheek_chrom']] = np.nan
        actual_hr, actual_provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        self.assert_core_equal(actual_hr, expected_hr)
        pd.testing.assert_frame_equal(actual_provenance, expected_provenance)

    def test_interpolation_is_any_used_roi_and_observation_is_all_used_rois(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        gap = np.arange(360, 365)
        trace.loc[gap, 'forehead_valid'] = False
        roi_wave.loc[gap, 'forehead_observed'] = False
        roi_wave.loc[gap, 'forehead_interpolated'] = True
        # The other actually used ROI is observed; it must not hide the gap.
        table, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        expected_observed = np.ones(len(wave), bool); expected_observed[gap] = False
        expected_interpolated = ~expected_observed
        np.testing.assert_array_equal(provenance.observed, expected_observed)
        np.testing.assert_array_equal(provenance.interpolated, expected_interpolated)
        self.assertTrue(table.accepted.all())
        oracle = legacy_estimate(wave, pd.DataFrame({'rgb_valid': expected_observed}),
            expected_interpolated, FPS, 10, 1, 42, 210)
        self.assert_core_equal(table, oracle)
        np.testing.assert_allclose(table.interpolated_fraction, oracle.interpolated_fraction)

    def test_excess_interpolation_is_not_hidden_by_finite_waveform(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        trace.loc[300:419, 'forehead_valid'] = False
        roi_wave.loc[300:419, 'forehead_observed'] = False
        roi_wave.loc[300:419, 'forehead_interpolated'] = True
        table, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        self.assertTrue(np.isfinite(wave).all())
        oracle = legacy_estimate(wave, pd.DataFrame({'rgb_valid': provenance.observed}),
            provenance.interpolated.to_numpy(bool), FPS, 10, 1, 42, 210)
        self.assert_core_equal(table, oracle)
        self.assertTrue((table.status=='insufficient_observed_rgb').any())
        self.assertTrue(table.loc[~table.accepted, 'ridge_bpm'].isna().all())

    def test_neighbor_contributors_propagate_only_inside_their_actual_interval(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        # Right cheek contributes only to proposal 4, covering [4s, 14s).
        selected = diagnostics.window_index.eq(4) & diagnostics.roi.eq('right_cheek')
        diagnostics.loc[selected, 'waveform_weight'] = .25
        proposals.loc[4, 'waveform_roi_count'] = 3
        # Both positions are missing for this ROI, but only the first position
        # actually receives its samples through overlap-add.
        gaps = [240, 600]
        trace.loc[gaps, 'right_cheek_valid'] = False
        roi_wave.loc[gaps, 'right_cheek_observed'] = False
        roi_wave.loc[gaps, 'right_cheek_interpolated'] = True
        _, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        expected = np.zeros(len(wave), bool); expected[240] = True
        np.testing.assert_array_equal(provenance.interpolated, expected)
        np.testing.assert_array_equal(provenance.observed, ~expected)

    def test_neighbor_covered_output_preserves_rejected_proposal(self):
        case = fixture()
        wave, trace, roi_wave, proposals, diagnostics = case
        wi = 7
        proposals.loc[wi, ['accepted','waveform_generated']] = False
        proposals.loc[wi, 'status'] = 'ambiguous_consensus'
        proposals.loc[wi, ['spectral_peak_bpm','ridge_bpm']] = np.nan
        proposals.loc[wi, 'waveform_roi_count'] = 0
        diagnostics.loc[diagnostics.window_index.eq(wi), 'waveform_weight'] = 0.
        snapshot = proposals.copy(deep=True)
        table, provenance = evaluate(case)
        self.assertTrue(table.loc[wi,'accepted'])
        self.assertFalse(table.loc[wi,'waveform_generated'])
        self.assertTrue(table.loc[wi,'neighbor_covered'])
        self.assertTrue(provenance.covered.all())
        pd.testing.assert_frame_equal(proposals, snapshot)

    def test_positive_diagnostics_cover_only_their_own_time_interval(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        wi = 3
        proposals['waveform_generated'] = proposals.index==wi
        diagnostics.loc[~diagnostics.window_index.eq(wi), 'waveform_weight'] = 0.
        # An intentionally finite input outside the supported interval must not
        # turn undocumented samples into accepted HR output.
        table, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        expected_coverage = np.zeros(len(wave), bool)
        expected_coverage[90:390] = True
        np.testing.assert_array_equal(provenance.covered, expected_coverage)
        self.assertEqual(table.index[table.accepted].tolist(), [wi])
        self.assertLess(abs(table.loc[wi,'ridge_bpm']-84), 1.)

    def test_no_waveform_never_yields_accepted_hr(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture()
        wave[:] = np.nan
        table, _ = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        self.assertFalse(table.accepted.any())
        self.assertTrue(table.ridge_bpm.isna().all())
        self.assertTrue(table.spectral_peak_bpm.isna().all())

    def test_diffuse_rejection_masks_final_hr_but_retains_raw_peak_diagnostic(self):
        wave, trace, roi_wave, proposals, diagnostics = fixture(seconds=24)
        wave[:] = 0.
        wave[360] = 1.
        table, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
        oracle = legacy_estimate(wave, pd.DataFrame({'rgb_valid':provenance.observed}),
            provenance.interpolated.to_numpy(bool), FPS, 10, 1, 42, 210)
        central = table.time_s.between(8., 16.)
        self.assertEqual(int(central.sum()), 9)
        self.assertTrue(oracle.loc[central,'status'].eq('diffuse_spectrum').all())
        self.assertTrue(oracle.loc[central,'spectral_peak_bpm'].notna().all())
        self.assertFalse(table.loc[central,'accepted'].any())
        self.assertTrue(table.loc[~table.accepted, ['spectral_peak_bpm','ridge_bpm']].isna().all().all())
        self.assert_core_equal(table, oracle)
        np.testing.assert_allclose(table.raw_spectral_peak_bpm, oracle.spectral_peak_bpm,
                                   rtol=0, atol=1e-9, equal_nan=True)

    def test_no_positive_contributor_does_not_authorize_finite_waveform(self):
        case = fixture()
        case[-1]['waveform_weight'] = 0.
        table, provenance = evaluate(case)
        self.assertFalse(provenance.covered.any())
        self.assertFalse(table.accepted.any())
        self.assertTrue(table.ridge_bpm.isna().all())

    def test_positive_contributions_with_missing_window_indices_are_rejected(self):
        case = fixture()
        diagnostics = case[-1]
        missing = diagnostics.window_index.eq(7) & (diagnostics.waveform_weight>0)
        diagnostics.loc[missing, 'window_index'] = np.nan
        # Neighbors still cover the full sample mask, so a coverage-equality
        # check alone cannot detect silently dropped positive contributions.
        with self.assertRaises(ValueError):
            evaluate(case)

    def test_equal_length_roi_time_axis_mismatch_is_rejected(self):
        case = fixture()
        case[2]['time_s'] = case[2].time_s + 1/FPS
        self.assertEqual(len(case[1]), len(case[2]))
        with self.assertRaises(ValueError):
            evaluate(case)

    def test_empty_and_short_inputs_have_typed_empty_hr_tables(self):
        for seconds in [0, 3]:
            with self.subTest(seconds=seconds):
                wave, trace, roi_wave, proposals, diagnostics = fixture(seconds=seconds)
                table, provenance = evaluate((wave,trace,roi_wave,proposals,diagnostics))
                self.assertEqual(len(table), 0)
                self.assertEqual(len(provenance), len(trace))
                self.assertEqual(table.accepted.dtype, np.dtype(bool))
                # This avoids the previous short-input np.isfinite failure.
                self.assertEqual(len(np.isfinite(table.ridge_bpm)), 0)
                self.assertEqual(len(np.isfinite(table.spectral_peak_bpm)), 0)

    def test_reference_like_columns_are_ignored_and_inputs_not_mutated(self):
        expected, expected_provenance = evaluate(fixture())
        case = fixture()
        for frame in case[1:]:
            frame['reference_bpm'] = 177.
            frame['ground_truth_bpm'] = 43.
            frame['reference_quality'] = 1.
        snapshots = [case[0].copy()] + [x.copy(deep=True) for x in case[1:]]
        actual, actual_provenance = evaluate(case)
        self.assert_core_equal(actual, expected)
        pd.testing.assert_frame_equal(actual_provenance, expected_provenance)
        np.testing.assert_array_equal(case[0], snapshots[0])
        for actual_input, snapshot in zip(case[1:], snapshots[1:]):
            pd.testing.assert_frame_equal(actual_input, snapshot)

    def test_saved_csv_signal_reproduces_the_same_legacy_hr(self):
        case = fixture(bpm=102, proposal_bpm=60)
        table, provenance = evaluate(case)
        csv = pd.DataFrame({'time_s':case[1].time_s, 'base':case[0],
                            'observed':provenance.observed,
                            'interpolated':provenance.interpolated,
                            'covered':provenance.covered}).to_csv(index=False)
        saved = pd.read_csv(StringIO(csv))
        self.assertTrue(saved.covered.all())
        oracle = legacy_estimate(saved.base.to_numpy(),
            pd.DataFrame({'rgb_valid':saved.observed}), saved.interpolated.to_numpy(bool),
            FPS, 10, 1, 42, 210)
        self.assert_core_equal(table, oracle)
        self.assertLess(np.max(abs(table.ridge_bpm-102)), 1.)


if __name__ == '__main__':
    unittest.main()
