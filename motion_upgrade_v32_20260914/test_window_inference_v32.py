"""Controlled numerical integration tests; no human video or reference input."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Support plain unittest discovery both in the installed project and staging.
_project = Path(__file__).resolve().parent.parent
if not (_project / 'motion_upgrade_v31_20260912').is_dir():
    _project = Path('/home/fengbujue/项目/rppg识别')
for _folder in ('motion_upgrade_v28_20260912', 'motion_upgrade_v31_20260912'):
    sys.path.insert(0, str(_project / _folder))

import window_inference_v32 as core


def fixture(seconds=14., fps=30., old_bpm=96., new_bpm=105., eligible=(2,)):
    time = np.arange(round(seconds*fps))/fps
    old_signal = np.sin(2*np.pi*old_bpm/60*time)
    new_signal = np.sin(2*np.pi*new_bpm/60*time)
    trace = pd.DataFrame(dict(frame=np.arange(len(time)), time_s=time,
        motion_x=0., motion_y=0., face_x0=.2, face_x1=.6, face_y0=.2, face_y1=.6))
    tables = {branch: pd.DataFrame(dict(time_s=time)) for branch in ('baseline', 'tracked')}
    channels = {}
    for roi in core.proposals_v31.ROIS:
        trace[roi+'_valid'] = True
        trace[roi+'_quality'] = .8
        trace[roi+'_pixel_source'] = 'tracked_ratio'
        for branch in tables:
            tables[branch][roi+'_observed'] = True
            tables[branch][roi+'_interpolated'] = False
            for method in ('pos', 'chrom'):
                channels[f'{branch}/{roi}/{method}'] = new_signal+.05*old_signal
    starts = np.arange(0, len(time)-round(10*fps)+1, round(fps))
    def wave(signal):
        return pd.DataFrame(dict(time_s=time, base=signal, covered=True, observed=True, interpolated=False))
    def hr(value):
        return pd.DataFrame(dict(time_s=starts/fps+5., window_start_s=starts/fps,
            window_end_s=starts/fps+10., ridge_bpm=value, accepted=True))
    old_hr, new_hr = hr(old_bpm), hr(new_bpm)
    if eligible is not None:
        outside = ~np.isin(np.arange(len(starts)), eligible)
        new_hr.loc[outside, ['accepted', 'ridge_bpm']] = [False, np.nan]
    return dict(trace=trace, fps=fps, old_wave=wave(old_signal), old_hr=old_hr,
                candidate_wave=wave(new_signal), candidate_hr=new_hr, channels=channels, tables=tables)


def execute(root, inputs, motion_guard=True):
    paths = {}
    for name in ('old_wave', 'old_hr', 'candidate_wave', 'candidate_hr'):
        path = root/(name+'.csv')
        inputs[name].to_csv(path, index=False)
        paths[name+'_path'] = path
    with patch.object(core, '_original_roi_inputs', return_value=(inputs['channels'], inputs['tables'])):
        hr, decisions, audit = core.infer_windows(**paths, trace=inputs['trace'], fps=inputs['fps'],
                                                out=root/'out', motion_guard=motion_guard)
    return hr, decisions, audit


class IndependentWindowTests(unittest.TestCase):
    def test_one_short_candidate_changes_without_altering_overlapping_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = fixture()
            hr, decisions, audit = execute(root, inputs)
            self.assertEqual(audit['final_eligible_windows'], 1)
            self.assertGreater(hr.ridge_bpm.iloc[2], 102.)
            self.assertTrue(hr.accepted.all())
            np.testing.assert_array_equal(hr.ridge_bpm.iloc[[0,1,3,4]], [96.]*4)
            manifest = pd.read_csv(root/'out/windows_manifest.csv')
            old = pd.read_csv(root/'old_wave.csv')
            for i in (0,1,3,4):
                segment = pd.read_csv(root/'out'/manifest.waveform_file.iloc[i])
                np.testing.assert_array_equal(segment.base, old.base.iloc[i*30:i*30+300])
            self.assertTrue(decisions.final_motion_guard_pass.iloc[2])
            self.assertEqual((root/'out/waveform.csv').read_bytes(), (root/'old_wave.csv').read_bytes())
            self.assertFalse(audit['continuous_context_is_replacement_HR_source'])

    def test_repeated_file_read_reconstructs_measured_candidates_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr, _, _ = execute(root, fixture())
            manifest = pd.read_csv(root/'out/windows_manifest.csv')
            for row in manifest.itertuples():
                file = root/'out'/row.waveform_file
                self.assertEqual(core.sha256(file), row.sha256)
                segment = pd.read_csv(file)
                candidates, emission = core.score_candidates(segment.base, 30., core.GRID)
                self.assertTrue(np.isfinite(emission[int(hr.ridge_bpm.iloc[row.window_index]-42)]))
                self.assertTrue(any(hr.ridge_bpm.iloc[row.window_index] in item['supported_bpm'] for item in candidates))

    def test_high_motion_consensus_is_rejected_by_main_but_exposed_in_ablation(self):
        for guard, expected in ((True, 0), (False, 1)):
            with tempfile.TemporaryDirectory() as tmp:
                inputs = fixture(old_bpm=90., new_bpm=132., seconds=10., eligible=None)
                inputs['trace']['motion_x'] = .8/30*np.sin(2*np.pi*132/60*inputs['trace'].time_s)
                _, decisions, audit = execute(Path(tmp), inputs, guard)
                self.assertTrue(decisions.proposal_eligible.all())
                self.assertGreater(decisions.candidate_direct_motion_risk.iloc[0], .5)
                self.assertEqual(audit['final_eligible_windows'], expected)

    def test_unknown_motion_cannot_be_treated_as_zero_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            for key in ('motion_x', 'motion_y', 'face_x0', 'face_x1', 'face_y0', 'face_y1'):
                inputs['trace'][key] = np.nan
            _, decisions, audit = execute(Path(tmp), inputs)
            self.assertTrue(decisions.proposal_eligible.iloc[2])
            self.assertFalse(decisions.motion_available.iloc[2])
            self.assertEqual(audit['final_eligible_windows'], 0)

    def test_only_direct_profile_is_used_by_motion_veto(self):
        self.assertTrue(core._motion_pass(True, .2, .1))
        self.assertFalse(core._motion_pass(False, 0., 0.))
        self.assertFalse(core._motion_pass(True, .8, .5))
        self.assertFalse(core._motion_pass(True, .2, .21))
        self.assertTrue(core._motion_pass(True, .2, .2+5e-10))

    def test_final_state_risk_masks_neighboring_unsafe_peak(self):
        real = core._state_evidence
        def high_risk_neighbor(*args, **kwargs):
            evidence = real(*args, **kwargs)
            evidence['available'] = True
            evidence['old_risk'] = .3
            evidence['profile'] = np.zeros(len(core.GRID))
            evidence['profile'][core.GRID < 105.] = .8
            evidence['motion_pass'] = evidence['profile'] < .5
            return evidence
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, '_state_evidence', side_effect=high_risk_neighbor):
            hr, decisions, audit = execute(Path(tmp), fixture())
            self.assertEqual(audit['final_eligible_windows'], 1)
            self.assertGreaterEqual(hr.ridge_bpm.iloc[2], 105.)
            self.assertLess(decisions.final_candidate_direct_motion_risk.iloc[2], .5)

    def test_final_dp_output_cannot_bypass_forbidden_state(self):
        actual = core.constrained_evidence_path
        def corrupt_path(emissions, grid, step, anchors):
            path, reacquired = actual(emissions, grid, step, anchors)
            for i in np.flatnonzero(~np.isfinite(anchors)):
                path[i] = int(np.flatnonzero(grid == 150.)[0])
            return path, reacquired
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'constrained_evidence_path', side_effect=corrupt_path):
            with self.assertRaisesRegex(core.ProtectionError, 'forbidden or unsupported'):
                execute(Path(tmp), fixture())

    def test_original_reject_remains_nan_and_breaks_jump_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            inputs['old_hr'].loc[1, ['accepted', 'ridge_bpm']] = [False, np.nan]
            hr, _, audit = execute(Path(tmp), inputs)
            self.assertFalse(hr.accepted.iloc[1])
            self.assertTrue(np.isnan(hr.ridge_bpm.iloc[1]))
            self.assertEqual(audit['accepted_windows'], 4)
            self.assertEqual(len(pd.read_csv(Path(tmp)/'out/windows_manifest.csv')), 5)

    def test_wrong_candidate_clock_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            inputs['candidate_wave']['time_s'] += .01
            with self.assertRaisesRegex(ValueError, 'sample clock'):
                execute(Path(tmp), inputs)

    def test_forged_old_accepted_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture(eligible=())
            inputs['old_hr']['ridge_bpm'] = 180.
            with self.assertRaisesRegex(core.ProtectionError, 'no actual saved-window spectral support'):
                execute(Path(tmp), inputs)

    def test_candidate_hr_without_actual_waveform_support_reverts(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture(old_bpm=90., new_bpm=132., seconds=10., eligible=None)
            inputs['candidate_wave']['base'] = np.sin(2*np.pi*180/60*inputs['trace'].time_s)
            hr, _, audit = execute(Path(tmp), inputs)
            self.assertEqual(audit['initial_eligible_windows'], 1)
            self.assertEqual(audit['final_eligible_windows'], 0)
            self.assertEqual(hr.ridge_bpm.iloc[0], 90.)
            records = json.loads((Path(tmp)/'out/guard_history.json').read_text())
            self.assertEqual(records[0]['quality_failures'], [0])

    def test_candidate_low_observation_quality_reverts_without_relaxing_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            inputs['candidate_wave']['observed'] = False
            inputs['candidate_wave']['interpolated'] = True
            hr, _, audit = execute(Path(tmp), inputs)
            self.assertEqual(audit['final_eligible_windows'], 0)
            np.testing.assert_array_equal(hr.ridge_bpm, inputs['old_hr'].ridge_bpm)

    def test_candidate_gaps_never_filled_and_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            inputs['candidate_wave'].loc[150, ['base', 'covered', 'observed']] = [np.nan, False, False]
            _, _, audit = execute(Path(tmp), inputs)
            self.assertEqual(audit['initial_eligible_windows'], 0)

    def test_new_jump_reverts_selected_endpoint_without_overlap_cascade(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture(old_bpm=90., new_bpm=132.)
            hr, _, audit = execute(Path(tmp), inputs)
            self.assertEqual(audit['initial_eligible_windows'], 1)
            self.assertEqual(audit['final_eligible_windows'], 0)
            np.testing.assert_array_equal(hr.ridge_bpm, inputs['old_hr'].ridge_bpm)
            records = json.loads((Path(tmp)/'out/guard_history.json').read_text())
            self.assertEqual(records[0]['withdrawn_windows'], [2])
        selected = np.ones(5, bool)
        np.testing.assert_array_equal(core.withdraw_selected_endpoints(selected, [2]), [True, True, False, True, True])

    def test_same_physical_roi_cannot_supply_two_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = fixture()
            for key in inputs['channels']:
                if '/forehead/' not in key:
                    inputs['channels'][key] = inputs['old_wave'].base.to_numpy()
            _, decisions, audit = execute(Path(tmp), inputs)
            self.assertEqual(decisions.physical_roi_support_count.iloc[2], 1)
            self.assertEqual(audit['initial_eligible_windows'], 0)

    def test_core_does_not_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            execute(root, fixture())
            before = (root/'out/heart_rate.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                execute(root, fixture())
            self.assertEqual((root/'out/heart_rate.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
