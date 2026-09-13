"""Synthetic integration checks; production admission is isolated by a stub."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from protected_inference_v31 import infer_protected
from protected_waveform_router import ProtectionError, read_saved_protected_hr
from test_protected_waveform_router import inputs


class ProtectedIntegrationTests(unittest.TestCase):
    def run_case(self, eligible, seconds, injected_failure=False):
        old, hr, candidate, trace, fps = inputs(seconds=seconds)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            old_path, hr_path = folder/'original.csv', folder/'original_hr.csv'
            candidate_path, candidate_hr_path = folder/'candidate.csv', folder/'candidate_hr.csv'
            old.to_csv(old_path, index=False)
            hr.to_csv(hr_path, index=False)
            candidate.to_csv(candidate_path, index=False)
            hr.to_csv(candidate_hr_path, index=False)
            calls = []
            def readout(*args, **kwargs):
                calls.append(1)
                if injected_failure and len(calls) == 1:
                    raise ProtectionError('Synthetic anchor validation failure', [5])
                return read_saved_protected_hr(*args, **kwargs)
            with patch('proposal_evidence_v31.build_proposals', return_value=pd.DataFrame(
                    {'proposal_eligible': eligible})), patch(
                    'protected_waveform_router.read_saved_protected_hr', side_effect=readout):
                wave, final, decisions, audit = infer_protected(old_path, hr_path,
                    candidate_path, candidate_hr_path, trace, fps, folder/'output', 'direct_motion')
            history = json.loads((folder/'output/guard_history.json').read_text())
            original = pd.read_csv(old_path)
            original_hr = pd.read_csv(hr_path)
            fixed = ~decisions.eligible.to_numpy(bool)
            np.testing.assert_array_equal(final.accepted, original_hr.accepted)
            np.testing.assert_array_equal(final.ridge_bpm[fixed], original_hr.ridge_bpm[fixed])
            protected = wave.protected_sample.to_numpy(bool)
            for name in ('time_s', 'base', 'covered', 'observed', 'interpolated', 'source'):
                np.testing.assert_array_equal(wave[name][protected], original[name][protected])
            return wave, final, decisions, audit, history, len(calls)

    def test_no_effect_admission_demotes_and_anchors_every_hr(self):
        eligible = np.zeros(11, bool)
        eligible[5] = True
        _, final, decisions, audit, _, _ = self.run_case(eligible, 20)
        self.assertFalse(decisions.eligible.any())
        self.assertEqual(audit['initial_eligible_windows'], 1)
        self.assertEqual(audit['modified_samples'], 0)
        self.assertTrue(final.protected_anchor_verified.all())

    def test_protection_exception_with_indices_rolls_back_and_terminates(self):
        _, _, decisions, audit, history, calls = self.run_case(
            np.ones(26, bool), 35, injected_failure=True)
        self.assertGreater(calls, 1)
        self.assertGreater(len(history), 1)
        self.assertLessEqual(len(history), 28)
        self.assertTrue(decisions.withdrawn_by_runtime_guard.any())
        self.assertTrue(audit['protected_waveform_and_HR_preserved'])
        counts = [row['eligible_windows'] for row in history]
        self.assertTrue(all(b < a for a, b in zip(counts, counts[1:])))


if __name__ == '__main__':
    unittest.main()
