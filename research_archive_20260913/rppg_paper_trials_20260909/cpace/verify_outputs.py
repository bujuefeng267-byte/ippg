"""Post-run consistency check; no reference file or tuning."""
import json
import numpy as np
import pandas as pd
from run_cpace_trial import HERE, CASES, code_hashes, dump, final_hr, sha

protocol = json.loads((HERE/'protocol_before_inference.json').read_text())
assert code_hashes() == protocol['code_hashes']
records = []
for variant in ('results', 'results_no_homodyne'):
    for case in CASES:
        output = HERE/variant/case
        meta = json.loads((output/'summary.json').read_text())
        wave = pd.read_csv(output/'waveform.csv')
        hr = pd.read_csv(output/'heart_rate.csv')
        assert meta['reference_used'] is False
        assert len(wave) == meta['frames']
        np.testing.assert_allclose(wave.time_s, np.arange(len(wave))/meta['fps'], rtol=0, atol=1e-8)
        finite = np.isfinite(wave.base)
        np.testing.assert_array_equal(wave.covered.to_numpy(bool), finite)
        assert not (wave.observed & wave.interpolated).any()
        np.testing.assert_array_equal(wave.observed | wave.interpolated, finite)
        assert hr.loc[~hr.accepted, ['spectral_peak_bpm', 'ridge_bpm']].isna().all().all()
        recomputed = final_hr(wave, meta['fps'])
        pd.testing.assert_frame_equal(hr, recomputed, check_dtype=False, rtol=0, atol=1e-9)
        assert meta['code_hashes'] == protocol['code_hashes']
        assert sha(meta['source_trace']) == meta['trace_sha256']
        for name, expected in meta['output_hashes'].items():
            assert sha(output/name) == expected
        records.append({'variant': variant, 'case': case, 'saved_waveform_hr_recomputed_equal': True,
                        'sample_and_hr_times_valid': True, 'provenance_mask_valid': True,
                        'source_and_output_hashes_valid': True})
dump(HERE/'output_verification.json', {'passed': True, 'outputs_checked': len(records),
     'reference_used': False, 'records': records, 'script_sha256': sha(__file__)})
print('Verified all 12 outputs: exact saved-wave HR recomputation, masks, time axes, source/output hashes.')
