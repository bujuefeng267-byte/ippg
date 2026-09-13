"""Read-only smoke audit for a generic V27 run and its equivalent batch variant.

Does not read reference, evaluation, or synchronization files. The output receipt
is new and must not already exist. Numeric comparisons use fixed 1e-9 tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_v27 import readout


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(a, b):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    maximum = 0.
    discrete = []
    for col in a:
        x, y = a[col], b[col]
        np.testing.assert_array_equal(x.isna(), y.isna(), err_msg=col)
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            xx, yy = x.to_numpy(float), y.to_numpy(float)
            np.testing.assert_allclose(xx, yy, atol=1e-9, rtol=0, equal_nan=True, err_msg=col)
            mask = np.isfinite(xx) & np.isfinite(yy)
            if mask.any():
                maximum = max(maximum, float(abs(xx[mask]-yy[mask]).max()))
        else:
            np.testing.assert_array_equal(x.fillna('<missing>'), y.fillna('<missing>'), err_msg=col)
            discrete.append(col)
    return dict(rows=len(a), columns=len(a.columns), maximum_numeric_difference=maximum,
                discrete_columns_exact=discrete, all_missing_masks_exact=True, passed=True)


def audit(smoke, batch, output, reviewed_png_sha256):
    smoke, batch, output = map(Path, (smoke, batch, output))
    assert not output.exists(), 'Preserve existing QA receipts'
    meta = json.loads((smoke/'manifest.json').read_text())
    assert meta['status'] == 'complete'
    assert meta['reference_used'] is False and meta['physiological_accuracy_evaluated'] is False
    assert meta['computed_variants'] == ['late_psd_cluster']
    binding = json.loads((smoke/'frontend_binding.json').read_text())
    assert binding['extraction_mode'] == 'verified_frontend_cache'
    hashes = {str(smoke/'manifest.json'): sha(smoke/'manifest.json')}
    for name, expected in meta['outputs'].items():
        assert sha(smoke/name) == expected, name
        hashes[str(smoke/name)] = expected
    here = Path(__file__).resolve().parent
    for name, expected in meta['source_hashes'].items():
        assert sha(here/name) == expected, name
        hashes[str(here/name)] = expected
    variant = smoke/'variants/late_psd_cluster'
    assert sha(variant/'ppg_hr.png') == reviewed_png_sha256.lower(), 'Reviewed image differs'
    pairs = {'waveform.csv': 'waveform.csv', 'primary_hr.csv': 'heart_rate.csv',
             'secondary_hr.csv': 'fusion_heart_rate.csv', 'patch_diagnostics.csv': 'patch_diagnostics.csv'}
    comparisons = {}
    for own, old in pairs.items():
        hashes[str(batch/old)] = sha(batch/old)
        comparisons[own] = dict(byte_identical=sha(variant/own) == hashes[str(batch/old)],
                               **compare(pd.read_csv(variant/own), pd.read_csv(batch/old)))
    saved = pd.read_csv(variant/'waveform.csv')
    frames = pd.read_csv(smoke/'frontend/frame_trace.csv')
    replay = readout(saved, frames, meta['fps'])
    secondary = pd.read_csv(variant/'secondary_hr.csv')
    reread = compare(secondary, replay)
    primary = pd.read_csv(variant/'primary_hr.csv')
    for path, expected in hashes.items():
        assert sha(path) == expected, 'Input/source changed during audit: '+path
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, errors=[],
        reference_files_read=False, source_and_input_outputs_unchanged=True,
        source_sha256=sha(__file__), input_hashes=hashes,
        smoke=str(smoke), batch=str(batch), comparisons=comparisons,
        saved_wave_secondary_HR_replay=reread, frames=len(saved), fps=meta['fps'],
        planned_windows=len(primary), primary_accepted_windows=int(primary.accepted.sum()),
        secondary_accepted_windows=int(secondary.accepted.sum()),
        scope='Runtime, binding and source consistency only; no physiological accuracy assessment.',
        PNG_manual_review=dict(passed=True, path=str(variant/'ppg_hr.png'),
            reviewed_png_sha256=reviewed_png_sha256.lower(),
            observations='Three readable panels distinguish the saved relative-amplitude waveform, primary spatial patch HR statistic and secondary saved-wave HR; no reference; missing intervals remain blank; no clipping or label overlap observed.'))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'frames', 'planned_windows',
                                           'primary_accepted_windows', 'secondary_accepted_windows')}))
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reviewed-png-sha256', required=True,
                   help='SHA of the already visually inspected PNG; do not supply without inspecting')
    args = p.parse_args()
    audit(args.smoke, args.batch, args.output, args.reviewed_png_sha256)
