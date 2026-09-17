"""Small reference-free I/O helpers for the retained candidate scorer."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Input or code changed: ' + str(path))


def canonical(hr, n, fps):
    if not np.isfinite(fps) or fps <= 7:
        raise ValueError('FPS must support the existing 42–210 bpm search range')
    if {'reference_bpm', 'error_bpm', 'abs_error_bpm'} & set(hr.columns):
        raise ValueError('Pass inference heart_rate.csv, not a reference-paired evaluation table')
    required = {'ridge_bpm', 'accepted', 'status', 'evidence_candidates_json'}
    if not required.issubset(hr):
        raise ValueError('Missing inference fields: ' + str(sorted(required - set(hr))))
    table = hr.copy(deep=True)
    width, hop = round(10 * fps), round(fps)
    starts = np.arange(0, max(0, n - width + 1), hop)
    if len(table) != len(starts):
        raise ValueError('HR and waveform do not share the original 10 s / 1 s plan')
    for field, values in [('time_s', (starts + width / 2) / fps),
                          ('window_start_s', starts / fps),
                          ('window_end_s', (starts + width) / fps)]:
        if field in table:
            np.testing.assert_allclose(table[field], values, rtol=0, atol=1e-7)
        table[field] = values
    if not table.accepted.isin([True, False]).all():
        raise ValueError('Accepted must be an explicit boolean')
    finite = np.isfinite(table.ridge_bpm.to_numpy(float))
    if not np.array_equal(finite, table.accepted.to_numpy(bool)):
        raise ValueError('Accepted rows must have finite HR, rejected rows must remain missing')
    table['window_index'] = np.arange(len(table))
    return table
