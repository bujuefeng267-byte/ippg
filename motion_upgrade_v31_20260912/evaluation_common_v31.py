"""Frozen shared evaluation primitives; copied from validated V30, without V30 runner."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
P = Path("/home/fengbujue/项目/rppg识别")
OLD_CODE = P/"motion_upgrade_v28_20260912"
CORE_PATH = P/"batch_analysis_20260911/evaluate_batch.py"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value

def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def bools(series):
    assert series.notna().all() and series.isin([True, False, 0, 1]).all(), 'Explicit Boolean masks required'
    return series.to_numpy(bool)

def verify_hashes(bindings):
    assert isinstance(bindings, dict) and bindings, 'Nonempty hash bindings required'
    for name, expected in bindings.items():
        assert Path(name).is_absolute(), 'Hash keys must be absolute paths'
        assert sha(name) == expected, 'Bound input/source changed: '+str(name)

def protocol_bindings(value):
    """Find nested absolute-path -> SHA256 pairs without imposing a top schema."""
    result = {}
    def visit(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if isinstance(key, str) and Path(key).is_absolute() and isinstance(val, str) and len(val) == 64:
                    if key in result: assert result[key] == val, 'Conflicting frozen file binding'
                    result[key] = val
                visit(val)
        elif isinstance(item, list):
            for val in item: visit(val)
    visit(value)
    return result

def core_module():
    spec = importlib.util.spec_from_file_location('v31_reference_core', CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_signal_tables(wave, hr, frames, fps):
    assert len(wave) == frames and frames > 0 and np.isfinite(fps) and fps > 7
    np.testing.assert_allclose(wave.time_s, np.arange(frames)/fps, rtol=0, atol=1e-8)
    values = wave.base.to_numpy(float)
    assert not np.isinf(values).any()
    finite = np.isfinite(values)
    np.testing.assert_array_equal(finite, bools(wave.covered))
    observed, interpolated = bools(wave.observed), bools(wave.interpolated)
    assert not (observed & ~finite).any() and not (interpolated & ~finite).any()
    width, hop = round(10*fps), round(fps)
    starts = np.arange(0, frames-width+1, hop, dtype=int)
    assert len(hr) == len(starts)
    for key, expected in [('time_s', (starts+width/2)/fps), ('window_start_s', starts/fps),
                          ('window_end_s', (starts+width)/fps)]:
        np.testing.assert_allclose(hr[key], expected, rtol=0, atol=1e-8)
    accepted, y = bools(hr.accepted), hr.ridge_bpm.to_numpy(float)
    assert np.isfinite(y[accepted]).all() and np.isnan(y[~accepted]).all()
    assert ((y[accepted] >= 42) & (y[accepted] <= 210)).all()
    assert all(finite[a:a+width].all() for a in starts[accepted]), 'Accepted HR crosses missing samples'
    return values, finite, observed, interpolated, starts, accepted, y

def verify_measured_support(wave, hr, fps, starts, accepted):
    """Recompute original peak support from actual saved samples, without truth.

Motion scoring changes ranks, not peak detection or supported frequency grids.
This checks emitted HR has measured energy; it does not force V28 HR selection.
"""
    sys.path.insert(0, str(OLD_CODE))
    from evidence_hr import score_candidates
    from legacy_motion import estimate
    original_gates = estimate(wave.base.to_numpy(float), pd.DataFrame({'rgb_valid': bools(wave.observed)}),
        bools(wave.interpolated), fps, 10, 1, 42, 210)
    assert not (accepted & ~bools(original_gates.accepted)).any(), 'Accepted HR fails unchanged V28 validity gates'
    grid, width = np.arange(42., 211.), round(10*fps)
    for i in np.flatnonzero(accepted):
        claimed = json.loads(hr.iloc[i].evidence_candidates_json)
        y = float(hr.iloc[i].ridge_bpm)
        assert any(any(abs(float(bpm)-y) <= 1e-9 and float(power) >= .05-1e-12
            for bpm, power in zip(c['supported_bpm'], c['supported_relative_power'])) for c in claimed), 'Claimed HR lacks candidate support'
        measured, _ = score_candidates(wave.base.iloc[starts[i]:starts[i]+width].to_numpy(float), fps, grid)
        assert any(any(abs(float(bpm)-y) <= 1e-9 and float(power) >= .05-1e-12
            for bpm, power in zip(c['supported_bpm'], c['supported_relative_power'])) for c in measured), 'HR lacks actual saved-wave spectral support'

def group_error(y, reference, mask):
    valid = np.asarray(mask, bool) & np.isfinite(reference) & (reference > 0) & np.isfinite(y)
    errors = np.asarray(y)[valid]-np.asarray(reference)[valid]
    return dict(Nvalid=int(valid.sum()), Nwithin5=int((abs(errors) <= 5).sum()),
        MAE_bpm=float(abs(errors).mean()) if len(errors) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        P5_valid_pct=100*float((abs(errors) <= 5).mean()) if len(errors) else np.nan)

def compare(y, accepted, reference, old_y, old_accepted):
    common, new, lost = accepted & old_accepted, accepted & ~old_accepted, ~accepted & old_accepted
    groups = dict(common_new=group_error(y, reference, common), common_V28=group_error(old_y, reference, common),
                  newly_covered=group_error(y, reference, new), lost_V28=group_error(old_y, reference, lost))
    return dict(common_output_windows=int(common.sum()), new_output_windows=int(new.sum()),
                lost_output_windows=int(lost.sum()), **groups), (common, new, lost)

def pool_errors(rows):
    n = sum(row['Nvalid'] for row in rows)
    n5 = sum(row['Nwithin5'] for row in rows)
    return dict(Nvalid=n, Nwithin5=n5,
        MAE_bpm=sum(row['Nvalid']*row['MAE_bpm'] for row in rows if row['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(row['Nvalid']*row['RMSE_bpm']**2 for row in rows if row['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*n5/n if n else np.nan)

def aggregate(rows):
    result = pool_errors(rows)
    planned, nref = sum(r['Nplanned'] for r in rows), sum(r['Nref'] for r in rows)
    output = sum(r['Noutput'] for r in rows)
    result.update(Nplanned=planned, Noutput=output, Nref=nref,
        R5_all_reference_pct=100*result['Nwithin5']/nref if nref else np.nan,
        HR_coverage_pct=100*output/planned if planned else np.nan,
        waveform_time_coverage_pct=100*sum(r['waveform_finite_frames']/r['fps'] for r in rows)/
            sum(r['waveform_frames']/r['fps'] for r in rows),
        waveform_frame_coverage_pct=100*sum(r['waveform_finite_frames'] for r in rows)/sum(r['waveform_frames'] for r in rows))
    return result
