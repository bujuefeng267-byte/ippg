"""Read saved waveforms and independently recalculate both HR estimators.

This auditor does not import the production estimator, fusion, pixel tracker or
reference evaluator. SciPy Welch is applied directly to the saved signal; gates,
window plan and offline dynamic programming are independently expressed below.
No reference labels or videos are read. All input artifacts are hashed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


HERE = Path(__file__).resolve().parent
CASES = ('user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1')
MODES = ('bounded_tracking_gap10', 'bounded_tracking_gap15',
         'guarded_fusion_gap10', 'guarded_fusion_gap15')
VARIANTS = ('fusion', 'pos', 'chrom')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def flag(series):
    values = series.to_numpy()
    if series.isna().any() or not np.isin(values, [True, False, 0, 1]).all():
        raise AssertionError(f'Invalid Boolean column {series.name}')
    return values.astype(bool)


def equal_numeric(actual, expected, label, atol=1e-9):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape:
        raise AssertionError(f'{label}: shape differs {actual.shape} != {expected.shape}')
    if not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        raise AssertionError(f'{label}: missing/finite masks differ')
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol,
                               equal_nan=True, err_msg=label)


def offline_path(power, grid, step_seconds):
    """Viterbi recurrence expressed over one contiguous accepted run."""
    power = np.asarray(power, float)
    emission = np.log(np.maximum(power / np.maximum(power.max(1)[:, None], 1e-30), 1e-12))
    # Axis 0 is destination, axis 1 is predecessor (production uses transpose).
    speed = (grid[:, None] - grid[None, :]) / max(step_seconds, 1e-6)
    transition = -.05 * speed**2
    transition[np.abs(grid[:, None]-grid[None, :]) > 12*step_seconds + 1e-9] = -np.inf
    parents = []
    total = emission[0].copy()
    for current_emission in emission[1:]:
        possibilities = transition + total[None, :]
        predecessor = np.argmax(possibilities, axis=1)
        parents.append(predecessor)
        total = possibilities[np.arange(len(grid)), predecessor] + current_emission
    states = [int(np.argmax(total))]
    for predecessors in reversed(parents):
        states.append(int(predecessors[states[-1]]))
    return grid[np.array(states[::-1], int)]


def recompute(signal, observed, interpolated, fps, window_seconds, step_seconds, low, high):
    n_window, n_step = round(window_seconds*fps), round(step_seconds*fps)
    starts = list(range(0, len(signal)-n_window+1, n_step))
    grid = np.arange(low, high+.01, 1.)
    spectra, records = [], []
    for start in starts:
        stop = start+n_window
        values = signal[start:stop]
        seen, filled = float(np.mean(observed[start:stop])), float(np.mean(interpolated[start:stop]))
        peak, concentration, spectrum = np.nan, np.nan, np.zeros(len(grid))
        if not np.isfinite(values).all():
            status = 'gap_or_filter_edge'
        elif seen < .9 or filled > .1:
            status = 'insufficient_observed_rgb'
        elif np.std(values) < 1e-8:
            status = 'flat_signal'
        else:
            frequency, psd = welch(values, fs=fps, nperseg=n_window,
                                   nfft=max(2048, n_window), detrend='constant')
            spectrum = np.interp(grid/60., frequency, psd)
            peak = grid[int(np.argmax(spectrum))]
            concentration = spectrum[np.abs(grid-peak) <= 6].sum() / max(spectrum.sum(), 1e-30)
            status = 'diffuse_spectrum' if concentration < .12 else 'candidate_only'
        spectra.append(spectrum)
        records.append(dict(time_s=(start+n_window/2)/fps,
                            observed_fraction=seen, interpolated_fraction=filled,
                            spectral_peak_bpm=peak, peak_concentration=concentration,
                            accepted=status == 'candidate_only', status=status, ridge_bpm=np.nan))
    table = pd.DataFrame(records, columns=['time_s', 'observed_fraction', 'interpolated_fraction',
                       'spectral_peak_bpm', 'peak_concentration', 'accepted', 'status', 'ridge_bpm'])
    accepted = flag(table.accepted)
    index = 0
    while index < len(table):
        if not accepted[index]:
            index += 1
            continue
        stop = index+1
        while stop < len(table) and accepted[stop]:
            stop += 1
        table.loc[index:stop-1, 'ridge_bpm'] = offline_path(spectra[index:stop], grid, n_step/fps)
        index = stop
    return table, starts, n_window


def check_provenance(wave, trace, roi, proposals, diagnostics, hr, starts, n_window):
    n_frames = len(trace)
    covered = np.zeros(n_frames, bool)
    observed = np.ones(n_frames, bool)
    interpolated = np.zeros(n_frames, bool)
    generated = np.zeros(len(starts), bool)
    used_count = np.zeros(len(starts), int)
    weights = diagnostics.waveform_weight.to_numpy(float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise AssertionError('Invalid fusion contributor weights')
    for raw_index, group in diagnostics.loc[weights > 0].groupby('window_index'):
        index = int(raw_index)
        if index != raw_index or not 0 <= index < len(starts):
            raise AssertionError('Invalid contributor window index')
        rois = set(group.roi)
        if len(rois) < 2 or not flag(proposals.proposal_accepted)[index]:
            raise AssertionError('Positive weights lack accepted multi-ROI proposal')
        if proposals.waveform_roi_count.iloc[index] != len(rois):
            raise AssertionError('Proposal ROI count differs from positive weights')
        generated[index], used_count[index] = True, len(rois)
        section = slice(starts[index], starts[index]+n_window)
        covered[section] = True
        for name in rois:
            observed[section] &= flag(trace[f'{name}_valid'])[section]
            interpolated[section] |= flag(roi[f'{name}_interpolated'])[section]
    observed &= covered
    interpolated &= covered
    for key, expected in [('covered', covered), ('observed', observed), ('interpolated', interpolated)]:
        np.testing.assert_array_equal(flag(wave[key]), expected, err_msg=f'Fusion {key} provenance')
    np.testing.assert_array_equal(np.isfinite(wave.base), covered)
    np.testing.assert_array_equal(flag(hr.waveform_generated), generated)
    np.testing.assert_array_equal(flag(proposals.waveform_generated), generated)
    np.testing.assert_array_equal(hr.waveform_roi_count, used_count)
    fully_covered = np.array([covered[a:a+n_window].all() for a in starts])
    np.testing.assert_array_equal(flag(hr.neighbor_covered), fully_covered & ~generated)


def audit_case(directory, hashes):
    def csv(name):
        path = directory/name
        hashes[str(path)] = sha(path)
        return pd.read_csv(path)
    summary_path = directory/'summary.json'
    hashes[str(summary_path)] = sha(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for name,expected in summary['source_hashes'].items():
        if sha(HERE/name)!=expected:raise AssertionError(f'Inference source changed: {name}')
    trace = csv('frame_trace.csv')
    fps = float(summary['fps'])
    config = summary['config']
    window_s = float(config['window'])
    step_s = float(config['step'])
    np.testing.assert_array_equal(trace.frame, np.arange(len(trace)))
    equal_numeric(trace.time_s, np.arange(len(trace))/fps, 'Frame time axis', atol=1e-7)
    assert len(trace) == summary['frames']
    roi, proposals, diagnostics = csv('roi_waveforms.csv'), csv('fusion_proposals.csv'), csv('fusion_diagnostics.csv')
    equal_numeric(roi.time_s, trace.time_s, 'ROI waveform time axis')
    checked = []
    for variant in VARIANTS:
        wave = csv(f'{variant}_waveform.csv')
        hr = csv(f'{variant}_heart_rate.csv')
        equal_numeric(wave.time_s, trace.time_s, f'{variant} waveform time axis')
        original, starts, n_window = recompute(wave.base.to_numpy(float), flag(wave.observed),
            flag(wave.interpolated), fps, window_s, step_s,
            float(config['min_bpm']), float(config['max_bpm']))
        if variant == 'fusion':
            check_provenance(wave, trace, roi, proposals, diagnostics, hr, starts, n_window)
            equal_numeric(hr.raw_spectral_peak_bpm, original.spectral_peak_bpm, 'Raw local peak')
            original.loc[~flag(original.accepted), 'spectral_peak_bpm'] = np.nan
            equal_numeric(proposals.time_s, original.time_s, 'Proposal window centers')
            equal_numeric(hr.window_start_s, np.asarray(starts)/fps, 'HR window starts')
            equal_numeric(hr.window_end_s, (np.asarray(starts)+n_window)/fps, 'HR window ends')
        else:
            np.testing.assert_array_equal(flag(wave.observed), flag(trace.rgb_valid))
        for name in ['time_s', 'observed_fraction', 'interpolated_fraction',
                     'spectral_peak_bpm', 'ridge_bpm', 'peak_concentration']:
            equal_numeric(hr[name], original[name], f'{variant}.{name}')
        np.testing.assert_array_equal(flag(hr.accepted), flag(original.accepted))
        np.testing.assert_array_equal(hr.status.to_numpy(), original.status.to_numpy())
        stat = summary['variants'][variant]
        assert stat['total_windows'] == len(hr)
        assert stat['accepted_windows'] == int(flag(original.accepted).sum())
        equal_numeric(stat['finite_waveform_fraction'], float(np.isfinite(wave.base).mean()), 'Saved coverage')
        checked.append(dict(variant=variant, planned_windows=len(hr), accepted_windows=int(flag(hr.accepted).sum()),
                            finite_wave_samples=int(np.isfinite(wave.base).sum()), total_frames=len(wave),
                            independently_recomputed_estimators=['local_peak', 'offline_ridge']))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root', type=Path, default=HERE/'validation')
    parser.add_argument('--baseline-smoke', type=Path,
                        help='Audit six existing baseline cases without issuing a candidate receipt')
    args = parser.parse_args()
    hashes, records, errors = {}, [], []
    jobs = [('baseline_smoke', case, args.baseline_smoke/case) for case in CASES] if args.baseline_smoke else [
        (mode, case, args.validation_root/mode/case) for mode, case in itertools.product(MODES, CASES)]
    for mode, case, directory in jobs:
        try:
            records.append(dict(mode=mode, case=case, checked=audit_case(directory, hashes), passed=True))
            print(f'PASS {mode}/{case}: saved POS, CHROM and fusion -> local/DP HR', flush=True)
        except Exception as exc:
            errors.append(dict(mode=mode, case=case, error=f'{type(exc).__name__}: {exc}'))
            print(f'FAIL {mode}/{case}: {exc}', flush=True)
    changed = [path for path, expected in hashes.items() if sha(path) != expected]
    if changed:
        errors.append(dict(error='Inputs changed during audit', paths=changed))
    report = dict(passed=not errors, checked_cases=len(CASES), checked_modes=1 if args.baseline_smoke else len(MODES),
                  checked_variants=list(VARIANTS), checked_estimators=['local_peak', 'offline_ridge'],
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  methodology='Independent saved-waveform Welch, gate and offline DP recomputation; fusion provenance rebuilt from positive contributor weights; no production estimator imported.',
                  qa_script_sha256=sha(__file__), reference_used=False, input_sha256=hashes,
                  hr_sha256={}, results=records, errors=errors)
    if args.baseline_smoke:
        output = HERE/'baseline_waveform_qa_smoke.json'
    else:
        output = args.validation_root/'waveform_replay_qa.json'
        protocol_path=args.validation_root/'protocol_before_validation.json'
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        report['source_hashes']=protocol['source_hashes']
        report['protocol_sha256']=sha(protocol_path)
        for name,expected in report['source_hashes'].items():
            if sha(HERE/name)!=expected:raise AssertionError(f'Frozen source changed: {name}')
        for mode, case, variant in itertools.product(MODES, CASES, VARIANTS):
            rel = f'{mode}/{case}/{variant}_heart_rate.csv'
            absolute = str(args.validation_root/rel)
            if absolute in hashes:
                report['hr_sha256'][rel] = hashes[absolute]
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=not errors, audited_cases=len(records), output=str(output), errors=errors),
                     ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
