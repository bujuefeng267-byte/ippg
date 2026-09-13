"""Independent saved-output V25 evidence HR receipt; no reference or video reads.

Production estimators are never imported. The earlier independent V24 auditor
supplies only its separately expressed legacy Welch/gate/DP and provenance QA.
This file independently re-expresses V25 motion evidence, candidate scoring and
reacquisition DP, checking every serialized candidate and supported output.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD_QA = HERE/'qa_recompute_waveform_hr_v24.py'
if not OLD_QA.exists():
    OLD_QA = HERE.parent/'rppg_motion_v24/qa_recompute_waveform_hr.py'
spec = importlib.util.spec_from_file_location('independent_v24_receipt', OLD_QA)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
DEFAULT_ROOT = Path('/home/fengbujue/项目/rppg识别/results/data1_6_v25_20260911')
PARAMS = dict(min_relative_power=.05, min_prominence=.02, peak_distance_bpm=6.,
    candidate_radius_bpm=3., max_candidates=8, spectral_weight=.35,
    periodic_weight=1.10, harmonic_weight=1., motion_weight=1.25,
    third_harmonic_weight=.35, transition_penalty=.05,
    max_rate_bpm_per_s=12., reacquisition_cost=3., candidate_offset_penalty=.025)


def compare(actual, expected, label):
    """Check nested candidate diagnostics without trusting stored score fields."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), label+' fields differ'
        for key in expected:
            compare(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label+' length differs'
        for i, (got, want) in enumerate(zip(actual, expected)):
            compare(got, want, label+f'[{i}]')
    elif expected is None or isinstance(expected, (bool, str)):
        assert actual == expected, label+f': {actual} != {expected}'
    else:
        old.equal_numeric(actual, expected, label, atol=2e-7)


def stretches(mask):
    offset = 0
    for valid, run in itertools.groupby(np.asarray(mask, bool)):
        length = sum(1 for _ in run)
        if valid:
            yield offset, offset+length
        offset += length


def independent_motion(trace, start, stop, fps, grid):
    """Physical unit conversion and gap-preserving segment spectra."""
    block = trace.iloc[start:stop]
    def column(name):
        return (pd.to_numeric(block[name], errors='coerce').to_numpy(float)
                if name in block else np.full(len(block), np.nan))
    height = column('face_y1')-column('face_y0')
    delta = np.column_stack([column('motion_x'), column('motion_y')])
    valid = np.isfinite(delta).all(axis=1) & np.isfinite(height) & (height > 1e-6)
    unavailable = (np.zeros(len(grid)), None, 0., False)
    if valid.mean() < .5:
        return unavailable
    velocity = np.full_like(delta, np.nan)
    velocity[valid] = delta[valid] / height[valid, None] * fps
    powers, lengths = [], []
    for left, right in stretches(valid):
        size = right-left
        if size < max(4, int(np.ceil(3.*fps-1e-9))):
            continue
        total_power = np.zeros(len(grid))
        for axis in range(2):
            values = velocity[left:right, axis]
            if np.std(values) < 1e-10:
                continue
            fft_length = max(8192, 2**int(np.ceil(np.log2(max(4*size, 256*fps)))))
            hz, density = welch(values, fs=fps, window='hann', nperseg=size,
                                noverlap=0, nfft=fft_length, detrend='constant')
            total_power += np.interp(grid/60., hz, density)
        powers.append(total_power)
        lengths.append(size)
    if sum(lengths)/len(block) < .5:
        return unavailable
    speed_squared = np.mean(np.sum(velocity[valid]**2, axis=1))
    strength = float(speed_squared/(speed_squared+.25**2))
    reliability = float(sum(lengths)/len(block))
    combined = sum(p*n for p, n in zip(powers, lengths))/sum(lengths)
    normalized = combined/combined.max() if combined.max() > 1e-20 else np.zeros(len(grid))
    return np.clip(normalized*strength*reliability, 0, 1), strength, reliability, True


def correlation(values, fps, bpm):
    period = 60*fps/bpm
    coordinates = np.arange(int(np.ceil(period)), len(values))
    if len(coordinates) < 4:
        return 0.
    pair = np.stack([values[coordinates], np.interp(coordinates-period,
                                  np.arange(len(values)), values)])
    pair -= pair.mean(axis=1, keepdims=True)
    normalizer = np.sqrt(np.sum(pair[0]**2)*np.sum(pair[1]**2))
    return float(np.clip(np.sum(pair[0]*pair[1])/normalizer, -1, 1)) if normalizer > 1e-15 else 0.


def candidates_from_saved(values, fps, grid, motion):
    frequencies, psd = welch(values, fs=fps, nperseg=len(values),
                             nfft=max(2048, len(values)), detrend='constant')
    band = np.interp(grid/60, frequencies, psd)
    relative = band/band.max()
    locs, props = find_peaks(np.concatenate(([0], relative, [0])),
        height=PARAMS['min_relative_power'], prominence=PARAMS['min_prominence'],
        distance=round(PARAMS['peak_distance_bpm']))
    peak_rows = [(int(loc-1), float(prominence)) for loc, prominence in
                 zip(locs, props['prominences']) if 0 < loc <= len(grid)]
    peak_rows.sort(key=lambda row: (-relative[row[0]], grid[row[0]]))
    profile, strength, reliability, available = motion
    rows = []
    emissions = np.full(len(grid), -np.inf)
    for index, prominence in peak_rows[:PARAMS['max_candidates']]:
        bpm = float(grid[index])
        full_corr = correlation(values, fps, bpm)
        halves = (values[:len(values)//2], values[len(values)//2:])
        split_corr = sum(correlation(half, fps, bpm) for half in halves)/2
        periodic = max(0., .75*full_corr+.25*split_corr)
        h2 = float(np.interp(2*bpm/60, frequencies, psd, right=0)/band.max())
        h3 = float(np.interp(3*bpm/60, frequencies, psd, right=0)/band.max())
        harmonic = float(min(1., np.sqrt(relative[index]*max(h2, 0))+
                                    PARAMS['third_harmonic_weight']*np.sqrt(relative[index]*max(h3, 0))))
        overlap = float(profile[index])
        score = float(PARAMS['spectral_weight']*np.log(relative[index])+
            PARAMS['periodic_weight']*periodic+PARAMS['harmonic_weight']*harmonic-
            PARAMS['motion_weight']*overlap)
        support = np.flatnonzero((np.abs(grid-bpm) <= PARAMS['candidate_radius_bpm']) &
                                (relative >= PARAMS['min_relative_power']))
        for state in support:
            emissions[state] = max(emissions[state], score-
                PARAMS['candidate_offset_penalty']*(grid[state]-bpm)**2)
        rows.append(dict(bpm=bpm, relative_power=float(relative[index]),
            peak_concentration=float(band[np.abs(grid-bpm) <= 6].sum()/band.sum()),
            prominence=prominence, period_correlation=full_corr,
            split_period_correlation=split_corr, periodic_support=periodic,
            harmonic2_relative_power=h2, harmonic3_relative_power=h3, harmonic_support=harmonic,
            motion_profile=overlap, motion_strength=strength, motion_reliability=reliability,
            motion_overlap=overlap, motion_available=available, evidence_score=score,
            supported_bpm=grid[support].tolist(), supported_relative_power=relative[support].tolist()))
    rows.sort(key=lambda row: (-row['evidence_score'], row['bpm']))
    return rows, emissions


def independent_path(emissions, grid, step_s):
    """Destination-first recurrence and backward reconstruction."""
    distance = np.abs(grid[:, None]-grid[None, :])
    ordinary = -PARAMS['transition_penalty']*(distance/step_s)**2
    ordinary[distance > PARAMS['max_rate_bpm_per_s']*step_s+1e-9] = -np.inf
    cost = np.maximum(ordinary, -PARAMS['reacquisition_cost'])
    score, parents = np.asarray(emissions[0]).copy(), []
    for emission in emissions[1:]:
        alternatives = cost+score[None, :]
        parent = alternatives.argmax(axis=1)
        parents.append(parent)
        score = alternatives[np.arange(len(grid)), parent]+emission
    states = [int(score.argmax())]
    for parent in reversed(parents):
        states.append(int(parent[states[-1]]))
    states = np.asarray(states[::-1], int)
    reacquired = np.zeros(len(states), bool)
    reacquired[1:] = ordinary[states[1:], states[:-1]] < -PARAMS['reacquisition_cost']
    return states, reacquired


def serialized_emissions(candidates, grid):
    """Saved candidate scalars retain original floating precision for tie replay."""
    scores = np.full(len(grid), -np.inf)
    for row in candidates:
        for frequency in row['supported_bpm']:
            state = int(np.flatnonzero(grid == frequency)[0])
            scores[state] = max(scores[state], row['evidence_score']-
                PARAMS['candidate_offset_penalty']*(frequency-row['bpm'])**2)
    return scores


def objective(states, emissions, grid, step_s):
    score = sum(emissions[i][state] for i, state in enumerate(states))
    for prior, current in zip(states[:-1], states[1:]):
        distance = abs(grid[current]-grid[prior])
        ordinary = (-PARAMS['transition_penalty']*(distance/step_s)**2 if
                    distance <= PARAMS['max_rate_bpm_per_s']*step_s+1e-9 else -np.inf)
        score += max(ordinary, -PARAMS['reacquisition_cost'])
    return float(score)


def audit(directory, hashes, tie_events, preservation):
    def read(name):
        path = directory/name
        hashes[str(path)] = old.sha(path)
        return pd.read_csv(path)
    path = directory/'summary.json'
    hashes[str(path)] = old.sha(path)
    summary = json.loads(path.read_text())
    for name, expected in summary['source_hashes'].items():
        assert old.sha(HERE/name) == expected, 'Frozen source changed: '+name
    config = summary['config']
    assert config['hr_mode'] == 'evidence'
    expected_motion = 'legacy' if directory.parent.name == 'stage4_preserve_waveform' else 'reliable'
    assert config['motion_evidence'] == expected_motion
    assert summary['reference_identity'] is None and not config.get('reference_ubfc')
    fps = float(summary['fps'])
    trace, roi = read('frame_trace.csv'), read('roi_waveforms.csv')
    proposals, diagnostics = read('fusion_proposals.csv'), read('fusion_diagnostics.csv')
    old.equal_numeric(trace.frame, np.arange(len(trace)), 'Frames')
    old.equal_numeric(trace.time_s, np.arange(len(trace))/fps, 'Trace time', atol=1e-7)
    assert len(trace) == summary['frames']
    old.equal_numeric(roi.time_s, trace.time_s, 'ROI time')
    result = []
    for variant in ('pos', 'chrom', 'fusion'):
        wave, hr = read(variant+'_waveform.csv'), read(variant+'_heart_rate.csv')
        old.equal_numeric(wave.time_s, trace.time_s, variant+' time')
        legacy, starts, window = old.recompute(wave.base.to_numpy(float),
            old.flag(wave.observed), old.flag(wave.interpolated), fps,
            config['window'], config['step'], config['min_bpm'], config['max_bpm'])
        if variant == 'fusion':
            old.check_provenance(wave, trace, roi, proposals, diagnostics, hr, starts, window)
            old.equal_numeric(hr.raw_spectral_peak_bpm, legacy.spectral_peak_bpm, 'Raw peak')
            old.equal_numeric(hr.legacy_ridge_bpm, legacy.ridge_bpm, 'Retained legacy DP')
            legacy.loc[~old.flag(legacy.accepted), 'spectral_peak_bpm'] = np.nan
            old.equal_numeric(hr.window_start_s, np.asarray(starts)/fps, 'Window starts')
            old.equal_numeric(hr.window_end_s, (np.asarray(starts)+window)/fps, 'Window ends')
            old.equal_numeric(proposals.time_s, hr.time_s, 'Proposal centers')
        else:
            np.testing.assert_array_equal(old.flag(wave.observed), old.flag(trace.rgb_valid))
        for column in ('time_s', 'observed_fraction', 'interpolated_fraction',
                       'spectral_peak_bpm', 'peak_concentration'):
            old.equal_numeric(hr[column], legacy[column], variant+'.'+column)
        np.testing.assert_array_equal(old.flag(hr.accepted), old.flag(legacy.accepted))
        np.testing.assert_array_equal(hr.status, legacy.status)
        if variant != 'fusion':
            old.equal_numeric(hr.ridge_bpm, legacy.ridge_bpm, variant+' legacy DP')
        else:
            grid = np.arange(config['min_bpm'], config['max_bpm']+.01, 1.)
            emissions, candidates, exact_emissions = [], [], []
            for i, start in enumerate(starts):
                if not bool(legacy.accepted.iloc[i]):
                    assert json.loads(hr.evidence_candidates_json.iloc[i]) == []
                    assert hr.evidence_candidate_count.iloc[i] == 0
                    assert hr.evidence_tracker_state.iloc[i] == 'unavailable'
                    assert not bool(hr.evidence_motion_available.iloc[i])
                    assert not bool(hr.evidence_reacquired.iloc[i])
                    for col in ('ridge_bpm', 'evidence_score', 'evidence_local_bpm',
                                'evidence_selected_relative_power', 'evidence_selected_peak_bpm'):
                        assert pd.isna(hr[col].iloc[i]), 'Rejected window contains '+col
                    candidates.append([])
                    emissions.append(np.full(len(grid), -np.inf))
                    exact_emissions.append(np.full(len(grid), -np.inf))
                    continue
                motion = independent_motion(trace, start, start+window, fps, grid)
                rows, emission = candidates_from_saved(wave.base.to_numpy(float)[start:start+window], fps, grid, motion)
                assert rows, 'Accepted waveform has no supported candidate'
                serialized = json.loads(hr.evidence_candidates_json.iloc[i])
                compare(serialized, rows, f'window{i}.candidates')
                assert len(rows) == hr.evidence_candidate_count.iloc[i]
                assert rows[0]['motion_available'] == bool(hr.evidence_motion_available.iloc[i])
                old.equal_numeric(hr.evidence_local_bpm.iloc[i], rows[0]['bpm'], 'Evidence local')
                margin = rows[0]['evidence_score']-rows[1]['evidence_score'] if len(rows) > 1 else np.nan
                old.equal_numeric(hr.evidence_runner_up_margin.iloc[i], margin, 'Evidence margin', atol=2e-7)
                candidates.append(rows)
                emissions.append(emission)
                exact_emissions.append(serialized_emissions(serialized, grid))
            for left, right in stretches(old.flag(legacy.accepted)):
                step_s = round(config['step']*fps)/fps
                recomputed_states, _ = independent_path(emissions[left:right], grid, step_s)
                states, reacquired = independent_path(exact_emissions[left:right], grid, step_s)
                old.equal_numeric(hr.ridge_bpm.iloc[left:right], grid[states], 'Saved-candidate exact DP replay')
                # At score ties ~machine precision, mathematically equivalent
                # summation orders can pick different best paths. Require the
                # saved path to attain the independently recomputed optimum;
                # retain every disagreement instead of concealing it in a BPM
                # tolerance. Exact serialized-candidate replay is mandatory.
                saved_score = objective(states, emissions[left:right], grid, step_s)
                best_score = objective(recomputed_states, emissions[left:right], grid, step_s)
                old.equal_numeric(saved_score, best_score, 'Independently optimal path objective', atol=1e-10)
                different = np.flatnonzero(states != recomputed_states)
                if len(different):
                    maximum_delta = max(float(np.max(abs(a[np.isfinite(a)]-b[np.isfinite(b)])))
                        for a, b in zip(emissions[left:right], exact_emissions[left:right]))
                    tie_events.append(dict(stage=directory.parent.name, case=directory.name,
                        contiguous_start_window=left, contiguous_stop_window=right,
                        differing_window_indices=(different+left).tolist(),
                        saved_bpm=grid[states[different]].tolist(),
                        independently_recomputed_bpm=grid[recomputed_states[different]].tolist(),
                        saved_path_objective=saved_score, independently_optimal_objective=best_score,
                        objective_difference=best_score-saved_score,
                        maximum_emission_roundoff=maximum_delta,
                        saved_candidate_json_dp_exact_replay=True,
                        interpretation='Floating-point tied optima; output is supported and optimal, but the best sequence is not numerically unique.'))
                np.testing.assert_array_equal(old.flag(hr.evidence_reacquired.iloc[left:right]), reacquired)
                for offset, state in enumerate(states):
                    i, bpm = left+offset, float(grid[state])
                    eligible = [row for row in candidates[i] if bpm in row['supported_bpm']]
                    assert eligible, 'DP produced unsupported state'
                    selected = max(eligible, key=lambda row: row['evidence_score']-
                        PARAMS['candidate_offset_penalty']*(row['bpm']-bpm)**2)
                    power = selected['supported_relative_power'][selected['supported_bpm'].index(bpm)]
                    assert power >= .05 and abs(bpm-selected['bpm']) <= 3
                    expected = dict(evidence_score=emissions[i][state],
                        evidence_selected_relative_power=power, evidence_selected_peak_bpm=selected['bpm'],
                        evidence_selected_periodic_support=selected['periodic_support'],
                        evidence_selected_motion_overlap=selected['motion_overlap'])
                    for name, value in expected.items():
                        old.equal_numeric(hr[name].iloc[i], value, name, atol=2e-7)
                    expected_state = 'acquired' if not offset else 'reacquired' if reacquired[offset] else 'tracked'
                    assert hr.evidence_tracker_state.iloc[i] == expected_state
        stat = summary['variants'][variant]
        assert stat['total_windows'] == len(hr) and stat['accepted_windows'] == int(legacy.accepted.sum())
        old.equal_numeric(stat['finite_waveform_fraction'], np.isfinite(wave.base).mean(), 'Coverage')
        result.append(dict(variant=variant, frames=len(trace), planned_windows=len(hr),
            accepted_windows=int(legacy.accepted.sum()), finite_wave_samples=int(np.isfinite(wave.base).sum()),
            independently_recomputed=['local_peak', 'legacy_DP'] if variant != 'fusion' else
                ['local_peak', 'retained_legacy_DP', 'motion_profile', 'all_candidate_evidence', 'evidence_DP', 'reacquisition']))
    if directory.parent.name == 'stage2_hr':
        prior = directory.parent.parent/'stage1_motion'/directory.name
        for name in ('frame_trace.csv', 'roi_waveforms.csv', 'fusion_waveform.csv',
                     'fusion_proposals.csv', 'fusion_diagnostics.csv', 'branch_routing.csv'):
            hashes[str(prior/name)] = old.sha(prior/name)
            hashes[str(directory/name)] = old.sha(directory/name)
            assert hashes[str(prior/name)] == hashes[str(directory/name)], 'Stage2 changed '+name
    if directory.parent.name == 'stage4_preserve_waveform':
        baseline = directory.parent.parent.parent/'data1_6_20260911'/directory.name/'inference'
        replay = directory.parent.parent/'v24_replay'/directory.name
        preservation_row = dict(case=directory.name, files={}, accepted_mask_exactly_equal=True)
        for name in ('fusion_waveform.csv', 'frame_trace.csv', 'roi_waveforms.csv',
                     'fusion_proposals.csv', 'fusion_diagnostics.csv', 'branch_routing.csv'):
            hashes[str(baseline/name)] = old.sha(baseline/name)
            hashes[str(directory/name)] = old.sha(directory/name)
            hashes[str(replay/name)] = old.sha(replay/name)
            original, current, repeated = pd.read_csv(baseline/name), pd.read_csv(directory/name), pd.read_csv(replay/name)
            # New diagnostic columns are allowed; every original numerical
            # waveform, geometry, sampling and contributor field agrees within
            # numeric precision. Cache CSV reload adds tiny floating roundoff
            # compared with original V24; same-round legacy replay is exact.
            pd.testing.assert_frame_equal(current[repeated.columns], repeated, check_exact=True)
            maximum = 0.
            for column in original.columns:
                left, right = current[column], original[column]
                if pd.api.types.is_bool_dtype(right) or not pd.api.types.is_numeric_dtype(right):
                    pd.testing.assert_series_equal(left, right, check_exact=True)
                else:
                    old.equal_numeric(left, right, 'Original V24 '+name+'.'+column, atol=1e-9)
                    finite = np.isfinite(right.to_numpy(float))
                    if finite.any():
                        maximum = max(maximum, float(np.max(abs(left.to_numpy(float)[finite]-right.to_numpy(float)[finite]))))
            preservation_row['files'][name] = dict(exactly_equal_to_same_round_legacy_replay=True,
                original_v24_maximum_absolute_numeric_difference=maximum,
                original_v24_absolute_tolerance=1e-9, original_discrete_and_missingness_masks_equal=True)
        baseline_hr = baseline/'fusion_heart_rate.csv'
        hashes[str(baseline_hr)] = old.sha(baseline_hr)
        np.testing.assert_array_equal(old.flag(hr.accepted), old.flag(pd.read_csv(baseline_hr).accepted))
        preservation.append(preservation_row)
    return result


def selftest():
    """Audit recurrence against brute force, then check known-energy boundaries."""
    grid = np.array([42., 84., 168.])
    emission = np.array([[1., 0., -np.inf], [0., .3, -np.inf], [-np.inf, 0., 5.]])
    states, jumps = independent_path(emission, grid, 1.)
    def total(path):
        score = sum(emission[i, state] for i, state in enumerate(path))
        for a, b in zip(path[:-1], path[1:]):
            distance = abs(grid[b]-grid[a])
            score += max(-.05*distance**2 if distance <= 12 else -np.inf, -3.)
        return score
    best = max(itertools.product(range(3), repeat=3), key=total)
    assert tuple(states) == best and jumps[-1]
    grid = np.arange(42., 211.)
    t = np.arange(300)/30
    rows, emissions = candidates_from_saved(np.sin(2*np.pi*3*t), 30., grid,
                                            (np.zeros(len(grid)), None, 0., False))
    assert rows[0]['bpm'] == 180. and not np.isfinite(emissions[grid == 90]).any()
    for fps in (30., 180.):
        t = np.arange(round(10*fps))/fps
        trace = pd.DataFrame(dict(motion_x=.1*np.sin(2*np.pi*1.4*t)*.3/fps,
                                 motion_y=0., face_y0=.2, face_y1=.5))
        profile, strength, reliability, available = independent_motion(trace, 0, len(trace), fps, grid)
        assert available and reliability == 1. and grid[profile.argmax()] == 84.
        assert abs(strength-(.005/(.005+.25**2))) < 1e-10
    print('Independent QA selftest passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--stages', nargs='+', default=['stage2_hr', 'stage3_routing', 'stage4_preserve_waveform'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.root/'qa_v25_evidence.json'
    if args.selftest:
        selftest()
        return
    # Defaults are independently recorded, with an AST-only drift check.
    tree = ast.parse((HERE/'evidence_hr.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvidenceConfig')
    recorded = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert recorded == PARAMS, 'Evidence defaults changed; independent QA needs review'
    protocol_path = args.root/'protocol_before_validation.json'
    protocol = json.loads(protocol_path.read_text())
    hashes = {str(protocol_path): old.sha(protocol_path), str(OLD_QA): old.sha(OLD_QA)}
    records, errors, tie_events, preservation = [], [], [], []
    for stage in args.stages:
        runs_path = args.root/stage/'runs.json'
        hashes[str(runs_path)] = old.sha(runs_path)
        runs = json.loads(runs_path.read_text())
        assert len(runs) == 6 and all(row['returncode'] == 0 for row in runs)
        assert {row['case'] for row in runs} == {f'data{i}' for i in range(1, 7)}
        for case in sorted(row['case'] for row in runs):
            try:
                result = audit(args.root/stage/case, hashes, tie_events, preservation)
                records.append(dict(stage=stage, case=case, passed=True, branches=result))
                print(f'PASS {stage}/{case}: masks, waveforms, local and DP, every candidate', flush=True)
            except Exception as exc:
                errors.append(dict(stage=stage, case=case, error=f'{type(exc).__name__}: {exc}'))
                print(f'FAIL {stage}/{case}: {exc}', flush=True)
    changed = [path for path, checksum in hashes.items() if old.sha(path) != checksum]
    if changed:
        errors.append(dict(error='Audit inputs changed', paths=changed))
    for name, expected in protocol['source_hashes'].items():
        assert old.sha(HERE/name) == expected, 'Frozen source changed: '+name
    report = dict(passed=not errors, checked_utc=datetime.now(timezone.utc).isoformat(),
        methodology=__doc__, reference_used=False, raw_video_read=False,
        source_hashes=protocol['source_hashes'], independent_evidence_config=PARAMS,
        qa_script_sha256=old.sha(__file__), input_sha256=hashes,
        audit_cases=len(records), results=records, floating_point_tied_paths=tie_events,
        stage4_waveform_preservation=preservation,
        all_serialized_candidate_paths_replayed_exactly=not errors,
        all_recomputed_paths_unique_and_identical=not errors and not tie_events,
        errors=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=not errors, cases=len(records), output=str(args.output), errors=errors), ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
