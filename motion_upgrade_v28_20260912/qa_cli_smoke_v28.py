"""Compare generic CLI caches against frozen batch signals; no reference files."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v28_20260912'
V25 = P/'results/data1_6_v25_20260911/stage4_preserve_waveform'
V26 = P/'results/data1_6_v26_20260911/component_harmonics'
WAVE_ABS_TOL = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_wave(actual, expected):
    a, b = pd.read_csv(actual), pd.read_csv(expected)
    same_clock = len(a) == len(b) and np.allclose(a.time_s, b.time_s, atol=1e-8, rtol=0)
    masks = {k: bool(np.array_equal(a[k], b[k])) for k in ('covered', 'observed', 'interpolated')}
    finite_a, finite_b = np.isfinite(a.base), np.isfinite(b.base)
    same_finite = bool(np.array_equal(finite_a, finite_b))
    common = finite_a & finite_b
    difference = np.abs(a.base[common].to_numpy()-b.base[common].to_numpy())
    maximum = float(difference.max()) if len(difference) else 0.
    result = dict(actual=str(actual), expected=str(expected), actual_sha256=sha(actual),
        expected_sha256=sha(expected), frames=len(a), same_clock=same_clock,
        same_finite_mask=same_finite, sampling_masks_equal=masks, max_abs_error=maximum,
        tolerance=WAVE_ABS_TOL, passed=bool(same_clock and same_finite and all(masks.values())
                                          and maximum <= WAVE_ABS_TOL))
    return result


def compare_hr(actual, expected):
    a, b = pd.read_csv(actual), pd.read_csv(expected)
    clock = len(a) == len(b) and np.allclose(a.time_s, b.time_s, atol=1e-8, rtol=0)
    accepted = bool(np.array_equal(a.accepted, b.accepted))
    finite = bool(np.array_equal(np.isfinite(a.ridge_bpm), np.isfinite(b.ridge_bpm)))
    values = bool(np.allclose(a.ridge_bpm, b.ridge_bpm, atol=0., rtol=0., equal_nan=True))
    return dict(actual=str(actual), expected=str(expected), actual_sha256=sha(actual),
        expected_sha256=sha(expected), planned_windows=len(a), same_clock=clock,
        accepted_equal=accepted, finite_mask_equal=finite, hr_exact=values,
        passed=bool(clock and accepted and finite and values))


def main():
    rows = []
    for case in ('data1', 'data5'):
        out = R/f'cli_smoke_{case}'
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['reference_used'] is False
        assert manifest['summary']['frames'] == manifest['video_probe']['decoded_frames']
        bound_outputs = all(sha(out/name) == digest for name, digest in manifest['outputs'].items())
        assert bound_outputs
        groups = [
            ('final', out, R/'direct_guard'/case, 'waveform.csv', 'heart_rate.csv'),
            ('fallback', out/'v25_fallback', V25/case, 'fusion_waveform.csv', 'fusion_heart_rate.csv'),
            ('candidate', out/'v26_harmonic_candidate', V26/case, 'waveform.csv', 'heart_rate.csv')]
        comparisons = {name: dict(waveform=compare_wave(local/'waveform.csv', historic/wave),
                                  heart_rate=compare_hr(local/'heart_rate.csv', historic/hr))
                       for name, local, historic, wave, hr in groups}
        actual_route = pd.read_csv(out/'routing_decisions.csv')
        batch_route = pd.read_csv(R/'direct_guard'/case/'routing_decisions.csv')
        route_equal = bool(np.array_equal(actual_route.route_eligible, batch_route.route_eligible))
        row = dict(case=case, decoded_frames=manifest['video_probe']['decoded_frames'],
            source_hashes=manifest['source_hashes'], manifest_sha256=sha(out/'manifest.json'),
            outputs_bound=bound_outputs, route_eligible_equal=route_equal,
            route_eligible_windows=int(actual_route.route_eligible.sum()), comparisons=comparisons)
        row['passed'] = route_equal and all(c[k]['passed'] for c in comparisons.values()
                                            for k in ('waveform', 'heart_rate'))
        rows.append(row)
    result = dict(reference_used=False, saved_waveform_absolute_tolerance=WAVE_ABS_TOL,
                  hr_tolerance_bpm=0., cases=rows, passed=all(row['passed'] for row in rows))
    (R/'qa_cli_smoke_v28.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit('CLI smoke differs from frozen batch: inspect, do not silently relax tolerances')


if __name__ == '__main__':
    main()
