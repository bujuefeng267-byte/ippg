"""Read-only replay of actual measured-component/source-mask evidence.

No inference module or reference/HR label is imported. This reconstructs only
the stored proposal path from the original saved optical waveforms, never
selects a new HR path, and writes a separate QA receipt.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT/'results/data1_6_20260911'
ROOT = PROJECT/'results/data1_6_v26_20260911/component_consensus'
HERE = Path(__file__).resolve().parent
ROIS = ('forehead', 'left_cheek', 'right_cheek')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def correlation(x, y):
    x, y = x-x.mean(), y-y.mean()
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x, y)/norm) if norm > 1e-12 else 0.


def component(x, fps, frequency_bpm, cfg):
    # Independently solve linear detrending, rather than call production helper.
    t = np.arange(len(x), dtype=float)
    design = np.column_stack([np.ones(len(x)), t])
    residual = x-design@np.linalg.lstsq(design, x, rcond=None)[0]
    padded = np.pad(residual, (len(x), len(x)), mode='reflect')
    bpm = np.fft.rfftfreq(len(padded), 1/fps)*60
    distance = abs(bpm-frequency_bpm)
    gain = np.zeros(len(bpm))
    low, high = cfg['pass_half_width_bpm'], cfg['stop_half_width_bpm']
    gain[distance <= low] = 1.
    slope = (distance > low) & (distance < high)
    gain[slope] = (1+np.cos(np.pi*(distance[slope]-low)/(high-low)))/2
    filtered = np.fft.irfft(np.fft.rfft(padded)*gain, n=len(padded))
    return filtered[len(x):2*len(x)]


def main():
    receipt = ROOT.parent/'qa_component_signal_evidence_v26.json'
    if receipt.exists():
        raise FileExistsError('Preserve prior QA receipts.')
    protocol = json.loads((ROOT/'protocol_before_run.json').read_text())
    cfg = protocol['component_config']
    assert protocol['reference_used'] is False and protocol['post_diagnosis_development'] is True
    assert cfg['min_rois'] == 2
    errors, checked, hashes = [], [], {}
    for name, expected in protocol['source_hashes'].items():
        assert sha(HERE/name) == expected, 'Frozen source changed: '+name
        hashes[str(HERE/name)] = expected
    for case in [f'data{i}' for i in range(1, 7)]:
        directory, original = ROOT/case, BASE/case/'inference'
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['source_hashes'] == protocol['source_hashes']
        assert summary['reference_used'] is False and summary['hr_from_saved_waveform'] is True
        trace = pd.read_csv(original/'frame_trace.csv')
        roi_wave = pd.read_csv(original/'baseline_roi_waveforms.csv')
        tracked = pd.read_csv(original/'roi_waveforms.csv')
        wave = pd.read_csv(directory/'waveform.csv')
        proposals = pd.read_csv(directory/'component_proposals.csv')
        records = json.loads((directory/'all_roi_candidates.json').read_text())
        channels = {}
        for branch, table in [('baseline',roi_wave), ('tracked',tracked)]:
            np.testing.assert_allclose(table.time_s, trace.time_s, rtol=0, atol=1e-8)
            for roi in ROIS:
                for suffix in ['observed', 'interpolated']:
                    np.testing.assert_array_equal(table[f'{roi}_{suffix}'], roi_wave[f'{roi}_{suffix}'])
                for method in ['pos','chrom']:
                    channels[f'{branch}/{roi}/{method}'] = table[f'{roi}_{method}'].to_numpy(float)
        n, fps = len(trace), float(summary['fps'])
        width, hop = round(10*fps), round(fps)
        starts = np.arange(0,n-width+1,hop)
        assert len(starts) == len(proposals)
        by_window = {i: {} for i in range(len(starts))}
        for record in records:
            by_window[record['window_index']].setdefault(record['channel'], []).append(record)
        total, weights = np.zeros(n), np.zeros(n)
        observed, interpolated = np.ones(n,bool), np.zeros(n,bool)
        taper = np.hanning(width+2)[1:-1]
        generated_count = 0
        for wi, start in enumerate(starts):
            stop = start+width
            row = proposals.iloc[wi]
            expected_keys = json.loads(row.channels)
            assert len(expected_keys) == len({key.split('/')[1] for key in expected_keys})
            if not row.proposal_supported:
                assert not row.generated and np.isnan(row.proposal_bpm) and expected_keys == []
                continue
            target = float(row.proposal_bpm)
            best = {roi: (-np.inf, None) for roi in ROIS}
            for key, candidates in sorted(by_window[wi].items()):
                branch, roi, method = key.split('/')
                assert roi in ROIS and method in ('pos','chrom') and branch in ('baseline','tracked')
                assert np.isfinite(channels[key][start:stop]).all()
                assert roi_wave[f'{roi}_observed'].iloc[start:stop].mean() >= cfg['min_observed']
                assert trace[f'{roi}_quality'].iloc[start:stop].mean() >= cfg['min_quality']
                if branch == 'tracked':
                    source = trace[f'{roi}_pixel_source'].iloc[start:stop]
                    assert (source == 'tracked_ratio').mean() >= cfg['min_tracked_fraction']
                    assert source.isin(['baseline_reset','numerical_reset']).mean() <= cfg['max_reset_fraction']
                score = max((c['evidence_score']-.025*(target-c['bpm'])**2
                             for c in candidates if target in c['supported_bpm']),default=-np.inf)
                if score > best[roi][0]:
                    best[roi] = (score,key)
            supported_rois = [roi for roi in ROIS if best[roi][1] is not None]
            assert len(supported_rois) >= cfg['min_rois']
            reconstructed, keys = [], []
            for roi in ROIS:
                key = best[roi][1]
                if key is None:
                    continue
                x = component(channels[key][start:stop], fps, target, cfg)
                scale = x.std()
                if scale < 1e-8:
                    continue
                reconstructed.append(x/scale)
                keys.append(key)
            used, generated = [], False
            if len(reconstructed) >= cfg['min_rois']:
                corr = np.array([[correlation(x,y) for y in reconstructed] for x in reconstructed])
                anchor = int(np.argmax(abs(corr).sum(axis=1)))
                used = [j for j in range(len(keys)) if abs(corr[anchor,j]) >= cfg['min_component_correlation']]
                generated = len(used) >= cfg['min_rois']
                if generated:
                    chosen_keys = [keys[j] for j in used]
                    if chosen_keys != expected_keys:
                        errors.append(f'{case}/{wi}: actual contributing channel set differs')
                    value = np.mean([reconstructed[j]*(1 if corr[anchor,j] >= 0 else -1) for j in used],axis=0)
                    existing = weights[start:stop] > 0
                    if existing.sum() >= round(fps) and correlation(value[existing],total[start:stop][existing]/weights[start:stop][existing]) < 0:
                        value = -value
                    total[start:stop] += value*taper
                    weights[start:stop] += taper
                    for key in chosen_keys:
                        roi = key.split('/')[1]
                        observed[start:stop] &= roi_wave[f'{roi}_observed'].iloc[start:stop].to_numpy(bool)
                        interpolated[start:stop] |= roi_wave[f'{roi}_interpolated'].iloc[start:stop].to_numpy(bool)
                    generated_count += 1
            assert bool(row.generated) == generated
            assert int(row.contributing_rois) == (len(used) if generated else 0)
        covered = weights > 0
        expected = np.full(n,np.nan)
        expected[covered] = total[covered]/weights[covered]
        observed &= covered
        interpolated &= covered
        for key, value in [('covered',covered),('observed',observed),('interpolated',interpolated)]:
            np.testing.assert_array_equal(wave[key].to_numpy(bool), value)
        np.testing.assert_array_equal(np.isfinite(wave.base),covered)
        delta = float(abs(wave.base.to_numpy(float)[covered]-expected[covered]).max()) if covered.any() else 0.
        if not np.allclose(wave.base, expected, rtol=1e-8, atol=1e-8, equal_nan=True):
            errors.append(case+': measured-coefficient waveform replay differs')
        checked.append(dict(case=case,planned_windows=len(starts),generated_windows=generated_count,
            frames=n,covered_frames=int(covered.sum()),maximum_absolute_waveform_difference=delta,
            discrete_source_masks_exact=True,physical_ROI_votes_and_actual_contributors_verified=True))
        for name, expected_sha in protocol['input_hashes'][case].items():
            assert sha(original/name) == expected_sha
            hashes[str(original/name)] = expected_sha
        for name in ['waveform.csv','heart_rate.csv','component_proposals.csv','all_roi_candidates.json','summary.json']:
            hashes[str(directory/name)] = sha(directory/name)
        print(json.dumps(checked[-1]),flush=True)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors,errors=errors,
        qa_source_sha256=sha(__file__),cases=checked,protocol_sha256=sha(ROOT/'protocol_before_run.json'),
        frozen_sources_verified=True,original_source_inputs_verified=True,reference_files_read=False,
        source_hashes=hashes,interpretation=[
            'Reconstruction preserves measured complex Fourier coefficients after detrending and reflection padding; no sinusoid/oscillator is generated.',
            'The adaptive frequency choice, per-window normalization, phase/polarity alignment, narrow filtering and overlap addition make the entire procedure nonlinear; this is not original waveform morphology/amplitude recovery.',
            'Repeated POS/CHROM and baseline/tracking channels can compete within a physical ROI but cannot add independent votes.',
            'Reference-frequency SNR after an adaptive narrow filter would be a frequency-selection proxy, not independent evidence of physiological waveform fidelity.',
            'Input hashes checked now supplement prior frozen protocol; this QA receipt does not retroactively preregister the post-diagnosis candidate.'])
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],errors=errors,receipt=str(receipt))),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
