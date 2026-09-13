"""Score exported neural BVP using the classical window/reference protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_rppg_motion import runs, bandpass, estimate, reference_metrics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--reference', type=Path)
a = p.parse_args()
info = json.loads((a.output/'neural_summary.json').read_text())
data = pd.read_csv(a.output/'neural_waveform.csv')
trace = pd.read_csv(info['trace'])
fps = info['fps']
wave = data.bvp_raw.to_numpy().copy()
for start, stop in runs(np.isfinite(wave)):
    if stop-start >= round(4*fps):
        wave[start:stop] = bandpass(wave[start:stop], fps, 42, 210)
    else:
        wave[start:stop] = np.nan
table = estimate(wave, trace, data.interpolated_roi.to_numpy(bool), fps)
result = dict(total_windows=len(table), accepted_windows=int(table.accepted.sum()),
              protocol='10 s windows, 1 s step, 42-210 bpm; no fitted time shift')
if a.reference:
    result['reference'] = reference_metrics(table, a.reference, 10)
    comparisons = {}
    for method in ('pos', 'chrom'):
        baseline = pd.read_csv(Path(info['trace']).parent/f'{method}_heart_rate.csv')
        assert np.allclose(table.time_s, baseline.time_s)
        mask = table.accepted & baseline.accepted & np.isfinite(table.reference_bpm)
        values = {}
        for label, source in [('neural', table), (method, baseline)]:
            errors = source.loc[mask, 'ridge_bpm']-table.loc[mask, 'reference_bpm']
            values[label] = dict(n=int(mask.sum()), mae_bpm=float(errors.abs().mean()),
                                 rmse_bpm=float(np.sqrt((errors**2).mean())))
        comparisons[method] = values
    result['common_window_ridge_comparison'] = comparisons
table.to_csv(a.output/'neural_heart_rate.csv', index=False)
(a.output/'neural_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
