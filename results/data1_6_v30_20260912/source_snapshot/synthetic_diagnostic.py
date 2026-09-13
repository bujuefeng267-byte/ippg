"""Read-only V28 mechanism probes, using generated signals only.

Does not read videos, reference values, prior prediction tables or evaluations.
Alternative normalization/contrast are diagnostic arithmetic, not model changes.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch

SOURCE = Path('/home/fengbujue/项目/rppg识别/motion_upgrade_v28_20260912')
sys.path.insert(0, str(SOURCE))
from evidence_hr import DEFAULT_CONFIG, score_candidates, evidence_path, _period_correlation
from motion_evidence import motion_evidence


def main():
    fps, seconds = 30., 10.
    t = np.arange(round(fps*seconds))/fps
    grid = np.arange(42., 211.)
    motion_rows = []
    for motion_bpm in (20., 30., 60., 120., 240.):
        speed = .5
        velocity = np.sqrt(2)*speed*np.sin(2*np.pi*motion_bpm/60*t)
        trace = pd.DataFrame(dict(time_s=t, face_y0=.2, face_y1=.6,
            motion_x=velocity*.4/fps, motion_y=np.zeros(len(t))))
        actual = motion_evidence(trace, 0, len(t), fps, grid)
        nfft=max(8192, 2**int(np.ceil(np.log2(max(4*len(t),256*fps)))))
        f,p=welch(velocity,fs=fps,window='hann',nperseg=len(t),noverlap=0,nfft=nfft,detrend='constant')
        candidate_band=np.interp(grid/60,f,p)
        absolute_profile=candidate_band/p.max()*actual['strength']*actual['reliability']
        band=(f*60>=grid[0])&(f*60<=grid[-1])
        motion_rows.append(dict(motion_bpm=motion_bpm,
            candidate_band_peak_bpm=float(grid[np.argmax(actual['profile'])]),
            current_max_motion_risk=float(np.max(actual['profile'])),
            full_spectrum_normalized_max_risk=float(np.max(absolute_profile)),
            max_candidate_band_psd_over_full_peak=float(candidate_band.max()/p.max()),
            band_energy_fraction=float(np.trapz(p[band],f[band])/np.trapz(p,f)),
            strength=float(actual['strength']), reliability=float(actual['reliability'])))
    for row in motion_rows:
        if row['motion_bpm'] in (20.,30.,240.):
            assert row['current_max_motion_risk'] > .75
            assert row['full_spectrum_normalized_max_risk'] < .01
    # Same measured waveform supports two physically different explanations.
    # Synthetic component amplitudes are known; no inferred reference is used.
    segment=np.sin(2*np.pi*108/60*t)+.5*np.sin(2*np.pi*54/60*t)
    candidates, emission=score_candidates(segment,fps,grid)
    harmonic_pair=[]
    for candidate in candidates:
        if min(abs(candidate['bpm']-54),abs(candidate['bpm']-108)) <= 2:
            bpm=candidate['bpm']
            # A bounded contrast experiment with no added threshold. It is not
            # presumed more accurate: it rejects some real harmonic-dominant pulses.
            period=60*fps/bpm
            correlation_half_period=_period_correlation(segment,fps,2*bpm)
            contrast=max(0., candidate['period_correlation']-correlation_half_period)
            hypothetical_score=(candidate['evidence_score']-
                DEFAULT_CONFIG.periodic_weight*candidate['periodic_support']+
                DEFAULT_CONFIG.periodic_weight*min(1.,contrast))
            harmonic_pair.append(dict(**candidate,
                correlation_at_half_period=correlation_half_period,
                fundamental_contrast_diagnostic=contrast,
                hypothetical_contrast_score=hypothetical_score))
    assert abs(candidates[0]['bpm']-54)<=2
    low=min(harmonic_pair,key=lambda r:abs(r['bpm']-54))
    high=min(harmonic_pair,key=lambda r:abs(r['bpm']-108))
    assert low['relative_power'] < high['relative_power']
    assert low['evidence_score'] > high['evidence_score']
    # Verify the DP's intended cost, rather than presenting it as an index bug.
    # An isolated two-window candidate advantage cannot pay two jump costs.
    scores=np.full((12,len(grid)),-np.inf)
    a,b=int(54-grid[0]),int(108-grid[0])
    scores[:,a]=1.;scores[:,b]=0.
    scores[5:7,a]=0.;scores[5:7,b]=1.
    path,reacquired=evidence_path(scores,grid,1.)
    assert np.all(grid[path]==54)
    result=dict(source=str(SOURCE),reference_read=False,production_code_changed=False,
        fps=fps,seconds=seconds,motion_band_normalization=motion_rows,
        subharmonic_pair=harmonic_pair,
        same_waveform_ambiguity=[
            '108 bpm pulse plus a 54 bpm nuisance component of half amplitude',
            '54 bpm pulse whose second harmonic has twice the fundamental amplitude'],
        dp_check=dict(trajectory_bpm=grid[path].tolist(), reacquisition_flags=reacquired.tolist(),
            isolated_advantage_score_units=2.,two_jump_cost_score_units=6.,
            interpretation='Intended smoothing preference, not an indexing or unsupported-frequency bug'),
        source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                       for name in ('evidence_hr.py','motion_evidence.py','component_harmonics_v26.py')})
    out=Path(__file__).with_name('synthetic_diagnostic_results.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(motion_band_normalization=motion_rows,
        pair=[{k:r[k] for k in ('bpm','relative_power','periodic_support','harmonic_support','evidence_score',
                               'fundamental_contrast_diagnostic','hypothetical_contrast_score')} for r in harmonic_pair],
        dp_path=grid[path].tolist(),output=str(out)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
