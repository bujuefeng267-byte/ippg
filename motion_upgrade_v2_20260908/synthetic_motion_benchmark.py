"""Deterministic RGB-mixture stress checks, NOT a human motion dataset.

Known pulse and nuisance components are mixed at RGB level, before POS/CHROM.
Scenario definitions are fixed independently of real-video reference scores.
No model sees the pulse frequency or truth; truth is used only after inference.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
import pandas as pd
from scipy.signal import welch

from legacy_motion import estimate, make_waveforms
from motion_fusion import FusionConfig,fuse_windows

HERE=Path(__file__).resolve().parent
ROIS=['forehead','left_cheek','right_cheek']
FPS=30.; DURATION=60.; SEED=20260908
SCENARIOS={
    'clean':dict(pulse_bpm=96.,motion_bpm=132.,motion_amplitudes=[0.,0.,0.]),
    'one_patch_motion':dict(pulse_bpm=96.,motion_bpm=132.,motion_amplitudes=[.12,0.,0.]),
    'unequal_shared_motion':dict(pulse_bpm=96.,motion_bpm=132.,motion_amplitudes=[.12,.025,.005]),
    'global_motion':dict(pulse_bpm=96.,motion_bpm=132.,motion_amplitudes=[.08,.08,.08]),
    'pulse_motion_frequency_overlap':dict(pulse_bpm=96.,motion_bpm=96.,motion_amplitudes=[.12,.025,.005]),
}

def build_trace(config,seed):
    rng=np.random.default_rng(seed);t=np.arange(round(FPS*DURATION))/FPS
    phi=2*np.pi*config['pulse_bpm']/60*t
    pulse=np.sin(phi)+.2*np.sin(2*phi+.4)
    nuisance=np.sin(2*np.pi*config['motion_bpm']/60*t+.7)
    data=dict(frame=np.arange(len(t)),time_s=t,rgb_valid=np.ones(len(t),bool),
              motion_x=.01*nuisance if max(config['motion_amplitudes']) else np.zeros(len(t)),
              motion_y=np.zeros(len(t)))
    arrays=[]
    for i,(roi,amplitude) in enumerate(zip(ROIS,config['motion_amplitudes'])):
        baseline=np.array([140.,115.,100.])*(1+.05*i)
        # Pulse-colour modulation, chromatic nuisance, slow illumination and sensor noise.
        rgb=baseline[None,:]*(1+pulse[:,None]*np.array([.006,.018,.003])[None,:]
                             +amplitude*nuisance[:,None]*np.array([.8,.2,.5])[None,:]
                             +.01*np.sin(2*np.pi*.1*t[:,None]))
        rgb+=rng.normal(0,.05,size=rgb.shape)
        arrays.append(rgb)
        for c,values in zip('rgb',rgb.T): data[f'{roi}_{c}']=values
        data[f'{roi}_valid']=np.ones(len(t),bool);data[f'{roi}_quality']=np.ones(len(t))
    merged=np.mean(arrays,axis=0)
    for c,values in zip('rgb',merged.T):data[c]=values
    return pd.DataFrame(data),pulse

def ref_snr(x,hr):
    f,p=welch(x,fs=FPS,window='hann',nperseg=len(x),noverlap=0,nfft=8192,detrend='constant')
    b=(f>=.7)&(f<=3.5);h=b&(np.abs(f-hr/60)<=.1)
    return float(10*np.log10(max(p[h].sum(),1e-30)/max(p[b&~h].sum(),1e-30)))

def main():
    out=HERE/'synthetic_stress';out.mkdir(exist_ok=False)
    protocol=dict(seed=SEED,fps=FPS,duration_s=DURATION,scenarios=SCENARIOS,fusion_config=asdict(FusionConfig()),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation='Synthetic RGB mixtures only: no actual video, geometry, blur or human generalization.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];details=[]
    for i,(case,config) in enumerate(SCENARIOS.items()):
        trace,truth=build_trace(config,SEED+i)
        signals={};tables={};waves={}
        for method in ['pos','chrom']:
            wave,_,filled=make_waveforms(trace,FPS,method,.1,42,210)
            table=estimate(wave,trace,filled,FPS)
            waves[method]=wave;tables[method]=table
        for roi in ROIS:
            local=trace.copy()
            local[['r','g','b']]=trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy()
            for method in ['pos','chrom']:
                signals[f'{roi}_{method}']=make_waveforms(local,FPS,method,.1,42,210)[0]
        tables['fusion'],waves['fusion'],_=fuse_windows(signals,trace,FPS)
        for method,table in tables.items():
            for estimator,col in ([('forward','ridge_bpm')] if method=='fusion' else
                                  [('local_peak','spectral_peak_bpm'),('offline_dp','ridge_bpm')]):
                mask=table.accepted&table[col].notna()
                error=table.loc[mask,col].to_numpy()-config['pulse_bpm']
                snrs=[]
                for item in table.itertuples():
                    a=round(item.time_s*FPS-150);b=a+300
                    wave=waves[method][a:b]
                    value=ref_snr(wave,config['pulse_bpm']) if item.accepted and len(wave)==300 and np.isfinite(wave).all() else np.nan
                    if np.isfinite(value):snrs.append(value)
                    prediction=getattr(item,col)
                    details.append(dict(case=case,method=method,estimator=estimator,time_s=item.time_s,
                                        accepted=item.accepted,status=item.status,reference_bpm=config['pulse_bpm'],
                                        prediction_bpm=prediction,snr_ref_h1_db=value))
                rows.append(dict(case=case,method=method,estimator=estimator,n_planned=len(table),n_valid=len(error),
                                 C_out_pct=100*mask.mean(),MAE_bpm=float(np.abs(error).mean()) if len(error) else None,
                                 RMSE_bpm=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                                 P5_pct=float(np.mean(abs(error)<=5)*100) if len(error) else None,
                                 R5_pct=float(np.sum(abs(error)<=5)/len(table)*100),
                                 SNR_ref_H1_median_db=float(np.median(snrs)) if snrs else None))
        assert len({len(t) for t in tables.values()})==1
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(details).to_csv(out/'windows.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
