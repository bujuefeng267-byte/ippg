"""Reference-free joint ROI candidate tracking and measured-component reconstruction.

An experimental narrowband output, not a claim of PPG morphology recovery.
The saved component is a linear Fourier filtering of actual input windows;
no oscillator or target/reference frequency is accepted. Physical ROIs vote
once, regardless of correlated RGB method or baseline/tracking branch count.
"""
from dataclasses import dataclass,asdict
import json
import numpy as np
import pandas as pd
from scipy.signal import detrend
from evidence_hr import score_candidates,evidence_path
from legacy_motion import runs
from motion_evidence import motion_evidence

ROIS=('forehead','left_cheek','right_cheek')
@dataclass(frozen=True)
class ComponentConfig:
    min_rois:int=2
    min_observed:float=.9
    min_quality:float=.2
    pass_half_width_bpm:float=6.
    stop_half_width_bpm:float=12.
    min_component_correlation:float=.15
    min_tracked_fraction:float=.60
    max_reset_fraction:float=.05

def measured_component(x,fps,bpm,config=ComponentConfig()):
    """Preserve measured complex coefficients in a raised-cosine pulse band."""
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<4 or not np.isfinite(x).all():raise ValueError('Finite 1D samples required')
    if not 0<bpm<fps*30:raise ValueError('Invalid frequency')
    # Reflection is a numerical boundary condition, not extra observed time.
    padded=np.pad(detrend(x,type='linear'),(len(x),len(x)),mode='reflect')
    f=np.fft.rfftfreq(len(padded),1/fps)*60;distance=np.abs(f-bpm)
    gain=np.zeros(len(f));gain[distance<=config.pass_half_width_bpm]=1
    edge=(distance>config.pass_half_width_bpm)&(distance<config.stop_half_width_bpm)
    gain[edge]=.5*(1+np.cos(np.pi*(distance[edge]-config.pass_half_width_bpm)/(config.stop_half_width_bpm-config.pass_half_width_bpm)))
    return np.fft.irfft(np.fft.rfft(padded)*gain,n=len(padded))[len(x):2*len(x)]

def correlate(a,b):
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else 0.

def fuse_components(channels,trace,roi_wave,fps,window_s=10,step_s=1,min_bpm=42,max_bpm=210,config=ComponentConfig()):
    """channels: branch/ROI/method -> real previously saved broadband samples."""
    n=len(trace);window=round(window_s*fps);step=round(step_s*fps);starts=np.arange(0,n-window+1,step)
    grid=np.arange(min_bpm,max_bpm+.01,1.);emissions=[];supports=[];records=[]
    for wi,a in enumerate(starts):
        b=a+window;motion=motion_evidence(trace,a,b,fps,grid)
        per_roi={roi:np.full(len(grid),-np.inf) for roi in ROIS};source_map={roi:[None]*len(grid) for roi in ROIS}
        for key,full in sorted(channels.items()):
            branch,roi,method=key.split('/');x=np.asarray(full[a:b],float)
            observed=np.asarray(roi_wave[f'{roi}_observed'][a:b],bool)
            quality=np.asarray(trace[f'{roi}_quality'][a:b],float)
            if not np.isfinite(x).all() or observed.mean()<config.min_observed or np.mean(quality)<config.min_quality:continue
            if branch=='tracked':
                source=trace[f'{roi}_pixel_source'].iloc[a:b]
                if ((source=='tracked_ratio').mean()<config.min_tracked_fraction or source.isin(['baseline_reset','numerical_reset']).mean()>config.max_reset_fraction):continue
            candidates,score=score_candidates(x,fps,grid,motion)
            for candidate in candidates:records.append(dict(window_index=wi,channel=key,**candidate))
            choose=score>per_roi[roi]
            for k in np.flatnonzero(choose):source_map[roi][k]=key
            per_roi[roi][choose]=score[choose]
        matrix=np.stack([per_roi[roi] for roi in ROIS]);count=np.isfinite(matrix).sum(0)
        # Equal physical-ROI aggregation. Repeated methods never add votes.
        joint=np.full(len(grid),-np.inf)
        qualified=count>=config.min_rois
        joint[qualified]=np.sum(np.where(np.isfinite(matrix[:,qualified]),matrix[:,qualified],0),axis=0)/count[qualified]
        emissions.append(joint);supports.append(source_map)
    usable=np.array([np.isfinite(e).any() for e in emissions],bool);path=np.full(len(starts),np.nan);reacquired=np.zeros(len(starts),bool)
    for a,b in runs(usable):
        chosen,re=evidence_path(emissions[a:b],grid,step/fps);path[a:b]=grid[chosen];reacquired[a:b]=re
    total=np.zeros(n);weight=np.zeros(n);observed=np.ones(n,bool);filled=np.zeros(n,bool);details=[]
    taper=np.hanning(window+2)[1:-1]
    for wi,a in enumerate(starts):
        b=a+window;proposal=path[wi];row=dict(window_index=wi,time_s=(a+window/2)/fps,proposal_bpm=proposal,
            proposal_supported=bool(usable[wi]),reacquired=bool(reacquired[wi]),generated=False,contributing_rois=0,channels='[]')
        if not usable[wi]:details.append(row);continue
        k=int(round(proposal-min_bpm));waves=[];keys=[]
        for roi in ROIS:
            key=supports[wi][roi][k]
            if key is None:continue
            x=measured_component(channels[key][a:b],fps,proposal,config)
            scale=np.std(x)
            if scale<1e-8:continue
            waves.append(x/scale);keys.append(key)
        if len(waves)>=config.min_rois:
            # Highest total absolute correlation is a reference-free medoid.
            corr=np.array([[correlate(x,y) for y in waves] for x in waves]);anchor=int(np.argmax(np.abs(corr).sum(1)))
            used=[j for j in range(len(waves)) if abs(corr[anchor,j])>=config.min_component_correlation]
            if len(used)>=config.min_rois:
                wave=np.mean([waves[j]*(1 if corr[anchor,j]>=0 else -1) for j in used],axis=0)
                existing=weight[a:b]>0
                if existing.sum()>=round(fps) and correlate(wave[existing],total[a:b][existing]/weight[a:b][existing])<0:wave=-wave
                total[a:b]+=wave*taper;weight[a:b]+=taper
                selected=[keys[j] for j in used]
                for key in selected:
                    roi=key.split('/')[1];observed[a:b]&=np.asarray(roi_wave[f'{roi}_observed'][a:b],bool);filled[a:b]|=np.asarray(roi_wave[f'{roi}_interpolated'][a:b],bool)
                row.update(generated=True,contributing_rois=len(used),channels=json.dumps(selected),score=float(emissions[wi][k]))
        details.append(row)
    covered=weight>0;out=np.full(n,np.nan);np.divide(total,weight,out=out,where=covered);observed&=covered;filled&=covered
    waveform=pd.DataFrame(dict(time_s=trace.time_s,base=out,covered=covered,observed=observed,interpolated=filled))
    return waveform,pd.DataFrame(details),records
