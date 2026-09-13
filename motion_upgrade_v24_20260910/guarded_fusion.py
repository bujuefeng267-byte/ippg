"""Reference-free routing of original and bounded tracking waveform proposals.

Branches must independently pass the unchanged multi-ROI waveform gates.
Correlated branches do not count as extra ROIs or independent methods. Large
frequency disagreements preserve the original branch; no past HR is filled.
The final HR is subsequently derived from the saved overlap-added waveform.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from motion_fusion import FusionConfig, fuse_windows, _correlation


@dataclass(frozen=True)
class GuardConfig:
    min_tracked_fraction: float = 0.60
    max_reset_fraction: float = 0.05
    switch_score_ratio: float = 1.15
    max_branch_disagreement_bpm: float = 12.0

    def __post_init__(self):
        if not 0<self.min_tracked_fraction<=1 or not 0<=self.max_reset_fraction<=1:
            raise ValueError('Tracking/reset fractions must be bounded')
        if not np.isfinite(self.switch_score_ratio) or self.switch_score_ratio<=1:
            raise ValueError('A switch must require stronger evidence')
        if not np.isfinite(self.max_branch_disagreement_bpm) or self.max_branch_disagreement_bpm<=0:
            raise ValueError('Branch disagreement bound must be positive')


ROUTING_COLUMNS=['window_index','time_s','selected_branch','selection_reason',
    'baseline_bpm','tracked_bpm','baseline_score','tracked_score',
    'baseline_generated','tracked_generated','tracking_eligible','minimum_tracked_fraction',
    'second_largest_tracked_fraction','maximum_reset_fraction',
    'baseline_motion_overlap','tracked_motion_overlap']


def tracking_eligibility(trace,start,stop,config=GuardConfig(),contributing_rois=None):
    fractions=[];resets=[]
    for roi in FusionConfig().roi_names:
        source=trace[f'{roi}_pixel_source'].iloc[start:stop].astype(str)
        fractions.append(float((source=='tracked_ratio').mean()))
        resets.append(float(source.isin(['baseline_reset','numerical_reset']).mean()))
    allowed=set(FusionConfig().roi_names if contributing_rois is None else contributing_rois)
    usable=[roi in allowed and f>=config.min_tracked_fraction and r<=config.max_reset_fraction
            for roi,f,r in zip(FusionConfig().roi_names,fractions,resets)]
    return sum(usable)>=2, fractions, resets


def choose_branch(baseline,tracked,baseline_generated,tracked_generated,tracking_ok,
                  config=GuardConfig()):
    """Quality is an uncalibrated engineering score, never accuracy probability."""
    score=lambda row: float(row['quality_proxy'])*(1-.25*float(row['motion_overlap']))*(1-.5*float(row['runner_up_ratio']))
    bs,ts=score(baseline),score(tracked)
    if not tracked_generated or not tracking_ok:
        return ('baseline','tracking_quality_fallback',bs,ts) if baseline_generated else ('none','no_qualified_branch',bs,ts)
    if not baseline_generated:
        return 'tracked','baseline_unavailable',bs,ts
    difference=abs(float(tracked['ridge_bpm'])-float(baseline['ridge_bpm']))
    if difference>config.max_branch_disagreement_bpm:
        return 'baseline','frequency_disagreement_fallback',bs,ts
    if ts>=config.switch_score_ratio*bs:
        return 'tracked','stronger_qualified_evidence',bs,ts
    return 'baseline','baseline_evidence_retained',bs,ts


def guarded_fuse(baseline_signals,tracked_signals,trace,fps,window_s=10,step_s=1,
                 min_bpm=42,max_bpm=210,config=FusionConfig(),guard=GuardConfig()):
    branches={name:fuse_windows(signals,trace,fps,window_s,step_s,min_bpm,max_bpm,
                                config=config,return_segments=True)
              for name,signals in [('baseline',baseline_signals),('tracked',tracked_signals)]}
    base_rows=branches['baseline'][0];track_rows=branches['tracked'][0]
    n=len(trace);window=round(window_s*fps);step=round(step_s*fps)
    taper=np.hanning(window+2)[1:-1]
    wave_sum=np.zeros(n);wave_weight=np.zeros(n)
    selected=[];routing=[];diags=[]
    for wi,start in enumerate(range(0,n-window+1,step)):
        stop=start+window
        tracked_diag=branches['tracked'][2]
        contributing_rois=tracked_diag.loc[(tracked_diag.window_index==wi)&
            (tracked_diag.waveform_weight>0),'roi'].unique()
        ok,fractions,resets=tracking_eligibility(trace,start,stop,guard,contributing_rois)
        b,t=base_rows.iloc[wi],track_rows.iloc[wi]
        generated={k:wi in v[3] for k,v in branches.items()}
        name,reason,bs,ts=choose_branch(b,t,generated['baseline'],generated['tracked'],ok,guard)
        row=(t if name=='tracked' else b).copy()
        if name=='none':
            row['accepted']=False;row['waveform_roi_count']=0
            row['ridge_bpm']=row['spectral_peak_bpm']=np.nan
        selected.append(row)
        routing.append(dict(window_index=wi,time_s=b.time_s,selected_branch=name,selection_reason=reason,
            baseline_bpm=b.ridge_bpm,tracked_bpm=t.ridge_bpm,baseline_score=bs,tracked_score=ts,
            baseline_generated=generated['baseline'],tracked_generated=generated['tracked'],
            tracking_eligible=ok,minimum_tracked_fraction=min(fractions),
            second_largest_tracked_fraction=sorted(fractions)[1],maximum_reset_fraction=max(resets),
            baseline_motion_overlap=b.motion_overlap,tracked_motion_overlap=t.motion_overlap))
        if name!='none':
            segment=branches[name][3][wi]
            wave=segment['waveform'].copy()
            existing=wave_weight[start:stop]>0
            if existing.sum()>=round(fps):
                prior=wave_sum[start:stop][existing]/wave_weight[start:stop][existing]
                if _correlation(wave[existing],prior)<0:wave=-wave
            weights=taper*segment['quality']
            wave_sum[start:stop]+=wave*weights;wave_weight[start:stop]+=weights
        for branch,(_,_,d,_) in branches.items():
            subset=d.loc[d.window_index==wi].copy()
            subset['branch']=branch
            subset['original_channel']=subset.channel
            subset['method']=branch+'_'+subset.method
            subset['channel']=subset.roi+'_'+subset.method
            if branch!=name:
                subset['waveform_weight']=0.;subset['output_accepted']=False
            diags.append(subset)
    out=np.full(n,np.nan)
    np.divide(wave_sum,wave_weight,out=out,where=wave_weight>0)
    table=pd.DataFrame(selected,columns=base_rows.columns)
    table['accepted']=table.accepted.astype(bool)
    diagnostics=pd.concat(diags,ignore_index=True) if diags else branches['baseline'][2].copy()
    routing=pd.DataFrame(routing,columns=ROUTING_COLUMNS)
    table.attrs.update(guard_config=asdict(guard),quality_proxy_is_probability=False)
    return table,out,diagnostics,routing,branches
