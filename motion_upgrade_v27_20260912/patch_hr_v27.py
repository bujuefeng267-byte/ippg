"""Independent patch-median / complete-PSD clustering experiment.

Inspired by the documented pyVHR comparison, independently implemented here:
https://github.com/phuselab/pyVHR
https://github.com/phuselab/pyVHR/blob/master/results/cfg/PURE_clustering.cfg
No pyVHR/GPL implementation is copied or imported. This is not a reproduction
of its 100-patch, 8-second, CircleClustering/Gaussian-fit protocol.

Primary HR aggregates patch HRs. The separately saved waveform is a weighted
sum of actual broadband inputs, not a waveform synthesized from that HR.
No reference, motion-spectrum penalty, temporal correction, or HR filling is
used by the primary two aggregation methods.
"""
from dataclasses import asdict, dataclass
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch

REGIONS = ('forehead', 'left_cheek', 'right_cheek')


@dataclass(frozen=True)
class PatchHRConfig:
    window_s: float = 10.
    step_s: float = 1.
    min_bpm: float = 42.
    max_bpm: float = 210.
    grid_step_bpm: float = 1.
    welch_nfft: int = 8192
    min_observed: float = .90
    min_quality: float = .20
    max_gap_s: float = .10
    min_tracked_fraction: float = .60
    max_reset_fraction: float = .05
    min_regions: int = 2
    concentration_radius_bpm: float = 6.
    width_scale_bpm: float = 12.
    complete_link_cosine_distance: float = .25
    flat_std: float = 1e-8

    def __post_init__(self):
        finite = [v for v in asdict(self).values()]
        if not np.isfinite(finite).all():
            raise ValueError('All configuration values must be finite')
        if not (self.window_s > 0 and self.step_s > 0 and 0 < self.min_bpm < self.max_bpm):
            raise ValueError('Invalid window/range')
        if self.grid_step_bpm != 1. or self.welch_nfft < 4:
            raise ValueError('A one-BPM grid and at least four FFT points are required')
        for key in ['min_observed','min_quality','min_tracked_fraction','max_reset_fraction']:
            if not 0 <= getattr(self,key) <= 1:
                raise ValueError(key+' must be in [0,1]')
        if not (self.max_gap_s >= 0 and self.concentration_radius_bpm > 0 and self.width_scale_bpm > 0 and self.flat_std > 0):
            raise ValueError('Invalid gap, spectral, or flatness setting')
        if self.min_regions not in (2,3) or not 0 < self.complete_link_cosine_distance < 1:
            raise ValueError('Require two or three regions and a cosine cutoff between zero and one')


DEFAULT_CONFIG = PatchHRConfig()


def protocol_description(config=DEFAULT_CONFIG):
    return dict(config=asdict(config), primary_HR_source='spatial_aggregation_of_saved_patch_BVP',
        representative='One eligible channel per physical patch; maximize H1 peak concentration/(1+FWHM/12), tie by channel key.',
        median='Median of patch HRs within each region, then ordinary median of region medians; even-count median may lie between actual peaks.',
        psd_cluster='Complete-link clustering of the full L2-normalized PSD on the BPM grid; not HR-distance clustering.',
        cluster_score='Region-balanced L1 PSD concentration/(1+FWHM/12), multiplied by supported region count/3.',
        fft='max(8192, next_power_of_two(max(4*n_window,256*fps))); zero padding only interpolates the spectrum, not the physical frequency resolution.',
        waveform='Broadband measured inputs, window demeaning/std normalization, region-balanced polarity medoid, positive Hann overlap-add; no frequency shifting or sinusoid generation.',
        provenance='observed=ALL actual contributors; interpolated=ANY actual contributor; uncovered remains NaN.',
        no_reference_used=True, no_primary_motion_penalty=True, no_primary_temporal_correction=True,
        difference_from_author='12 anatomical patches, 10s/1s, 42–210 BPM, complete-link cosine clusters, region-balanced aggregation; not pyVHR protocol reproduction.',
        inspiration_urls=['https://github.com/phuselab/pyVHR',
            'https://github.com/phuselab/pyVHR/blob/master/results/html/MAE_run_on_dataset_PURE.html'])


def _runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def _bool_array(values,n,label):
    a=np.asarray(values)
    if a.shape!=(n,) or pd.isna(a).any() or not np.isin(a,[True,False,0,1]).all():
        raise ValueError(label+' must contain exactly N finite booleans')
    return a.astype(bool)


def _half_power_width(grid,power,peak):
    half=power[peak]/2
    lo=hi=int(peak)
    while lo>0 and power[lo-1]>=half:lo-=1
    while hi<len(power)-1 and power[hi+1]>=half:hi+=1
    left=float(grid[lo]);right=float(grid[hi])
    if lo>0 and power[lo]!=power[lo-1]:
        left=float(grid[lo-1]+(half-power[lo-1])/(power[lo]-power[lo-1])*(grid[lo]-grid[lo-1]))
    if hi<len(power)-1 and power[hi]!=power[hi+1]:
        right=float(grid[hi]+(power[hi]-half)/(power[hi]-power[hi+1])*(grid[hi+1]-grid[hi]))
    return max(float(grid[1]-grid[0]),right-left)


def _spectrum_summary(grid,power,config):
    peak=int(np.argmax(power))
    bpm=float(grid[peak])
    mass=float(power.sum())
    concentration=float(power[abs(grid-bpm)<=config.concentration_radius_bpm].sum()/mass)
    width=_half_power_width(grid,power,peak)
    return dict(peak_bpm=bpm,peak_concentration=concentration,peak_fwhm_bpm=width,
        spectral_score=concentration/(1+width/config.width_scale_bpm))


def patch_spectrum(samples,fps,config=DEFAULT_CONFIG):
    """One fixed strongest Welch peak and the entire PSD shape; no labels."""
    x=np.asarray(samples,float)
    if x.ndim!=1 or len(x)<4 or not np.isfinite(x).all():
        raise ValueError('Finite one-dimensional samples required')
    if not np.isfinite(fps) or fps*30<=config.max_bpm:
        raise ValueError('Sample rate must exceed twice the upper pulse frequency')
    if x.std()<config.flat_std:
        return None
    grid=np.arange(config.min_bpm,config.max_bpm+.001,config.grid_step_bpm)
    nfft=max(config.welch_nfft,1 << int(np.ceil(np.log2(max(4*len(x),256*fps)))))
    frequency,power=welch(x,fs=fps,window='hann',nperseg=len(x),noverlap=0,
        nfft=nfft,detrend='constant')
    band=np.interp(grid/60.,frequency,power)
    norm=float(np.linalg.norm(band));mass=float(band.sum())
    if not np.isfinite(band).all() or norm<=0 or mass<=0:return None
    return dict(grid=grid,power_fraction=band/mass,shape=band/norm,
        **_spectrum_summary(grid,band,config))


def complete_link_clusters(shapes,cutoff=.25):
    """Deterministic agglomeration using maximum FULL-PSD cosine distance.

Clusters merge only when every cross-cluster pair meets the cutoff, preventing
single-link bridge chaining. No peak frequencies are passed to this function.
"""
    x=np.asarray(shapes,float)
    if x.ndim!=2 or not np.isfinite(x).all() or not 0<cutoff<1:
        raise ValueError('Finite spectral matrix and cosine cutoff required')
    if len(x)==0:return []
    lengths=np.linalg.norm(x,axis=1)
    if np.any(lengths<=0):raise ValueError('Nonzero spectra required')
    x=x/lengths[:,None]
    distances=np.clip(1-x@x.T,0,2)
    clusters=[(i,) for i in range(len(x))]
    while len(clusters)>1:
        options=[]
        for a in range(len(clusters)):
            for b in range(a+1,len(clusters)):
                d=float(distances[np.ix_(clusters[a],clusters[b])].max())
                if d<=cutoff+1e-12:
                    options.append((d,clusters[a],clusters[b],a,b))
        if not options:break
        _,_,_,a,b=min(options)
        merged=tuple(sorted(clusters[a]+clusters[b]))
        clusters=[c for i,c in enumerate(clusters) if i not in (a,b)]+[merged]
        clusters.sort()
    return [list(c) for c in clusters]


def _correlation(x,y):
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/den) if den>1e-12 else 0.


def _broadband_fusion(selected,channels,start,stop):
    """Measured samples only. The selected HR is intentionally not an argument."""
    values=[]
    for item in selected:
        x=np.asarray(channels[item['channel']][start:stop],float)
        values.append((x-x.mean())/x.std())
    values=np.asarray(values)
    groups={region:[i for i,item in enumerate(selected) if item['region']==region]
        for region in REGIONS if any(item['region']==region for item in selected)}
    corr=np.array([[_correlation(x,y) for y in values] for x in values])
    medoid_scores=np.mean([np.mean(abs(corr[:,idx]),axis=1) for idx in groups.values()],axis=0)
    anchor=int(np.argmax(medoid_scores))
    coefficients=np.empty(len(selected))
    for idx in groups.values():
        coefficients[idx]=1/(len(groups)*len(idx))
    signs=np.where(corr[anchor]>=0,1.,-1.)
    output=np.sum(values*(coefficients*signs)[:,None],axis=0)
    return output,coefficients,signs,selected[anchor]['channel']


def _regional_median(selected):
    medians={region:float(np.median([r['peak_bpm'] for r in selected if r['region']==region]))
        for region in REGIONS if any(r['region']==region for r in selected)}
    return float(np.median(list(medians.values()))),medians


def _prepare(channels,patch_regions,observed,interpolated,quality,tracking_sources,frame_trace,fps,config):
    if not isinstance(channels,Mapping) or not isinstance(patch_regions,Mapping):
        raise ValueError('Channels and physical patch metadata must be mappings')
    n=len(frame_trace)
    if n<1 or not np.isfinite(fps) or fps*30<=config.max_bpm:
        raise ValueError('A nonempty trace with a valid FPS is required')
    if 'time_s' not in frame_trace:raise ValueError('Trace needs time_s')
    np.testing.assert_allclose(frame_trace.time_s,np.arange(n)/fps,rtol=0,atol=1e-7)
    specs={};masks={};qs={};sources={};safe_fill={}
    for patch,region in sorted(patch_regions.items()):
        if not isinstance(patch,str) or '/' in patch or region not in REGIONS:
            raise ValueError('Each physical patch has one fixed anatomical region')
        obs=_bool_array(observed[patch],n,patch+'/observed')
        filled=_bool_array(interpolated[patch],n,patch+'/interpolated')
        if np.any(obs&filled):raise ValueError('Observed and interpolated are mutually exclusive')
        q=np.asarray(quality[patch],float)
        if q.shape!=(n,) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
            raise ValueError('Quality must be finite in [0,1] for every source frame')
        src=np.asarray(tracking_sources[patch],object)
        if src.shape!=(n,) or pd.isna(src).any():raise ValueError('Tracking sources need N nonmissing entries')
        short=np.zeros(n,bool);limit=int(np.floor(config.max_gap_s*fps+1e-9))
        for a,b in _runs(~obs):
            if a>0 and b<n and b-a<=limit:short[a:b]=True
        safe_fill[patch]=short&filled
        masks[patch]=(obs,filled);qs[patch]=q;sources[patch]=src
    for key,values in sorted(channels.items()):
        parts=key.split('/')
        if len(parts)!=3:raise ValueError('Channel key must be branch/patch_id/method')
        branch,patch,method=parts
        if branch not in ('baseline','tracked') or patch not in patch_regions or method not in ('pos','chrom'):
            raise ValueError('Unsupported channel metadata')
        x=np.asarray(values,float)
        if x.shape!=(n,):raise ValueError('Each channel must have N samples')
        specs[key]=(branch,patch,method,x)
    return n,specs,masks,qs,sources,safe_fill


def infer_patch_hr(channels,patch_regions,observed,interpolated,quality,tracking_sources,
                   frame_trace,fps,mode='median',config=DEFAULT_CONFIG):
    """Return (actual broadband waveform, direct HR table, diagnostic records).

Input channels are saved continuous BVPs, keyed baseline/<patch_id>/pos or
tracked/<patch_id>/chrom. Each metadata dictionary maps physical patch ID to
an N-sample array. Additional trace/reference columns are ignored.
"""
    if mode not in ('median','psd_cluster'):raise ValueError('mode must be median or psd_cluster')
    n,specs,masks,qs,sources,safe_fill=_prepare(channels,patch_regions,observed,interpolated,
        quality,tracking_sources,frame_trace,fps,config)
    window,hop=round(config.window_s*fps),round(config.step_s*fps)
    if window<4 or hop<1:raise ValueError('Window/step contain too few samples')
    starts=np.arange(0,n-window+1,hop,dtype=int)
    total=np.zeros(n);weights=np.zeros(n);out_observed=np.ones(n,bool);out_interpolated=np.zeros(n,bool)
    taper=np.hanning(window+2)[1:-1]
    hr_rows=[];diagnostics=[]
    for wi,start in enumerate(starts):
        stop=start+window;representatives={}
        row=dict(window_index=wi,time_s=(start+window/2)/fps,window_start_s=start/fps,
            window_end_s=stop/fps,accepted=False,ridge_bpm=np.nan,raw_estimate_bpm=np.nan,
            status='insufficient_regions',hr_source='spatial_aggregation_of_saved_patch_BVP',
            aggregation_mode=mode,eligible_patches=0,eligible_regions=0,contributing_patches=0,
            contributing_regions=0,waveform_generated=False,selected_channels_json='[]',
            region_medians_json='{}',estimate_near_contributor_peak=False,
            region_median_spread_bpm=np.nan,selected_cluster_score=np.nan)
        for key,(branch,patch,method,x) in specs.items():
            obs,fill=masks[patch];status='eligible'
            fraction=float(obs[start:stop].mean());qmean=float(qs[patch][start:stop].mean())
            src=sources[patch][start:stop]
            tf=float((src=='tracked_ratio').mean());reset=float(np.isin(src,['baseline_reset','numerical_reset']).mean())
            if not np.isfinite(x[start:stop]).all():status='nonfinite_bvp'
            elif fraction<config.min_observed:status='insufficient_observed'
            elif not np.all(obs[start:stop]|safe_fill[patch][start:stop]):status='unverified_or_long_interpolation'
            elif qmean<config.min_quality:status='low_quality'
            elif branch=='tracked' and (tf<config.min_tracked_fraction or reset>config.max_reset_fraction):status='tracking_ineligible'
            spectrum=patch_spectrum(x[start:stop],fps,config) if status=='eligible' else None
            if status=='eligible' and spectrum is None:status='flat_or_zero_band_signal'
            diagnostic=dict(record_type='channel',window_index=wi,channel=key,patch_id=patch,
                region=patch_regions[patch],branch=branch,method=method,status=status,
                observed_fraction=fraction,mean_quality=qmean,tracked_fraction=tf,reset_fraction=reset,
                representative=False)
            if spectrum is not None:
                diagnostic.update({k:spectrum[k] for k in ('peak_bpm','peak_concentration','peak_fwhm_bpm','spectral_score')})
                item=dict(channel=key,patch_id=patch,region=patch_regions[patch],**spectrum,
                    diagnostic_index=len(diagnostics))
                # Sorted input keys + strict greater preserve deterministic ties.
                if patch not in representatives or item['spectral_score']>representatives[patch]['spectral_score']:
                    representatives[patch]=item
            diagnostics.append(diagnostic)
        reps=[representatives[p] for p in sorted(representatives)]
        for item in reps:diagnostics[item['diagnostic_index']]['representative']=True
        available_regions={item['region'] for item in reps}
        row.update(eligible_patches=len(reps),eligible_regions=len(available_regions))
        selected=[]
        if len(available_regions)>=config.min_regions:
            if mode=='median':selected=reps
            else:
                row['status']='no_multiregion_psd_cluster'
                clusters=complete_link_clusters([item['shape'] for item in reps],config.complete_link_cosine_distance)
                options=[]
                for ci,indices in enumerate(clusters):
                    members=[reps[i] for i in indices]
                    groups={r:[item for item in members if item['region']==r] for r in REGIONS}
                    groups={r:v for r,v in groups.items() if v}
                    regional_psds=[np.mean([item['power_fraction'] for item in v],axis=0) for v in groups.values()]
                    average_psd=np.mean(regional_psds,axis=0)
                    spectral=_spectrum_summary(reps[0]['grid'],average_psd,config)
                    score=spectral['spectral_score']*len(groups)/len(REGIONS)
                    qualified=len(groups)>=config.min_regions
                    diagnostics.append(dict(record_type='cluster',window_index=wi,cluster_index=ci,
                        patch_ids_json=json.dumps([item['patch_id'] for item in members]),
                        regions_json=json.dumps(list(groups)),patch_count=len(members),region_count=len(groups),
                        qualified=qualified,cluster_score=score,selected=False,**spectral))
                    if qualified:options.append((score,tuple(item['patch_id'] for item in members),members,len(diagnostics)-1))
                if options:
                    winner=sorted(options,key=lambda option:(-option[0],option[1]))[0]
                    row['selected_cluster_score']=winner[0];selected=winner[2]
                    diagnostics[winner[3]]['selected']=True
        if selected:
            estimate,region_medians=_regional_median(selected)
            wave,coefficients,signs,anchor=_broadband_fusion(selected,channels,start,stop)
            existing=weights[start:stop]>0;window_sign=1.
            if existing.sum()>=round(fps) and _correlation(wave[existing],total[start:stop][existing]/weights[start:stop][existing])<0:
                wave=-wave;window_sign=-1.
            assert np.isfinite(wave).all()
            total[start:stop]+=wave*taper;weights[start:stop]+=taper
            for item,coefficient,sign in zip(selected,coefficients,signs):
                patch=item['patch_id'];obs,fill=masks[patch]
                out_observed[start:stop]&=obs[start:stop];out_interpolated[start:stop]|=fill[start:stop]
                diagnostics.append(dict(record_type='contributor',window_index=wi,channel=item['channel'],
                    patch_id=patch,region=item['region'],coefficient=float(coefficient),
                    polarity=float(sign*window_sign),medoid_channel=anchor,peak_bpm=item['peak_bpm']))
            row.update(accepted=True,ridge_bpm=estimate,raw_estimate_bpm=estimate,status='accepted',
                contributing_patches=len(selected),contributing_regions=len(region_medians),waveform_generated=True,
                selected_channels_json=json.dumps([item['channel'] for item in selected]),
                region_medians_json=json.dumps(region_medians),
                estimate_near_contributor_peak=any(abs(item['peak_bpm']-estimate)<=3 for item in selected),
                region_median_spread_bpm=max(region_medians.values())-min(region_medians.values()))
        hr_rows.append(row)
    covered=weights>0;output=np.full(n,np.nan);np.divide(total,weights,out=output,where=covered)
    out_observed&=covered;out_interpolated&=covered
    waveform=pd.DataFrame(dict(time_s=frame_trace.time_s.to_numpy(float),base=output,
        covered=covered,observed=out_observed,interpolated=out_interpolated))
    columns=['window_index','time_s','window_start_s','window_end_s','accepted','ridge_bpm','raw_estimate_bpm',
        'status','hr_source','aggregation_mode','eligible_patches','eligible_regions','contributing_patches',
        'contributing_regions','waveform_generated','selected_channels_json','region_medians_json',
        'estimate_near_contributor_peak','region_median_spread_bpm','selected_cluster_score']
    table=pd.DataFrame(hr_rows,columns=columns)
    for col in ['accepted','waveform_generated','estimate_near_contributor_peak']:table[col]=table[col].astype(bool)
    for col in ['time_s','window_start_s','window_end_s','ridge_bpm','raw_estimate_bpm','region_median_spread_bpm','selected_cluster_score']:
        table[col]=table[col].astype(float)
    table.attrs.update(protocol_description(config),primary_HR_is_fused_waveform_readout=False)
    return waveform,table,pd.DataFrame(diagnostics)


def fusion_readout_diagnostic(waveform,frame_trace,fps,estimator,config=DEFAULT_CONFIG):
    """Optional injected frozen V25 estimator; never replaces primary median HR."""
    result=estimator(waveform.base.to_numpy(float),pd.DataFrame({'rgb_valid':waveform.observed.to_numpy(bool)}),
        waveform.interpolated.to_numpy(bool),fps,config.window_s,config.step_s,config.min_bpm,config.max_bpm,
        motion_trace=frame_trace)
    if 'spectral_peak_bpm' in result:
        result['raw_spectral_peak_bpm']=result.spectral_peak_bpm
        result.loc[~result.accepted,'spectral_peak_bpm']=np.nan
    result.loc[~result.accepted,'ridge_bpm']=np.nan
    result['hr_source']='diagnostic_V25_readout_of_saved_broadband_patch_fusion'
    return result
