"""Preserve local RGB until an explicit early/late aggregation boundary.

Reuse the frozen V25 complementary anchor and POS/CHROM preprocessing. Each
three-patch adapter batch occupies the three existing region slots separately;
the anchor's merged RGB fields are never used as a patch measurement.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from anchored_reconstruction import anchored_reconstruction, AnchorConfig
from legacy_motion import make_waveforms

REGIONS=('forehead','left_cheek','right_cheek')
PATCHES=tuple(f'{region}_{i}' for region in REGIONS for i in range(4))

def mask(values):
    x=np.asarray(values)
    if x.dtype==bool:return x
    if np.isin(x,[0,1]).all():return x.astype(bool)
    raise ValueError('Expected explicit Boolean/0/1 mask')

def validate_inputs(patch_trace,frame_trace,fps):
    if not np.isfinite(fps) or fps<=7:raise ValueError('FPS cannot support the fixed HR band')
    n=len(frame_trace)
    if n==0 or not np.array_equal(frame_trace.frame,np.arange(n)):
        raise ValueError('Complete original frame clock required')
    np.testing.assert_allclose(frame_trace.time_s,np.arange(n)/fps,atol=1e-8,rtol=0)
    if len(patch_trace)!=12*n or patch_trace[['frame','patch_id']].duplicated().any():
        raise ValueError('Expected exactly twelve distinct patch observations per original frame')
    if set(patch_trace.patch_id)!=set(PATCHES):raise ValueError('Unexpected patch identities')
    blocks={}
    for name in PATCHES:
        b=patch_trace.loc[patch_trace.patch_id==name].sort_values('frame').reset_index(drop=True)
        if not np.array_equal(b.frame,np.arange(n)):raise ValueError('Missing patch frame rows')
        np.testing.assert_allclose(b.time_s,frame_trace.time_s,atol=1e-8,rtol=0)
        if not (b.region==name.rsplit('_',1)[0]).all():raise ValueError('Patch parent changed')
        valid=mask(b.valid);raw=b[[f'raw_{c}' for c in 'rgb']].to_numpy(float)
        if not np.array_equal(np.isfinite(raw).all(1),valid):raise ValueError('Raw RGB validity mismatch')
        if (raw[valid]<=0).any():raise ValueError('Raw colors must be positive')
        q=b.quality.to_numpy(float)
        if not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid patch quality')
        blocks[name]=b
    return blocks

def anchor_patches(patch_trace,frame_trace,fps):
    blocks=validate_inputs(patch_trace,frame_trace,fps);result=[]
    for number in range(4):
        adapter=frame_trace.copy()
        for c in 'rgb':adapter[c]=0.
        for region in REGIONS:
            b=blocks[f'{region}_{number}']
            adapter[f'{region}_valid']=mask(b.valid)
            for c in 'rgb':
                adapter[f'baseline_{region}_{c}']=b[f'raw_{c}'].to_numpy(float)
                adapter[f'{region}_{c}']=b[f'tracked_{c}'].to_numpy(float)
                adapter[f'{region}_pixel_log_delta_{c}']=b[f'pixel_log_delta_{c}'].to_numpy(float)
            adapter[f'{region}_pixel_source']=b.pixel_source.to_numpy()
            adapter[f'{region}_pixel_tracks']=b.pixel_tracks.to_numpy(int)
        anchored=anchored_reconstruction(adapter,fps,AnchorConfig())
        for region in REGIONS:
            b=blocks[f'{region}_{number}'].copy()
            for c in 'rgb':b[f'anchored_{c}']=anchored[f'{region}_{c}'].to_numpy(float)
            b['anchored_source']=anchored[f'{region}_pixel_source'].to_numpy()
            b['anchor_correction_log']=anchored[f'{region}_anchor_correction_log'].to_numpy(float)
            result.append(b)
    return pd.concat(result,ignore_index=True).sort_values(['frame','patch_id']).reset_index(drop=True)

def spatial_inputs(anchored_trace,frame_trace,mode):
    """Early changes only the location of spatial averaging, before BVP."""
    if mode not in ('early','late'):raise ValueError('Choose early or late RGB aggregation')
    by_patch={name:anchored_trace.loc[anchored_trace.patch_id==name].sort_values('frame').reset_index(drop=True)
              for name in PATCHES}
    if mode=='late':return by_patch
    combined={};n=len(frame_trace)
    for region in REGIONS:
        rows=[by_patch[f'{region}_{i}'] for i in range(4)]
        valid=np.stack([mask(b.valid) for b in rows]);count=valid.sum(0)
        b=pd.DataFrame(dict(frame=np.arange(n),time_s=frame_trace.time_s.to_numpy(),patch_id=region,region=region,
                            valid=count>0,quality=np.mean([x.quality.to_numpy(float) for x in rows],axis=0)))
        for kind in ('raw','anchored'):
            colors=np.stack([x[[f'{kind}_{c}' for c in 'rgb']].to_numpy(float) for x in rows])
            averaged=np.full((n,3),np.nan)
            np.divide(np.where(valid[...,None],colors,0.).sum(0),count[:,None],out=averaged,where=count[:,None]>0)
            for j,c in enumerate('rgb'):b[f'{kind}_{c}']=averaged[:,j]
        source=np.stack([x.anchored_source.to_numpy(str) for x in rows])
        tracked=((source=='tracked_ratio')|~valid).all(0)&(count>0)
        reset=((source=='baseline_reset')|(source=='numerical_reset'))&valid
        status=np.full(n,'missing',dtype=object);status[count>0]='baseline_ratio_fallback'
        status[tracked]='tracked_ratio';status[reset.any(0)]='baseline_reset'
        b['anchored_source']=status;b['subpatch_count']=count
        combined[region]=b
    return combined

def prepare_signals(anchored_trace,frame_trace,fps,mode):
    inputs=spatial_inputs(anchored_trace,frame_trace,mode)
    channels={};regions={};observed={};filled_masks={};quality={};sources={}
    wave=pd.DataFrame({'time_s':frame_trace.time_s})
    for name,b in inputs.items():
        regions[name]=str(b.region.iloc[0]);observed[name]=mask(b.valid)
        quality[name]=b.quality.to_numpy(float);sources[name]=b.anchored_source.to_numpy(str)
        one=frame_trace[['time_s','motion_x','motion_y']].copy()
        one['rgb_valid']=observed[name]
        masks=[]
        for branch,kind in [('baseline','raw'),('tracked','anchored')]:
            for c in 'rgb':one[c]=b[f'{kind}_{c}'].to_numpy(float)
            for method in ('pos','chrom'):
                base,_,filled=make_waveforms(one,fps,method,.1,42,210)
                key=f'{branch}/{name}/{method}';channels[key]=base;wave[key]=base;masks.append(filled)
        for other in masks[1:]:np.testing.assert_array_equal(masks[0],other)
        filled_masks[name]=masks[0]
        wave[f'{name}/observed']=observed[name];wave[f'{name}/interpolated']=masks[0]
        wave[f'{name}/quality']=quality[name];wave[f'{name}/tracking_source']=sources[name]
    metadata=dict(spatial_mode=mode,patch_regions=regions,channel_order=list(channels),anchor=asdict(AnchorConfig()),
                  methods=['pos','chrom'],window_s=10,step_s=1,band_bpm=[42,210],max_gap_s=.1,
                  preprocessing='unchanged V25 make_waveforms for each actual raw/anchored RGB series')
    return (channels,regions,observed,filled_masks,quality,sources),wave,metadata

def read_saved_signals(path,metadata):
    wave=pd.read_csv(path);regions=metadata['patch_regions']
    channels={key:wave[key].to_numpy(float) for key in metadata['channel_order']}
    obs={key:mask(wave[f'{key}/observed']) for key in regions}
    interp={key:mask(wave[f'{key}/interpolated']) for key in regions}
    quality={key:wave[f'{key}/quality'].to_numpy(float) for key in regions}
    source={key:wave[f'{key}/tracking_source'].to_numpy(str) for key in regions}
    return channels,regions,obs,interp,quality,source
