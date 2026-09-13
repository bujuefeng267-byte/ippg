#!/usr/bin/env python3
"""V2.2 offline multi-ROI rPPG with sub-pixel RGB aggregation.

No reference HR is supplied to extraction, waveform fusion or candidate selection.
The optional reference is applied only AFTER all predictions have been produced.
Both filtering/overlap fusion and the final DP tracker are offline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from legacy_motion import estimate, make_waveforms, reference_metrics
from motion_frontend import extract
from motion_fusion import FusionConfig, fuse_windows
from waveform_hr import estimate_fused_waveform

ROIS=('forehead','left_cheek','right_cheek')
HERE=Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes():
    names=['analyze_motion_v2.py','analyze_rppg.py','legacy_motion.py','motion_frontend.py',
           'motion_fusion.py','stable_groups.py','waveform_hr.py']
    return {name:digest(HERE/name) for name in names}

def validate_trace(trace, fps):
    if len(trace)==0 or not np.isfinite(fps) or fps<5:
        raise ValueError('Empty trace or invalid FPS')
    if trace.time_s.duplicated().any() or not np.all(np.diff(trace.time_s)>0):
        raise ValueError('Trace timestamps must be unique and increasing')
    if not np.array_equal(trace.frame.to_numpy(),np.arange(len(trace))):
        raise ValueError('Trace frame indices must be complete and ordered')
    if not np.allclose(trace.time_s,np.arange(len(trace))/fps,rtol=0,atol=1e-7):
        raise ValueError('Trace timestamps do not match FPS and frame indices')
    if not np.array_equal(np.isfinite(trace[['r','g','b']]).all(axis=1),trace.rgb_valid.to_numpy(bool)):
        raise ValueError('Merged RGB validity mismatch')
    for roi in ROIS:
        cols=[f'{roi}_{c}' for c in 'rgb']
        finite=np.isfinite(trace[cols]).all(axis=1)
        valid=trace[f'{roi}_valid'].to_numpy(bool)
        if not np.array_equal(finite,valid):
            raise ValueError(f'{roi} validity does not match independent RGB')
        q=trace[f'{roi}_quality'].to_numpy()
        if not np.isfinite(q).all() or np.any((q<0)|(q>1)):
            raise ValueError(f'{roi} quality must be finite in [0,1]')

def per_roi_signals(trace, fps, gap, low, high):
    """Same signal/filter/edge rules as V1, independently for each skin patch."""
    signals={}; wave=pd.DataFrame({'time_s':trace.time_s})
    for roi in ROIS:
        local=trace.copy()
        local[['r','g','b']]=trace[[f'{roi}_{c}' for c in 'rgb']].to_numpy()
        local['rgb_valid']=trace[f'{roi}_valid']
        for method in ['pos','chrom']:
            base,_,filled=make_waveforms(local,fps,method,gap,low,high)
            key=f'{roi}_{method}'
            signals[key]=base
            wave[key]=base
        wave[f'{roi}_observed']=trace[f'{roi}_valid']
        wave[f'{roi}_interpolated']=filled
    return signals,wave

def summary_for(table):
    valid=table.accepted.to_numpy(bool)
    return dict(total_windows=len(table),accepted_windows=int(valid.sum()),
                accepted_fraction=float(valid.mean()) if len(valid) else None,
                status_counts={str(k):int(v) for k,v in table.status.value_counts().items()})

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video',type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--cache',type=Path,help='Reuse an exact V2 frontend trace with matching identity')
    ap.add_argument('--rgb-aggregation',choices=['trimmed_mean','median'],default='trimmed_mean',
                    help='Fixed 10%% per-channel trimmed mean; median is the controlled baseline')
    ap.add_argument('--max-seconds',type=float)
    ap.add_argument('--max-gap',type=float,default=.1)
    ap.add_argument('--window',type=float,default=10)
    ap.add_argument('--step',type=float,default=1)
    ap.add_argument('--min-bpm',type=float,default=42)
    ap.add_argument('--max-bpm',type=float,default=210)
    ap.add_argument('--reference-ubfc',type=Path)
    args=ap.parse_args()
    frozen_sources=source_hashes()
    requested_window_s=args.window
    if args.window<6 or args.step<=0 or args.max_gap<0 or not 0<args.min_bpm<args.max_bpm:
        ap.error('Invalid window, step, gap or HR range')
    if args.max_seconds is not None and args.max_seconds<=0:
        ap.error('max-seconds must be positive')
    stat=args.video.stat()
    identity=dict(video=str(args.video.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                  max_seconds=args.max_seconds,
                  rgb_aggregation=args.rgb_aggregation,
                  frontend_hashes={n:digest(HERE/n) for n in ['motion_frontend.py','legacy_motion.py','analyze_rppg.py']})
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    if args.cache:
        metadata=json.loads(args.cache.with_suffix('.json').read_text(encoding='utf-8'))
        if metadata['identity']!=identity:
            raise ValueError('Cache does not match video and frozen frontend implementation')
        if metadata['trace_sha256']!=digest(args.cache):
            raise ValueError('Cached trace content checksum mismatch')
        trace=pd.read_csv(args.cache); fps=metadata['fps']
        if len(trace)!=metadata['n_frames']:
            raise ValueError('Cached trace frame count mismatch')
    else:
        trace,fps=extract(args.video,max_seconds=args.max_seconds,qa_dir=args.output/'qa',
                          rgb_aggregation=args.rgb_aggregation)
    validate_trace(trace,fps)
    # Rounding 6 s at fractional FPS can produce slightly less than 6 s.
    # Round this minimum boundary upward consistently for every branch.
    if round(args.window*fps)<6*fps:
        args.window=float(np.ceil(6*fps)/fps)
    trace.to_csv(args.output/'frame_trace.csv',index=False)
    (args.output/'frame_trace.json').write_text(json.dumps(dict(identity=identity,fps=fps,n_frames=len(trace),
        trace_sha256=digest(args.output/'frame_trace.csv')),indent=2),encoding='utf-8')
    tables={}; waveforms={}
    # Matched-frontend ablation: original merged RGB and original V1 signal/HR logic.
    for method in ['pos','chrom']:
        base,_,filled=make_waveforms(trace,fps,method,args.max_gap,args.min_bpm,args.max_bpm)
        table=estimate(base,trace,filled,fps,args.window,args.step,args.min_bpm,args.max_bpm)
        tables[method]=table; waveforms[method]=base
        pd.DataFrame(dict(time_s=trace.time_s,base=base,observed=trace.rgb_valid,
                          interpolated=filled)).to_csv(args.output/f'{method}_waveform.csv',index=False)
    signals,roi_wave=per_roi_signals(trace,fps,args.max_gap,args.min_bpm,args.max_bpm)
    roi_wave.to_csv(args.output/'roi_waveforms.csv',index=False)
    config=FusionConfig()
    proposals,wave,diagnostics=fuse_windows(signals,trace,fps,args.window,args.step,args.min_bpm,args.max_bpm,
                                       config=config)
    proposals['waveform_generated']=proposals.accepted & (proposals.waveform_roi_count>=config.min_rois)
    fused,provenance=estimate_fused_waveform(wave,trace,roi_wave,proposals,diagnostics,fps,
        args.window,args.step,args.min_bpm,args.max_bpm)
    if not np.array_equal(np.isfinite(wave),provenance.covered.to_numpy(bool)):
        raise RuntimeError('Saved waveform samples do not match documented fusion contributors')
    if not np.array_equal(proposals.waveform_generated,fused.waveform_generated):
        raise RuntimeError('Proposal generation flags do not match actual contributors')
    tables['fusion']=fused; waveforms['fusion']=wave
    provenance.insert(1,'base',wave)
    provenance.to_csv(args.output/'fusion_waveform.csv',index=False)
    proposals.rename(columns={'spectral_peak_bpm':'proposal_spectral_peak_bpm',
        'ridge_bpm':'consensus_bpm','accepted':'proposal_accepted','status':'proposal_status'}).to_csv(
            args.output/'fusion_proposals.csv',index=False)
    diagnostics.rename(columns={'output_accepted':'proposal_accepted'}).to_csv(
        args.output/'fusion_diagnostics.csv',index=False)
    variants={}
    for name,table in tables.items():
        # Empty HR tables still need numeric columns for reference evaluation.
        table['accepted']=table.accepted.astype(bool)
        for column in ('time_s','spectral_peak_bpm','ridge_bpm'):
            table[column]=table[column].astype(float)
        info=summary_for(table)
        if args.reference_ubfc:
            info['reference']=reference_metrics(table,args.reference_ubfc,args.window)
        info['finite_waveform_fraction']=float(np.isfinite(waveforms[name]).mean())
        if name=='fusion':
            info['waveform_generated_windows']=int(table.waveform_generated.sum())
            info['waveform_generated_fraction']=float(table.waveform_generated.mean()) if len(table) else None
            info['neighbor_covered_windows']=int(table.neighbor_covered.sum())
            info['proposal_accepted_windows']=int(table.proposal_accepted.sum())
            info['hr_tracker']='offline_DP_on_saved_fused_waveform'
        variants[name]=info
        table.to_csv(args.output/f'{name}_heart_rate.csv',index=False)
    if source_hashes()!=frozen_sources:
        raise RuntimeError('Inference source changed during execution; results are not a frozen run')
    reference_identity=None
    if args.reference_ubfc:
        reference_identity=dict(path=str(args.reference_ubfc.resolve()),size=args.reference_ubfc.stat().st_size,
                                sha256=digest(args.reference_ubfc))
    summary=dict(version='multi_roi_v2.2',identity=identity,source_hashes=frozen_sources,fps=fps,frames=len(trace),
                 effective_window_s=round(args.window*fps)/fps,effective_step_s=round(args.step*fps)/fps,
                 requested_window_s=requested_window_s,
                 nominal_reference_window_s=args.window,reference_identity=reference_identity,
                 observed_rgb_fraction=float(trace.rgb_valid.mean()),
                 roi_observed_fraction={roi:float(trace[f'{roi}_valid'].mean()) for roi in ROIS},
                 source_counts={str(k):int(v) for k,v in trace.source.value_counts().items()},
                 config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 fusion_config=asdict(config),variants=variants,elapsed_processing_seconds=time.perf_counter()-started,
                 limitations=['Waveforms use offline zero-phase filtering and overlap fusion.',
                              'Final local/DP HR is estimated from the saved fused waveform; DP uses future windows.',
                              'Consensus frequencies are proposals only. Coverage is not accuracy.',
                              'Sampling provenance uses ALL observed and ANY interpolated actual contributors.',
                              'Provenance does not describe full upstream filter support or physiological morphology.',
                              'accepted and proposal quality_proxy are engineering decisions, not accuracy probabilities.',
                              'Reference data is used exclusively for post-hoc evaluation.'])
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout='constrained')
    axes[0].plot(trace.time_s,trace.rgb_valid.astype(int),color='#6b7280',lw=.8,label='Merged RGB available')
    axes[0].set(ylabel='Sampling',ylim=(-.1,1.2),title='V2.2: sampling and offline HR from actual waveforms')
    axes[0].legend(loc='lower right')
    for name,color in [('pos','#8f8f8f'),('chrom','#b8b8b8'),('fusion','#21618c')]:
        t=tables[name]
        axes[1].plot(t.time_s,t.ridge_bpm.where(t.accepted),label=name,color=color,lw=1.2)
    if args.reference_ubfc:
        axes[1].plot(fused.time_s,fused.reference_bpm,'k--',label='UBFC device HR')
    axes[1].set(xlabel='Window center, video elapsed seconds',ylabel='HR (bpm)',ylim=(args.min_bpm,args.max_bpm))
    axes[1].legend(ncol=4);axes[1].grid(alpha=.2)
    fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(video=str(args.video),frames=len(trace),variants=variants,
                         elapsed_processing_seconds=summary['elapsed_processing_seconds']),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':
    main()
