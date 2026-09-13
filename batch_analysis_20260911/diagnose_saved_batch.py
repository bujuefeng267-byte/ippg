"""Post-hoc fixed-protocol diagnostics only; no inference, retiming or tuning."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.signal import welch


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def stats(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return dict(n=len(v),mean=float(v.mean()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,
                min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None)


def flags(s):
    assert s.notna().all() and s.isin([True,False,0,1]).all()
    return s.to_numpy(bool)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-root',type=Path,required=True);args=ap.parse_args()
    root=args.batch_root; hashes={str(Path(__file__).resolve()):sha(Path(__file__))}
    output=root/'qa/frequency_diagnosis.json'; windows_file=root/'qa/frequency_diagnosis_windows.csv'
    assert not output.exists() and not windows_file.exists(), 'Do not overwrite earlier diagnostic evidence'
    def csv(p):hashes[str(p)]=sha(p);return pd.read_csv(p)
    def js(p):hashes[str(p)]=sha(p);return json.loads(p.read_text())
    rows=[];reports=[]
    for case in [f'data{i}' for i in range(1,7)]:
        folder=root/case;run=folder/'inference';ev=folder/'evaluation'
        summary=js(run/'summary.json');metric=js(ev/'metrics.json')
        hr=csv(run/'fusion_heart_rate.csv');paired=csv(ev/'paired_windows.csv')
        trace=csv(run/'frame_trace.csv');routing=csv(run/'branch_routing.csv')
        diag=csv(run/'fusion_diagnostics.csv');methods=csv(ev/'method_diagnostics.csv')
        proposals={b:csv(run/f'{b}_branch_proposals.csv') for b in ('baseline','tracked')}
        fps=summary['fps'];w=round(10*fps);step=round(fps);grid=np.arange(42.,211.)
        starts=list(range(0,len(trace)-w+1,step));accepted=flags(hr.accepted)
        valid=accepted&flags(paired.reference_valid)
        assert len(hr)==len(paired)==len(routing)==len(starts)
        for i,a in enumerate(starts):
            profiles=[];peak_bpm=[];motion_std=[]
            for axis in ['motion_x','motion_y']:
                m=trace[axis].to_numpy(float)[a:a+w]
                if not np.isfinite(m).all() or np.std(m)<1e-8:continue
                nfft=max(2048,2**int(np.ceil(np.log2(len(m)*4))))
                f,p=welch(m,fs=fps,window='hann',nperseg=len(m),noverlap=0,nfft=nfft,detrend='constant')
                p=np.interp(grid/60,f,p)
                if p.max()>0:profiles.append(p/p.max());peak_bpm.append(grid[np.argmax(p)]);motion_std.append(np.std(m))
            profile=np.max(profiles,axis=0) if profiles else None
            ref=float(paired.reference_bpm.iloc[i]);dp=float(hr.ridge_bpm.iloc[i]);local=float(hr.spectral_peak_bpm.iloc[i])
            r=dict(case=case,window_index=i,time_s=float(hr.time_s.iloc[i]),accepted=bool(accepted[i]),
                   reference_valid=bool(np.isfinite(ref)),reference_bpm=ref,dp_bpm=dp,local_bpm=local,
                   dp_error=dp-ref,local_error=local-ref,dp_to_reference_ratio=dp/ref,
                   abs_dp_minus_2reference=abs(dp-2*ref),abs_dp_minus_half_reference=abs(dp-ref/2),
                   motion_axes_eligible=len(profiles),motion_x_or_y_peak_bpm=';'.join(map(str,peak_bpm)),
                   maximum_motion_std=max(motion_std) if motion_std else np.nan,
                   selected_branch=routing.selected_branch.iloc[i],selection_reason=routing.selection_reason.iloc[i],
                   baseline_score=float(routing.baseline_score.iloc[i]),tracked_score=float(routing.tracked_score.iloc[i]))
            for label,target in [('dp',dp),('reference',ref),('local',local)]:
                r[f'motion_relative_power_at_{label}']=float(np.interp(target,grid,profile)) if profile is not None and np.isfinite(target) else np.nan
            d=diag.loc[diag.window_index.eq(i)]
            for branch in ('baseline','tracked'):
                p=proposals[branch].iloc[i];value=float(p.ridge_bpm)
                generated=bool(p.accepted) and int(p.waveform_roi_count)>=2
                bd=d.loc[d.branch.eq(branch)&d.candidate_weight.gt(0)]
                r[f'{branch}_generated']=generated;r[f'{branch}_proposal_bpm']=value
                r[f'{branch}_proposal_error']=value-ref
                r[f'{branch}_motion_overlap']=float(p.motion_overlap)
                r[f'{branch}_candidate_near_reference_rois']=int(bd.loc[(bd.candidate_bpm-ref).abs().le(6),'roi'].nunique()) if np.isfinite(ref) else 0
            rows.append(r)
        current=pd.DataFrame(rows).loc[lambda x:x.case.eq(case)];paired_rows=current.loc[current.accepted&current.reference_valid]
        mismatch=current.loc[current.selection_reason.eq('frequency_disagreement_fallback')&current.reference_valid]
        comparison=mismatch.loc[mismatch.baseline_generated&mismatch.tracked_generated]
        error_b=comparison.baseline_proposal_error.abs();error_t=comparison.tracked_proposal_error.abs()
        diag_channels=diag.drop_duplicates(['window_index','branch','roi','method'])
        report=dict(case=case,metrics=metric,method_errors=methods.to_dict('records'),
                    branch_selection_counts=routing.selected_branch.value_counts().to_dict(),
                    branch_reason_counts=routing.selection_reason.value_counts().to_dict(),
                    dp_error=stats(paired_rows.dp_error),dp_frequency=stats(paired_rows.dp_bpm),
                    reference_frequency=stats(paired_rows.reference_bpm),dp_to_reference_ratio=stats(paired_rows.dp_to_reference_ratio),
                    near_double_reference_within6bpm_count=int(paired_rows.abs_dp_minus_2reference.le(6).sum()),
                    near_half_reference_within6bpm_count=int(paired_rows.abs_dp_minus_half_reference.le(6).sum()),
                    eligible_motion_windows=int(paired_rows.motion_axes_eligible.gt(0).sum()),
                    motion_relative_power_at_dp=stats(paired_rows.motion_relative_power_at_dp),
                    motion_relative_power_at_reference=stats(paired_rows.motion_relative_power_at_reference),
                    high_motion_overlap_at_dp_above_point8_count=int(paired_rows.motion_relative_power_at_dp.ge(.8).sum()),
                    maximum_motion_std=stats(paired_rows.maximum_motion_std),
                    baseline_candidate_at_reference_atleast2rois=int(paired_rows.baseline_candidate_near_reference_rois.ge(2).sum()),
                    tracked_candidate_at_reference_atleast2rois=int(paired_rows.tracked_candidate_near_reference_rois.ge(2).sum()),
                    disagreement_fallback=dict(n=len(comparison),baseline_proposal_MAE=error_b.mean(),tracked_proposal_MAE=error_t.mean(),
                        tracked_proposal_closer_count=int((error_t<error_b).sum()),baseline_proposal_closer_count=int((error_b<error_t).sum())),
                    channel_status_counts=diag_channels.groupby(['branch','channel_status']).size().to_dict())
        reports.append(report)
        print(json.dumps(clean({k:v for k,v in report.items() if k not in ['metrics','method_errors','channel_status_counts']})),flush=True)
    for p,expected in hashes.items():assert sha(Path(p))==expected,'Input changed during diagnostics'
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(windows_file,index=False)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=reports,input_sha256=hashes,
                reference_used_for_posthoc_diagnostics_only=True,inference_run=False,offset_or_threshold_selection=False,
                methodology='Motion PSD uses complete nonflat x/y tracks and the frozen fusion frequency grid/normalization; no missing motion filled. Fixed 6 bpm neighborhoods are descriptive, not tuned to results.',
                limitations=['Motion spectrum overlap is not proof of motion causality; normalized power can be high for weak movement.',
                    'Reference clock remains the fixed mtime-based estimate.',
                    'Proposal-frequency errors describe routing inputs; they are not final waveform HR and do not predict the effect of switching branches.',
                    'No-qualification windows can inherit waveform coverage from accepted neighbors.'])
    output.write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(str(output),flush=True)


if __name__=='__main__':main()
