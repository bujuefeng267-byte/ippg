"""Reference-only evaluation of frozen data1--6 inference; never selects an offset."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT=Path('/home/fengbujue/项目/rppg识别/results/data1_6_20260911')
SHIFTS=(-5,-2,-1,0,1,2,5)

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.bool_,)):return bool(value)
    return value

def save(path,data):
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def bools(s):
    assert s.notna().all() and s.isin([True,False,0,1]).all()
    return s.to_numpy(bool)

def spans(mask):
    edge=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))

def score(pred,accepted,ref):
    output=accepted & np.isfinite(pred); eligible=np.isfinite(ref)&(ref>0);valid=output&eligible
    error=pred[valid]-ref[valid];ae=np.abs(error)
    n,no,nr,nv=len(pred),int(output.sum()),int(eligible.sum()),int(valid.sum())
    n5,n10=int((ae<=5).sum()),int((ae<=10).sum())
    sd=float(np.std(error,ddof=1)) if nv>1 else np.nan
    bias=float(np.mean(error)) if nv else np.nan
    return dict(Nplanned=n,Noutput=no,Nref=nr,Nvalid=nv,Nwithin5=n5,Nwithin10=n10,
        hr_output_coverage_pct=100*no/n if n else np.nan,
        paired_reference_coverage_pct=100*nv/nr if nr else np.nan,
        MAE_bpm=float(ae.mean()) if nv else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(error**2))) if nv else np.nan,
        Bias_bpm=bias,MAPE_pct=float(np.mean(ae/ref[valid])*100) if nv else np.nan,
        P5_valid_pct=100*n5/nv if nv else np.nan,P10_valid_pct=100*n10/nv if nv else np.nan,
        R5_all_reference_pct=100*n5/nr if nr else np.nan,
        R10_all_reference_pct=100*n10/nr if nr else np.nan,
        P95_abs_error_bpm=float(np.quantile(ae,.95)) if nv else np.nan,
        Max_abs_error_bpm=float(ae.max()) if nv else np.nan,
        LoA_lower_descriptive_bpm=bias-1.96*sd,LoA_upper_descriptive_bpm=bias+1.96*sd)

def reference_windows(raw,start_ns,starts,ends,shift):
    t=(raw.host_utc_ns.to_numpy(np.int64)-np.int64(start_ns)).astype(float)/1e9-shift
    y=raw.hr_bpm.to_numpy(float)
    good=np.isfinite(y)&(y>0);t,y=t[good],y[good]
    assert len(t)>1 and np.all(np.diff(t)>0)
    tail=float(np.median(np.diff(t)))
    result=[]
    for a,b in zip(starts,ends):
        take=(t>=a)&(t<b);points=t[take];n=int(take.sum())
        maxgap=float(np.max(np.diff(np.r_[a,points,b])))
        status='ok'
        if a<t[0] or b>t[-1]+tail:status='outside_reference_recording'
        elif n<2:status='too_few_reference_values'
        elif maxgap>2:status='reference_gap_over_2s_including_edges'
        result.append(dict(reference_bpm=float(y[take].mean()) if status=='ok' else np.nan,
            reference_valid=status=='ok',reference_status=status,reference_samples=n,
            reference_max_gap_s=maxgap))
    return pd.DataFrame(result)

def evaluate(data):
    case=data['case']; folder=ROOT/case;run=folder/'inference'
    if not (folder/'run_record.json').exists():return None
    if json.loads((folder/'run_record.json').read_text())['exit_code']!=0:return None
    out=folder/'evaluation';out.mkdir(exist_ok=True)
    summary=json.loads((run/'summary.json').read_text())
    config=summary['config']
    for key,value in dict(algorithm_mode='guarded_fusion',pixel_mode='tracking_screened',
            max_gap=.1,window=10,step=1,min_bpm=42,max_bpm=210).items():
        assert config[key]==value, f'Unexpected inference parameter {key}'
    hr=pd.read_csv(run/'fusion_heart_rate.csv');wave=pd.read_csv(run/'fusion_waveform.csv')
    trace=pd.read_csv(run/'frame_trace.csv')
    fps=float(summary['fps']);n=int(summary['frames']);duration=n/fps
    assert len(trace)==len(wave)==n
    w=round(10*fps);step=round(fps);startframes=np.arange(0,n-w+1,step)
    assert len(hr)==len(startframes)
    centers=(startframes+w/2)/fps;starts=startframes/fps;ends=(startframes+w)/fps
    np.testing.assert_allclose(hr.time_s,centers,atol=1e-8,rtol=0)
    np.testing.assert_allclose(wave.time_s,np.arange(n)/fps,atol=1e-8,rtol=0)
    accepted=bools(hr.accepted);pred=hr.ridge_bpm.to_numpy(float)
    assert np.isnan(pred[~accepted]).all() and np.isfinite(pred[accepted]).all()
    wave_y=wave.base.to_numpy(float);finite=np.isfinite(wave_y)
    assert all(finite[a:a+w].all() for a in startframes[accepted])
    assert np.array_equal(bools(wave.covered),finite)
    refs=[Path(f['path']) for f in data['files'] if Path(f['path']).name=='polar_hr.csv']
    assert len(refs)==1
    raw=pd.read_csv(refs[0]);assert raw.host_utc_ns.notna().all()
    assert not raw.host_utc_ns.duplicated().any() and np.all(np.diff(raw.host_utc_ns)>0)
    end_epoch=data['video']['extended_unix_mtime_s']
    if end_epoch is None:raise ValueError('No independent archived UTC timestamp; alignment cannot be guessed')
    # Integer arithmetic preserves epoch precision; match the previous data1 rule.
    origin_ns=int(end_epoch)*1_000_000_000-int(round(duration*1_000_000_000))
    align=dict(case=case,video_start_utc_ns=origin_ns,
        video_start_beijing_estimate=datetime.fromtimestamp(origin_ns/1e9,timezone(timedelta(hours=8))).isoformat(),
        archived_video_end_utc_s=end_epoch,decoded_frames=n,fps=fps,duration_s=duration,
        method='Original ZIP extended Unix mtime minus decoded duration; no predicted HR used.',
        sync_status='estimated_from_archived_mtime_not_hardware_synchronized',
        assumption='mtime approximates recording end; flush latency and clock differences unknown.',
        primary_shift_s=0,sensitivity_shifts_s=list(SHIFTS),
        shift_interpretation='Positive shift moves video later on reference clock; not confidence limits or offset optimization.',
        reference_path=str(refs[0]),reference_sha256=sha(refs[0]),
        reference_time='host_utc_ns',reference_value='hr_bpm',
        aggregation='Arithmetic mean of finite positive notifications in each half-open video window.',
        quality_rule='Window within reference recording, >=2 notifications, <=2s gap including window edges.',
        original_video_sha256=data['video']['sha256'])
    save(out/'alignment.json',align)
    sensitivities=[];primary=None
    for shift in SHIFTS:
        ref=reference_windows(raw,origin_ns,starts,ends,shift)
        row=dict(case=case,shift_s=shift,**score(pred,accepted,ref.reference_bpm.to_numpy(float)))
        sensitivities.append(row)
        if shift==0:primary=ref
    pd.DataFrame(sensitivities).to_csv(out/'alignment_sensitivity.csv',index=False)
    joined=pd.DataFrame(dict(window_index=np.arange(len(hr)),time_s=centers,
        window_start_s=starts,window_end_s=ends,accepted=accepted,estimated_bpm=pred,
        local_spectral_bpm=hr.spectral_peak_bpm,output_status=hr.status))
    joined=pd.concat([joined,primary],axis=1)
    valid=accepted&joined.reference_valid.to_numpy(bool)
    joined['error_bpm']=np.where(valid,pred-joined.reference_bpm,np.nan)
    joined['abs_error_bpm']=np.abs(joined.error_bpm)
    snrs=[]
    for a,ref in zip(startframes,joined.reference_bpm):
        part=wave_y[a:a+w]
        if not np.isfinite(ref) or not np.isfinite(part).all():snrs.append(np.nan);continue
        f,p=welch(part,fs=fps,window='hann',nperseg=w,noverlap=0,nfft=max(8192,2**int(np.ceil(np.log2(w)))),detrend=False)
        band=(f>=.7)&(f<=3.5);heart=band&(np.abs(f-ref/60)<=.1)
        signal=p[heart].sum();noise=p[band&~heart].sum()
        snrs.append(10*np.log10(signal/noise) if signal>0 and noise>0 else np.nan)
    joined['reference_frequency_snr_db']=snrs
    joined.to_csv(out/'paired_windows.csv',index=False)
    base=score(pred,accepted,primary.reference_bpm.to_numpy(float))
    stream=next(s for s in data['ffprobe']['streams'] if s['codec_type']=='video')
    m=dict(case=case,video_name=Path(data['video']['path']).name,
        fps=fps,width=stream['width'],height=stream['height'],frames=n,duration_s=duration,
        **base,face_detection_pct=100*bools(trace.face_detected).mean(),
        rgb_usable_pct=100*bools(trace.rgb_valid).mean(),
        waveform_coverage_pct=100*finite.mean(),waveform_finite_samples=int(finite.sum()),
        waveform_duration_s=finite.sum()/fps,
        waveform_interpolated_samples=int((finite&bools(wave.interpolated)).sum()),
        longest_waveform_run_s=max((b-a for a,b in spans(finite)),default=0)/fps,
        longest_hr_run_windows=max((b-a for a,b in spans(accepted)),default=0),
        hr_internal_breaks=max(0,len(spans(accepted))-1),
        reference_frequency_snr_median_db=float(np.nanmedian(snrs)) if np.isfinite(snrs).any() else None,
        reference_frequency_snr_windows=int(np.isfinite(snrs).sum()),
        predicted_mean_bpm=float(np.mean(pred[accepted])) if accepted.any() else None,
        paired_predicted_mean_bpm=float(np.mean(pred[valid])) if valid.any() else None,
        paired_reference_mean_bpm=float(joined.reference_bpm[valid].mean()) if valid.any() else None,
        predicted_min_bpm=float(np.min(pred[accepted])) if accepted.any() else None,
        predicted_max_bpm=float(np.max(pred[accepted])) if accepted.any() else None,
        output_status_counts=hr.status.value_counts().to_dict(),
        sync_status=align['sync_status'],reference_kind='Polar device HR; not ECG',
        sensitivity_MAE_min_bpm=min((x['MAE_bpm'] for x in sensitivities if np.isfinite(x['MAE_bpm'])),default=np.nan),
        sensitivity_MAE_max_bpm=max((x['MAE_bpm'] for x in sensitivities if np.isfinite(x['MAE_bpm'])),default=np.nan))
    if m['Nref'] and m['Nvalid']:
        np.testing.assert_allclose(m['R5_all_reference_pct'],m['paired_reference_coverage_pct']*m['P5_valid_pct']/100)
    save(out/'metrics.json',m)
    # All variants are reported as diagnostics, never selected using the reference.
    secondary=[]
    for variant in ['fusion','pos','chrom']:
        v=pd.read_csv(run/f'{variant}_heart_rate.csv'); ok=bools(v.accepted)
        for column in ['ridge_bpm','spectral_peak_bpm']:
            secondary.append(dict(case=case,variant=variant,estimator=column,
                **score(v[column].to_numpy(float),ok,primary.reference_bpm.to_numpy(float))))
    pd.DataFrame(secondary).to_csv(out/'method_diagnostics.csv',index=False)
    evidence_paths=[ROOT/'inputs_manifest.json',ROOT/'protocol_before_inference.json',
        ROOT/'evaluation_protocol.json',folder/'run_record.json',refs[0]]
    evidence_paths += sorted(run.glob('*.csv')) + sorted(run.glob('*.json'))
    save(out/'evaluation_provenance.json',dict(evaluator_sha256=sha(Path(__file__)),
        inputs={str(p):sha(p) for p in evidence_paths},
        snr_definition='Reference-frequency H1 +/-0.1Hz, 0.7-3.5Hz band, Hann full-window Welch, detrend=False, all reference-valid finite waveform windows; not identical to older SNR reports.',
        inference_source_hashes=summary.get('source_hashes'),reference_used_only_after_predictions=True))
    print(json.dumps(clean(m),ensure_ascii=False),flush=True)
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='*');args=ap.parse_args()
    protocol=ROOT/'evaluation_protocol.json'
    declared=dict(primary_algorithm='frozen V2.4 guarded_fusion',primary_estimator='ridge_bpm accepted windows',
        main_offset_policy='ZIP UTC mtime minus decoded duration, independently of predictions',
        shifts_s=list(SHIFTS),primary_shift_s=0,accuracy_thresholds_bpm=[5,10],
        P5_definition='Count(abs error<=5)/paired output windows',
        R5_definition='Count(abs error<=5)/all reference-valid planned windows; missing outputs remain failures',
        coverage_definition='HR: accepted/planned windows; waveform: finite samples/all decoded frames',
        reference_window='10s half-open sample support; mean positive finite device HR; min2 values and max2s gap including edges',
        no_waveform_truth_claim='Reference-frequency SNR is a frequency agreement proxy; waveform morphology accuracy unverified.',
        uncertainty='Overlapping windows and repeated subjects; descriptive metrics, no independent-window confidence claims.')
    if protocol.exists():assert json.loads(protocol.read_text())==declared
    else:save(protocol,declared)
    manifest=json.loads((ROOT/'inputs_manifest.json').read_text())
    for data in manifest:
        if not args.cases or data['case'] in args.cases:evaluate(data)
    all_metrics=[json.loads(p.read_text()) for p in sorted(ROOT.glob('data*/evaluation/metrics.json'))]
    if all_metrics:
        pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,dict)} for row in all_metrics]).to_csv(ROOT/'comparison_metrics.csv',index=False)
        save(ROOT/'comparison_metrics.json',all_metrics)

if __name__=='__main__':main()
