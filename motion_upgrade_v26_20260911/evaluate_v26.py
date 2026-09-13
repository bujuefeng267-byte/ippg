"""Fixed post-inference reference comparison versus V25, no offset optimization."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
import pandas as pd
P=Path('/home/fengbujue/项目/rppg识别');B=P/'results/data1_6_20260911';V25=P/'results/data1_6_v25_20260911/stage4_preserve_waveform';ROOT=P/'results/data1_6_v26_20260911'
spec=importlib.util.spec_from_file_location('reference_core',P/'batch_analysis_20260911/evaluate_batch.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(core.clean(x),ensure_ascii=False,indent=2,allow_nan=False))
def aggregate(rows):
    n=sum(r['Nvalid'] for r in rows);nr=sum(r['Nref'] for r in rows);nc=sum(r['common_windows'] for r in rows)
    return dict(Nref=nr,Nvalid=n,Nwithin5=sum(r['Nwithin5'] for r in rows),
        MAE_bpm=sum(r['Nvalid']*r['MAE_bpm'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['Nvalid']*r['RMSE_bpm']**2 for r in rows if r['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*sum(r['Nwithin5'] for r in rows)/n if n else np.nan,
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in rows)/nr,
        HR_coverage_pct=100*sum(r['Noutput'] for r in rows)/sum(r['Nplanned'] for r in rows),
        common_windows=nc,common_MAE_bpm=sum(r['common_windows']*r['common_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan,
        common_V25_MAE_bpm=sum(r['common_windows']*r['common_V25_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True);args=ap.parse_args();folder=ROOT/args.variant
    assert not (folder/'evaluation_summary.json').exists(),'Preserve completed evaluations'
    rows=[];sensitivity=[];hashes={str(Path(__file__)):sha(__file__),str(P/'batch_analysis_20260911/evaluate_batch.py'):sha(P/'batch_analysis_20260911/evaluate_batch.py')}
    for i in range(1,7):
        c=f'data{i}';d=folder/c;out=d/'evaluation';out.mkdir()
        wave=pd.read_csv(d/'waveform.csv');hr=pd.read_csv(d/'heart_rate.csv');old=pd.read_csv(V25/c/'fusion_heart_rate.csv')
        ow=pd.read_csv(V25/c/'fusion_waveform.csv');ref=pd.read_csv(B/c/'evaluation/paired_windows.csv');metadata=json.loads((B/c/'inference/summary.json').read_text())
        fps=metadata['fps'];frames=metadata['frames'];assert len(wave)==frames
        np.testing.assert_allclose(wave.time_s,ow.time_s,atol=1e-8,rtol=0);np.testing.assert_allclose(hr.time_s,ref.time_s,atol=1e-8,rtol=0)
        np.testing.assert_allclose(old.time_s,ref.time_s,atol=1e-8,rtol=0)
        ok=core.bools(hr.accepted);ook=core.bools(old.accepted);y=hr.ridge_bpm.to_numpy(float);oy=old.ridge_bpm.to_numpy(float);r=ref.reference_bpm.to_numpy(float)
        assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
        finite=np.isfinite(wave.base);starts=np.arange(0,frames-round(10*fps)+1,round(fps));assert len(starts)==len(hr)
        assert all(finite.iloc[a:a+round(10*fps)].all() for a in starts[ok])
        if 'covered' in wave:np.testing.assert_array_equal(finite,core.bools(wave.covered))
        common=ok&ook&np.isfinite(r);new=ok&~ook&np.isfinite(r);lost=~ok&ook&np.isfinite(r)
        m=core.score(y,ok,r);prior=core.score(oy,ook,r)
        row=dict(variant=args.variant,case=c,**m,V25_MAE_bpm=prior['MAE_bpm'],delta_MAE_bpm=m['MAE_bpm']-prior['MAE_bpm'],
            delta_P5_pp=m['P5_valid_pct']-prior['P5_valid_pct'],delta_R5_pp=m['R5_all_reference_pct']-prior['R5_all_reference_pct'],
            delta_hr_coverage_pp=m['hr_output_coverage_pct']-prior['hr_output_coverage_pct'],
            waveform_coverage_pct=100*finite.mean(),delta_waveform_coverage_pp=100*(finite.mean()-np.isfinite(ow.base).mean()),
            common_windows=int(common.sum()),new_windows=int(new.sum()),lost_windows=int(lost.sum()),
            common_MAE_bpm=np.abs(y[common]-r[common]).mean() if common.any() else np.nan,
            common_V25_MAE_bpm=np.abs(oy[common]-r[common]).mean() if common.any() else np.nan,
            new_MAE_bpm=np.abs(y[new]-r[new]).mean() if new.any() else np.nan,
            lost_V25_MAE_bpm=np.abs(oy[lost]-r[lost]).mean() if lost.any() else np.nan,
            mean_HR_bpm=float(np.mean(y[ok])) if ok.any() else np.nan,
            target_P5_95_met=bool(m['P5_valid_pct']>=95),status_counts=hr.status.value_counts().to_dict())
        paired=ref[['window_index','time_s','window_start_s','window_end_s','reference_bpm','reference_valid']].copy()
        paired['estimated_bpm']=y;paired['accepted']=ok;paired['V25_bpm']=oy;paired['V25_accepted']=ook
        paired['error_bpm']=np.where(ok,y-r,np.nan);paired['abs_error_bpm']=np.abs(paired.error_bpm)
        paired['status']=hr.status;paired['common']=common;paired['new']=new;paired['lost']=lost
        paired.to_csv(out/'paired_windows.csv',index=False);write(out/'metrics.json',row);rows.append(row)
        alignment=json.loads((B/c/'evaluation/alignment.json').read_text());raw=pd.read_csv(alignment['reference_path'])
        sr=[]
        for shift in [-5,-2,-1,0,1,2,5]:
            adjusted=core.reference_windows(raw,alignment['video_start_utc_ns'],ref.window_start_s,ref.window_end_s,shift)
            if shift==0:np.testing.assert_allclose(r,adjusted.reference_bpm,atol=1e-9,rtol=0,equal_nan=True)
            sr.append(dict(variant=args.variant,case=c,shift_s=shift,**core.score(y,ok,adjusted.reference_bpm.to_numpy(float))))
        pd.DataFrame(sr).to_csv(out/'alignment_sensitivity.csv',index=False);sensitivity+=sr
        for path in [d/'waveform.csv',d/'heart_rate.csv',B/c/'evaluation/paired_windows.csv',B/c/'evaluation/alignment.json',Path(alignment['reference_path'])]:hashes[str(path)]=sha(path)
    agg=aggregate(rows);prior=json.loads((V25/'stage_evaluation.json').read_text())['pooled']
    checks=dict(pooled_MAE=agg['MAE_bpm']<prior['MAE_bpm'],pooled_R5_gain=agg['R5_all_reference_pct']>=prior['R5_all_reference_pct']+5,
        each_MAE=all(r['delta_MAE_bpm']<=3 for r in rows),each_P5=all(r['delta_P5_pp']>=-3 for r in rows),each_R5=all(r['delta_R5_pp']>=-3 for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp']>=-3 for r in rows),each_waveform_coverage=all(r['delta_waveform_coverage_pp']>=-3 for r in rows),
        common_MAE=agg['common_MAE_bpm']<agg['common_V25_MAE_bpm'])
    report=dict(variant=args.variant,pooled=agg,cases=rows,promotion_checks=checks,promotion_pass=all(checks.values()),
        near_all_within_5bpm_target_met=all(r['target_P5_95_met'] for r in rows),scope='Previously seen development clips, not held-out validation; estimated original alignment',input_hashes=hashes)
    write(folder/'evaluation_summary.json',report);pd.DataFrame(rows).drop(columns=['status_counts']).to_csv(folder/'metrics.csv',index=False)
    print(json.dumps(core.clean(dict(variant=args.variant,pooled=agg,promotion_checks=checks,promotion_pass=all(checks.values()),
        cases=[{k:r[k] for k in ['case','MAE_bpm','P5_valid_pct','R5_all_reference_pct','hr_output_coverage_pct','waveform_coverage_pct']} for r in rows])),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
