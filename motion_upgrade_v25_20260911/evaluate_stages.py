"""Evaluate every V25 development stage against the frozen six-video baseline."""
from pathlib import Path
import argparse, hashlib, importlib.util, json
import numpy as np
import pandas as pd

PROJECT=Path('/home/fengbujue/项目/rppg识别')
BASE=PROJECT/'results/data1_6_20260911'
ROOT=PROJECT/'results/data1_6_v25_20260911'
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch_reference_core',PROJECT/'batch_analysis_20260911/evaluate_batch.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(core.clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def case(stage,case):
    directory=ROOT/stage/case; out=directory/'evaluation';out.mkdir(exist_ok=True)
    s=json.loads((directory/'summary.json').read_text())
    config=s['config'];assert config['window']==10 and config['step']==1 and config['max_gap']==.1
    hr=pd.read_csv(directory/'fusion_heart_rate.csv');wave=pd.read_csv(directory/'fusion_waveform.csv')
    prior_hr=pd.read_csv(BASE/case/'inference/fusion_heart_rate.csv')
    prior_wave=pd.read_csv(BASE/case/'inference/fusion_waveform.csv')
    reference=pd.read_csv(BASE/case/'evaluation/paired_windows.csv')
    np.testing.assert_allclose(hr.time_s,reference.time_s,atol=1e-8,rtol=0)
    np.testing.assert_allclose(wave.time_s,prior_wave.time_s,atol=1e-8,rtol=0)
    fps=s['fps'];frames=s['frames'];assert len(wave)==frames
    ok=core.bools(hr.accepted);old_ok=core.bools(prior_hr.accepted)
    y=hr.ridge_bpm.to_numpy(float);old_y=prior_hr.ridge_bpm.to_numpy(float)
    ref=reference.reference_bpm.to_numpy(float);eligible=np.isfinite(ref)
    assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
    finite=np.isfinite(wave.base.to_numpy(float));assert np.array_equal(finite,core.bools(wave.covered))
    starts=np.arange(0,frames-round(10*fps)+1,round(fps))
    assert len(starts)==len(hr)
    assert all(finite[a:a+round(10*fps)].all() for a in starts[ok])
    if stage=='v24_replay':
        np.testing.assert_array_equal(ok,old_ok)
        np.testing.assert_allclose(y,old_y,atol=1e-8,rtol=0,equal_nan=True)
        np.testing.assert_allclose(wave.base,prior_wave.base,atol=1e-8,rtol=0,equal_nan=True)
    common=ok&old_ok&eligible; new=ok&~old_ok&eligible;lost=~ok&old_ok&eligible
    metric=core.score(y,ok,ref)
    baseline=core.score(old_y,old_ok,ref)
    rows=dict(stage=stage,case=case,**metric,baseline_MAE_bpm=baseline['MAE_bpm'],
        delta_MAE_bpm=metric['MAE_bpm']-baseline['MAE_bpm'],
        delta_R5_pp=metric['R5_all_reference_pct']-baseline['R5_all_reference_pct'],
        delta_hr_coverage_pp=metric['hr_output_coverage_pct']-baseline['hr_output_coverage_pct'],
        waveform_coverage_pct=100*finite.mean(),
        baseline_waveform_coverage_pct=100*np.isfinite(prior_wave.base).mean(),
        delta_waveform_coverage_pp=100*(finite.mean()-np.isfinite(prior_wave.base).mean()),
        common_windows=int(common.sum()),new_windows=int(new.sum()),lost_windows=int(lost.sum()),
        common_MAE_bpm=float(np.abs(y[common]-ref[common]).mean()) if common.any() else np.nan,
        common_baseline_MAE_bpm=float(np.abs(old_y[common]-ref[common]).mean()) if common.any() else np.nan,
        new_MAE_bpm=float(np.abs(y[new]-ref[new]).mean()) if new.any() else np.nan,
        lost_baseline_MAE_bpm=float(np.abs(old_y[lost]-ref[lost]).mean()) if lost.any() else np.nan,
        correct_new_windows=int((new&(np.abs(y-ref)<=5)).sum()),
        lost_correct_windows=int((lost&(np.abs(old_y-ref)<=5)).sum()),
        mean_hr_bpm=float(y[ok].mean()) if ok.any() else np.nan,
        status_counts=hr.status.value_counts().to_dict(),
        branch_counts=s.get('branch_selection_counts'),
        sync_status='fixed_original_estimated_timestamps_not_hardware_sync')
    paired=reference[['window_index','time_s','window_start_s','window_end_s','reference_bpm','reference_valid']].copy()
    paired['accepted']=ok;paired['estimated_bpm']=y;paired['error_bpm']=np.where(ok&eligible,y-ref,np.nan)
    paired['abs_error_bpm']=np.abs(paired.error_bpm)
    paired['baseline_accepted']=old_ok;paired['baseline_bpm']=old_y;paired['common']=common
    paired['new']=new;paired['lost']=lost;paired['status']=hr.status
    paired.to_csv(out/'paired_windows.csv',index=False)
    alignment=json.loads((BASE/case/'evaluation/alignment.json').read_text())
    raw=pd.read_csv(alignment['reference_path'])
    sensitivity=[]
    for shift in (-5,-2,-1,0,1,2,5):
        adjusted=core.reference_windows(raw,alignment['video_start_utc_ns'],
            reference.window_start_s,reference.window_end_s,shift)
        if shift==0:np.testing.assert_allclose(ref,adjusted.reference_bpm,atol=1e-9,rtol=0,equal_nan=True)
        sensitivity.append(dict(stage=stage,case=case,shift_s=shift,
            **core.score(y,ok,adjusted.reference_bpm.to_numpy(float))))
    pd.DataFrame(sensitivity).to_csv(out/'alignment_sensitivity.csv',index=False)
    write(out/'metrics.json',rows)
    write(out/'provenance.json',dict(evaluator_sha256=sha(__file__),
        reference_core_sha256=sha(PROJECT/'batch_analysis_20260911/evaluate_batch.py'),
        frozen_reference_files={str(p):sha(p) for p in [BASE/case/'evaluation/paired_windows.csv',BASE/case/'evaluation/alignment.json']},
        inputs={str(p):sha(p) for p in [directory/'summary.json',directory/'fusion_waveform.csv',directory/'fusion_heart_rate.csv']},
        reference_used_only_after_inference=True))
    return rows

def pooled(rows):
    n=sum(r['Nvalid'] for r in rows);nr=sum(r['Nref'] for r in rows)
    nc=sum(r['common_windows'] for r in rows)
    return dict(Nvalid=n,Nref=nr,Nwithin5=sum(r['Nwithin5'] for r in rows),
        MAE_bpm=sum(r['MAE_bpm']*r['Nvalid'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['RMSE_bpm']**2*r['Nvalid'] for r in rows if r['Nvalid'])/n) if n else np.nan,
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in rows)/nr,
        hr_output_coverage_pct=100*sum(r['Noutput'] for r in rows)/sum(r['Nplanned'] for r in rows),
        common_MAE_bpm=sum(r['common_MAE_bpm']*r['common_windows'] for r in rows if r['common_windows'])/nc if nc else np.nan,
        common_baseline_MAE_bpm=sum(r['common_baseline_MAE_bpm']*r['common_windows'] for r in rows if r['common_windows'])/nc if nc else np.nan)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);args=ap.parse_args()
    rows=[case(args.stage,f'data{i}') for i in range(1,7)]
    agg=pooled(rows)
    baseline=json.loads((BASE/'comparison_metrics.json').read_text())
    n=sum(r['Nvalid'] for r in baseline);nr=sum(r['Nref'] for r in baseline)
    bmae=sum(r['MAE_bpm']*r['Nvalid'] for r in baseline)/n
    br5=100*sum(r['Nwithin5'] for r in baseline)/nr
    checks=dict(pooled_MAE_reduction=agg['MAE_bpm']<=.9*bmae,
        pooled_R5_gain=agg['R5_all_reference_pct']>=br5+5,
        all_case_MAE=all(r['delta_MAE_bpm']<=5 for r in rows),
        protected_MAE=all(r['delta_MAE_bpm']<=1 for r in rows if r['case'] in ('data2','data4')),
        protected_R5=all(r['delta_R5_pp']>=-5 for r in rows if r['case'] in ('data2','data4')),
        HR_coverage=all(r['delta_hr_coverage_pp']>=-5 for r in rows),
        waveform_coverage=all(r['delta_waveform_coverage_pp']>=-5 for r in rows),
        common_MAE=agg['common_MAE_bpm']<agg['common_baseline_MAE_bpm'])
    result=dict(stage=args.stage,pooled=agg,baseline_pooled_MAE_bpm=bmae,baseline_pooled_R5_pct=br5,
        adoption_checks=checks,adoption_pass=all(checks.values()),cases=rows,
        interpretation='Development regression only; not independently held-out validation.')
    write(ROOT/args.stage/'stage_evaluation.json',result)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]).to_csv(ROOT/args.stage/'stage_metrics.csv',index=False)
    print(json.dumps(core.clean(dict(stage=args.stage,pooled=agg,adoption_checks=checks,adoption_pass=all(checks.values()),
        per_case=[{k:r[k] for k in ['case','MAE_bpm','hr_output_coverage_pct','waveform_coverage_pct','R5_all_reference_pct','delta_MAE_bpm']} for r in rows])),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
