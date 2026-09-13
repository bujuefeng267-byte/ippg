"""Independent arithmetic/provenance QA, without importing the evaluator."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ESTIMATORS={'local_peak':'spectral_peak_bpm','offline_ridge':'ridge_bpm'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def equal(a,b,tol=1e-8):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)

def local_path(value):
    value=str(value)
    if sys.platform=='linux' and len(value)>2 and value[1]==':':
        value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value)

def score(pred,accepted,reference):
    pred=np.array(pred,float); ref=np.array(reference,float)
    output=np.array(accepted,bool)&np.isfinite(pred)
    eligible=np.isfinite(ref)&(ref>0)
    valid=output&eligible
    err=pred[valid]-ref[valid]
    n5=int((np.abs(err)<=5).sum())
    return dict(Nplanned=len(pred),Noutput=int(output.sum()),Nref=int(eligible.sum()),
        Nvalid=int(valid.sum()),Nwithin5=n5,C_out_pct=100*output.mean(),
        MAE_bpm=np.abs(err).mean() if len(err) else np.nan,
        RMSE_bpm=np.sqrt(np.mean(err**2)) if len(err) else np.nan,
        Bias_bpm=err.mean() if len(err) else np.nan,
        P5_valid_pct=100*n5/len(err) if len(err) else np.nan,
        R5_all_reference_pct=100*n5/eligible.sum() if eligible.any() else np.nan)

def main():
    ev=HERE/'evaluation'
    report=load(ev/'summary.json')
    m=pd.read_csv(ev/'metrics.csv');ref=pd.read_csv(ev/'reference_windows.csv')
    wave_metrics=pd.read_csv(ev/'waveform_metrics.csv')
    checked=0; waves=0; rejected_auxiliary_peaks=0
    raw_tables={}; independent_scores={}
    for row in m.to_dict('records'):
        case,ver,variant,est=(row[k] for k in ['case','version','variant','estimator'])
        directory=local_path(report['case_metadata'][case]['directories'][ver])
        hr=pd.read_csv(directory/f'{variant}_heart_rate.csv')
        raw_tables[case,ver,variant]=hr
        rr=ref[ref.case.eq(case)&ref.shift_s.eq(0)].sort_values('window_index')
        equal(hr.time_s,rr.time_s)
        accepted=hr.accepted.to_numpy(bool)
        pred=hr[ESTIMATORS[est]].to_numpy(float)
        if variant=='fusion' or est=='offline_ridge':
            assert not np.isfinite(pred[~accepted]).any(), 'Rejected final HR must be missing'
        else:
            rejected_auxiliary_peaks+=int(np.isfinite(pred[~accepted]).sum())
        actual=score(pred,accepted,rr.reference_bpm)
        independent_scores[case,ver,variant,est]=actual
        for k,value in actual.items():equal(value,row[k])
        checked+=1
        if est=='offline_ridge':
            wave=pd.read_csv(directory/f'{variant}_waveform.csv')
            ww=wave_metrics[wave_metrics.case.eq(case)&wave_metrics.version.eq(ver)&
                            wave_metrics.variant.eq(variant)&wave_metrics.scope.eq('own')].iloc[0]
            finite=np.isfinite(wave.base)
            equal(100*finite.mean(),ww.finite_sample_pct)
            equal(finite.sum(),ww.finite_sample_count)
            equal(finite.sum()/report['case_metadata'][case]['fps'],ww.finite_sample_duration_s)
            assert len(wave)==report['case_metadata'][case]['frames']
            waves+=1
    assert checked==216 and waves==108
    paired=pd.read_csv(ev/'paired_metrics.csv')
    independent_pairs={}
    for row in paired.to_dict('records'):
        case,before,after,variant,est,scope=(row[k] for k in ['case','before','after','variant','estimator','scope'])
        a,b=raw_tables[case,before,variant],raw_tables[case,after,variant]
        aa,bb=a.accepted.to_numpy(bool),b.accepted.to_numpy(bool)
        if scope=='common_before':chosen,mask=a,aa&bb
        elif scope=='common_after':chosen,mask=b,aa&bb
        elif scope=='added_after':chosen,mask=b,~aa&bb
        elif scope=='lost_before':chosen,mask=a,aa&~bb
        else:raise AssertionError(scope)
        rr=ref[ref.case.eq(case)&ref.shift_s.eq(0)].sort_values('window_index')
        actual=score(chosen[ESTIMATORS[est]],mask,rr.reference_bpm)
        for k,value in actual.items():equal(value,row[k])
        independent_pairs[case,before,after,variant,est,scope]=actual
    macros=pd.read_csv(ev/'macro_metrics.csv')
    for row in macros.to_dict('records'):
        before,after,side,scope,est=(row[k] for k in ['before','after','side','scope','estimator'])
        version=before if side=='before' else after
        if row['population']=='human_reference_UBFC_and_estimated_data1':
            scores=[independent_scores[c,version,'fusion',est] if scope=='own' else
                    independent_pairs[c,before,after,'fusion',est,'common_'+side] for c in ['ubfc','data1']]
            equal(np.mean([x['MAE_bpm'] for x in scores]),row['source_balanced_MAE_bpm'])
            equal(np.sqrt(np.mean([x['RMSE_bpm']**2 for x in scores])),row['source_balanced_RMSE_bpm'])
            equal(np.mean([x['R5_all_reference_pct'] for x in scores]),row['mean_R5_all_reference_pct'])
        elif row['population']=='five_human_captures':
            cases=['user0904','user0907','ubfc','kaggle_full','data1']
            equal(np.mean([independent_scores[c,version,'fusion','offline_ridge']['C_out_pct'] for c in cases]),row['mean_C_out_pct'])
            ww=wave_metrics[wave_metrics.case.isin(cases)&wave_metrics.version.eq(version)&
                            wave_metrics.variant.eq('fusion')&wave_metrics.scope.eq('own')]
            assert len(ww)==5
            equal(ww.finite_sample_pct.mean(),row['mean_finite_sample_pct'])
        else:raise AssertionError(row['population'])
    sensitivity=pd.read_csv(ev/'sensitivity.csv')
    assert set(sensitivity.shift_s)=={-5,-2,-1,0,1,2,5}
    assert len(sensitivity)==252
    for name,info in report['output_manifest'].items():assert sha(ev/name)==info['sha256']
    tests=load(HERE/'validation/tests.json')
    assert tests['passed'] and tests['run']>=85 and tests['errors']==tests['failures']==tests['skipped']==0
    for n,h in tests['test_hashes'].items():assert sha(HERE/n)==h
    for n,h in report['source_protocol_v24']['source_hashes'].items():assert sha(HERE/n)==h
    replay=load(HERE/'validation/waveform_replay_qa.json')
    assert replay['passed']
    gate=load(ev/'upgrade_gate.json')
    assert gate==report['upgrade_gate'] and gate['evaluated']
    result=dict(passed=True,checker_sha256=sha(__file__),metric_rows_recomputed=checked,
        paired_metric_rows_recomputed=len(paired),macro_rows_recomputed=len(macros),
        saved_waveforms_counted=waves,all_seven_data1_shifts_preserved=True,
        evaluated_files_and_sources_unchanged=True,rejected_final_fusion_and_DP_HR_remains_missing=True,
        rejected_auxiliary_POS_CHROM_diagnostic_peaks_excluded=rejected_auxiliary_peaks,
        tests_run=tests['run'],saved_waveform_replay_passed=True,
        performance_gate_passed=gate['default']['eligible_for_named_upgrade'],
        evidence={str(p.relative_to(HERE)):sha(p) for p in [ev/'summary.json',ev/'metrics.csv',
            ev/'reference_windows.csv',ev/'waveform_metrics.csv',ev/'upgrade_gate.json',
            HERE/'validation/waveform_replay_qa.json',HERE/'validation/tests.json']},
        limitations=['References retain original protocols, including estimated data1 alignment.',
                     'Regression arithmetic QA does not establish clinical or held-out validity.'])
    target=HERE/'independent_qa.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'independent_qa.md').write_text(
        '# V2.4 独立结果核对\n\n'
        f'核对通过：从实际 HR CSV 与独立读取的参考窗口重新计算 {checked} 行主要指标；'
        f'从实际波形 CSV 重新统计 {waves} 组有限样本、时长与覆盖率。\n\n'
        '另核对拒绝的融合 HR/DP HR 为空；POS/CHROM 的拒绝窗口诊断峰不计入输出或误差。'
        '七个 data1 偏移完整，评测文件与冻结源码哈希一致，'
        f'{tests["run"]} 个代码测试通过，全部保存波形的独立 HR 回算通过。\n\n'
        '本核对通过表示计算与文件可信；是否性能升级通过另见 evaluation/upgrade_gate.json。'
        'data1 同步仍是估计，所有视频是开发回归资料。\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
