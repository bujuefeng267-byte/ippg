"""Ground-truth-conditioned information audit, NEVER an inference estimator.

No predictions are generated or overwritten. Oracle hits only describe whether
the frozen spectra/candidate policy contain a reference-compatible possibility.
An oracle may cherry-pick noise and correlated channels: its coverage is NOT an
achieved accuracy, a guarantee of recoverability, or independent sensor support.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

PROJECT = Path('/home/fengbujue/项目/rppg识别')
V25 = PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
BASE = PROJECT/'results/data1_6_20260911'
OUT = PROJECT/'results/data1_6_v26_20260911/diagnostics'
GRID = np.arange(42., 211., 1.)
ROIS = ('forehead', 'left_cheek', 'right_cheek')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,(np.integer,int)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def write(path,data):
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def spectral_information(segment,fps,reference):
    x=np.asarray(segment,float)
    result=dict(finite_fraction=float(np.isfinite(x).mean()),spectral_available=False,
        status='nonfinite_window',reference_relative_power=np.nan,reference_H1_energy_fraction=np.nan,
        reference_H1_snr_db=np.nan,nearest_local_peak_distance_bpm=np.nan,
        reference_peak_rank_all_local=np.nan,reference_peak_rank_v25_qualified=np.nan,
        reference_matching_peak_relative_power=np.nan,
        top1_peak_oracle_hit=False,top3_peak_oracle_hit=False,top8_peak_oracle_hit=False,
        uncapped_qualified_peak_oracle_hit=False,v25_supported_state_oracle_hit=False,
        top8_strong_20pct_peak_oracle_hit=False,top8_strong_50pct_peak_oracle_hit=False,
        qualified_peak_count=0,diagnostic_top8_peaks_json='[]')
    if not np.isfinite(x).all():return result
    if np.std(x)<1e-8:
        result['status']='flat_window';return result
    # Exact frozen V25 candidate spectral convention, reimplemented here.
    f,p=welch(x,fs=fps,window='hann',nperseg=len(x),noverlap=0,
              nfft=max(2048,len(x)),detrend='constant')
    power=np.interp(GRID/60.,f,p);maximum=power.max()
    if maximum<=0:
        result['status']='zero_band_power';return result
    relative=power/maximum
    all_k,_=find_peaks(np.r_[0.,relative,0.])
    all_k=sorted([int(k)-1 for k in all_k if 0<=int(k)-1<len(GRID)],key=lambda k:(-relative[k],GRID[k]))
    kk,props=find_peaks(np.r_[0.,relative,0.],height=.05,prominence=.02,distance=6)
    qualified=sorted([(int(k)-1,float(prom)) for k,prom in zip(kk,props['prominences'])
                      if 0<=int(k)-1<len(GRID)],key=lambda kp:(-relative[kp[0]],GRID[kp[0]]))
    peaks=[];supported=[]
    for rank,(k,prom) in enumerate(qualified[:8],1):
        states=GRID[(np.abs(GRID-GRID[k])<=3)&(relative>=.05)]
        supported.extend(states)
        peaks.append(dict(relative_power_rank=rank,bpm=float(GRID[k]),relative_power=float(relative[k]),
            prominence=prom,peak_energy_fraction_6bpm=float(power[np.abs(GRID-GRID[k])<=6].sum()/power.sum())))
    # Separate diagnostic energy convention; matches baseline H1 SNR settings.
    hf,hp=welch(x,fs=fps,window='hann',nperseg=len(x),noverlap=0,nfft=max(8192,len(x)),detrend=False)
    band=(hf>=.7)&(hf<=3.5);h1=band&(np.abs(hf-reference/60.)<=.1)
    pulse,other=hp[h1].sum(),hp[band&~h1].sum()
    nearby_all=[j+1 for j,k in enumerate(all_k) if abs(GRID[k]-reference)<=5]
    nearby_qualified=[j+1 for j,(k,_) in enumerate(qualified) if abs(GRID[k]-reference)<=5]
    near_power=[relative[k] for k in all_k if abs(GRID[k]-reference)<=5]
    result.update(spectral_available=True,status='spectrum_computed',
        reference_relative_power=float(np.interp(reference,GRID,relative)),
        reference_H1_energy_fraction=float(pulse/(pulse+other)) if pulse+other>0 else np.nan,
        reference_H1_snr_db=float(10*np.log10(pulse/other)) if pulse>0 and other>0 else np.nan,
        nearest_local_peak_distance_bpm=min((abs(GRID[k]-reference) for k in all_k),default=np.nan),
        reference_peak_rank_all_local=min(nearby_all,default=np.nan),
        reference_peak_rank_v25_qualified=min(nearby_qualified,default=np.nan),
        reference_matching_peak_relative_power=float(max(near_power)) if near_power else np.nan,
        top1_peak_oracle_hit=any(abs(p['bpm']-reference)<=5 for p in peaks[:1]),
        top3_peak_oracle_hit=any(abs(p['bpm']-reference)<=5 for p in peaks[:3]),
        top8_peak_oracle_hit=any(abs(p['bpm']-reference)<=5 for p in peaks),
        uncapped_qualified_peak_oracle_hit=bool(nearby_qualified),
        v25_supported_state_oracle_hit=any(abs(v-reference)<=5 for v in supported),
        top8_strong_20pct_peak_oracle_hit=any(abs(p['bpm']-reference)<=5 and p['relative_power']>=.20 for p in peaks),
        top8_strong_50pct_peak_oracle_hit=any(abs(p['bpm']-reference)<=5 and p['relative_power']>=.50 for p in peaks),
        qualified_peak_count=len(qualified),diagnostic_top8_peaks_json=json.dumps(peaks,separators=(',',':')))
    return result


def add_signal(signals,label,kind,group,roi,values):
    signals.append(dict(channel=label,kind=kind,group=group,physical_roi=roi,values=np.asarray(values,float)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args()
    out=args.output
    if out.exists():raise FileExistsError('Preserve prior diagnostics; select a new output directory')
    out.mkdir(parents=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),diagnostic_script_sha256=sha(__file__),
        purpose='REFERENCE-CONDITIONED ORACLE / INFORMATION LIMIT ONLY; NOT INFERENCE OUTPUT OR ACHIEVED ACCURACY',
        data='Frozen six V25 stage4 recordings, same nominal ten-second windows and original estimated zero-shift reference.',
        candidate_policy=dict(grid_bpm=[42,210,1],window='Hann',nfft='max(2048,window samples)',detrend='constant',
            relative_power_min=.05,prominence_min=.02,peak_distance_bpm=6,max_peaks=8,supported_state_radius_bpm=3),
        oracle_definitions=dict(primary='Any actual retained local peak center within 5 bpm of the reference.',
            supported_state='Any original <=3 bpm candidate neighborhood state with >=5% peak-relative energy within 5 bpm of reference; more permissive than peak-center oracle.',
            two_roi='At least two physical ROIs with a matching peak, deduplicated across POS/CHROM and baseline/tracked branches; not independent sensors.'),
        energy_policy=dict(window='Hann full-window Welch',nfft='max(8192,window samples)',detrend=False,band_hz=[.7,3.5],H1_radius_hz=.1),
        raw_rgb_policy='Window fractional color x/mean(x)-1 for each saved channel, no missing-data interpolation; diagnostic spectra only.',
        motion_policy='Signed motion_x/y * fps / contemporaneous face height; full finite windows only; no interpolation or reference-derived adjustment.',
        prohibitions=['No new prediction column/HR file is produced.','No timing or candidate parameter is selected by reference error.','Oracle selection across channels cannot be reported as achieved accuracy.'],
        original_outputs_immutable=True,source_hashes={})
    write(out/'diagnostic_protocol.json',protocol)
    # Arithmetic sanity checks against numerical signals, not a human recording.
    time=np.arange(300)/30
    pure=spectral_information(np.sin(2*np.pi*1.4*time),30.,84.)
    competing=spectral_information(2*np.sin(2*np.pi*2.4*time)+np.sin(2*np.pi*1.4*time),30.,84.)
    assert pure['top1_peak_oracle_hit'] and competing['top8_peak_oracle_hit'] and not competing['top1_peak_oracle_hit']
    source_hashes={};channel_rows=[];window_rows=[];case_summaries=[]
    for case in [f'data{i}' for i in range(1,7)]:
        folder=V25/case;s=json.loads((folder/'summary.json').read_text());fps=float(s['fps']);frames=int(s['frames'])
        names=['summary.json','frame_trace.csv','baseline_roi_waveforms.csv','roi_waveforms.csv','pos_waveform.csv',
               'chrom_waveform.csv','fusion_waveform.csv','baseline_branch_waveform.csv','tracked_branch_waveform.csv','fusion_heart_rate.csv']
        tables={n:pd.read_csv(folder/n) for n in names if n.endswith('.csv')}
        for n in names:source_hashes[str(folder/n)]=sha(folder/n)
        refpath=BASE/case/'evaluation/paired_windows.csv';ref=pd.read_csv(refpath)
        source_hashes[str(refpath)]=sha(refpath)
        alignpath=BASE/case/'evaluation/alignment.json';source_hashes[str(alignpath)]=sha(alignpath)
        trace=tables['frame_trace.csv'];hr=tables['fusion_heart_rate.csv']
        np.testing.assert_allclose(hr.time_s,ref.time_s,rtol=0,atol=1e-8)
        for n,df in tables.items():
            if n!='fusion_heart_rate.csv':
                assert len(df)==frames
                np.testing.assert_allclose(df.time_s,np.arange(frames)/fps,rtol=0,atol=1e-8)
        signals=[]
        for branch,name in [('baseline','baseline_roi_waveforms.csv'),('tracked','roi_waveforms.csv')]:
            for roi in ROIS:
                for method in ['pos','chrom']:
                    add_signal(signals,branch+'_'+roi+'_'+method,'optical_waveform','roi_'+branch,roi,tables[name][roi+'_'+method])
        for name,group in [('pos','merged'),('chrom','merged'),('fusion','fusion_actual'),('baseline_branch','fusion_baseline'),('tracked_branch','fusion_tracked')]:
            add_signal(signals,name,'optical_waveform',group,None,tables[name+'_waveform.csv'].base)
        for prefix,branch in [('', 'tracked'),('baseline_','baseline')]:
            for roi in ROIS:
                for color in 'rgb':
                    add_signal(signals,'rgb_'+branch+'_'+roi+'_'+color,'raw_color','rgb_'+branch,roi,trace[prefix+roi+'_'+color])
            for color in 'rgb':
                add_signal(signals,'rgb_'+branch+'_merged_'+color,'raw_color','rgb_'+branch+'_merged',None,trace[prefix+color])
        height=trace.face_y1.to_numpy(float)-trace.face_y0.to_numpy(float)
        for axis in ['x','y']:
            velocity=np.divide(trace['motion_'+axis].to_numpy(float)*fps,height,
                out=np.full(frames,np.nan),where=np.isfinite(height)&(height>1e-6))
            add_signal(signals,'face_velocity_'+axis,'motion','motion',None,velocity)
        w,hop=round(10*fps),round(fps);starts=np.arange(0,frames-w+1,hop)
        assert len(starts)==len(ref)
        case_window=[]
        for wi,a in enumerate(starts):
            reference=float(ref.reference_bpm.iloc[wi]);eligible=bool(ref.reference_valid.iloc[wi])
            assert eligible==bool(np.isfinite(reference) and reference>0)
            if not eligible:continue
            result_rows=[]
            for sig in signals:
                part=sig['values'][a:a+w].copy()
                if sig['kind']=='raw_color' and np.isfinite(part).all() and abs(part.mean())>1e-12:
                    part=part/part.mean()-1.
                stats=spectral_information(part,fps,reference)
                row=dict(case=case,window_index=wi,time_s=float(ref.time_s.iloc[wi]),
                    window_start_s=float(ref.window_start_s.iloc[wi]),window_end_s=float(ref.window_end_s.iloc[wi]),
                    reference_bpm=reference,channel=sig['channel'],kind=sig['kind'],group=sig['group'],physical_roi=sig['physical_roi'],**stats)
                result_rows.append(row);channel_rows.append(row)
            optical=[r for r in result_rows if r['kind']=='optical_waveform']
            rois=[r for r in optical if r['physical_roi'] in ROIS]
            fused=next(r for r in optical if r['group']=='fusion_actual')
            out_ok=bool(hr.accepted.iloc[wi]);actual=float(hr.ridge_bpm.iloc[wi]) if out_ok else np.nan
            correct=out_ok and abs(actual-reference)<=5
            saved_candidates=json.loads(hr.evidence_candidates_json.iloc[wi])
            saved_hit=any(abs(float(v)-reference)<=5 for c in saved_candidates for v in c['supported_bpm'])
            matches={r['physical_roi'] for r in rois if r['top8_peak_oracle_hit']}
            strong_matches={r['physical_roi'] for r in rois if r['top8_strong_20pct_peak_oracle_hit']}
            if correct:category='actual_output_within_5bpm'
            elif saved_hit:category='existing_final_candidate_support_present_but_not_selected'
            elif fused['top8_peak_oracle_hit']:category='fixed_fusion_peak_present_but_not_available_to_current_final_readout'
            elif len(matches)>=2:category='reference_compatible_peaks_in_at_least_two_original_ROIs'
            elif matches:category='reference_compatible_peak_in_one_original_ROI_only'
            elif any(r['top8_peak_oracle_hit'] for r in optical):category='other_aggregate_waveform_peak_only'
            elif any(r['spectral_available'] for r in optical):category='no_peak_found_by_fixed_candidate_definition'
            else:category='no_complete_finite_optical_window'
            wr=dict(case=case,window_index=wi,time_s=float(ref.time_s.iloc[wi]),reference_bpm=reference,
                actual_V25_output_accepted=out_ok,actual_V25_ridge_bpm=actual,actual_V25_abs_error_bpm=abs(actual-reference) if out_ok else np.nan,
                actual_V25_within5=correct,saved_final_supported_state_oracle_hit=saved_hit,
                actual_output_near_half_reference=out_ok and abs(actual-reference/2)<=5,
                actual_output_near_double_reference=out_ok and abs(actual-reference*2)<=5,
                fusion_top8_peak_oracle_hit=fused['top8_peak_oracle_hit'],fusion_top1_peak_oracle_hit=fused['top1_peak_oracle_hit'],
                fusion_H1_energy_fraction=fused['reference_H1_energy_fraction'],fusion_H1_snr_db=fused['reference_H1_snr_db'],
                fusion_reference_peak_rank=fused['reference_peak_rank_v25_qualified'],
                physical_ROIs_with_peak=len(matches),physical_ROIs_with_20pct_peak=len(strong_matches),
                information_partition=category)
            groups={'all_optical_waveforms':optical,'original_ROI_waveforms':rois,
                'baseline_ROI_waveforms':[r for r in rois if r['group']=='roi_baseline'],
                'tracked_ROI_waveforms':[r for r in rois if r['group']=='roi_tracked'],
                'merged_POS_CHROM':[r for r in optical if r['group']=='merged'],
                'raw_RGB_diagnostic':[r for r in result_rows if r['kind']=='raw_color'],
                'motion_diagnostic':[r for r in result_rows if r['kind']=='motion']}
            for group,items in groups.items():
                wr[group+'_spectral_available']=any(r['spectral_available'] for r in items)
                for key in ['top1_peak_oracle_hit','top3_peak_oracle_hit','top8_peak_oracle_hit','v25_supported_state_oracle_hit','top8_strong_20pct_peak_oracle_hit','top8_strong_50pct_peak_oracle_hit']:
                    wr[group+'_'+key]=any(r[key] for r in items)
            window_rows.append(wr);case_window.append(wr)
        frame=pd.DataFrame(case_window)
        summary=dict(case=case,Nplanned=len(ref),Nreference_eligible=len(frame),actual_outputs=int(frame.actual_V25_output_accepted.sum()),
            actual_within5=int(frame.actual_V25_within5.sum()),saved_final_supported_state_oracle_hits=int(frame.saved_final_supported_state_oracle_hit.sum()),
            fusion_top8_peak_oracle_hits=int(frame.fusion_top8_peak_oracle_hit.sum()),
            any_original_ROI_peak_oracle_hits=int(frame.original_ROI_waveforms_top8_peak_oracle_hit.sum()),
            two_physical_ROI_peak_oracle_hits=int((frame.physical_ROIs_with_peak>=2).sum()),
            two_physical_ROI_20pct_peak_oracle_hits=int((frame.physical_ROIs_with_20pct_peak>=2).sum()),
            any_optical_waveform_peak_oracle_hits=int(frame.all_optical_waveforms_top8_peak_oracle_hit.sum()),
            any_optical_waveform_supported_state_oracle_hits=int(frame.all_optical_waveforms_v25_supported_state_oracle_hit.sum()),
            raw_RGB_diagnostic_peak_hits=int(frame.raw_RGB_diagnostic_top8_peak_oracle_hit.sum()),
            motion_reference_peak_coincidences=int(frame.motion_diagnostic_top8_peak_oracle_hit.sum()),
            near_half_reference_actual_outputs=int(frame.actual_output_near_half_reference.sum()),
            near_double_reference_actual_outputs=int(frame.actual_output_near_double_reference.sum()),
            information_partition=frame.information_partition.value_counts().to_dict(),
            fusion_H1_energy_median=float(frame.fusion_H1_energy_fraction.median()),
            fusion_H1_snr_median_db=float(frame.fusion_H1_snr_db.median()))
        assert sum(summary['information_partition'].values())==summary['Nreference_eligible']
        assert summary['actual_within5']<=summary['saved_final_supported_state_oracle_hits']
        case_summaries.append(summary)
        print(json.dumps(clean(summary),ensure_ascii=False),flush=True)
    channels=pd.DataFrame(channel_rows);windows=pd.DataFrame(window_rows)
    channels.to_csv(out/'channel_window_diagnostics.csv',index=False)
    windows.to_csv(out/'oracle_windows_REFERENCE_CONDITIONED_NOT_PREDICTIONS.csv',index=False)
    pd.DataFrame([{k:v for k,v in s.items() if not isinstance(v,dict)} for s in case_summaries]).to_csv(out/'oracle_summary.csv',index=False)
    grouped=[]
    for (case,group),df in channels.groupby(['case','group'],sort=True):
        available=df.loc[df.spectral_available]
        grouped.append(dict(case=case,group=group,channel_windows=len(df),spectral_channel_windows=len(available),
            reference_H1_energy_median=float(available.reference_H1_energy_fraction.median()),
            reference_H1_snr_median_db=float(available.reference_H1_snr_db.median()),
            reference_power_relative_to_strongest_median=float(available.reference_relative_power.median()),
            matched_peak_relative_power_median=float(available.reference_matching_peak_relative_power.median())))
    pd.DataFrame(grouped).to_csv(out/'spectral_energy_by_group.csv',index=False)
    limits=[
        'All oracle statistics consult the reference. They are evaluation-only ceilings for fixed candidate sets, not predictions or achieved accuracy.',
        'An oracle across many correlated channels can pick coincidental noise. Requiring two physical ROIs or >=20% peak power is a sensitivity view, not proof of physiological origin.',
        'Absence under this fixed spectral definition does not prove the original video lacks physiological information; another extraction method, genuine fundamental/harmonic structure, or imperfect synchronization may alter conclusions.',
        'RGB spectra can contain illumination/expression/motion energy; a reference-frequency RGB peak is not automatically PPG.',
        'Motion-frequency coincidence is not proof of artifact: real pulse and periodic movement can coincide.',
        'Raw color/motion diagnostics require a wholly finite window and do not interpolate missing samples; missing spectra are counted separately.',
        'The main original file-mtime alignment is retained; no offset is fitted. Unknown Polar notification delay and lack of shared clock markers limit all conclusions.',
        'Six previously inspected clips, overlapping windows and repeated participants are not held-out validation. No significance or population confidence interval is inferred.',
        'The generous supported-state oracle allows a <=3 bpm neighborhood around actual peaks, with original energy support; it must not be mislabeled as exact peak-center recovery.']
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose=protocol['purpose'],protocol_sha256=sha(out/'diagnostic_protocol.json'),
        diagnostic_script_sha256=sha(__file__),source_hashes=source_hashes,cases=case_summaries,
        channel_window_rows=len(channels),reference_window_rows=len(windows),originals_changed=False,new_predictions_generated=False,
        limits=limits)
    write(out/'information_summary.json',result)
    lines=['# V25 可用信号信息诊断（参考条件 oracle，绝非算法成绩）','',
        '只检查保存波形中是否存在与参考频率相容的候选。所有 oracle 结果都借助参考定位，不能用作实际准确率，也没有生成或替换任何预测。','',
        '|片段|参考窗|实际 ±5 bpm|最终已有候选支持上限|融合波形实际峰上限|任意原 ROI 实际峰上限|至少2个物理 ROI|至少2 ROI且峰≥20%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in case_summaries:
        lines.append('|'+ '|'.join(str(s[k]) for k in ['case','Nreference_eligible','actual_within5','saved_final_supported_state_oracle_hits','fusion_top8_peak_oracle_hits','any_original_ROI_peak_oracle_hits','two_physical_ROI_peak_oracle_hits','two_physical_ROI_20pct_peak_oracle_hits'])+'|')
    lines+=['','“已有候选支持”包含峰周围±3 bpm且有实际能量的允许状态；其他“实际峰”列要求局部峰中心距参考≤5 bpm，两者不同。分母固定为原参考合格计划窗；缺失窗口未被悄悄剔除。','',
        '## 可检验的后续路线','',
        '1. 对已有最终候选却选错的窗口，优先审计候选竞争、谐波/周期和路径重获，冻结全局规则后再测试，不能把 oracle 选中的频率写入推理。',
        '2. 对原 ROI 存在候选而融合波形缺失该候选的窗口，检查跨 ROI 的相消、谱峰竞争和融合前信号质量；实验必须保留原波形作为对照。',
        '3. 对固定波形集中也找不到相容峰的窗口，单纯改最终 HR 读出无法保证达到±5 bpm，需要新的像素/照明/运动补偿或外部同步采集证据。不存在峰只是本定义下未检出，不能直接断言视频没有脉动。',
        '4. 先补共享相机/Polar 同步标记和未用于本轮选择的新录像。候选覆盖上限不是独立人群准确率。','',
        '## 限制','']+['- '+x for x in limits]
    (out/'information_diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DONE '+str(out),flush=True)


if __name__=='__main__':main()
