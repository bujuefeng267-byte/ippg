"""Read-only report of all eight fixed V27 variants after inference completes.

No inference/reference-fitting imports. Reads the canonical evaluator schema,
checks summary arithmetic against saved paired windows, retains NaN gaps and
all six cases, and writes only a new report directory plus an optional summary.
This report generator is presentation code, not pre-inference model code.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

PROJECT=Path('/home/fengbujue/项目/rppg识别')
ROOT=PROJECT/'results/data1_6_v27_20260912'
V25=PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
CASES=tuple(f'data{i}' for i in range(1,7))
SPATIAL=('early_median','early_psd_cluster','late_median','late_psd_cluster')
READOUT=('fusion_nomotion_local','fusion_motion_local','fusion_nomotion_dp','fusion_motion_dp')
VARIANTS=SPATIAL+READOUT
LABELS={'V25':'V25 保留基线','early_median':'先合 RGB＋中位数',
    'early_psd_cluster':'先合 RGB＋PSD 聚类','late_median':'保留 patch＋中位数',
    'late_psd_cluster':'保留 patch＋PSD 聚类','fusion_nomotion_local':'同波形：无运动／局部',
    'fusion_motion_local':'同波形：运动／局部','fusion_nomotion_dp':'同波形：无运动／离线 DP',
    'fusion_motion_dp':'同波形：运动／离线 DP'}
SHORT={'V25':'V25','early_median':'Early\nmedian','early_psd_cluster':'Early\nPSD',
    'late_median':'Late\nmedian','late_psd_cluster':'Late\nPSD',
    'fusion_nomotion_local':'No motion\nlocal','fusion_motion_local':'Motion\nlocal',
    'fusion_nomotion_dp':'No motion\nDP','fusion_motion_dp':'Motion\nDP'}
CONTRASTS=(
    ('spatial','保留 patch（中位数固定）','early_median','late_median'),
    ('spatial','保留 patch（PSD 聚类固定）','early_psd_cluster','late_psd_cluster'),
    ('spatial','PSD 聚类（先合 RGB 固定）','early_median','early_psd_cluster'),
    ('spatial','PSD 聚类（保留 patch 固定）','late_median','late_psd_cluster'),
    ('readout','启用运动评分（局部读出固定）','fusion_nomotion_local','fusion_motion_local'),
    ('readout','启用运动评分（离线 DP 固定）','fusion_nomotion_dp','fusion_motion_dp'),
    ('readout','离线 DP（无运动评分固定）','fusion_nomotion_local','fusion_nomotion_dp'),
    ('readout','离线 DP（运动评分固定）','fusion_motion_local','fusion_motion_dp'))
GATE_LABELS={'pooled_MAE':'合并 MAE 下降','pooled_R5_gain':'合并 R5 至少提高 5 个百分点',
    'each_MAE':'每片 MAE 上升不超过 3 bpm','each_P5':'每片 P5 下降不超过 3 个百分点',
    'each_R5':'每片 R5 下降不超过 3 个百分点','each_HR_coverage':'每片 HR 覆盖下降不超过 3 个百分点',
    'each_waveform_coverage':'每片波形覆盖下降不超过 3 个百分点','common_MAE':'共同窗口合并 MAE 下降'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def save_json(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read_json(path,hashes):
    hashes[str(path.resolve())]=sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path,hashes):
    hashes[str(path.resolve())]=sha(path)
    return pd.read_csv(path)


def bools(values):
    a=np.asarray(values)
    assert not pd.isna(a).any() and np.isin(a,[True,False,0,1]).all(),'Invalid Boolean flag'
    return a.astype(bool)


def score(pred,accepted,ref):
    pred=np.asarray(pred,float);accepted=bools(accepted);ref=np.asarray(ref,float)
    assert pred.shape==accepted.shape==ref.shape
    assert np.isnan(pred[~accepted]).all() and np.isfinite(pred[accepted]).all()
    refok=np.isfinite(ref)&(ref>0);valid=accepted&refok
    error=pred[valid]-ref[valid];n=len(error);success=int((abs(error)<=5).sum())
    return dict(Nplanned=len(pred),Nref=int(refok.sum()),Noutput=int(accepted.sum()),Nvalid=n,Nwithin5=success,
        MAE_bpm=float(np.mean(abs(error))) if n else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(error**2))) if n else np.nan,
        Bias_bpm=float(error.mean()) if n else np.nan,
        P5_valid_pct=100*success/n if n else np.nan,
        R5_all_reference_pct=100*success/refok.sum() if refok.any() else np.nan,
        HR_coverage_pct=100*accepted.mean() if len(pred) else np.nan)


def assert_metrics(actual,reported,context):
    aliases={'HR_coverage_pct':'hr_output_coverage_pct'}
    for key,value in actual.items():
        field=key if key in reported else aliases.get(key)
        if field not in reported:continue
        expected=reported[field]
        expected=np.nan if expected is None else expected
        np.testing.assert_allclose(value,expected,rtol=1e-10,atol=1e-8,equal_nan=True,err_msg=context+'/'+key)


def pair_metrics(pred_a,ok_a,pred_b,ok_b,ref):
    ref=np.asarray(ref,float);a=np.asarray(pred_a,float);b=np.asarray(pred_b,float)
    va=bools(ok_a);vb=bools(ok_b);refok=np.isfinite(ref)&(ref>0)
    common=va&vb&refok;new=~va&vb&refok;lost=va&~vb&refok
    ea=a[common]-ref[common];eb=b[common]-ref[common]
    mean=lambda x:float(np.mean(x)) if len(x) else np.nan
    return dict(common_windows=int(common.sum()),new_windows=int(new.sum()),lost_windows=int(lost.sum()),
        common_A_MAE_bpm=mean(abs(ea)),common_B_MAE_bpm=mean(abs(eb)),
        common_delta_MAE_bpm=mean(abs(eb))-mean(abs(ea)),
        common_A_RMSE_bpm=np.sqrt(mean(ea**2)),common_B_RMSE_bpm=np.sqrt(mean(eb**2)),
        common_A_within5=int((abs(ea)<=5).sum()),common_B_within5=int((abs(eb)<=5).sum()),
        common_correct_to_wrong=int(((abs(ea)<=5)&(abs(eb)>5)).sum()),
        common_wrong_to_correct=int(((abs(ea)>5)&(abs(eb)<=5)).sum()),
        new_B_MAE_bpm=mean(abs(b[new]-ref[new])),lost_A_MAE_bpm=mean(abs(a[lost]-ref[lost])),
        new_B_within5=int((abs(b[new]-ref[new])<=5).sum()),
        lost_A_within5=int((abs(a[lost]-ref[lost])<=5).sum()))


def _stack(pairs,cases,column):return np.concatenate([pairs[c][column].to_numpy() for c in cases])


def load_all(root,baseline,hashes):
    # Check completeness before reading even the first score. An incomplete
    # run is not silently shown as a subset or an empty candidate.
    needed=[root/'protocol_before_inference.json']
    for variant in VARIANTS:
        needed += [root/variant/'evaluation_summary.json']
        for case in CASES:
            needed += [root/variant/case/name for name in ('heart_rate.csv','waveform.csv','fusion_heart_rate.csv','manifest.json')]
            needed += [root/variant/case/'evaluation'/name for name in ('paired_windows.csv','alignment_sensitivity.csv')]
    needed += [baseline/case/name for case in CASES for name in ('fusion_heart_rate.csv','fusion_waveform.csv')]
    missing=[str(p) for p in needed if not p.is_file()]
    if missing:raise FileNotFoundError('All eight complete six-video evaluations are required: '+json.dumps(missing,ensure_ascii=False))
    fixed=read_json(root/'protocol_before_inference.json',hashes)
    assert fixed['variants']==list(VARIANTS) and fixed['reference_used_in_inference'] is False
    reports={};pairs={};waves={};baseline_pairs={};baseline_waves={};case_rows=[];secondary=[];sensitivity=[]
    for variant in VARIANTS:
        report=read_json(root/variant/'evaluation_summary.json',hashes);reports[variant]=report
        assert report['variant']==variant and [r['case'] for r in report['cases']]==list(CASES)
        pairs[variant]={};waves[variant]={}
        for item in report['cases']:
            case=item['case'];folder=root/variant/case
            pair=read_csv(folder/'evaluation/paired_windows.csv',hashes)
            hr=read_csv(folder/'heart_rate.csv',hashes);wave=read_csv(folder/'waveform.csv',hashes)
            fh=read_csv(folder/'fusion_heart_rate.csv',hashes);manifest=read_json(folder/'manifest.json',hashes)
            assert manifest['status']=='complete' and manifest['reference_used'] is False
            assert manifest['case']==case and manifest['variant']==variant
            np.testing.assert_array_equal(pair.window_index,np.arange(len(pair)))
            np.testing.assert_allclose(pair.time_s,hr.time_s,rtol=0,atol=1e-8)
            np.testing.assert_allclose(pair.time_s,fh.time_s,rtol=0,atol=1e-8)
            np.testing.assert_allclose(pair.estimated_bpm,hr.ridge_bpm,rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(pair.fusion_readout_bpm,fh.ridge_bpm,rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(bools(pair.accepted),bools(hr.accepted))
            np.testing.assert_array_equal(bools(pair.fusion_readout_accepted),bools(fh.accepted))
            np.testing.assert_array_equal(bools(wave.covered),np.isfinite(wave.base))
            primary=score(pair.estimated_bpm,pair.accepted,pair.reference_bpm)
            assert_metrics(primary,item,case+'/'+variant+'/primary')
            second=score(pair.fusion_readout_bpm,pair.fusion_readout_accepted,pair.reference_bpm)
            assert_metrics(second,item['fusion_readout_metrics'],case+'/'+variant+'/secondary')
            wave_coverage=100*float(np.isfinite(wave.base).mean())
            np.testing.assert_allclose(wave_coverage,item['waveform_coverage_pct'],rtol=0,atol=1e-8)
            if case not in baseline_pairs:
                old=read_csv(baseline/case/'fusion_heart_rate.csv',hashes)
                oldwave=read_csv(baseline/case/'fusion_waveform.csv',hashes)
                np.testing.assert_allclose(old.time_s,pair.time_s,atol=1e-8,rtol=0)
                np.testing.assert_allclose(oldwave.time_s,wave.time_s,atol=1e-8,rtol=0)
                bp=pair[['window_index','time_s','window_start_s','window_end_s','reference_bpm','reference_valid']].copy()
                bp['estimated_bpm']=old.ridge_bpm;bp['accepted']=bools(old.accepted)
                baseline_pairs[case]=bp;baseline_waves[case]=oldwave
                bm=score(bp.estimated_bpm,bp.accepted,bp.reference_bpm)
                case_rows.append(dict(variant='V25',case=case,**bm,
                    waveform_coverage_pct=100*float(np.isfinite(oldwave.base).mean()),
                    HR_kind='saved_fused_wave_motion_dp',primary_hr_source='V25 saved fused waveform'))
            bp=baseline_pairs[case]
            np.testing.assert_allclose(pair.V25_bpm,bp.estimated_bpm,atol=0,rtol=0,equal_nan=True)
            np.testing.assert_array_equal(bools(pair.V25_accepted),bools(bp.accepted))
            for field in ('time_s','window_start_s','window_end_s','reference_bpm'):
                np.testing.assert_allclose(pair[field],bp[field],atol=1e-8,rtol=0,equal_nan=True)
            pm=pair_metrics(pair.V25_bpm,pair.V25_accepted,pair.estimated_bpm,pair.accepted,pair.reference_bpm)
            for field in ('common_windows','new_windows','lost_windows'):
                assert pm[field]==item[field],case+'/'+variant+'/'+field
            np.testing.assert_allclose(pm['common_B_MAE_bpm'],item['common_MAE_bpm'] if item['common_MAE_bpm'] is not None else np.nan,equal_nan=True)
            case_rows.append(dict(variant=variant,case=case,**primary,waveform_coverage_pct=wave_coverage,
                HR_kind='patch_spatial_primary' if variant in SPATIAL else 'saved_fused_wave_readout',
                primary_hr_source=item['primary_hr_source'],
                mean_patch_observed_pct=item['mean_patch_observed_pct'],
                at_least_two_regions_observed_pct=item['at_least_two_regions_observed_pct'],
                face_detection_pct=item['face_detection_pct'],**pm))
            both=bools(pair.accepted)&bools(pair.fusion_readout_accepted)
            secondary.append(dict(variant=variant,case=case,**second,
                common_primary_secondary_windows=int(both.sum()),
                primary_secondary_mean_abs_difference_bpm=float(np.mean(abs(pair.estimated_bpm[both]-pair.fusion_readout_bpm[both]))) if both.any() else np.nan))
            ss=read_csv(folder/'evaluation/alignment_sensitivity.csv',hashes)
            assert sorted(ss.shift_s)==[-5,-2,-1,0,1,2,5]
            sensitivity.append(ss)
            pairs[variant][case]=pair;waves[variant][case]=wave
        if variant in READOUT:
            for case in CASES:
                assert hashes[str((root/variant/case/'waveform.csv').resolve())]==hashes[str((root/'late_psd_cluster'/case/'waveform.csv').resolve())]
    frame=pd.DataFrame(case_rows)
    frame['variant']=pd.Categorical(frame.variant,categories=['V25',*VARIANTS],ordered=True)
    frame=frame.sort_values(['variant','case']).reset_index(drop=True)
    pairs={'V25':baseline_pairs,**pairs};waves={'V25':baseline_waves,**waves}
    return fixed,reports,pairs,waves,frame,pd.DataFrame(secondary),pd.concat(sensitivity,ignore_index=True)


def pooled_tables(reports,pairs,frame):
    rows=[];secondary=[];paired=[]
    for variant in ('V25',*VARIANTS):
        pp=pairs[variant];y=_stack(pp,CASES,'estimated_bpm');ok=_stack(pp,CASES,'accepted');ref=_stack(pp,CASES,'reference_bpm')
        m=score(y,ok,ref);row=dict(variant=variant,**m,
            waveform_coverage_macro_pct=float(frame[frame.variant==variant].waveform_coverage_pct.mean()))
        if variant!='V25':
            report=reports[variant];assert_metrics(m,report['pooled'],variant+'/pooled')
            row.update(promotion_pass=report['promotion_pass'],near_all_target_met=report['near_all_within_5bpm_target_met'],
                failed_checks='; '.join(k for k,v in report['promotion_checks'].items() if not v))
            fy=_stack(pp,CASES,'fusion_readout_bpm');fok=_stack(pp,CASES,'fusion_readout_accepted')
            sm=score(fy,fok,ref);assert_metrics(sm,report['fusion_readout_pooled'],variant+'/secondary pooled')
            both=bools(ok)&bools(fok)
            secondary.append(dict(variant=variant,**sm,common_primary_secondary_windows=int(both.sum()),
                primary_secondary_mean_abs_difference_bpm=float(np.mean(abs(y[both]-fy[both]))) if both.any() else np.nan))
            for scope,cs in [('pooled',CASES)]+[(c,(c,)) for c in CASES]:
                a=_stack(pairs['V25'],cs,'estimated_bpm');aok=_stack(pairs['V25'],cs,'accepted')
                b=_stack(pp,cs,'estimated_bpm');bok=_stack(pp,cs,'accepted');r=_stack(pp,cs,'reference_bpm')
                paired.append(dict(variant=variant,scope=scope,**pair_metrics(a,aok,b,bok,r)))
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(secondary),pd.DataFrame(paired)


def factor_tables(pairs):
    result=[]
    for family,label,a,b in CONTRASTS:
        for scope,cases in [('pooled',CASES)]+[(c,(c,)) for c in CASES]:
            ya=_stack(pairs[a],cases,'estimated_bpm');oa=_stack(pairs[a],cases,'accepted')
            yb=_stack(pairs[b],cases,'estimated_bpm');ob=_stack(pairs[b],cases,'accepted')
            r=_stack(pairs[a],cases,'reference_bpm')
            ma=score(ya,oa,r);mb=score(yb,ob,r)
            result.append(dict(family=family,contrast=label,variant_A=a,variant_B=b,scope=scope,
                **{f'delta_{k}':mb[k]-ma[k] for k in ('MAE_bpm','RMSE_bpm','P5_valid_pct','R5_all_reference_pct','HR_coverage_pct')},
                **pair_metrics(ya,oa,yb,ob,r)))
    return pd.DataFrame(result)


def setup_plotting():
    for path in (Path('/mnt/c/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/msyh.ttc')):
        if path.exists():
            font_manager.fontManager.addfont(str(path));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(path)).get_name();break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'figure.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def save_figure(fig,output,stem):
    fig.savefig(output/(stem+'.png'),dpi=170)
    fig.savefig(output/(stem+'.pdf'))
    plt.close(fig)


def metric_overview(pooled,output):
    fig,axs=plt.subplots(2,2,figsize=(14.7,9))
    metrics=[('MAE_bpm','心率 MAE（bpm；越低越好）'),('P5_valid_pct','P5：有效输出内 ±5 bpm（%）'),
        ('R5_all_reference_pct','R5：全部参考窗内 ±5 bpm（%）'),('HR_coverage_pct','有效心率输出覆盖（%）')]
    colors=['#708090']+['#3276A5']*4+['#AF6C2F']*4
    for ax,(column,title) in zip(axs.flat,metrics):
        vals=pooled[column].to_numpy(float);ax.bar(np.arange(len(vals)),vals,color=colors,width=.7)
        ax.set_xticks(np.arange(len(vals)),[SHORT[v] for v in pooled.variant],fontsize=8)
        ax.set_title(title,loc='left',fontsize=12,pad=10)
        limit=100 if column!='MAE_bpm' else max(10,float(np.nanmax(vals))*1.22)
        ax.set_ylim(0,limit*1.08)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for i,y in enumerate(vals):
            if np.isfinite(y):ax.text(i,y+limit*.014,f'{y:.1f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('固定八候选全量比较：空间聚合与波形读出分别评价',x=.055,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.025,'蓝色前四项：patch 统计 HR；棕色后四项：同一条保存融合波形的 HR。每项均包含同样六段开发视频。\nP5 的分母为有效输出，R5 的分母包含拒判窗；309 个重叠窗口不等于 309 个独立样本。',fontsize=10,color='#4B5563')
    fig.subplots_adjust(left=.055,right=.985,top=.91,bottom=.14,hspace=.38,wspace=.18)
    save_figure(fig,output,'all_eight_primary_metrics')


def factorial_plot(pooled,output,family):
    if family=='spatial':
        cells=[['early_median','early_psd_cluster'],['late_median','late_psd_cluster']]
        rows=['先合 RGB','保留 patch'];cols=['中位数','完整 PSD 聚类']
        title='空间／频谱 2×2：共享前端与每 patch 候选规则'
    else:
        cells=[['fusion_nomotion_local','fusion_nomotion_dp'],['fusion_motion_local','fusion_motion_dp']]
        rows=['无运动评分','有运动评分'];cols=['局部读出','离线 DP']
        title='运动／时间 2×2：四项使用完全相同的 late-PSD 保存波形'
    indexed=pooled.set_index('variant');fig,axs=plt.subplots(2,2,figsize=(11.6,8.5))
    for ax,(metric,label) in zip(axs.flat,[('MAE_bpm','MAE（bpm）'),('P5_valid_pct','P5（%）'),('R5_all_reference_pct','R5（%）'),('HR_coverage_pct','HR 覆盖（%）')]):
        values=np.array([[indexed.loc[v,metric] for v in row] for row in cells],float)
        # Numeric labels carry the result; a single light background avoids
        # implying that separately scaled colors are comparable effect sizes.
        ax.imshow(np.ones((2,2)),cmap='Blues',vmin=0,vmax=5)
        ax.set_xticks([0,1],cols);ax.set_yticks([0,1],rows);ax.tick_params(length=0)
        ax.set_title(label,loc='left',fontsize=12,pad=12)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=22,color='#163B56')
        ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.grid(which='minor',color='white',linewidth=4);ax.tick_params(which='minor',length=0)
    fig.suptitle(title,x=.04,ha='left',fontsize=16,fontweight='bold')
    fig.text(.04,.018,'各格为自身有效窗口指标；共同窗口与新增／丢失窗口的单因子差异另列在 factor_effects.csv。\n仅解释这些固定开发片段的算法对照，不代表独立人群验证。',fontsize=10,color='#4B5563')
    fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.12,hspace=.39,wspace=.40)
    save_figure(fig,output,'factorial_'+family)


def hr_figures(pairs,frame,output):
    plot_rows=[]
    for variant in VARIANTS:
        fig,axs=plt.subplots(3,2,figsize=(15.2,11),sharey=True)
        allref=np.concatenate([pairs[variant][c].reference_bpm.to_numpy(float) for c in CASES])
        upper=max(220,float(np.nanmax(allref))+10)
        for ax,case in zip(axs.flat,CASES):
            p=pairs[variant][case];t=p.time_s.to_numpy(float)
            ref=np.where(bools(p.reference_valid),p.reference_bpm,np.nan)
            y=p.estimated_bpm.to_numpy(float);old=pairs['V25'][case].estimated_bpm.to_numpy(float)
            ax.fill_between(t,ref-5,ref+5,color='#9099A5',alpha=.17,linewidth=0)
            ax.plot(t,ref,color='#20252C',lw=1.8,zorder=5)
            ax.plot(t,old,color='#3575A2',ls='--',lw=1.3,marker='o',ms=2.4,zorder=3)
            ax.plot(t,y,color='#B96A28',lw=1.5,marker='.',ms=4,zorder=4)
            s=frame[(frame.variant==variant)&(frame.case==case)].iloc[0]
            ax.set_title(f'{case}  MAE {s.MAE_bpm:.2f} bpm；P5 {s.P5_valid_pct:.1f}%\n有效 HR {int(s.Noutput)}/{int(s.Nplanned)}；R5 {s.R5_all_reference_pct:.1f}%',loc='left',fontsize=11,pad=10)
            ax.set_xlim(0,float(p.window_end_s.iloc[-1])+1);ax.set_ylim(35,upper)
            ax.set_xlabel('视频时间（s；10 s 窗中心）');ax.set_ylabel('心率（bpm）');ax.grid(axis='y',alpha=.18)
            for i in range(len(p)):
                plot_rows.append(dict(variant=variant,case=case,window_index=int(p.window_index.iloc[i]),
                    time_s=float(t[i]),Polar_reference_bpm=float(ref[i]),V25_bpm=float(old[i]),
                    primary_bpm=float(y[i]),primary_accepted=bool(p.accepted.iloc[i]),
                    fusion_secondary_bpm=float(p.fusion_readout_bpm.iloc[i]),
                    fusion_secondary_accepted=bool(p.fusion_readout_accepted.iloc[i])))
        kind='patch 统计 HR，不等同于展示融合波形的主频' if variant in SPATIAL else '由同一保存融合波形读出的 HR'
        fig.suptitle(f'{LABELS[variant]}｜固定展示六段视频',x=.065,ha='left',fontsize=17,fontweight='bold')
        fig.legend(handles=[Line2D([0],[0],color='#20252C',lw=1.8,label='Polar 设备 HR（固定参考）'),
            Line2D([0],[0],color='#3575A2',lw=1.3,ls='--',marker='o',ms=3,label='V25'),
            Line2D([0],[0],color='#B96A28',lw=1.5,marker='.',ms=5,label=LABELS[variant])],
            loc='upper left',bbox_to_anchor=(.065,.955),ncol=3,frameon=False,fontsize=10)
        fig.text(.065,.022,f'{kind}。灰带为参考 ±5 bpm 容限，并非置信区间。\n缺失输出保持断线；原时间戳估计同步不变；同一候选应用于全部六片，没有逐片挑选最优算法。',fontsize=10,color='#4B5563')
        fig.subplots_adjust(left=.075,right=.98,top=.865,bottom=.115,hspace=.56,wspace=.16)
        save_figure(fig,output,'six_video_hr_'+variant)
    pd.DataFrame(plot_rows).to_csv(output/'all_hr_plot_data.csv',index=False)


def wave_figures(waves,output):
    for variant in SPATIAL:
        fig,axs=plt.subplots(3,2,figsize=(15.2,10))
        for ax,case in zip(axs.flat,CASES):
            wave=waves[variant][case];t=wave.time_s.to_numpy(float);y=wave.base.to_numpy(float)
            ax.plot(t,y,lw=.6,color='#2D6C98')
            coverage=100*np.isfinite(y).mean()
            ax.set_title(f'{case}  保存波形覆盖 {coverage:.1f}%',loc='left',fontsize=12,pad=9)
            ax.set_xlim(0,t[-1]);ax.set_xlabel('视频时间（s）');ax.set_ylabel('base（任意单位）');ax.grid(axis='y',alpha=.16)
        fig.suptitle(f'{LABELS[variant]}｜真实贡献 patch 的保存融合波形',x=.06,ha='left',fontsize=17,fontweight='bold')
        fig.text(.06,.035,'直接绘制 waveform.csv 的全部 base 样本；NaN 保持缺口。没有用 HR 合成正弦、移动频率或补齐缺失。\n源窗口已做归一化与极性对齐，幅度为任意单位；波形覆盖不是形态准确率，也未证明恢复重搏切迹。\n四个运动／时间读出共享“保留 patch＋PSD 聚类”这一份波形，不计为四份独立信号证据。',fontsize=10,color='#4B5563')
        fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.16,hspace=.54,wspace=.22)
        save_figure(fig,output,'six_video_measured_wave_'+variant)


def number(value,digits=2):
    if value is None or pd.isna(value):return 'NA'
    return f'{value:.{digits}f}'


def md_table(headers,rows):
    esc=lambda x:str(x).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(esc,row))+' |' for row in rows])


def markdown_report(reports,pooled,frame,secondary,paired,factors,output,root):
    passing=[v for v in VARIANTS if reports[v]['promotion_pass']]
    near=[v for v in VARIANTS if reports[v]['near_all_within_5bpm_target_met']]
    if passing:
        verdict='本轮开发集门槛通过：'+ '、'.join(LABELS[v] for v in passing)+'。这是开发集结果，不等于已经部署或独立验证通过。'
    else:verdict='本轮 8 个固定候选均未通过完整升级门槛，继续保留 V25 推荐入口。'
    target=('；'.join(LABELS[v] for v in near)+' 达到每片 P5≥95% 的开发目标。') if near else '没有候选达到六片各自 P5≥95% 的目标。'
    spatial_second=secondary[secondary.variant.isin(SPATIAL)].dropna(subset=['primary_secondary_mean_abs_difference_bpm'])
    separation=''
    if len(spatial_second):
        sep=spatial_second.loc[spatial_second.primary_secondary_mean_abs_difference_bpm.idxmax()]
        separation=(f'主／二次读出必须区分：{LABELS[sep.variant]} 的二者共同有效窗有 {int(sep.common_primary_secondary_windows)} 个，'
            f'两种 HR 的平均绝对差为 {sep.primary_secondary_mean_abs_difference_bpm:.2f} bpm。这个差值是读出之间的差异，不是真值误差；不能用主 patch 指标代替融合波形读出指标。')
    lines=['# V27：局部 patch、PSD 聚类与心率读出对照','',verdict,'',target,
        '', '以下所有候选都使用相同六段旧开发视频。前四项主 HR 来自保存 patch BVP 的空间统计聚合；后四项从同一条保存融合波形读取 HR。它们的来源不同，不能把前四项主 HR 称为展示波形的准确主频。',
        '', separation,
        '', '## 8 个候选的主指标', '',
        'MAE／RMSE 在各自有参考且有效输出的窗口计算；P5＝±5 bpm 内数量／有效对照窗口，R5＝±5 bpm 内数量／全部参考窗口。HR 覆盖＝有效输出窗口／全部计划窗口。波形覆盖下表为六片宏平均，逐片值见后表。', '']
    rows=[]
    for r in pooled.itertuples():
        gate='保留基线' if r.variant=='V25' else ('通过' if reports[r.variant]['promotion_pass'] else '未通过')
        rows.append([LABELS[r.variant],f'{r.Nwithin5}/{r.Nvalid}/{r.Nref}',number(r.MAE_bpm),number(r.RMSE_bpm),
            number(r.P5_valid_pct),number(r.R5_all_reference_pct),number(r.HR_coverage_pct),number(r.waveform_coverage_macro_pct),gate])
    lines += [md_table(['固定分支','±5 内／有效／参考窗','MAE bpm','RMSE bpm','P5 %','R5 %','HR 覆盖 %','波形覆盖宏平均 %','升级门槛'],rows),
        '', '![全部八候选指标](<all_eight_primary_metrics.png>)', '',
        '合并值按窗口数计算，不是六片 MAE 的简单平均；窗口长 10 s、步长约 1 s，彼此高度重叠。309 个计划窗不是 309 次独立采集，也不能据此声称人群泛化或总体置信区间。',
        '', '## 空间／频谱和运动／时间的单因子对照', '',
        '空间 2×2 对照：early 先在父区域内平均局部 RGB，再做 POS／CHROM；late 保留局部 patch 到 BVP 后再聚合。两者使用相同前端测量和区域等权原则。PSD 聚类使用完整归一化谱形，不只是峰频距离分组。',
        '', '![空间与频谱对照](<factorial_spatial.png>)', '',
        '运动／时间 2×2 的四支使用字节相同的 late_psd_cluster 保存波形，只切换已有运动评分与局部／离线 DP 读出。DP 可利用未来窗口，不应称为实时心率。',
        '', '![运动与时间对照](<factorial_readout.png>)', '',
        '下表均为 B−A；MAE 差为负表示误差下降，R5／覆盖差为百分点。共同窗比较和新增／丢失窗同时展示，防止拒判改变分母制造改善。', '']
    rows=[]
    for r in factors[factors.scope=='pooled'].itertuples():
        rows.append([r.contrast,number(r.delta_MAE_bpm),number(r.common_delta_MAE_bpm),r.common_windows,
            r.new_windows,r.lost_windows,number(r.delta_R5_all_reference_pct),number(r.delta_HR_coverage_pct)])
    lines += [md_table(['单一改动 A→B','自身 ΔMAE','共同窗 ΔMAE','共同窗','新增窗','丢失窗','ΔR5 pp','Δ覆盖 pp'],rows),
        '', '这些是固定开发片段上的算法对照；不同片段的效果不构成独立临床因果结论。[全部逐片单因子表](<factor_effects.csv>) 保留每个方向，没有按视频选择最优分支。',
        '', '## 与 V25 的共同窗口比较', '',
        '每个候选分别与 V25 取共同有效参考窗，因此候选之间的共同集合仍可能不同。不能只按本表 MAE 排名而忽略其窗口数和全部参考窗 R5。', '']
    rows=[]
    for r in paired[paired.scope=='pooled'].itertuples():
        rows.append([LABELS[r.variant],r.common_windows,number(r.common_A_MAE_bpm),number(r.common_B_MAE_bpm),
            number(r.common_A_RMSE_bpm),number(r.common_B_RMSE_bpm),r.new_windows,r.lost_windows,
            number(r.new_B_MAE_bpm),number(r.lost_A_MAE_bpm)])
    lines += [md_table(['候选','共同窗','V25 MAE','候选 MAE','V25 RMSE','候选 RMSE','新增','丢失','新增窗 MAE','丢失窗原 MAE'],rows),
        '', '[每片 V25 配对、新增／丢失与成功窗转换](<v25_paired_common.csv>)。',
        '', '## 主 patch 统计 HR 与融合波形二次读出', '',
        '前四项主指标可以来自区域中位数，偶数中位数可能落在两个实际谱峰之间。保存融合波形仅由真实贡献 patch 的宽带信号归一化、极性对齐和 Hann 加权而来，没有按统计 HR 造波。以下二次读出统一使用 V25 的运动评分＋离线 DP，只作一致性诊断，不替代主分数。后四项的二次读出重复同一个 motion+DP 结果，不是独立候选。', '']
    rows=[]
    primary=pooled.set_index('variant')
    for r in secondary.itertuples():
        p=primary.loc[r.variant]
        rows.append([LABELS[r.variant],number(p.MAE_bpm),number(p.P5_valid_pct),number(p.HR_coverage_pct),
            number(r.MAE_bpm),number(r.P5_valid_pct),number(r.HR_coverage_pct),
            number(r.primary_secondary_mean_abs_difference_bpm)])
    lines += [md_table(['分支','主 MAE','主 P5 %','主覆盖 %','波形读出 MAE','波形读出 P5 %','波形读出覆盖 %','主／次共同窗 HR 差'],rows),
        '', '[逐片二次读出与主／次 HR 差异](<secondary_by_video.csv>)。',
        '', '## 六片视频全部保留', '']
    for case in CASES:
        lines += ['### '+case,'']
        rows=[]
        for r in frame[frame.case==case].itertuples():
            rows.append([LABELS[r.variant],f'{r.Nwithin5}/{r.Nvalid}/{r.Nref}',number(r.MAE_bpm),number(r.RMSE_bpm),
                number(r.P5_valid_pct),number(r.R5_all_reference_pct),number(r.HR_coverage_pct),number(r.waveform_coverage_pct)])
        lines += [md_table(['分支','±5 内／有效／参考窗','MAE','RMSE','P5 %','R5 %','HR 覆盖 %','波形覆盖 %'],rows),'']
    lines += ['## 心率折线和真实保存波形', '',
        '每张心率图固定展示同一个候选在全部六片上的表现，并与 V25、Polar 设备 HR 对照。灰带是 ±5 bpm 容限而非置信区间，缺失值保持断线。', '']
    for v in VARIANTS:
        lines += [f'### {LABELS[v]}', '', f'![{LABELS[v]} 六片心率](<six_video_hr_{v}.png>)', '']
    lines += ['保存波形的纵轴为任意单位，全部原样本直接绘图，缺口不填补。各窗已归一化且做极性对齐；没有合格同步接触式 PPG 形态真值，因此不能把波形覆盖或外观称为形态恢复准确率。', '']
    for v in SPATIAL:
        lines += [f'![{LABELS[v]} 六片保存波形](<six_video_measured_wave_{v}.png>)', '']
    lines += ['## 门槛未通过的具体项目与解释边界', '']
    rows=[]
    for v in VARIANTS:
        failed=[GATE_LABELS.get(k,k) for k,ok in reports[v]['promotion_checks'].items() if not ok]
        rows.append([LABELS[v],'；'.join(failed) or '完整门槛通过'])
    lines += [md_table(['分支','未通过项目'],rows), '',
        '候选误差上升、丢失正确窗口或覆盖降低，是本轮能够直接确认的现象。若主 patch HR 与融合波形读出明显不同，说明统计聚合值和单一融合信号之间存在不一致；这本身不能证明哪一种更接近真实生理波形。', '',
        '拒判状态记录的是算法关卡，不是互斥的生理病因。多个区域共享的周期运动仍可形成一致频谱；同脸 patch、POS／CHROM 和原／跟踪分支不是独立传感器，因此不能把一致投票当作确定的真实脉搏。这里不根据本次真值重新选择 patch、阈值、时间偏移或按片推荐算法。', '',
        '固定位置可视检查还提示一项待验证风险：本轮对 data1 的 10%／50%／90% 时刻 ROI 叠图观察到，部分上脸颊块靠近眼镜。源码 sample_polygon 的像素筛选只包含 20<RGB<245、几何可见面积和逐通道尾部裁剪，没有语义皮肤或眼镜遮罩。因而存在区域污染的可能；现有观察不能证明眼镜进入了哪些实际采样像素，更不能断言它造成了本次 HR 误差。本轮参数已冻结，未据此修改取样区域或筛选阈值。', '',
        '## 参考、同步和可复算文件', '',
        '参考沿用 Polar 设备 HR 和首次冻结的原视频时间轴；不是 ECG 真值。视频开始时间由原文件时间戳与完整录像时长估计，不是硬件同步。固定 ±1／2／5 s 敏感性全部保留，主成绩始终使用 0 s 原对齐，没有选择误差最小的偏移。', '',
        '[全部逐片主指标](<primary_by_video.csv>) · [全部合并主指标](<primary_pooled.csv>) · [二次读出合并表](<secondary_pooled.csv>) · [全部时间偏移敏感性](<alignment_sensitivity.csv>) · [绘图数据](<all_hr_plot_data.csv>) · [数据来源与校核记录](<report_provenance.json>)', '',
        '本实现借鉴 [pyVHR](https://github.com/phuselab/pyVHR) 的多 patch 和频谱聚类思路，使用独立实现、不同 patch 数和窗口协议，不是作者 benchmark 的完整复现，也不能将作者 PURE 结果直接视为这些视频的预期精度。', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    summary=['# V27 结果说明','',verdict,'',target,'',
        '本次固定比较 8 个全局分支，所有分支都展示六段旧开发视频；未按片挑选最好结果。前四项主 HR 为保存 patch BVP 的空间统计聚合，后四项为同一保存宽带融合波形的读出；波形二次 HR 分栏报告。', '',
        separation, '',
        '设备参考沿用原时间戳估计同步和全部计划窗；±1／2／5 s 仅作敏感性。P5 是有效输出内合格比例，R5 将拒判窗计入参考分母。波形覆盖不是形态准确率；本轮也不是独立受试者验证。', '',
        '[完整报告](<'+str((output/'report.md').relative_to(root)) +'>)' if output.is_relative_to(root) else '[完整报告](<'+str(output/'report.md')+'>)', '']
    return '\n'.join(summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--baseline',type=Path,default=V25)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--summary-output',type=Path)
    ap.add_argument('--confirmed-complete',action='store_true',help='Set only after all six original-video predictions and all eight evaluations complete.')
    a=ap.parse_args()
    if not a.confirmed_complete:ap.error('Await explicit completion confirmation; this flag prevents reading partial result scores.')
    root=a.root.resolve();baseline=a.baseline.resolve();output=(a.output or root/'final_report').resolve()
    summary_output=(a.summary_output or root/'V27结果说明.md').resolve()
    if output.exists() or summary_output.exists():raise FileExistsError('Preserve prior report/summary; use new paths for revisions.')
    hashes={str(Path(__file__).resolve()):sha(__file__)}
    fixed,reports,pairs,waves,frame,secondary_cases,sensitivity=load_all(root,baseline,hashes)
    pooled,secondary,paired=pooled_tables(reports,pairs,frame);factors=factor_tables(pairs)
    assert len(frame)==54 and len(pooled)==9 and len(secondary_cases)==48
    assert len(paired)==56 and len(factors)==56 and len(sensitivity)==336
    assert frame[frame.variant=='V25'].Nplanned.sum()==309
    output.mkdir(parents=True)
    for name,table in [('primary_by_video',frame),('primary_pooled',pooled),('secondary_by_video',secondary_cases),
        ('secondary_pooled',secondary),('v25_paired_common',paired),('factor_effects',factors),('alignment_sensitivity',sensitivity)]:
        table.to_csv(output/(name+'.csv'),index=False)
    setup_plotting();metric_overview(pooled,output)
    factorial_plot(pooled,output,'spatial');factorial_plot(pooled,output,'readout')
    hr_figures(pairs,frame,output);wave_figures(waves,output)
    summary=markdown_report(reports,pooled,frame,secondary,paired,factors,output,root)
    # Source identity is checked before delivery; presentation never mutates
    # source data or the frozen inference/evaluation code.
    for path,digest in hashes.items():assert sha(path)==digest,'Report input changed during rendering: '+path
    summary_output.parent.mkdir(parents=True,exist_ok=True)
    summary_output.write_text(summary,encoding='utf-8')
    save_json(output/'report_provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_root=root,baseline_root=baseline,source_hashes=hashes,variants=list(VARIANTS),cases=list(CASES),
        source_read_only=True,reference_offset_selected=False,per_video_method_selection=False,
        evaluated_at_original_fixed_shift_s=0,alignment_sensitivity_shifts_s=[-5,-2,-1,0,1,2,5],
        primary_by_video_rows=len(frame),secondary_by_video_rows=len(secondary_cases),
        primary_vs_saved_secondary_explicit=True,NaN_plot_gaps_preserved=True,
        metrics_independently_reconciled_from_saved_paired_windows=True,
        visual_inspection_performed=False,
        generated_figures=[p.name for p in sorted(output.glob('*.png'))],
        interpretation='Previously seen six-video development comparison; overlapping windows are not independent samples; estimated time alignment; arbitrary-unit measured fused waveform is not validated physiological morphology.',
        fixed_protocol_sha256=hashes[str((root/'protocol_before_inference.json').resolve())]))
    print(json.dumps(clean(dict(output=output,summary=summary_output,primary_rows=len(frame),
        promotion_pass=[v for v in VARIANTS if reports[v]['promotion_pass']],
        near_all_target_met=[v for v in VARIANTS if reports[v]['near_all_within_5bpm_target_met']],
        figures=len(list(output.glob('*.png'))),visual_inspection_pending=True)),ensure_ascii=False))


if __name__=='__main__':main()
