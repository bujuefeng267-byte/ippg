"""Read-only, reproducible static comparison of fixed global V26 candidates.

Outputs are explicitly provisional by default. Never chooses a method by case,
interpolates missing HR, reads unreviewed inference labels, or edits a result.
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

PROJECT = Path('/home/fengbujue/项目/rppg识别')
ROOT = PROJECT/'results/data1_6_v26_20260911'
V25 = PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
BASE = PROJECT/'results/data1_6_20260911'
CASES = [f'data{i}' for i in range(1,7)]
LABELS = {'V25':'V25', 'log_projection_baseline':'Log（原 ROI）',
    'log_projection_guarded':'Log（带跟踪）', 'component_consensus':'Component',
    'neural_efficientphys':'EfficientPhys', 'component_harmonics':'Component+H',
    'conservative_components':'保守回退'}
DEFAULT_VARIANTS = ['log_projection_baseline','log_projection_guarded','component_consensus','neural_efficientphys']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def load_hr(path):
    df = pd.read_csv(path)
    assert df.accepted.notna().all() and df.accepted.isin([True,False,0,1]).all()
    accepted = df.accepted.to_numpy(bool)
    values = df.ridge_bpm.to_numpy(float)
    assert np.isnan(values[~accepted]).all() and np.isfinite(values[accepted]).all()
    return df.time_s.to_numpy(float),values,accepted


def link(path):
    value=str(Path(path).resolve())
    if value.startswith('/home/'):
        value='//wsl.localhost/Ubuntu'+value
    elif value.startswith('/mnt/c/'):
        value='C:/'+value[len('/mnt/c/'):]
    return '<'+value+'>'


def extra_hr_figure(output, frame, variant, hashes, status_zh):
    label=LABELS.get(variant,variant)
    stem='six_video_hr_'+variant
    fig,axs=plt.subplots(3,2,figsize=(15.2,11),sharey=True)
    plot_rows=[]
    for ax,case in zip(axs.flat,CASES):
        paths=[BASE/case/'evaluation/paired_windows.csv',V25/case/'fusion_heart_rate.csv',ROOT/variant/case/'heart_rate.csv']
        ref=pd.read_csv(paths[0]);t,old,oldok=load_hr(paths[1]);tn,new,newok=load_hr(paths[2])
        np.testing.assert_allclose(t,tn,rtol=0,atol=1e-8)
        np.testing.assert_allclose(t,ref.time_s,rtol=0,atol=1e-8)
        reference=np.where(ref.reference_valid.to_numpy(bool),ref.reference_bpm.to_numpy(float),np.nan)
        ax.fill_between(t,reference-5,reference+5,color='#929AA4',alpha=.18,linewidth=0)
        ax.plot(t,reference,color='#24272D',linewidth=1.8,zorder=5)
        ax.plot(t,old,color='#2D639C',linewidth=1.45,linestyle='--',marker='o',markersize=2.4,zorder=3)
        ax.plot(t,new,color='#D07A2C',linewidth=1.5,marker='.',markersize=3.8,zorder=4)
        a=frame[(frame.variant=='V25')&(frame.case==case)].iloc[0]
        b=frame[(frame.variant==variant)&(frame.case==case)].iloc[0]
        ax.set_title(f'{case}   MAE：{a.MAE_bpm:.1f} → {b.MAE_bpm:.1f} bpm\n有效输出：V25 {int(a.Noutput)}/{int(a.Nplanned)}；{label} {int(b.Noutput)}/{int(b.Nplanned)}',loc='left',fontsize=11,pad=10)
        ax.set_xlim(0,float(ref.window_end_s.iloc[-1])+1);ax.set_ylim(35,220)
        ax.set_yticks([40,80,120,160,200]);ax.grid(axis='y',color='#E3E6EB',linewidth=.7)
        ax.set_xlabel('视频时间（s；10 s 窗中心）',fontsize=10);ax.set_ylabel('心率（bpm）',fontsize=10)
        for i in range(len(ref)):
            plot_rows.append(dict(case=case,window_index=int(ref.window_index.iloc[i]),time_s=float(t[i]),
                Polar_reference_bpm=float(reference[i]),V25_ridge_bpm=float(old[i]),candidate_ridge_bpm=float(new[i]),
                candidate_variant=variant,V25_accepted=bool(oldok[i]),candidate_accepted=bool(newok[i])))
        for path in paths:hashes[str(path)]=sha(path)
    fig.suptitle(f'固定对照：V25／{label}／Polar 设备心率｜{status_zh}',x=.06,ha='left',fontsize=17,fontweight='bold')
    legend=[Line2D([0],[0],color='#24272D',lw=1.8,label='Polar 设备 HR（固定参考）'),
        Line2D([0],[0],color='#2D639C',lw=1.45,ls='--',marker='o',ms=3,label='V25'),
        Line2D([0],[0],color='#D07A2C',lw=1.5,marker='.',ms=5,label=label+' 候选')]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.06,.955),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.018,'灰带为参考 ±5 bpm 的误差容限，并非置信区间。缺失预测保持断线；六段使用同一候选。\n图中展示不构成升级推荐；参考采用原时间戳估计同步，所有候选均需满足逐片保护条件。',fontsize=10,color='#525B66')
    fig.subplots_adjust(left=.075,right=.98,top=.865,bottom=.115,hspace=.58,wspace=.16)
    fig.savefig(output/(stem+'.png'),dpi=180);fig.savefig(output/(stem+'.pdf'));plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(output/(stem+'_plot_data.csv'),index=False)
    return output/(stem+'.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants',nargs='+',default=DEFAULT_VARIANTS)
    ap.add_argument('--output',type=Path,default=ROOT/'report_provisional')
    ap.add_argument('--status',choices=['provisional','final'],default='provisional')
    ap.add_argument('--comparison-variant',default='component_consensus')
    ap.add_argument('--additional-comparison-variants',nargs='*',default=[])
    args = ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier report; use a new output directory.')
    assert len(set(args.variants)) == len(args.variants)
    assert args.comparison_variant in args.variants, 'HR comparison must be one explicitly included global candidate.'
    comparison_label=LABELS.get(args.comparison_variant,args.comparison_variant)
    assert all(v in args.variants and v!=args.comparison_variant for v in args.additional_comparison_variants)
    assert len(set(args.additional_comparison_variants))==len(args.additional_comparison_variants)
    output = args.output
    output.mkdir(parents=True)
    for font in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/mnt/c/Windows/Fonts/msyh.ttc']:
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'savefig.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B6BBC3',
        'text.color':'#202630','axes.labelcolor':'#202630','xtick.color':'#49515B','ytick.color':'#49515B'})
    hashes = {str(Path(__file__).resolve()):sha(__file__)}
    baseline_path = V25/'stage_evaluation.json'
    baseline = json.loads(baseline_path.read_text())
    hashes[str(baseline_path)] = sha(baseline_path)
    rows, reports, variant_order = [], {}, ['V25']+args.variants
    for r in baseline['cases']:
        rows.append({**r,'variant':'V25','promotion_pass':None,'near_all_target_met':None})
    for variant in args.variants:
        path = ROOT/variant/'evaluation_summary.json'
        report = json.loads(path.read_text())
        assert report['variant'] == variant and sorted(r['case'] for r in report['cases']) == CASES
        hashes[str(path)] = sha(path)
        reports[variant] = report
        for r in report['cases']:
            rows.append({**r,'variant':variant,'promotion_pass':report['promotion_pass'],
                'near_all_target_met':report['near_all_within_5bpm_target_met']})
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['variant','case']).any()
    frame['label'] = frame.variant.map(lambda v:LABELS.get(v,v))
    for _,r in frame.iterrows():
        np.testing.assert_allclose(r.R5_all_reference_pct,100*r.Nwithin5/r.Nref,atol=1e-9)
        np.testing.assert_allclose(r.P5_valid_pct,100*r.Nwithin5/r.Nvalid if r.Nvalid else np.nan,atol=1e-9,equal_nan=True)
    core = ['variant','label','case','Nplanned','Noutput','Nref','Nvalid','Nwithin5','MAE_bpm','RMSE_bpm',
        'P5_valid_pct','R5_all_reference_pct','hr_output_coverage_pct','waveform_coverage_pct',
        'common_windows','common_MAE_bpm','common_V25_MAE_bpm','new_windows','new_MAE_bpm',
        'lost_windows','lost_V25_MAE_bpm','promotion_pass','near_all_target_met']
    frame.reindex(columns=core).to_csv(output/'all_fixed_candidates.csv',index=False)
    summary_rows=[]
    for variant in variant_order:
        group = frame[frame.variant==variant]
        n = int(group.Nvalid.sum());nr=int(group.Nref.sum());correct=int(group.Nwithin5.sum())
        summary_rows.append(dict(variant=variant,label=LABELS.get(variant,variant),Nref=nr,Nvalid=n,Nwithin5=correct,
            MAE_bpm=float((group.MAE_bpm*group.Nvalid).sum()/n),
            RMSE_bpm=float(np.sqrt((group.RMSE_bpm**2*group.Nvalid).sum()/n)),
            P5_valid_pct=100*correct/n,R5_all_reference_pct=100*correct/nr,
            HR_coverage_pct=100*group.Noutput.sum()/group.Nplanned.sum(),
            waveform_coverage_macro_pct=float(group.waveform_coverage_pct.mean()),
            promotion_pass=reports[variant]['promotion_pass'] if variant!='V25' else None,
            near_all_target_met=bool((group.P5_valid_pct>=95).all())))
    totals = pd.DataFrame(summary_rows)
    totals.to_csv(output/'pooled_fixed_candidates.csv',index=False)

    status_zh = '临时汇总' if args.status=='provisional' else '本轮开发汇总'
    spec = [('MAE_bpm','MAE（bpm，越低越好）','YlOrBr',None),
        ('RMSE_bpm','RMSE（bpm，越低越好）','YlOrBr',None),
        ('P5_valid_pct','P5（%，分母：有参考的有效输出）','Blues',100),
        ('R5_all_reference_pct','R5（%，分母：所有参考合格窗）','Blues',100),
        ('hr_output_coverage_pct','心率输出覆盖率（%，有效输出／计划窗）','Blues',100),
        ('waveform_coverage_pct','rPPG 波形可用率（%，有限样本／总帧数）','Blues',100)]
    fig,axs=plt.subplots(3,2,figsize=(16,11.5))
    for ax,(key,title,cmap,ceiling) in zip(axs.flat,spec):
        values=frame.pivot(index='variant',columns='case',values=key).reindex(index=variant_order,columns=CASES).to_numpy(float)
        limit=ceiling if ceiling is not None else float(np.nanmax(values))*1.06
        ax.imshow(values,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=limit)
        ax.set_xticks(range(6),CASES)
        ax.set_yticks(range(len(variant_order)),[LABELS.get(v,v) for v in variant_order],fontsize=10)
        ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=12)
        ax.tick_params(axis='both',length=0,pad=7)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val=values[i,j]
                ax.text(j,i,f'{val:.1f}' if np.isfinite(val) else 'NA',ha='center',va='center',fontsize=11,
                    color='white' if np.isfinite(val) and val>limit*.61 else '#202630')
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle(f'V25 与固定全局候选：六段录像逐片比较｜{status_zh}',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.text(.04,.018,'P5 不是所有时段的准确率；R5 将缺失输出保留在分母。波形可用率不代表波形真值准确率。\n相同 10 s 窗／约 1 s 步长，沿用估计同步；六段已用于开发，重叠窗口不视为独立实验。',fontsize=10,color='#525B66')
    fig.subplots_adjust(left=.17,right=.98,top=.92,bottom=.11,hspace=.48,wspace=.55)
    fig.savefig(output/'metrics_overview.png',dpi=180)
    fig.savefig(output/'metrics_overview.pdf')
    plt.close(fig)

    plot_rows=[]
    fig,axs=plt.subplots(3,2,figsize=(15.2,11),sharey=True)
    for ax,case in zip(axs.flat,CASES):
        refpath=BASE/case/'evaluation/paired_windows.csv'
        ref=pd.read_csv(refpath)
        oldpath=V25/case/'fusion_heart_rate.csv'
        newpath=ROOT/args.comparison_variant/case/'heart_rate.csv'
        t,old,oldok=load_hr(oldpath)
        tn,new,newok=load_hr(newpath)
        np.testing.assert_allclose(t,tn,rtol=0,atol=1e-8)
        np.testing.assert_allclose(t,ref.time_s,rtol=0,atol=1e-8)
        reference=np.where(ref.reference_valid.to_numpy(bool),ref.reference_bpm.to_numpy(float),np.nan)
        # NaN remains in all three plotting arrays; do not drop or interpolate it.
        ax.fill_between(t,reference-5,reference+5,color='#929AA4',alpha=.18,linewidth=0)
        ax.plot(t,reference,color='#24272D',linewidth=1.8,zorder=5)
        ax.plot(t,old,color='#2D639C',linewidth=1.45,linestyle='--',marker='o',markersize=2.4,zorder=3)
        ax.plot(t,new,color='#D07A2C',linewidth=1.5,marker='.',markersize=3.8,zorder=4)
        a=frame[(frame.variant=='V25')&(frame.case==case)].iloc[0]
        b=frame[(frame.variant==args.comparison_variant)&(frame.case==case)].iloc[0]
        ax.set_title(f'{case}   MAE：{a.MAE_bpm:.1f} → {b.MAE_bpm:.1f} bpm\n有效输出：V25 {int(a.Noutput)}/{int(a.Nplanned)}；{comparison_label} {int(b.Noutput)}/{int(b.Nplanned)}',loc='left',fontsize=11,pad=10)
        end=float(ref.window_end_s.iloc[-1])
        ax.set_xlim(0,end+1)
        ax.set_ylim(35,220)
        ax.set_yticks([40,80,120,160,200])
        ax.grid(axis='y',color='#E3E6EB',linewidth=.7)
        ax.set_xlabel('视频时间（s；10 s 窗中心）',fontsize=10)
        ax.set_ylabel('心率（bpm）',fontsize=10)
        for i in range(len(ref)):
            plot_rows.append(dict(case=case,window_index=int(ref.window_index.iloc[i]),time_s=float(t[i]),
                Polar_reference_bpm=float(reference[i]),V25_ridge_bpm=float(old[i]),candidate_ridge_bpm=float(new[i]),
                candidate_variant=args.comparison_variant,V25_accepted=bool(oldok[i]),candidate_accepted=bool(newok[i])))
        for path in [refpath,oldpath,newpath]:hashes[str(path)]=sha(path)
    fig.suptitle(f'固定对照：V25／{comparison_label}／Polar 设备心率｜{status_zh}',x=.06,ha='left',fontsize=17,fontweight='bold')
    legend=[Line2D([0],[0],color='#24272D',lw=1.8,label='Polar 设备 HR（固定参考）'),
        Line2D([0],[0],color='#2D639C',lw=1.45,ls='--',marker='o',ms=3,label='V25'),
        Line2D([0],[0],color='#D07A2C',lw=1.5,marker='.',ms=5,label=comparison_label+' 候选')]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.06,.955),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.018,'灰带为参考 ±5 bpm 的误差容限，并非置信区间。缺失预测保持断线；没有按视频选最优方法。\n参考为 Polar 设备通知均值，采用原文件时间戳估计同步；曲线一致性不等于 ECG 验证或 PPG 形态恢复。',fontsize=10,color='#525B66')
    fig.subplots_adjust(left=.075,right=.98,top=.865,bottom=.115,hspace=.58,wspace=.16)
    fig.savefig(output/'six_video_hr_fixed_comparison.png',dpi=180)
    fig.savefig(output/'six_video_hr_fixed_comparison.pdf')
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(output/'six_video_hr_plot_data.csv',index=False)
    extra_figures={v:extra_hr_figure(output,frame,v,hashes,status_zh) for v in args.additional_comparison_variants}

    def fmt(x,digits=2):return f'{x:.{digits}f}' if pd.notna(x) else 'NA'
    all_failed=all(not report['promotion_pass'] for report in reports.values())
    decision=('本轮所有新增实验均未通过预设升级门槛，继续推荐 V25；“基本所有 HR 误差≤±5 bpm”的目标仍未达到。'
        if all_failed else '本文件汇总本轮所有指定的固定候选；通过部分改进门槛不等于已达到每片 P5≥95% 的目标。')
    lines=[f'# rPPG V26 {status_zh}：固定候选与 V25 对照','',
        ('这是临时版，最终追加候选尚未汇齐；此处不作最终升级决策。' if args.status=='provisional' else decision),'',
        '所有算法在六段视频上统一运行，未按视频挑选最优输出。用户目标按每片有效输出中至少 95% 的误差不超过 ±5 bpm 检查，同时报告全参考窗口成功率和输出覆盖率。','',
        '|固定候选|MAE bpm|RMSE bpm|±5 窗数／全部参考窗|P5 %|R5 %|HR 覆盖 %|六片 P5≥95%|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summary_rows:
        lines.append(f"|{r['label']}|{fmt(r['MAE_bpm'])}|{fmt(r['RMSE_bpm'])}|{r['Nwithin5']}/{r['Nref']}|{fmt(r['P5_valid_pct'])}|{fmt(r['R5_all_reference_pct'])}|{fmt(r['HR_coverage_pct'])}|{'达到' if r['near_all_target_met'] else '未达到'}|")
    lines+=['','合并 MAE 按有效配对窗加权，RMSE 从各窗平方误差合并；它们不代表六名独立受试者或独立重复实验。','',
        f'![六片固定候选比较]({link(output/"metrics_overview.png")})','',
        '图中名称对应同一套全局规则：Log（原 ROI）仅用原始 ROI 的对数颜色投影；Log（带跟踪）加入既有跟踪支路；Component 在物理 ROI 层面竞争频率后重建实测窄带成分；Component+H 增加固定的运动谐波证据；EfficientPhys 使用固定 PURE 预训练权重与本地 30 Hz 输入适配，未训练或择优换权重；若表中有“保守回退”，表示最后追加的统一运动风险条件回退实验。','',
        '## 逐片数值','',
        '|片段|固定候选|MAE bpm|P5 %|R5 %|HR 覆盖 %|rPPG 波形可用 %|有效／计划窗|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        for variant in variant_order:
            r=frame[(frame.case==case)&(frame.variant==variant)].iloc[0]
            lines.append(f'|{case}|{r.label}|{fmt(r.MAE_bpm)}|{fmt(r.P5_valid_pct)}|{fmt(r.R5_all_reference_pct)}|{fmt(r.hr_output_coverage_pct)}|{fmt(r.waveform_coverage_pct)}|{int(r.Noutput)}/{int(r.Nplanned)}|')
    lines+=['','## 与 V25 共同窗口的比较','',
        '共同窗口能减少“只拒绝难例就显得更准确”的影响。新增和丢失窗口仍须与覆盖率一起看。','',
        '|固定候选|共同窗数|共同窗 MAE bpm|同窗 V25 MAE bpm|升级门槛|',
        '|---|---:|---:|---:|---|']
    for variant in args.variants:
        r=reports[variant];p=r['pooled']
        lines.append(f"|{LABELS.get(variant,variant)}|{p['common_windows']}|{fmt(p['common_MAE_bpm'])}|{fmt(p['common_V25_MAE_bpm'])}|{'通过' if r['promotion_pass'] else '未通过'}|")
    lines+=['','未通过的具体升级保护条件：','',
        '|固定候选|失败条件|','|---|---|']
    gate_labels={'pooled_MAE':'合并 MAE 必须下降','pooled_R5_gain':'合并 R5 至少提升 5 个百分点',
        'each_MAE':'每片 MAE 增加不超过 3 bpm','each_P5':'每片 P5 下降不超过 3 个百分点',
        'each_R5':'每片 R5 下降不超过 3 个百分点','each_HR_coverage':'每片 HR 覆盖下降不超过 3 个百分点',
        'each_waveform_coverage':'每片波形可用率下降不超过 3 个百分点','common_MAE':'合并共同窗 MAE 必须下降'}
    for variant in args.variants:
        failed=[gate_labels.get(k,k) for k,v in reports[variant]['promotion_checks'].items() if not v]
        lines.append(f"|{LABELS.get(variant,variant)}|{'；'.join(failed) if failed else '无'}|")
    lines+=['','## 六段心率曲线','',
        f'主图展示统一固定的 V25、{comparison_label} 和 Polar 三条对照。谐波候选虽然在本轮产生最多的 ±5 bpm 窗口，但它不是升级推荐；六段全部展示，未挑选表现较好的视频。图中箭头是各自有效输出 MAE，不是共同窗口 MAE。所有心率使用原 10 s 窗和约 1 s 步长，NaN 保持断线。','',
        f'![六段固定心率对照]({link(output/"six_video_hr_fixed_comparison.png")})','']
    for variant,path in extra_figures.items():
        lines += [f'### {LABELS.get(variant,variant)}的统一六段对照','',
            '此图同样使用固定的全局候选。保守回退旨在保持既有输出范围，但仍须检查每片误差，不能用其他片段的收益抵消明显回退。','',
            f'![{LABELS.get(variant,variant)}六段心率对照]({link(path)})','']
    lines+=['## 指标与证据边界','',
        '- P5 = 误差≤5 bpm 的配对输出数／有参考的有效输出数；R5 = 同一成功数／所有参考合格计划窗。缺失输出不会被算成正确。',
        '- HR 覆盖率 = 接受且有限的心率输出数／计划窗数。波形可用率 = 有限波形样本数／视频总帧数，不能称为波形真值准确率。',
        '- 参考是 Polar 设备 HR 通知在每个半开窗口的均值，不是直接 ECG。同步沿用原 AVI 存档 UTC 修改时间减解码时长的显式估计，未用预测误差拟合偏移。所有候选评分均另报告 −5、−2、−1、0、1、2、5 s 敏感性，没有取最优时移。',
        '- Component 从已有实测复频谱系数重建窄带成分，整体包含自适应选频、归一化和极性对齐。不能据此宣称原始脉搏相位、振幅、重搏切迹或形态恢复。',
        '- 六段录像此前已被反复检查和用于开发；窗口重叠且可能有重复参与者，不能当作 309 个独立训练或验证样本，不给总体显著性或人群精度承诺。',
        '- 神经模型采用当前项目的裁剪、30 Hz 重采样及统一 HR 读出适配，这不是作者论文基准协议的完整复现，也不能据本批结果给网络架构作普遍排名。',
        '- 单纯降低 MAE 的部分进展，与基本所有输出进入 ±5 bpm、以及满足升级保护条件是不同结论。','',
        '## 文件与来源','',
        f'- [所有逐片指标 CSV]({link(output/"all_fixed_candidates.csv")})',
        f'- [合并指标 CSV]({link(output/"pooled_fixed_candidates.csv")})',
        f'- [心率绘图原数据 CSV]({link(output/"six_video_hr_plot_data.csv")})',
        f'- [输入哈希与构建记录]({link(output/"report_manifest.json")})',
        '- 链接使用本机 Ubuntu 的 Windows 可访问绝对路径；如需在其他电脑分享，应复制整个报告目录并另行转换链接。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status=args.status,variants=variant_order,
        fixed_HR_comparators=['V25',args.comparison_variant,'Polar'],case_specific_best_selection=False,
        additional_fixed_HR_comparators=args.additional_comparison_variants,
        missing_HR_interpolated=False,source_hashes=hashes,
        axis_policy='Common 35–220 bpm y axis for six HR panels; per-video actual relative time.',
        display_link_policy='Windows-accessible //wsl.localhost/Ubuntu absolute links, angle-bracket wrapped; metadata retains Linux paths.',
        report_content_checked_against_source=True,rendered_visual_review='Pending human/agent inspection of generated PNGs',
        source_hashes_recorded_post_run=True)
    (output/'report_manifest.json').write_text(json.dumps(clean(manifest),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),variants=variant_order,case_rows=len(frame),plot_rows=len(plot_rows),status=args.status)),flush=True)


if __name__=='__main__':main()
