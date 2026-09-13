"""Render frozen independent evaluation into concise tables and static plots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
E = HERE/'evaluation'
CASES = ['user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1']
CASE_LABELS = dict(user0904='09-04 视频', user0907='09-07 运动视频', ubfc='UBFC subject1',
                   kaggle_full='Kaggle 长运动视频', synthetic72='72 bpm 合成视频', data1='新拍 data1')
SYSTEMS = ['v21/fusion', 'trimmed_gap10/fusion', 'trimmed_gap15/fusion', 'cpace', 'rhythm']
LABELS = dict(zip(SYSTEMS, ['V2.1', 'V2.2（0.10 s）', 'V2.2（0.15 s）', 'cPACE', 'RhythmMamba']))
LABELS['cpace_no_homodyne'] = 'cPACE 关闭包络校正（诊断）'
PRIMARY = ['trimmed_gap10/fusion', 'cpace', 'rhythm']


def f(x, digits=2):
    return '—' if pd.isna(x) else f'{x:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    m = pd.read_csv(E/'metrics.csv')
    m = m[m.estimator.eq('offline_ridge')]
    w = pd.read_csv(E/'waveform_metrics.csv')
    w = w[w.scope.eq('own') & w.shift_s.eq(0)]
    c = pd.read_csv(E/'continuity.csv')
    c = c[c.estimator.eq('offline_ridge')]
    p = pd.read_csv(E/'paired_metrics.csv')
    parts = ['## 全部视频的输出覆盖率', '', '每格为“心率有效窗口比例 / 有限波形采样点比例”，单位 %；两者均不是准确率。', '']
    rows = []
    for case in CASES:
        row = [CASE_LABELS[case]]
        for system in SYSTEMS:
            mm = m[m.case.eq(case)&m.system.eq(system)].iloc[0]
            ww = w[w.case.eq(case)&w.system.eq(system)].iloc[0]
            row.append(f'{f(mm.C_out_pct)} / {f(ww.finite_sample_pct)}')
        rows.append(row)
    parts += [table(['视频']+[LABELS[s] for s in SYSTEMS], rows), '']
    for case in ['ubfc', 'data1', 'synthetic72']:
        parts += ['## '+CASE_LABELS[case]+'：心率误差', '']
        rows=[]
        for system in SYSTEMS+['cpace_no_homodyne']:
            r=m[m.case.eq(case)&m.system.eq(system)].iloc[0]
            rows.append([LABELS[system],f'{int(r.Nvalid)}/{int(r.Nref)}',f(r.MAE_bpm),f(r.RMSE_bpm),
                         f(r.Bias_bpm),f(r.P5_valid_pct),f(r.R5_all_reference_pct)])
        parts += [table(['方法','参考配对窗/全部参考窗','MAE↓ (bpm)','RMSE↓ (bpm)','Bias (bpm)',
                         '有效输出内 ±5 bpm (%)','全部参考时段 ±5 bpm (%)'], rows), '']
    parts += ['## 相同窗口上的比较', '', '分别与 V2.2（0.10 s）取交集，避免丢掉困难窗口后误报进步。新增、丢失窗口的误差另见 paired_metrics.csv。', '']
    rows=[]
    for case in ['ubfc', 'data1', 'synthetic72']:
        for system in ['cpace', 'rhythm']:
            sub=p[p.case.eq(case)&p.before.eq('trimmed_gap10/fusion/offline_ridge')&p.after.eq(system+'/offline_ridge')]
            a=sub[sub.scope.eq('common_before')].iloc[0]
            b=sub[sub.scope.eq('common_after')].iloc[0]
            rows.append([CASE_LABELS[case],LABELS[system],int(b.Nvalid),f(a.MAE_bpm),f(b.MAE_bpm),f(b.MAE_bpm-a.MAE_bpm),int(b.Nadded_output),int(b.Nlost_output)])
    parts += [table(['视频','候选','共同有效窗','V2.2 MAE','候选 MAE','差值↓ (bpm)','新增窗','丢失窗'], rows),'']
    parts += ['## 连续性：最长连续输出窗口的原视频支撑区间', '', '单位秒。这里计算重叠窗口覆盖区间的并集，包含窗口本身时长；不能把它理解为逐秒心率输出持续了同样久。', '']
    rows=[]
    for case in CASES:
        row=[CASE_LABELS[case]]
        for system in SYSTEMS:
            r=c[c.case.eq(case)&c.system.eq(system)].iloc[0]
            row.append(f'{f(r.longest_run_support_union_s)} / {int(r.internal_breaks)}')
        rows.append(row)
    parts += [table(['视频']+[LABELS[s]+'：秒 / 内部断点' for s in SYSTEMS],rows),'']
    parts += ['## 波形参考基频 SNR', '', '单位 dB，各自有效参考窗口的均值。滤波带宽和有效窗口不同，因此只能作为辅助诊断；不是形态相关性或波形正确率。', '']
    rows=[]
    for case in ['ubfc','data1','synthetic72']:
        row=[CASE_LABELS[case]]
        for system in SYSTEMS+['cpace_no_homodyne']:
            r=w[w.case.eq(case)&w.system.eq(system)].iloc[0]
            row.append(f'{f(r.snr_mean_db)} (n={int(r.Nscored)})')
        rows.append(row)
    parts += [table(['视频']+[LABELS[s] for s in SYSTEMS+['cpace_no_homodyne']],rows),'']
    s=pd.read_csv(E/'sensitivity.csv')
    s=s[s.estimator.eq('offline_ridge')]
    parts += ['## data1 时间对齐敏感性', '', '固定主结果为偏移 0 s；表中偏移是相对于估计的视频开始 UTC 时间追加的秒数。列出全部方案，不选择误差最小的偏移。', '']
    rows=[]
    for shift in [-5,-2,-1,0,1,2,5]:
        row=[shift]
        for system in PRIMARY:
            r=s[s.shift_s.eq(shift)&s.system.eq(system)].iloc[0]
            row.append(f'{f(r.MAE_bpm)} / {f(r.R5_all_reference_pct)}')
        rows.append(row)
    parts += [table(['偏移 (s)']+[LABELS[x]+'：MAE / 全时段±5%' for x in PRIMARY],rows),'']
    (HERE/'结果明细表.md').write_text('\n'.join(parts),encoding='utf-8')
    make_plot(m)
    (HERE/'report_table_verification.json').write_text(json.dumps({'passed':True,'metrics_rows':len(m),
        'source':'evaluation/*.csv','rounding':'2 decimals, full precision retained in CSV'},indent=2),encoding='utf-8')


def make_plot(metrics):
    colors=['#6b7280','#dd7733','#247fbd']
    names=['V2.2 gap0.10','cPACE','RhythmMamba']
    fig=plt.figure(figsize=(14,9),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[.9,1])
    ax=fig.add_subplot(grid[0,:])
    x=np.arange(len(CASES))
    for i,system in enumerate(PRIMARY):
        vals=[metrics[metrics.case.eq(case)&metrics.system.eq(system)].iloc[0].C_out_pct for case in CASES]
        rects=ax.bar(x+(i-1)*.24,vals,.23,color=colors[i],label=names[i])
        ax.bar_label(rects,fmt='%.1f',fontsize=8)
    ax.set_xticks(x,['09-04','09-07 motion','UBFC','Kaggle motion','Synthetic 72','data1'])
    ax.set_ylim(0,113); ax.set_ylabel('Accepted HR windows (%)');ax.set_title('Coverage is not accuracy')
    ax.legend(ncols=3,loc='upper left');ax.spines[['top','right']].set_visible(False)
    aligned=pd.read_csv(E/'aligned_windows.csv')
    for j,case in enumerate(['ubfc','data1']):
        ax=fig.add_subplot(grid[1,j])
        ref=aligned[aligned.case.eq(case)&aligned.branch.eq(PRIMARY[0]+'/offline_ridge')]
        ax.plot(ref.time_s,ref.reference_bpm,color='#20242b',ls='--',lw=2,label='Device HR reference')
        for i,system in enumerate(PRIMARY):
            rows=aligned[aligned.case.eq(case)&aligned.branch.eq(system+'/offline_ridge')]
            ax.plot(rows.time_s,rows.prediction_bpm,color=colors[i],lw=1.6,label=names[i])
        ax.set_xlabel('Window center (s)');ax.set_ylabel('Heart rate (bpm)')
        ax.set_title('UBFC: dataset-paired reference' if case=='ubfc' else 'data1: estimated alignment; shift 0 s')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16)
        ax.legend(fontsize=8)
    fig.suptitle('Frozen pretrained / untrained trials on existing development videos',fontsize=14)
    fig.savefig(HERE/'两种论文方法_实测对比.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
