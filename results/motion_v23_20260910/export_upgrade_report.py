"""Write a concise Chinese report from the frozen evaluated numbers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
LABELS={'user0904':'0904 视频','user0907':'0907 运动视频','ubfc':'UBFC（有参考）',
        'kaggle_full':'Kaggle 完整运动视频','synthetic72':'合成 72 bpm','data1':'data1（估计同步参考）'}

def fmt(v,decimals=2):return 'NA' if pd.isna(v) else f'{v:.{decimals}f}'
def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                       *['| '+' | '.join(map(str,r))+' |' for r in rows]])

def main():
    ev=HERE/'evaluation'
    m=pd.read_csv(ev/'metrics.csv');w=pd.read_csv(ev/'waveform_metrics.csv')
    p=pd.read_csv(ev/'paired_metrics.csv');src=pd.read_csv(ev/'pixel_source_metrics.csv')
    d=pd.read_csv(ev/'waveform_deltas.csv');s=pd.read_csv(ev/'sensitivity.csv')
    continuity=pd.read_csv(ev/'continuity.csv')
    summary=json.loads((ev/'summary.json').read_text())
    gate=summary['upgrade_gate']
    default_ok=gate['default']['eligible_for_named_upgrade']
    b,n='trimmed_gap10','tracking_screened_gap10'
    def metric(case,version,est='offline_ridge'):
        return m[m.case.eq(case)&m.version.eq(version)&m.variant.eq('fusion')&m.estimator.eq(est)].iloc[0]
    def wave(case,version):
        return w[w.case.eq(case)&w.version.eq(version)&w.variant.eq('fusion')&w.scope.eq('own')].iloc[0]
    main_rows=[]
    for case,label in LABELS.items():
        a,z=metric(case,b),metric(case,n);aw,zw=wave(case,b),wave(case,n)
        main_rows.append([label,f'{fmt(a.MAE_bpm)} → {fmt(z.MAE_bpm)}',
            f'{fmt(a.RMSE_bpm)} → {fmt(z.RMSE_bpm)}',
            f'{int(a.Noutput)}/{int(a.Nplanned)} → {int(z.Noutput)}/{int(z.Nplanned)}',
            f'{fmt(a.C_out_pct)} → {fmt(z.C_out_pct)}',
            f'{fmt(aw.finite_sample_pct)} → {fmt(zw.finite_sample_pct)}',
            f'{fmt(a.R5_all_reference_pct)} → {fmt(z.R5_all_reference_pct)}'])
    title='通过默认升级门槛' if default_ok else '未通过默认升级门槛；保留独立实验版本'
    parts=['# 皮肤像素跟踪与运动筛选：接入及实测结果',
        f'**结论：{title}。**',
        '主要比较是跑前固定的 V2.2 gap0.10 与 V2.3 跟踪＋筛选 gap0.10；'
        '未根据结果逐片换方法、降低质量门槛或重新选择参考时间偏移。下表均为实际融合波形读出的离线 DP 心率。',
        md_table(['视频','MAE ↓（bpm）','RMSE ↓（bpm）','HR 输出窗数','HR 覆盖率 ↑（%）',
                  '波形覆盖率 ↑（%）','R5 ↑（%）'],main_rows),
        'NA 表示没有可用参考，不能判断准确率。R5 是全部可评分参考窗中误差≤5 bpm的比例，'
        '同时反映误差和拒绝输出。波形覆盖率按实际保存的有限样本计数，包含已标记插值及邻窗贡献，不是波形形态准确率。',
        '**同一批窗口是否变准**',]
    common=[]
    for case in ['ubfc','data1']:
        for est in ['offline_ridge','local_peak']:
            q=p[p.case.eq(case)&p.before.eq(b)&p.after.eq(n)&p.variant.eq('fusion')&p.estimator.eq(est)]
            a=q[q.scope.eq('common_before')].iloc[0];z=q[q.scope.eq('common_after')].iloc[0]
            added=q[q.scope.eq('added_after')].iloc[0] if 'added_after' in set(q.scope) else None
            common.append([LABELS[case],est,int(a.Nvalid),f'{fmt(a.MAE_bpm)} → {fmt(z.MAE_bpm)}',
                           f'{fmt(a.RMSE_bpm)} → {fmt(z.RMSE_bpm)}'])
    parts += [md_table(['视频','估计器','共同可评分窗','共同 MAE（bpm）','共同 RMSE（bpm）'],common),
        'local_peak 为当前窗口的频谱峰，offline_ridge 为使用未来窗口的离线 DP。两者都完整报告，默认升级门槛同时检查。',
        '**两种像素方法与两个缺帧设置全部保留**']
    allrows=[]
    for case,label in LABELS.items():
        for v in summary['versions']:
            x=metric(case,v);y=wave(case,v)
            allrows.append([label,v,fmt(x.MAE_bpm),fmt(x.RMSE_bpm),fmt(x.C_out_pct),fmt(y.finite_sample_pct)])
    parts.append(md_table(['视频','配置','MAE（bpm）','RMSE（bpm）','HR覆盖率（%）','波形覆盖率（%）'],allrows))
    track_rows=[]
    for case,label in LABELS.items():
        q=src[src.case.eq(case)&src.version.eq(n)&src.source.eq('tracked_ratio')]
        v=q.set_index('roi').all_video_frames_pct.reindex(['forehead','left_cheek','right_cheek'],fill_value=0)
        track_rows.append([label,*[fmt(x) for x in v]])
    snr_rows=[]
    for case in ['ubfc','data1']:
        a,z=wave(case,b),wave(case,n)
        diff=d[d.case.eq(case)&d.variant.eq('fusion')&d.before.eq(b)&d.after.eq(n)].iloc[0]
        snr_rows.append([LABELS[case],fmt(a.snr_mean_db),fmt(z.snr_mean_db),
                         int(diff.Npaired_snr),fmt(diff.mean_paired_delta_snr_db)])
    outage_rows=[]
    for case,label in LABELS.items():
        q=continuity[continuity.case.eq(case)&continuity.variant.eq('fusion')]
        a=q[q.version.eq(b)].iloc[0];z=q[q.version.eq(n)].iloc[0]
        outage_rows.append([label,f'{fmt(a.longest_rejected_grid_run_s)} → {fmt(z.longest_rejected_grid_run_s)}',
            f'{fmt(wave(case,b).longest_missing_run_s)} → {fmt(wave(case,n).longest_missing_run_s)}'])
    parts += ['**像素跟踪使用情况**',
        '逐 ROI 的真实跟踪成功按 pixel_source=tracked_ratio 统计；fallback/reset/missing 单独计数。'
        '完整比例见 evaluation/pixel_source_metrics.csv。检测可用率保持一致不代表所有帧的像素都跟踪成功。',
        md_table(['视频','额头跟踪占比（%）','左侧颊区（%）','右侧颊区（%）'],track_rows),
        '**连续性：最长中断**',
        md_table(['视频','最长拒绝 HR 窗格持续时间（s）','最长波形缺失时间（s）'],outage_rows),
        'HR 一列按连续拒绝的决策窗数量×实际步长计算，波形一列按连续缺失帧/FPS计算；均包含首尾缺失。',
        '**波形质量与参考边界**',
        md_table(['视频','原版自身有效窗 SNR（dB）','新版自身有效窗 SNR（dB）','共同评分窗','共同窗 SNR 变化（dB）'],snr_rows),
        '参考 H1 SNR 采用与 V2.2 相同的参考心率频带；共同窗变化见 evaluation/waveform_deltas.csv。'
        '没有合格同步的波形形态真值，r_wave 继续为 NA。data1 仍按原协议估计同步；'
        '所有 −5、−2、−1、0、1、2、5 秒敏感性均见 evaluation/sensitivity.csv，未挑最好偏移。',
        '**如何理解本次结果**',
        'data1 的局部参考 H1 SNR 改善，但 HR 误差同时增加；新版 DP 心率在主对齐的全部38个有效窗都偏低。'
        '共同36窗的误差也变大，因此不能将退步仅归因于新增2个困难窗口。'
        '七个相同偏移对照中两种像素方法的 data1 MAE 均比原版更高。'
        '这说明本次小块变化重建/筛选还不足以可靠分离脉动。积分漂移、空间采样变化和共同干扰是需进一步消融的可能因素，尚未证明单一根因。'
        '不能用上一轮其他模型的高频运动锁定现象解释本次偏低结果。',
        '**验证与安装**',
        f'{summary["QA"]["tests"]["run"]} 个合成与回归测试通过，六个来源共 24 次候选运行完成。'
        '保存波形的独立 HR 回算、旧基线复算、冻结几何核对及指标独立算术复核均单独保留证据。',
        '实验入口：`./run_motion_v23_experimental.sh 视频路径 --output 新结果目录`。'
        '是否安装完成以 deployment.json 与 deployment_verification.json 为准。'
        '旧版本入口保留；默认方案仅在性能和安装验证均通过后切换。',
        '**实验局限**',
        '五段真人录像加一个合成控制全部是已观察过的开发回归资料，不能当作独立测试集或六名独立受试者。'
        '本实现使用相邻帧配对皮肤小块，避免整段轨迹必须全程存活；输出颜色是相对变化重建。'
        '共同周期运动仍可能被保留，积分漂移、插值及回退采样也需要考虑。',
        '详细门槛与全部失败项：evaluation/upgrade_gate.json；新增/丢失窗：evaluation/paired_metrics.csv；'
        '连续中断：evaluation/continuity.csv；代码核对：independent_qa.md。']
    (HERE/'像素跟踪接入与实测报告.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(title)
    print(md_table(['视频','MAE','RMSE','输出窗','HR覆盖%','波形覆盖%','R5%'],main_rows))

if __name__=='__main__':main()
