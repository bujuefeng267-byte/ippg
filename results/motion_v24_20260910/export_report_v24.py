"""Render V24 results without selecting a post-hoc winning configuration."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
LABELS={'user0904':'0904 视频','user0907':'0907 运动视频','ubfc':'UBFC（有参考）',
        'kaggle_full':'Kaggle 完整视频','synthetic72':'合成72 bpm','data1':'data1（估计同步）'}
def f(v):return 'NA' if pd.isna(v) else f'{v:.2f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
    *['| '+' | '.join(map(str,row))+' |' for row in rows]])
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    ev=HERE/'evaluation'
    summary=read(ev/'summary.json');gate=read(ev/'upgrade_gate.json')
    m=pd.read_csv(ev/'metrics.csv');w=pd.read_csv(ev/'waveform_metrics.csv')
    p=pd.read_csv(ev/'paired_metrics.csv');macro=pd.read_csv(ev/'macro_metrics.csv')
    tests=read(HERE/'validation/tests.json');replay=read(HERE/'validation/waveform_replay_qa.json')
    qa=read(HERE/'independent_qa.json')
    assert qa['passed'] and replay['passed'] and tests['passed']
    def metric(c,v,e='offline_ridge'):
        return m[m.case.eq(c)&m.version.eq(v)&m.variant.eq('fusion')&m.estimator.eq(e)].iloc[0]
    def waveform(c,v):return w[w.case.eq(c)&w.version.eq(v)&w.variant.eq('fusion')&w.scope.eq('own')].iloc[0]
    def result_table(b,n):
        rows=[]
        for c,label in LABELS.items():
            a,z=metric(c,b),metric(c,n);aw,zw=waveform(c,b),waveform(c,n)
            rows.append([label,f'{f(a.MAE_bpm)} → {f(z.MAE_bpm)}',f'{f(a.RMSE_bpm)} → {f(z.RMSE_bpm)}',
                f'{int(a.Noutput)}/{int(a.Nplanned)} → {int(z.Noutput)}/{int(z.Nplanned)}',
                f'{f(a.C_out_pct)} → {f(z.C_out_pct)}',f'{f(aw.finite_sample_pct)} → {f(zw.finite_sample_pct)}',
                f'{f(a.R5_all_reference_pct)} → {f(z.R5_all_reference_pct)}'])
        return table(['视频','MAE↓ bpm','RMSE↓ bpm','HR有效窗','HR覆盖↑ %','波形覆盖↑ %','R5↑ %'],rows)
    default=gate['default']['eligible_for_named_upgrade']
    continuity=gate['separate_continuity_option']['eligible_for_named_upgrade']
    parts=['# V2.4：论文依据、代码修改与完整视频实测',
        '**结论：'+('预先固定的主要性能门槛通过。' if default else '尚未通过预先固定的主要性能门槛，不能宣称精度和覆盖率已全面提高。')+'**',
        '核查与开发日期：2026-09-10。本轮保留像素跟踪、局部异常筛选、跟踪失败回退；主要候选在运行前固定为guarded_fusion_gap10，原V2.2同gap为基线。没有按视频、真值或时间偏移选择算法。',
        '## 代码改了什么',
        '1. **四阶低频校正**：对逐帧配对颜色增量的累积误差进行0.15Hz因果互补校正，每个ROI在缺口后独立重置。\n'
        '2. **独立保留原版分支**：新旧分支分别通过同样的多ROI/POS/CHROM门槛；实际跟踪比例不足、证据不够或主频相差超过12 bpm时保留合格原版。\n'
        '3. **融合实际波形再读心率**：所有切换原因、贡献区域、观察与插值状态留档，不用延续心率数字填空。',
        '这些是结合论文原则和合成反例的自有实现，并非复现某篇深度网络。四阶校正减少一阶方案的高频运动泄漏，但仍存在启动延迟和共同干扰；低误差及高覆盖必须由实片验证。',
        '## 同配置比较：缺帧上限0.10秒',result_table('trimmed_gap10','guarded_fusion_gap10'),
        'MAE/RMSE越低越好。HR覆盖率=有效输出窗口数/完整计划窗口数；波形覆盖率=实际保存有限样本/完整视频帧数，包含标明的短插值和邻窗贡献。R5以全部可评分参考窗口为分母统计误差≤5 bpm的比例。NA表示缺少参考，不等于0误差或准确。',
        '## 与此前连续性设置比较：缺帧上限0.15秒',result_table('trimmed_gap15','guarded_fusion_gap15'),
        '独立连续性选项门槛：'+('通过。' if continuity else '未通过。')+'两种gap各与相同gap的V2.2比较，未混用旧版0.10与新版0.15。',
        '## 相同窗口上的心率误差']
    rows=[]
    for c in ['ubfc','data1']:
        for e in ['offline_ridge','local_peak']:
            q=p[p.case.eq(c)&p.before.eq('trimmed_gap10')&p.after.eq('guarded_fusion_gap10')&p.variant.eq('fusion')&p.estimator.eq(e)]
            a=q[q.scope.eq('common_before')].iloc[0];z=q[q.scope.eq('common_after')].iloc[0]
            rows.append([LABELS[c],e,int(a.Nvalid),f'{f(a.MAE_bpm)} → {f(z.MAE_bpm)}',f'{f(a.RMSE_bpm)} → {f(z.RMSE_bpm)}'])
    parts += [table(['视频','读出方式','共同可评分窗','MAE bpm','RMSE bpm'],rows),
        '离线DP使用未来窗口，local_peak是当前窗口频谱峰。两种读出都报告，不能把离线结果称实时准确率。',
        '## 四个预定义候选全部结果']
    data_pairs=p[p.case.eq('data1')&p.before.eq('trimmed_gap10')&p.after.eq('guarded_fusion_gap10')&p.variant.eq('fusion')]
    add_dp=data_pairs[data_pairs.scope.eq('added_after')&data_pairs.estimator.eq('offline_ridge')].iloc[0]
    add_local=data_pairs[data_pairs.scope.eq('added_after')&data_pairs.estimator.eq('local_peak')].iloc[0]
    common_old=data_pairs[data_pairs.scope.eq('common_before')&data_pairs.estimator.eq('offline_ridge')].iloc[0]
    common_new=data_pairs[data_pairs.scope.eq('common_after')&data_pairs.estimator.eq('offline_ridge')].iloc[0]
    explanation=(f'data1新增{int(add_dp.Nvalid)}个可评分窗口；新增窗DP平均绝对误差为{f(add_dp.MAE_bpm)} bpm，'
        f'局部峰误差为{f(add_local.MAE_bpm)} bpm。共同{int(common_old.Nvalid)}窗DP误差从'
        f'{f(common_old.MAE_bpm)}变为{f(common_new.MAE_bpm)} bpm。因此自身有效窗平均误差下降，不能解释为原窗口识别变准。')
    parts.insert(3,explanation)
    rows=[]
    for c,label in LABELS.items():
        for v in summary['versions']:
            a=metric(c,v);z=waveform(c,v)
            rows.append([label,v,f(a.MAE_bpm),f(a.RMSE_bpm),f(a.C_out_pct),f(z.finite_sample_pct)])
    parts.append(table(['视频','配置','MAE bpm','RMSE bpm','HR覆盖 %','波形覆盖 %'],rows))
    failures=[]
    for key,item in [('主要候选',gate['default']),('独立连续性选项',gate['separate_continuity_option'])]:
        for r in item['failed_checks']:
            failures.append([key,r['check'],f(r.get('before')),f(r.get('after'))])
    parts += ['## 未满足的升级条件',table(['候选','条件','原版','新版'],failures) if failures else '以上两项主门槛均通过。',
        '主门槛要求两段真人参考的等权MAE与源均衡RMSE下降、各参考例不退；五段真人平均HR与波形覆盖均上升、每片不退。共同窗、R5和local读出同时检查。最长空缺及H1参考频带SNR另见evaluation/continuity.csv、waveform_metrics.csv和V2.3兼容门槛。',
        '## 论文和开源项目',
        '- [2026光流引导区域选择](https://www.sciencedirect.com/science/article/pii/S1746809425018117)：支持对齐与稳定区域选择，但其网络成绩不能直接套用本程序。\n'
        '- [CVPR 2026 rPPG-VQA](https://arxiv.org/abs/2604.11156)：借鉴多方法质量共识，也保留共同运动/闪烁会误导共识的边界。\n'
        '- [ME-rPPG作者项目](https://github.com/KegangWangCCNU/ME-rPPG)、[作者关联open-rppg](https://github.com/KegangWangCCNU/open-rppg)：作为后续固定模型接口的调研对象，本轮没有将其权重冒充已安装模型。',
        '完整论文日期、发布内容、训练集重叠风险和许可核查见research_v24.md、sources.json。没有取得新仓库提交SHA的条目已明确标注，未宣称完成源码复现。',
        '## 额外尝试：对光流灰度图作局部照明归一化',
        '首轮评分冻结后，又完成48组预定义合成对照，检验光流是否会把脉动亮度当作位移。'
        '结果没有支持部署此方案：强纹理下脉动幅度无一致改善；弱纹理下出现跟踪率接近100%但假位移增加、脉动明显衰减。'
        '因此没有修改被冻结的V2.4跟踪代码，也没有将其推进完整视频实验或安装。全部组合保留在audit/photometric_flow_diagnosis.json。'
        '这只验证该候选的失败范围，不能认定它就是data1的实际退步原因。',
        '[PulseCam论文](https://www.nature.com/articles/s41598-020-61576-0)提供亮度假设问题的补充依据；'
        '该文为2020年研究，完整灌注估计还使用外部血氧仪波形，本项目没有复制该参考辅助估计或引用其误差作为本项目成绩。',
        '## 结果可信度与使用范围',
        f'{tests["run"]}项测试通过、24次候选推理完成、72份保存波形的心率独立回算通过。主要指标、共同窗和宏平均由另一脚本独立核算；路由和采样来源另有审计记录。',
        '所有五段真人视频加一个合成控制均为已观察过的开发回归资料。data1沿用估计同步，所有−5、−2、−1、0、1、2、5秒偏移完整保留，没有挑最好对齐。UBFC参考为设备心率；没有合格同步波形真值，形态相关性r_wave仍为NA。',
        '实验入口和安装后原视频验证见installation_usage.md及deployment_verification.json；性能门槛和安装一致性是不同检查。旧入口是否改变以部署记录为准。']
    target=HERE/'V24_论文依据与实测报告.md'
    target.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(target),default_gate_passed=default,continuity_gate_passed=continuity),ensure_ascii=False))
    print(result_table('trimmed_gap10','guarded_fusion_gap10'))

if __name__=='__main__':main()
